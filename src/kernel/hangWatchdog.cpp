#include "kernel/hangWatchdog.h"

#include "common/common.h"
#include "common/logging/log.h"
#include "common/singleton.h"
#include "common/stringUtils.h"
#include "loader/runtimeLinker.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <fmt/format.h>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h>
#include <dbghelp.h>
#include <tlhelp32.h>
#else
#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Kernel::HangWatchdog {

namespace {

constexpr uint32_t MaxSlots         = 1024;
constexpr int64_t  StallReportNs    = 5'000'000'000;  // first report after 5 s without progress
constexpr int64_t  RepeatReportNs   = 30'000'000'000; // then every 30 s
constexpr uint32_t MaxReports       = 4;
constexpr int64_t  MinBlockedWaitNs = 2'000'000'000;

struct Slot {
	std::atomic<int64_t>  start_ns {0}; // 0 = not waiting; published last with release
	std::atomic<uint64_t> host_tid {0};
	const char*           kind   = nullptr; // string literal
	uint64_t              arg0   = 0;
	uint64_t              arg1   = 0;
	uint64_t              caller = 0;
	char                  name[48] {};
	char                  thread_name[32] {};
};

std::array<Slot, MaxSlots> g_slots;
std::atomic_uint32_t       g_slot_count {0};
std::atomic_uint64_t       g_progress {0};
std::once_flag             g_start_once;

std::mutex                                                     g_reporters_mutex;
std::vector<std::pair<std::string, std::function<void(std::string&)>>> g_reporters;

thread_local Slot*    t_slot  = nullptr;
thread_local uint32_t t_depth = 0;

int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

uint64_t CurrentHostThreadId() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return GetCurrentThreadId();
#elif defined(__APPLE__)
	uint64_t id = 0;
	pthread_threadid_np(nullptr, &id);
	return id;
#else
	return static_cast<uint64_t>(syscall(SYS_gettid));
#endif
}

Slot* CurrentSlot() {
	if (t_slot == nullptr) {
		const auto index = g_slot_count.fetch_add(1, std::memory_order_relaxed);
		if (index >= MaxSlots) {
			return nullptr;
		}
		t_slot = &g_slots[index];
		t_slot->host_tid.store(CurrentHostThreadId(), std::memory_order_relaxed);
	}
	return t_slot;
}

void CopyName(char* dst, size_t size, const char* src) {
	if (src == nullptr) {
		dst[0] = 0;
		return;
	}
	size_t i = 0;
	for (; i + 1 < size && src[i] != 0; i++) {
		dst[i] = src[i];
	}
	dst[i] = 0;
}

void WatchdogThread();

} // namespace

WaitScope::WaitScope(const char* kind, const char* name, uint64_t arg0, uint64_t arg1,
                     const void* guest_caller) noexcept {
	if (t_depth++ != 0) {
		return;
	}
	auto* slot = CurrentSlot();
	if (slot == nullptr) {
		return;
	}
	m_outer      = true;
	slot->kind   = kind;
	slot->arg0   = arg0;
	slot->arg1   = arg1;
	slot->caller = reinterpret_cast<uint64_t>(guest_caller);
	CopyName(slot->name, sizeof(slot->name), name);
	slot->start_ns.store(NowNs() | 1, std::memory_order_release);
}

WaitScope::~WaitScope() {
	t_depth--;
	if (m_outer && t_slot != nullptr) {
		t_slot->start_ns.store(0, std::memory_order_release);
	}
}

void SetCurrentThreadName(const char* name) noexcept {
	auto* slot = CurrentSlot();
	if (slot != nullptr) {
		CopyName(slot->thread_name, sizeof(slot->thread_name), name);
	}
}

void NotifyProgress() noexcept {
	g_progress.fetch_add(1, std::memory_order_relaxed);
	std::call_once(g_start_once, [] { std::thread(WatchdogThread).detach(); });
}

void AddReporter(const char* name, std::function<void(std::string&)> reporter) {
	std::lock_guard lock(g_reporters_mutex);
	g_reporters.emplace_back(name, std::move(reporter));
}

namespace {

[[maybe_unused]] std::map<uint64_t, std::string> GuestThreadNames() {
	std::map<uint64_t, std::string> names;
	const auto count = std::min(g_slot_count.load(std::memory_order_relaxed), MaxSlots);
	for (uint32_t i = 0; i < count; i++) {
		const auto& slot = g_slots[i];
		if (slot.thread_name[0] != 0) {
			char copy[sizeof(slot.thread_name)];
			std::memcpy(copy, slot.thread_name, sizeof(copy));
			copy[sizeof(copy) - 1] = 0;
			names[slot.host_tid.load(std::memory_order_relaxed)] = copy;
		}
	}
	return names;
}

std::string DescribeGuestAddress(uint64_t addr) {
	if (addr == 0) {
		return "0";
	}
	auto* rt   = Common::Singleton<Loader::RuntimeLinker>::Instance();
	bool  busy = false;
	if (auto* program = rt->TryFindProgramByAddr(addr, &busy); program != nullptr) {
		return fmt::format("{}+0x{:x}",
		                   Common::FilenameWithoutDirectory(
		                       Common::PathToGenericString(program->file_name)),
		                   addr - program->base_vaddr);
	}
	return fmt::format("0x{:016x}{}", addr, busy ? " (linker busy)" : "");
}

void ReportBlockedWaits(std::string& out, int64_t now) {
	struct Row {
		uint64_t    tid;
		double      age_s;
		std::string text;
	};
	std::vector<Row> rows;
	const auto       count = std::min(g_slot_count.load(std::memory_order_relaxed), MaxSlots);
	for (uint32_t i = 0; i < count; i++) {
		const auto& slot = g_slots[i];
		for (int attempt = 0; attempt < 3; attempt++) {
			const auto start = slot.start_ns.load(std::memory_order_acquire);
			if (start == 0 || now - start < MinBlockedWaitNs) {
				break;
			}
			char name[sizeof(slot.name)];
			char thread_name[sizeof(slot.thread_name)];
			std::memcpy(name, slot.name, sizeof(name));
			std::memcpy(thread_name, slot.thread_name, sizeof(thread_name));
			name[sizeof(name) - 1]               = 0;
			thread_name[sizeof(thread_name) - 1] = 0;
			const auto* kind   = slot.kind;
			const auto  arg0   = slot.arg0;
			const auto  arg1   = slot.arg1;
			const auto  caller = slot.caller;
			if (slot.start_ns.load(std::memory_order_acquire) != start) {
				continue;
			}
			Row row;
			row.tid   = slot.host_tid.load(std::memory_order_relaxed);
			row.age_s = static_cast<double>(now - start) / 1e9;
			row.text  = fmt::format(
                "  tid={:<6} {:7.1f}s  {:<14} name=\"{}\" arg0=0x{:x} arg1=0x{:x} caller={} "
                "thread=\"{}\"\n",
                row.tid, row.age_s, (kind != nullptr ? kind : "?"), name, arg0, arg1,
                DescribeGuestAddress(caller), thread_name);
			rows.push_back(std::move(row));
			break;
		}
	}
	std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.age_s > b.age_s; });
	out += fmt::format("Blocked waits (older than {} s): {}\n", MinBlockedWaitNs / 1000000000,
	                   rows.size());
	for (const auto& row: rows) {
		out += row.text;
	}
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

struct ModuleRange {
	uint64_t    begin = 0;
	uint64_t    end   = 0;
	std::string name;
};

std::vector<ModuleRange> HostModules() {
	std::vector<ModuleRange> modules;
	HANDLE                   snapshot =
	    CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, GetCurrentProcessId());
	if (snapshot == INVALID_HANDLE_VALUE) {
		return modules;
	}
	MODULEENTRY32W entry {};
	entry.dwSize = sizeof(entry);
	for (BOOL ok = Module32FirstW(snapshot, &entry); ok; ok = Module32NextW(snapshot, &entry)) {
		ModuleRange range;
		range.begin = reinterpret_cast<uint64_t>(entry.modBaseAddr);
		range.end   = range.begin + entry.modBaseSize;
		for (const wchar_t* p = entry.szModule; *p != 0; p++) {
			range.name.push_back(*p < 128 ? static_cast<char>(*p) : '?');
		}
		modules.push_back(std::move(range));
	}
	CloseHandle(snapshot);
	return modules;
}

const ModuleRange* FindModule(const std::vector<ModuleRange>& modules, uint64_t addr) {
	for (const auto& m: modules) {
		if (addr >= m.begin && addr < m.end) {
			return &m;
		}
	}
	return nullptr;
}

bool ReadMemory(uint64_t addr, void* dst, size_t size) {
	SIZE_T read = 0;
	return ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(addr), dst, size,
	                         &read) != 0 &&
	       read == size;
}

// Plain data only: called while the target thread is suspended, so it must not allocate.
int WalkStack(CONTEXT ctx, uint64_t* frames, int max_frames, const ModuleRange* modules,
              size_t module_count) {
	int n = 0;
	for (int i = 0; i < max_frames; i++) {
		const uint64_t rip = ctx.Rip;
		if (rip == 0) {
			break;
		}
		frames[n++] = rip;
		bool host   = false;
		for (size_t m = 0; m < module_count; m++) {
			if (rip >= modules[m].begin && rip < modules[m].end) {
				host = true;
				break;
			}
		}
		if (host) {
			DWORD64 image_base = 0;
			auto*   function   = RtlLookupFunctionEntry(rip, &image_base, nullptr);
			if (function != nullptr) {
				PVOID   handler_data = nullptr;
				DWORD64 frame        = 0;
				RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, rip, function, &ctx,
				                 &handler_data, &frame, nullptr);
			} else {
				uint64_t ret = 0;
				if (!ReadMemory(ctx.Rsp, &ret, sizeof(ret))) {
					break;
				}
				ctx.Rip = ret;
				ctx.Rsp += 8;
			}
		} else {
			// Guest code: follow the frame-pointer chain.
			const uint64_t fp = ctx.Rbp;
			uint64_t       saved[2] {};
			if (fp < ctx.Rsp || fp - ctx.Rsp > (64ull << 20u) || !ReadMemory(fp, saved, 16)) {
				break;
			}
			ctx.Rbp = saved[0];
			ctx.Rip = saved[1];
			ctx.Rsp = fp + 16;
		}
	}
	return n;
}

int WalkStackSafe(const CONTEXT& ctx, uint64_t* frames, int max_frames,
                  const ModuleRange* modules, size_t module_count) {
	__try {
		return WalkStack(ctx, frames, max_frames, modules, module_count);
	} __except (EXCEPTION_EXECUTE_HANDLER) {
		return 0;
	}
}

class Symbolizer {
public:
	Symbolizer() {
		m_dbghelp = LoadLibraryW(L"dbghelp.dll");
		if (m_dbghelp == nullptr) {
			return;
		}
		auto init = reinterpret_cast<decltype(&SymInitialize)>(
		    reinterpret_cast<void*>(GetProcAddress(m_dbghelp, "SymInitialize")));
		auto options = reinterpret_cast<decltype(&SymSetOptions)>(
		    reinterpret_cast<void*>(GetProcAddress(m_dbghelp, "SymSetOptions")));
		m_from_addr = reinterpret_cast<decltype(&SymFromAddr)>(
		    reinterpret_cast<void*>(GetProcAddress(m_dbghelp, "SymFromAddr")));
		m_line = reinterpret_cast<decltype(&SymGetLineFromAddr64)>(
		    reinterpret_cast<void*>(GetProcAddress(m_dbghelp, "SymGetLineFromAddr64")));
		if (init == nullptr || options == nullptr || m_from_addr == nullptr) {
			return;
		}
		options(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
		m_ok = init(GetCurrentProcess(), nullptr, TRUE) != 0;
	}

	std::string Name(uint64_t addr) {
		if (!m_ok) {
			return {};
		}
		alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + 512] {};
		auto*                     symbol = reinterpret_cast<SYMBOL_INFO*>(buffer);
		symbol->SizeOfStruct             = sizeof(SYMBOL_INFO);
		symbol->MaxNameLen               = 511;
		DWORD64 displacement             = 0;
		if (m_from_addr(GetCurrentProcess(), addr, &displacement, symbol) == 0) {
			return {};
		}
		std::string result = fmt::format("{}+0x{:x}", symbol->Name, displacement);
		if (m_line != nullptr) {
			IMAGEHLP_LINE64 line {};
			line.SizeOfStruct  = sizeof(line);
			DWORD line_disp    = 0;
			if (m_line(GetCurrentProcess(), addr, &line_disp, &line) != 0 &&
			    line.FileName != nullptr) {
				result += fmt::format(" [{}:{}]",
				                      Common::FilenameWithoutDirectory(std::string(line.FileName)),
				                      line.LineNumber);
			}
		}
		return result;
	}

private:
	HMODULE                                m_dbghelp   = nullptr;
	decltype(&SymFromAddr)                 m_from_addr = nullptr;
	decltype(&SymGetLineFromAddr64)        m_line      = nullptr;
	bool                                   m_ok        = false;
};

std::string DescribeAddress(uint64_t addr, const std::vector<ModuleRange>& modules,
                            Symbolizer& symbolizer) {
	if (const auto* m = FindModule(modules, addr); m != nullptr) {
		auto symbol = symbolizer.Name(addr);
		return fmt::format("{}+0x{:x}{}{}", m->name, addr - m->begin, symbol.empty() ? "" : " ",
		                   symbol);
	}
	return DescribeGuestAddress(addr);
}

std::string ThreadDescription(HANDLE thread) {
	using get_description_t = HRESULT(WINAPI*)(HANDLE, PWSTR*);
	static auto get_description = reinterpret_cast<get_description_t>(reinterpret_cast<void*>(
	    GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "GetThreadDescription")));
	if (get_description == nullptr) {
		return {};
	}
	PWSTR wide = nullptr;
	if (FAILED(get_description(thread, &wide)) || wide == nullptr) {
		return {};
	}
	std::string result;
	for (const wchar_t* p = wide; *p != 0 && result.size() < 48; p++) {
		result.push_back(*p < 128 ? static_cast<char>(*p) : '?');
	}
	LocalFree(wide);
	return result;
}

uint64_t ThreadCpu100ns(HANDLE thread) {
	FILETIME creation {};
	FILETIME exit {};
	FILETIME kernel {};
	FILETIME user {};
	if (GetThreadTimes(thread, &creation, &exit, &kernel, &user) == 0) {
		return 0;
	}
	const auto k = (static_cast<uint64_t>(kernel.dwHighDateTime) << 32u) | kernel.dwLowDateTime;
	const auto u = (static_cast<uint64_t>(user.dwHighDateTime) << 32u) | user.dwLowDateTime;
	return k + u;
}

void ReportThreadSamples(std::string& out) {
	constexpr int SampleRounds = 40;
	constexpr int SampleGapMs  = 50;
	constexpr int MaxFrames    = 24;

	struct ThreadInfo {
		DWORD                             tid    = 0;
		HANDLE                            handle = nullptr;
		uint64_t                          cpu0   = 0;
		uint64_t                          cpu1   = 0;
		std::array<uint64_t, SampleRounds> rips {};
		int                               samples = 0;
		std::array<uint64_t, MaxFrames>   frames {};
		int                               frame_count = 0;
	};

	const auto modules = HostModules();

	std::vector<ThreadInfo> threads;
	HANDLE                  snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (snapshot == INVALID_HANDLE_VALUE) {
		out += "Thread sampling unavailable\n";
		return;
	}
	THREADENTRY32 entry {};
	entry.dwSize      = sizeof(entry);
	const auto pid    = GetCurrentProcessId();
	const auto self   = GetCurrentThreadId();
	for (BOOL ok = Thread32First(snapshot, &entry); ok; ok = Thread32Next(snapshot, &entry)) {
		if (entry.th32OwnerProcessID != pid || entry.th32ThreadID == self) {
			continue;
		}
		HANDLE handle = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT |
		                               THREAD_QUERY_LIMITED_INFORMATION | THREAD_QUERY_INFORMATION,
		                           FALSE, entry.th32ThreadID);
		if (handle == nullptr) {
			continue;
		}
		ThreadInfo info;
		info.tid    = entry.th32ThreadID;
		info.handle = handle;
		threads.push_back(info);
	}
	CloseHandle(snapshot);

	for (auto& t: threads) {
		t.cpu0 = ThreadCpu100ns(t.handle);
	}
	const auto wall0 = NowNs();
	for (int round = 0; round < SampleRounds; round++) {
		for (auto& t: threads) {
			if (SuspendThread(t.handle) == static_cast<DWORD>(-1)) {
				continue;
			}
			CONTEXT ctx {};
			ctx.ContextFlags = CONTEXT_CONTROL;
			if (GetThreadContext(t.handle, &ctx) != 0) {
				t.rips[t.samples++] = ctx.Rip;
			}
			ResumeThread(t.handle);
		}
		Sleep(SampleGapMs);
	}
	const auto wall1 = NowNs();
	for (auto& t: threads) {
		t.cpu1 = ThreadCpu100ns(t.handle);
	}
	const double wall_100ns = static_cast<double>(wall1 - wall0) / 100.0;

	// One full stack per thread that is busy or blocked; walking happens while suspended.
	for (auto& t: threads) {
		if (SuspendThread(t.handle) == static_cast<DWORD>(-1)) {
			continue;
		}
		CONTEXT ctx {};
		ctx.ContextFlags = CONTEXT_FULL;
		if (GetThreadContext(t.handle, &ctx) != 0) {
			t.frame_count =
			    WalkStackSafe(ctx, t.frames.data(), MaxFrames, modules.data(), modules.size());
		}
		ResumeThread(t.handle);
	}

	// Everything below runs with all threads resumed.
	Symbolizer symbolizer;
	const auto guest_names = GuestThreadNames();
	std::sort(threads.begin(), threads.end(), [](const ThreadInfo& a, const ThreadInfo& b) {
		return (a.cpu1 - a.cpu0) > (b.cpu1 - b.cpu0);
	});

	out += fmt::format("Threads: {} (sorted by CPU use over {:.1f} s)\n", threads.size(),
	                   wall_100ns / 1e7);
	for (auto& t: threads) {
		const double cpu = wall_100ns > 0 ? 100.0 * static_cast<double>(t.cpu1 - t.cpu0) / wall_100ns
		                                  : 0.0;
		std::string guest_name;
		if (auto it = guest_names.find(t.tid); it != guest_names.end()) {
			guest_name = it->second;
		}
		out += fmt::format("== tid={} cpu={:.0f}% desc=\"{}\" guest=\"{}\"\n", t.tid, cpu,
		                   ThreadDescription(t.handle), guest_name);

		std::map<uint64_t, int> histogram;
		for (int i = 0; i < t.samples; i++) {
			histogram[t.rips[i]]++;
		}
		std::vector<std::pair<int, uint64_t>> top;
		for (const auto& [rip, n]: histogram) {
			top.emplace_back(n, rip);
		}
		std::sort(top.rbegin(), top.rend());
		if (cpu >= 5.0) {
			for (size_t i = 0; i < std::min<size_t>(top.size(), 6); i++) {
				out += fmt::format("     rip {:>2}/{}  {}\n", top[i].first, t.samples,
				                   DescribeAddress(top[i].second, modules, symbolizer));
			}
		}
		for (int i = 0; i < t.frame_count; i++) {
			out += fmt::format("     #{:<2} {}\n", i,
			                   DescribeAddress(t.frames[i], modules, symbolizer));
		}
		CloseHandle(t.handle);
	}
}

#else

void ReportThreadSamples(std::string& out) {
	out += "Thread sampling is only implemented on Windows\n";
}

#endif

void WriteReport(const std::string& report) {
	std::fwrite(report.data(), 1, report.size(), stdout);
	std::fflush(stdout);
	LOGF("%s", report.c_str());
	if (auto* file = std::fopen("kyty_hang_report.txt", "ab"); file != nullptr) {
		std::fwrite(report.data(), 1, report.size(), file);
		std::fclose(file);
	}
}

void WatchdogThread() {
	uint64_t last_progress    = g_progress.load(std::memory_order_relaxed);
	int64_t  last_change_ns   = NowNs();
	int64_t  last_report_ns   = 0;
	uint32_t reports_in_stall = 0;
	uint32_t total_reports    = 0;

	for (;;) {
		std::this_thread::sleep_for(std::chrono::milliseconds(500));
		const auto now      = NowNs();
		const auto progress = g_progress.load(std::memory_order_relaxed);
		if (progress != last_progress) {
			if (reports_in_stall != 0) {
				WriteReport(fmt::format("=== Kyty hang watchdog: progress resumed after {:.1f} s\n",
				                        static_cast<double>(now - last_change_ns) / 1e9));
			}
			last_progress    = progress;
			last_change_ns   = now;
			reports_in_stall = 0;
			continue;
		}
		const auto stalled = now - last_change_ns;
		if (stalled < StallReportNs || total_reports >= MaxReports * 4 ||
		    reports_in_stall >= MaxReports ||
		    (reports_in_stall != 0 && now - last_report_ns < RepeatReportNs)) {
			continue;
		}
		reports_in_stall++;
		total_reports++;
		last_report_ns = now;

		std::string report = fmt::format(
		    "\n=== Kyty hang watchdog report #{}: no frame progress for {:.1f} s (progress "
		    "counter {}) ===\n",
		    total_reports, static_cast<double>(stalled) / 1e9, progress);
		// Write each section as soon as it is ready, so a section that gets stuck cannot hide
		// the ones before it.
		ReportBlockedWaits(report, now);
		WriteReport(report);
		{
			std::lock_guard lock(g_reporters_mutex);
			for (auto& [name, reporter]: g_reporters) {
				report = fmt::format("--- {} ---\n", name);
				reporter(report);
				WriteReport(report);
			}
		}
		report.clear();
		ReportThreadSamples(report);
		report += "=== end of hang report ===\n";
		WriteReport(report);
	}
}

} // namespace

} // namespace Kernel::HangWatchdog
