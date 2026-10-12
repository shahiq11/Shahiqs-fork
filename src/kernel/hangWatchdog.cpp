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
#include <condition_variable>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <fmt/format.h>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h>
#include <dbghelp.h>
#include <psapi.h>
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

	struct ActivityEntry {
		const char*          what = nullptr; // string literal
		uint64_t             arg0 = 0;
		uint64_t             arg1 = 0;
		std::atomic<int64_t> start_ns {0};
	};
	static constexpr uint32_t            MaxActivities = 8;
	std::array<ActivityEntry, MaxActivities> activities {};
	std::atomic_uint32_t                 activity_depth {0};
};

std::array<Slot, MaxSlots> g_slots;
std::atomic_uint32_t       g_slot_count {0};
std::atomic_uint64_t       g_progress {0};
std::atomic_uint64_t       g_flips {0};
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

Activity::Activity(const char* what, uint64_t arg0, uint64_t arg1) noexcept {
	auto* slot = CurrentSlot();
	if (slot == nullptr) {
		return;
	}
	const auto depth = slot->activity_depth.load(std::memory_order_relaxed);
	m_pushed         = true;
	if (depth < Slot::MaxActivities) {
		auto& entry = slot->activities[depth];
		entry.what  = what;
		entry.arg0  = arg0;
		entry.arg1  = arg1;
		entry.start_ns.store(NowNs() | 1, std::memory_order_relaxed);
	}
	slot->activity_depth.store(depth + 1, std::memory_order_release);
}

Activity::~Activity() {
	if (m_pushed && t_slot != nullptr) {
		const auto depth = t_slot->activity_depth.load(std::memory_order_relaxed);
		if (depth != 0) {
			t_slot->activity_depth.store(depth - 1, std::memory_order_release);
		}
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

void NotifyFlip() noexcept {
	g_flips.fetch_add(1, std::memory_order_relaxed);
	NotifyProgress();
}

void AppendDiagnostic(const char* file_name, const std::string& text) {
	static std::mutex mutex;
	std::lock_guard   lock(mutex);
	if (auto* file = std::fopen(file_name, "ab"); file != nullptr) {
		std::fwrite(text.data(), 1, text.size(), file);
		std::fclose(file);
	}
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

// Threads that have been inside the same marked activity (shader compile, GPU packet, ...) for a
// long time: these are busy, not waiting.
void ReportLongActivities(std::string& out, int64_t now) {
	std::string rows;
	uint32_t    found = 0;
	const auto  count = std::min(g_slot_count.load(std::memory_order_relaxed), MaxSlots);
	for (uint32_t i = 0; i < count; i++) {
		const auto& slot  = g_slots[i];
		const auto  depth = std::min(slot.activity_depth.load(std::memory_order_acquire),
		                             Slot::MaxActivities);
		if (depth == 0) {
			continue;
		}
		const auto outer_start = slot.activities[0].start_ns.load(std::memory_order_relaxed);
		if (outer_start == 0 || now - outer_start < MinBlockedWaitNs) {
			continue;
		}
		char thread_name[sizeof(slot.thread_name)];
		std::memcpy(thread_name, slot.thread_name, sizeof(thread_name));
		thread_name[sizeof(thread_name) - 1] = 0;
		found++;
		rows += fmt::format("  tid={} thread=\"{}\"\n", slot.host_tid.load(), thread_name);
		for (uint32_t d = 0; d < depth; d++) {
			const auto& e     = slot.activities[d];
			const auto  start = e.start_ns.load(std::memory_order_relaxed);
			rows += fmt::format("    [{}] {:7.1f}s in {:<28} arg0=0x{:x} arg1=0x{:x}\n", d,
			                    static_cast<double>(now - start) / 1e9,
			                    (e.what != nullptr ? e.what : "?"), e.arg0, e.arg1);
		}
	}
	out += fmt::format("Long-running activities (older than {} s): {}\n",
	                   MinBlockedWaitNs / 1000000000, found);
	out += rows;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

struct ModuleRange {
	uint64_t    begin = 0;
	uint64_t    end   = 0;
	std::string name;
};

// Uses only the loader's module list (no Toolhelp snapshot, which can block).
std::vector<ModuleRange> HostModules() {
	std::vector<ModuleRange> modules;
	std::vector<HMODULE>     handles(1024);
	DWORD                    needed = 0;
	if (K32EnumProcessModules(GetCurrentProcess(), handles.data(),
	                          static_cast<DWORD>(handles.size() * sizeof(HMODULE)), &needed) == 0) {
		return modules;
	}
	handles.resize(std::min<size_t>(handles.size(), needed / sizeof(HMODULE)));
	for (auto* handle: handles) {
		MODULEINFO info {};
		if (K32GetModuleInformation(GetCurrentProcess(), handle, &info, sizeof(info)) == 0) {
			continue;
		}
		char name[MAX_PATH] {};
		K32GetModuleBaseNameA(GetCurrentProcess(), handle, name, MAX_PATH);
		ModuleRange range;
		range.begin = reinterpret_cast<uint64_t>(info.lpBaseOfDll);
		range.end   = range.begin + info.SizeOfImage;
		range.name  = name;
		modules.push_back(std::move(range));
	}
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
		options(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES |
		        SYMOPT_NO_PROMPTS | SYMOPT_FAIL_CRITICAL_ERRORS);
		// Search only next to the executable: never a symbol server.
		wchar_t exe[MAX_PATH] {};
		GetModuleFileNameW(nullptr, exe, MAX_PATH);
		std::string dir;
		for (const wchar_t* p = exe; *p != 0; p++) {
			dir.push_back(*p < 128 ? static_cast<char>(*p) : '?');
		}
		if (auto pos = dir.find_last_of("\\/"); pos != std::string::npos) {
			dir.resize(pos);
		}
		m_ok = init(GetCurrentProcess(), dir.c_str(), TRUE) != 0;
	}

	std::string Name(uint64_t addr) {
		std::lock_guard lock(m_mutex);
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
			line.SizeOfStruct = sizeof(line);
			DWORD line_disp   = 0;
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
	HMODULE                         m_dbghelp   = nullptr;
	decltype(&SymFromAddr)          m_from_addr = nullptr;
	decltype(&SymGetLineFromAddr64) m_line      = nullptr;
	bool                            m_ok        = false;
	std::mutex                      m_mutex;
};

// DbgHelp must be initialized once per process and is not thread-safe.
Symbolizer& SharedSymbolizer() {
	static Symbolizer symbolizer;
	return symbolizer;
}

std::string DescribeAddress(uint64_t addr, const std::vector<ModuleRange>& modules,
                            Symbolizer* symbolizer) {
	if (const auto* m = FindModule(modules, addr); m != nullptr) {
		auto symbol = (symbolizer != nullptr ? symbolizer->Name(addr) : std::string {});
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

std::atomic<const char*> g_sampler_phase {"not started"};

// While a thread is suspended this only copies registers and raw stack memory; all analysis
// happens after every thread is resumed, so no lock held by a suspended thread is ever needed.
void ReportThreadSamples(std::string& out) {
	constexpr int    SampleRounds = 40;
	constexpr int    SampleGapMs  = 50;
	constexpr size_t StackBytes   = 48 * 1024;
	constexpr int    MaxFrames    = 32;

	struct ThreadInfo {
		DWORD                              tid    = 0;
		HANDLE                             handle = nullptr;
		uint64_t                           cpu0   = 0;
		uint64_t                           cpu1   = 0;
		std::array<uint64_t, SampleRounds> rips {};
		int                                samples = 0;
		uint64_t                           rip     = 0;
		uint64_t                           rsp     = 0;
		uint64_t                           rbp     = 0;
		std::vector<uint64_t>              stack;
		size_t                             stack_qwords = 0;
	};

	g_sampler_phase = "enumerating modules";
	const auto modules = HostModules();

	g_sampler_phase = "enumerating threads";
	std::vector<ThreadInfo> threads;
	HANDLE                  snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (snapshot == INVALID_HANDLE_VALUE) {
		out += "Thread sampling unavailable\n";
		return;
	}
	THREADENTRY32 entry {};
	entry.dwSize    = sizeof(entry);
	const auto pid  = GetCurrentProcessId();
	const auto self = GetCurrentThreadId();
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
		info.stack.resize(StackBytes / 8);
		threads.push_back(std::move(info));
	}
	CloseHandle(snapshot);

	g_sampler_phase = "sampling instruction pointers";
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

	g_sampler_phase = "copying stacks";
	for (auto& t: threads) {
		if (SuspendThread(t.handle) == static_cast<DWORD>(-1)) {
			continue;
		}
		CONTEXT ctx {};
		ctx.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
		if (GetThreadContext(t.handle, &ctx) != 0) {
			t.rip = ctx.Rip;
			t.rsp = ctx.Rsp;
			t.rbp = ctx.Rbp;
			// Copy as much of the stack as is readable, shrinking at the top of the stack.
			for (size_t bytes = StackBytes; bytes >= 4096; bytes /= 2) {
				SIZE_T read = 0;
				if (ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(t.rsp),
				                      t.stack.data(), bytes, &read) != 0) {
					t.stack_qwords = read / 8;
					break;
				}
			}
		}
		ResumeThread(t.handle);
	}

	// Everything below runs with all threads resumed.
	g_sampler_phase = "loading symbols";
	Symbolizer& symbolizer = SharedSymbolizer();
	g_sampler_phase = "formatting";
	const auto guest_names = GuestThreadNames();
	std::sort(threads.begin(), threads.end(), [](const ThreadInfo& a, const ThreadInfo& b) {
		return (a.cpu1 - a.cpu0) > (b.cpu1 - b.cpu0);
	});

	auto is_code = [&](uint64_t v) {
		if (v < 0x10000) {
			return false;
		}
		if (FindModule(modules, v) != nullptr) {
			return true;
		}
		bool  busy = false;
		auto* rt   = Common::Singleton<Loader::RuntimeLinker>::Instance();
		return rt->TryFindProgramByAddr(v, &busy) != nullptr;
	};

	out += fmt::format("Threads: {} (sorted by CPU use over {:.1f} s)\n", threads.size(),
	                   wall_100ns / 1e7);
	for (auto& t: threads) {
		const double cpu = wall_100ns > 0
		                       ? 100.0 * static_cast<double>(t.cpu1 - t.cpu0) / wall_100ns
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
		const bool busy_thread = cpu >= 5.0;
		if (busy_thread) {
			for (size_t i = 0; i < std::min<size_t>(top.size(), 8); i++) {
				out += fmt::format("     rip {:>2}/{}  {}\n", top[i].first, t.samples,
				                   DescribeAddress(top[i].second, modules, &symbolizer));
			}
		}

		// Frame-pointer chain over the copied stack (host code is built with /Oy-).
		const uint64_t stack_lo = t.rsp;
		const uint64_t stack_hi = t.rsp + t.stack_qwords * 8;
		auto read_stack = [&](uint64_t addr, uint64_t* value) {
			if (addr < stack_lo || addr + 8 > stack_hi || (addr & 7u) != 0) {
				return false;
			}
			*value = t.stack[(addr - stack_lo) / 8];
			return true;
		};
		out += fmt::format("     #0  {}\n", DescribeAddress(t.rip, modules, &symbolizer));
		uint64_t fp = t.rbp;
		for (int i = 1; i < MaxFrames; i++) {
			uint64_t next_fp = 0;
			uint64_t ret     = 0;
			if (!read_stack(fp, &next_fp) || !read_stack(fp + 8, &ret) || ret == 0) {
				break;
			}
			out += fmt::format("     #{:<2} {}\n", i, DescribeAddress(ret, modules, &symbolizer));
			if (next_fp <= fp) {
				break;
			}
			fp = next_fp;
		}
		// For busy threads also list code addresses found anywhere on the top of the stack, in
		// case the frame chain is broken by code built without frame pointers.
		if (busy_thread) {
			int listed = 0;
			for (size_t i = 0; i < std::min<size_t>(t.stack_qwords, 2048) && listed < 24; i++) {
				const auto value = t.stack[i];
				if (is_code(value)) {
					out += fmt::format("     scan +0x{:<5x} {}\n", i * 8,
					                   DescribeAddress(value, modules, &symbolizer));
					listed++;
				}
			}
		}
		CloseHandle(t.handle);
	}
	g_sampler_phase = "done";
}


// Periodic sampling profile of busy threads, written to kyty_profile.txt. Shows where frame time
// goes when the game is running (not stalled): self time per function and the hottest stacks.
void ProfileBusyThreads(std::string& out, int duration_ms) {
	constexpr int GapMs     = 2;
	constexpr int MaxFrames = 16;

	struct Sample {
		std::array<uint64_t, MaxFrames> frames {};
		int                             count = 0;
	};
	struct ThreadInfo {
		DWORD               tid    = 0;
		HANDLE              handle = nullptr;
		uint64_t            cpu0   = 0;
		uint64_t            cpu1   = 0;
		std::vector<Sample> samples;
	};

	const auto modules = HostModules();
	std::vector<ThreadInfo> threads;
	HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (snapshot == INVALID_HANDLE_VALUE) {
		return;
	}
	THREADENTRY32 entry {};
	entry.dwSize    = sizeof(entry);
	const auto pid  = GetCurrentProcessId();
	const auto self = GetCurrentThreadId();
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
		info.samples.reserve(static_cast<size_t>(duration_ms / GapMs) + 16);
		threads.push_back(std::move(info));
	}
	CloseHandle(snapshot);

	// Pre-pass: only sample threads that actually use CPU (keeps the overhead low).
	for (auto& t: threads) {
		t.cpu0 = ThreadCpu100ns(t.handle);
	}
	Sleep(200);
	std::vector<ThreadInfo*> busy;
	for (auto& t: threads) {
		if (ThreadCpu100ns(t.handle) - t.cpu0 >= 100000) { // >= 10 ms of CPU in 200 ms
			busy.push_back(&t);
		}
	}
	for (auto* t: busy) {
		t->cpu0 = ThreadCpu100ns(t->handle);
	}
	const auto wall0 = NowNs();
	while ((NowNs() - wall0) / 1000000 < duration_ms) {
		for (auto* t: busy) {
			if (SuspendThread(t->handle) == static_cast<DWORD>(-1)) {
				continue;
			}
			CONTEXT ctx {};
			ctx.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
			if (GetThreadContext(t->handle, &ctx) != 0) {
				Sample sample;
				sample.frames[sample.count++] = ctx.Rip;
				uint64_t fp = ctx.Rbp;
				while (sample.count < MaxFrames && fp >= ctx.Rsp && fp - ctx.Rsp < (8ull << 20u)) {
					uint64_t pair[2] {};
					SIZE_T   read = 0;
					if (ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(fp),
					                      pair, sizeof(pair), &read) == 0 ||
					    read != sizeof(pair) || pair[1] == 0 || pair[0] <= fp) {
						break;
					}
					sample.frames[sample.count++] = pair[1];
					fp                            = pair[0];
				}
				t->samples.push_back(sample);
			}
			ResumeThread(t->handle);
		}
		Sleep(GapMs);
	}
	const double wall_100ns = static_cast<double>(NowNs() - wall0) / 100.0;
	for (auto* t: busy) {
		t->cpu1 = ThreadCpu100ns(t->handle);
	}

	Symbolizer& symbolizer = SharedSymbolizer();
	const auto guest_names = GuestThreadNames();
	std::unordered_map<uint64_t, std::string> names;
	auto function_of = [&](uint64_t addr) -> const std::string& {
		auto it = names.find(addr);
		if (it != names.end()) {
			return it->second;
		}
		std::string name;
		if (const auto* m = FindModule(modules, addr); m != nullptr) {
			name = symbolizer.Name(addr);
			if (auto plus = name.rfind("+0x"); plus != std::string::npos) {
				name.resize(plus);
			}
			if (auto bracket = name.find(" ["); bracket != std::string::npos) {
				name.resize(bracket);
			}
			name = name.empty() ? fmt::format("{}+0x{:x}", m->name, (addr - m->begin) & ~0xffull)
			                    : m->name.substr(0, m->name.find('.')) + "!" + name;
		} else {
			// Guest code: bucket by 256 bytes.
			name = DescribeGuestAddress(addr & ~0xffull);
		}
		return names.emplace(addr, std::move(name)).first->second;
	};

	std::sort(busy.begin(), busy.end(), [](const ThreadInfo* a, const ThreadInfo* b) {
		return (a->cpu1 - a->cpu0) > (b->cpu1 - b->cpu0);
	});
	for (auto* t: busy) {
		const double cpu = 100.0 * static_cast<double>(t->cpu1 - t->cpu0) / wall_100ns;
		std::string guest_name;
		if (auto it = guest_names.find(t->tid); it != guest_names.end()) {
			guest_name = it->second;
		}
		out += fmt::format("== tid={} cpu={:.0f}% thread=\"{}\" samples={}\n", t->tid, cpu,
		                   guest_name, t->samples.size());
		if (t->samples.empty()) {
			continue;
		}
		std::unordered_map<std::string, int> self_counts;
		std::unordered_map<std::string, int> total_counts;
		std::unordered_map<std::string, int> stack_counts;
		for (const auto& sample: t->samples) {
			self_counts[function_of(sample.frames[0])]++;
			std::vector<const std::string*> seen;
			std::string                     stack;
			for (int i = 0; i < sample.count; i++) {
				const auto& name = function_of(sample.frames[i]);
				if (std::find(seen.begin(), seen.end(), &name) == seen.end()) {
					seen.push_back(&name);
					total_counts[name]++;
				}
				if (i < 6) {
					stack += (i == 0 ? "" : " <- ") + name;
				}
			}
			stack_counts[stack]++;
		}
		auto print_top = [&](const char* title, std::unordered_map<std::string, int>& counts,
		                     size_t limit) {
			std::vector<std::pair<int, std::string>> sorted;
			for (auto& [name, n]: counts) {
				sorted.emplace_back(n, name);
			}
			std::sort(sorted.rbegin(), sorted.rend());
			out += fmt::format("  {}:\n", title);
			for (size_t i = 0; i < std::min(limit, sorted.size()); i++) {
				out += fmt::format("    {:5.1f}%  {}\n",
				                   100.0 * sorted[i].first / static_cast<double>(t->samples.size()),
				                   sorted[i].second);
			}
		};
		print_top("self", self_counts, 15);
		print_top("inclusive", total_counts, 25);
		print_top("stacks", stack_counts, 8);
	}
	for (auto& t: threads) {
		CloseHandle(t.handle);
	}
}

// Runs the sampler on its own thread so a problem inside it cannot stop the report.
void ReportThreadSamplesBounded(std::string& out) {
	struct Shared {
		std::mutex              mutex;
		std::condition_variable cv;
		bool                    done = false;
		std::string             text;
	};
	auto shared = std::make_shared<Shared>();
	std::thread([shared] {
		std::string text;
		ReportThreadSamples(text);
		std::lock_guard lock(shared->mutex);
		shared->text = std::move(text);
		shared->done = true;
		shared->cv.notify_all();
	}).detach();
	std::unique_lock lock(shared->mutex);
	if (shared->cv.wait_for(lock, std::chrono::seconds(30), [&] { return shared->done; })) {
		out += shared->text;
	} else {
		out += fmt::format("Thread sampler did not finish within 30 s (stuck while: {})\n",
		                   g_sampler_phase.load());
	}
}

#else

void ReportThreadSamplesBounded(std::string& out) {
	out += "Thread sampling is only implemented on Windows\n";
}

void ProfileBusyThreads(std::string& out, int /*duration_ms*/) {
	out += "Profiling is only implemented on Windows\n";
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
	const int64_t start_ns    = NowNs();
	int64_t  next_profile_ns  = 40'000'000'000;
	int64_t  perf_ns          = start_ns;
	uint64_t perf_flips       = g_flips.load(std::memory_order_relaxed);
	uint64_t perf_progress    = last_progress;

	for (;;) {
		std::this_thread::sleep_for(std::chrono::milliseconds(500));
		const auto now      = NowNs();
		if (now - perf_ns >= 5'000'000'000) {
			const auto flips    = g_flips.load(std::memory_order_relaxed);
			const auto progress = g_progress.load(std::memory_order_relaxed);
			const auto seconds  = static_cast<double>(now - perf_ns) / 1e9;
			AppendDiagnostic("kyty_perf_log.txt",
			                 fmt::format("t={:7.1f}s flips/s={:6.1f} progress/s={:6.1f} total_flips={}\n",
			                             static_cast<double>(now - start_ns) / 1e9,
			                             static_cast<double>(flips - perf_flips) / seconds,
			                             static_cast<double>(progress - perf_progress) / seconds,
			                             flips));
			perf_ns       = now;
			perf_flips    = flips;
			perf_progress = progress;
		}
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
		if (stalled < StallReportNs && now - start_ns >= next_profile_ns) {
			next_profile_ns += 60'000'000'000;
			std::string profile = fmt::format("\n=== profile at t={:.0f}s ===\n",
			                                  static_cast<double>(now - start_ns) / 1e9);
			ProfileBusyThreads(profile, 3000);
			AppendDiagnostic("kyty_profile.txt", profile);
			continue;
		}
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
		ReportLongActivities(report, now);
		WriteReport(report);
		{
			std::lock_guard lock(g_reporters_mutex);
			for (auto& [name, reporter]: g_reporters) {
				report = fmt::format("--- {} ---\n", name);
				reporter(report);
				WriteReport(report);
			}
		}
		WriteReport("--- thread samples (takes a few seconds) ---\n");
		report.clear();
		ReportThreadSamplesBounded(report);
		report += "=== end of hang report ===\n";
		WriteReport(report);
	}
}

} // namespace

} // namespace Kernel::HangWatchdog
