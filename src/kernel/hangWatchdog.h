#ifndef EMULATOR_INCLUDE_EMULATOR_KERNEL_HANGWATCHDOG_H_
#define EMULATOR_INCLUDE_EMULATOR_KERNEL_HANGWATCHDOG_H_

#include <cstdint>
#include <functional>
#include <string>

// Diagnostic hang watchdog.
//
// When the guest stops making frame progress for a few seconds, a background thread writes a
// report to the console, the log, and "kyty_hang_report.txt" in the working directory. The report
// lists guest threads that are blocked in emulated kernel waits (with the guest call site), the
// state of the guest GPU queues, and, on Windows, a CPU-usage and instruction-pointer sample of
// every thread so busy-wait loops can be located.
namespace Kernel::HangWatchdog {

// Marks the current thread as blocked for the lifetime of the scope. Cheap enough to use on every
// blocking path: no allocation and no global lock.
class WaitScope final {
public:
	WaitScope(const char* kind, const char* name, uint64_t arg0 = 0, uint64_t arg1 = 0,
	          const void* guest_caller = nullptr) noexcept;
	~WaitScope();
	WaitScope(const WaitScope&)            = delete;
	WaitScope& operator=(const WaitScope&) = delete;

private:
	bool m_outer = false;
};

// Marks what the current thread is busy doing (not waiting), for example compiling a shader.
// Scopes nest; reports show the whole stack of activities and how long each has run.
class Activity final {
public:
	explicit Activity(const char* what, uint64_t arg0 = 0, uint64_t arg1 = 0) noexcept;
	~Activity();
	Activity(const Activity&)            = delete;
	Activity& operator=(const Activity&) = delete;

private:
	bool m_pushed = false;
};

// Appends a line to a diagnostics file in the working directory (thread safe).
void AppendDiagnostic(const char* file_name, const std::string& text);

// Records the current thread's guest name so reports can show it.
void SetCurrentThreadName(const char* name) noexcept;

// Called whenever the guest makes forward progress (frame boundaries, flips). Starts the watchdog
// thread on first use.
void NotifyProgress() noexcept;

// Called for every completed video-out flip; feeds the FPS log (kyty_perf_log.txt).
void NotifyFlip() noexcept;

// Adds a callback that appends subsystem state to a hang report. Called from the watchdog thread
// while no thread is suspended; it must not block indefinitely.
void AddReporter(const char* name, std::function<void(std::string&)> reporter);

} // namespace Kernel::HangWatchdog

#endif /* EMULATOR_INCLUDE_EMULATOR_KERNEL_HANGWATCHDOG_H_ */
