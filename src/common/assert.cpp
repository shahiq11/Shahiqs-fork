#include "common/assert.h"
#include "common/recoverable.h"

#include "common/logging/log.h"
#include "common/subsystems.h"
#include "kytyGitVersion.h"

#include <csetjmp>
#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <string>

namespace Common {

namespace {

// Builds with C++ exceptions (the Windows /EHsc build) unwind recoverable errors with an
// exception, so destructors run and Windows' stack unwinder is never asked to longjmp through
// frames with cleanups. Builds without exceptions jump back with longjmp.
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
constexpr bool RecoverWithExceptions = true;
#else
constexpr bool RecoverWithExceptions = false;
#endif

struct RecoveryPoint {
	std::jmp_buf   buffer {};
	std::string    message;
	RecoveryPoint* previous = nullptr;
};

struct RecoverableError {
	std::string message;
};

thread_local RecoveryPoint* t_recovery = nullptr;

// Reports a fatal error raised inside RunRecoverable and returns to it.
[[noreturn]] void Recover(const char* title, std::string_view text, const char* file, int line) {
	auto* point   = t_recovery;
	auto  message = fmt::format("{} {} ({}:{})", title, text, file, line);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
	(void)point;
	throw RecoverableError {std::move(message)};
#else
	t_recovery     = point->previous;
	point->message = std::move(message);
	std::longjmp(point->buffer, 1);
#endif
}

} // namespace

bool RunRecoverable(void (*function)(void*), void* context, std::string* error) {
	RecoveryPoint point;
	point.previous = t_recovery;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
	static_assert(RecoverWithExceptions);
	t_recovery = &point;
	try {
		function(context);
	} catch (RecoverableError& recovered) {
		t_recovery = point.previous;
		if (error != nullptr) *error = std::move(recovered.message);
		return false;
	}
	t_recovery = point.previous;
	return true;
#else
	static_assert(!RecoverWithExceptions);
	if (setjmp(point.buffer) != 0) {
		// Recover() already restored the enclosing recovery point.
		if (error != nullptr) *error = point.message;
		return false;
	}
	t_recovery = &point;
	function(context);
	t_recovery = point.previous;
	return true;
#endif
}

static std::string BuildFatalReport(const char* title, std::string_view text, const char* file,
                                    int line) {
	return fmt::format("--- Build ---\n{}\n{}\n{} in {}:{}\n", KYTY_BUILD_LABEL, title, text, file,
	                   line);
}

static int DbgReport(const char* title, std::string_view text, const char* file, int line) {
	Log::WriteFatal(BuildFatalReport(title, text, file, line));
	Subsystems::EmergencyShutdownActive();
	return 1;
}

int DbgExitIfHandler(const char* expr, const char* file, int line) {
	if (t_recovery != nullptr) Recover("Error:", fmt::format("condition ({}) is true", expr), file, line);
	return DbgReport("--- Fatal Error ---", fmt::format("Error: condition ({}) is true", expr),
	                 file, line);
}

int DbgNotImplementedHandler(const char* expr, const char* file, int line) {
	if (t_recovery != nullptr) Recover("Not implemented:", expr, file, line);
	return DbgReport("--- Fatal Error ---", fmt::format("Not implemented ({})", expr), file, line);
}

int DbgExitHandler(const char* file, int line, std::string_view text) {
	if (t_recovery != nullptr) Recover("Error:", text, file, line);
	Log::WriteFatal(BuildFatalReport("--- Error ---", text, file, line));
	return 1;
}

int DbgExitHandler(const char* file, int line, fmt::text_style style, std::string_view text) {
	if (t_recovery != nullptr) Recover("Error:", text, file, line);
	Log::WriteFatal(style, BuildFatalReport("--- Error ---", text, file, line));
	return 1;
}

void DbgExit(int status) {
	Subsystems::EmergencyShutdownActive();
	std::fflush(nullptr);
	std::_Exit(status);
}

} // namespace Common
