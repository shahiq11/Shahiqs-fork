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

struct RecoveryPoint {
	std::jmp_buf   buffer {};
	std::string    message;
	RecoveryPoint* previous = nullptr;
};

thread_local RecoveryPoint* t_recovery = nullptr;

// Reports a fatal error raised inside RunRecoverable and returns to it.
[[noreturn]] void Recover(const char* title, std::string_view text, const char* file, int line) {
	auto* point    = t_recovery;
	t_recovery     = point->previous;
	point->message = fmt::format("{} {} ({}:{})", title, text, file, line);
	std::longjmp(point->buffer, 1);
}

} // namespace

bool RunRecoverable(void (*function)(void*), void* context, std::string* error) {
	RecoveryPoint point;
	point.previous = t_recovery;
	if (setjmp(point.buffer) != 0) {
		// Recover() already restored the enclosing recovery point.
		if (error != nullptr) *error = point.message;
		return false;
	}
	t_recovery = &point;
	function(context);
	t_recovery = point.previous;
	return true;
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
