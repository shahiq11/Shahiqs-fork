#ifndef KYTY_COMMON_RECOVERABLE_H_
#define KYTY_COMMON_RECOVERABLE_H_

#include <string>
#include <type_traits>

namespace Common {

// Runs `function` so that a fatal error (EXIT, EXIT_IF, EXIT_NOT_IMPLEMENTED) raised on this
// thread inside it does not stop the emulator: the error is returned in `error` and the call
// returns false. Abandoned frames run no destructors, so use this only around work that holds
// no locks and whose partial results the caller discards (for example translating one guest
// shader). Nested calls are allowed. Implemented in assert.cpp.
bool RunRecoverable(void (*function)(void*), void* context, std::string* error);

template <typename Function>
bool RunRecoverable(Function&& function, std::string* error) {
	using Callable = std::remove_reference_t<Function>;
	return RunRecoverable([](void* context) { (*static_cast<Callable*>(context))(); },
	                      const_cast<void*>(static_cast<const void*>(&function)), error);
}

} // namespace Common

#endif /* KYTY_COMMON_RECOVERABLE_H_ */
