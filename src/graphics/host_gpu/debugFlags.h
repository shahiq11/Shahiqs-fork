#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_DEBUGFLAGS_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_DEBUGFLAGS_H_

#include <cstdint>
#include <string_view>

// Experiment switches read once from "kyty_debug_flags.txt" in the working directory, one per
// line ("name" or "name=value", '#' starts a comment). Used to isolate problems on a test machine
// without rebuilding, for example:
//   no_texture_residency            load every broad-heap texture (old behavior)
//   skip_shader=07236179d8c19973    do not issue draws/dispatches that use this shader
namespace Libs::Graphics::DebugFlags {

[[nodiscard]] bool Has(std::string_view name);
// True when "skip_shader=<hex hash>" lists the shader.
[[nodiscard]] bool SkipShader(uint64_t shader_hash);

} // namespace Libs::Graphics::DebugFlags

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_DEBUGFLAGS_H_ */
