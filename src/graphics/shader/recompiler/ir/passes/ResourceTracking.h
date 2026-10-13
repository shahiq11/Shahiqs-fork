#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCETRACKING_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCETRACKING_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

// Resolves native descriptor sources, plans their scalar reads, and assigns dense resource bindings.
void TrackResources(Program& program, const Decoder::Program& decoded, const CFG::Graph& native_cfg);

// When enabled (the emulator does; tests keep it off), an image descriptor the host cannot follow
// reads a null texture instead of failing the shader.
void SetUntraceableImagesAsNull(bool enabled) noexcept;

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCETRACKING_H_ */
