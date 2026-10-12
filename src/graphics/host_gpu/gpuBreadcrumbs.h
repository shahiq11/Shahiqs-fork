#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GPUBREADCRUMBS_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GPUBREADCRUMBS_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

namespace Libs::Graphics {

struct GraphicContext;

// Locates the GPU command that crashed the device (VK_ERROR_DEVICE_LOST). Every draw and
// dispatch writes a "started" marker before it and a "finished" marker after it (through
// VK_AMD_buffer_marker when available); after a device loss the commands between the two marker
// values are reported with their shader hashes.
namespace GpuBreadcrumbs {

[[nodiscard]] const char* ExtensionName() noexcept;
void                      Init(GraphicContext& graphics, bool extension_enabled);

// Returns the command id to pass to End (0 when breadcrumbs are off).
[[nodiscard]] uint32_t Begin(vk::CommandBuffer command, const char* kind, uint64_t hash0,
                             uint64_t hash1 = 0, uint32_t x = 0, uint32_t y = 0, uint32_t z = 0);
void                   End(vk::CommandBuffer command, uint32_t id);

// Prints the commands that were running when the device was lost (console, log and
// kyty_gpu_crash.txt). Safe to call more than once; only the first call reports.
void ReportDeviceLost();

} // namespace GpuBreadcrumbs

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GPUBREADCRUMBS_H_ */
