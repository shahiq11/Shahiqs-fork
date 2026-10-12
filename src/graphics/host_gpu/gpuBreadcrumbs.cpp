#include "graphics/host_gpu/gpuBreadcrumbs.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "kernel/hangWatchdog.h"

#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fmt/format.h>
#include <mutex>
#include <string>

namespace Libs::Graphics::GpuBreadcrumbs {

namespace {

struct Entry {
	uint32_t id      = 0;
	char     kind[12] {};
	uint64_t hash0   = 0;
	uint64_t hash1   = 0;
	uint32_t x       = 0;
	uint32_t y       = 0;
	uint32_t z       = 0;
};

constexpr uint32_t RingSize = 1u << 16u;

struct State {
	bool                        enabled    = false;
	vk::Buffer                  buffer     = nullptr;
	VmaAllocation               allocation = nullptr;
	volatile uint32_t*          mapped     = nullptr;
	std::atomic<uint32_t>       next_id {1};
	std::mutex                  mutex;
	std::array<Entry, RingSize> ring {};
	std::atomic_flag            reported = ATOMIC_FLAG_INIT;
};

State& GetState() {
	static State state;
	return state;
}

} // namespace

const char* ExtensionName() noexcept {
	return VK_AMD_BUFFER_MARKER_EXTENSION_NAME;
}

void Init(GraphicContext& graphics, bool extension_enabled) {
	auto& state = GetState();
	if (!extension_enabled || graphics.allocator == nullptr) {
		std::printf("GPU crash breadcrumbs: unavailable (no %s)\n", ExtensionName());
		return;
	}
	VkBufferCreateInfo buffer_info {};
	buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.size  = 256;
	buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VmaAllocationCreateInfo allocation_info {};
	allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	allocation_info.flags =
	    VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	allocation_info.requiredFlags =
	    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	VkBuffer          buffer = VK_NULL_HANDLE;
	VmaAllocationInfo result_info {};
	if (vmaCreateBuffer(graphics.allocator, &buffer_info, &allocation_info, &buffer,
	                    &state.allocation, &result_info) != VK_SUCCESS ||
	    result_info.pMappedData == nullptr) {
		std::printf("GPU crash breadcrumbs: could not create the marker buffer\n");
		return;
	}
	state.buffer = buffer;
	state.mapped = static_cast<volatile uint32_t*>(result_info.pMappedData);
	state.mapped[0] = 0;
	state.mapped[1] = 0;
	state.enabled   = true;
	std::printf("GPU crash breadcrumbs: enabled\n");
}

uint32_t Begin(vk::CommandBuffer command, const char* kind, uint64_t hash0, uint64_t hash1,
               uint32_t x, uint32_t y, uint32_t z) {
	auto& state = GetState();
	if (!state.enabled || !command) {
		return 0;
	}
	auto id = state.next_id.fetch_add(1, std::memory_order_relaxed);
	if (id == 0) id = state.next_id.fetch_add(1, std::memory_order_relaxed);
	{
		std::scoped_lock lock(state.mutex);
		auto&            entry = state.ring[id % RingSize];
		entry.id               = id;
		std::strncpy(entry.kind, kind, sizeof(entry.kind) - 1u);
		entry.kind[sizeof(entry.kind) - 1u] = '\0';
		entry.hash0 = hash0;
		entry.hash1 = hash1;
		entry.x     = x;
		entry.y     = y;
		entry.z     = z;
	}
	// Offset 0: last command the GPU front end reached.
	command.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eTopOfPipe, state.buffer, 0, id);
	return id;
}

void End(vk::CommandBuffer command, uint32_t id) {
	auto& state = GetState();
	if (!state.enabled || id == 0 || !command) {
		return;
	}
	// Offset 4: last command whose work (and all work before it) finished.
	command.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eBottomOfPipe, state.buffer, 4, id);
}

void ReportDeviceLost() {
	auto& state = GetState();
	if (state.reported.test_and_set()) {
		return;
	}
	std::string text;
	if (!state.enabled) {
		text = "GPU crash: breadcrumbs unavailable; the crashing command is unknown\n";
	} else {
		const uint32_t started  = state.mapped[0];
		const uint32_t finished = state.mapped[1];
		const uint32_t issued   = state.next_id.load() - 1u;
		text = fmt::format("GPU crash: commands issued up to #{}, front end reached #{}, all work "
		                   "finished through #{}\n",
		                   issued, started, finished);
		std::scoped_lock lock(state.mutex);
		const auto first = finished + 1u;
		const auto last  = std::max(started, first);
		uint32_t   shown = 0;
		for (uint32_t id = finished; id <= last && shown < 48u; ++id, ++shown) {
			const auto& entry = state.ring[id % RingSize];
			if (entry.id != id) continue;
			text += fmt::format("  {} #{} {} shader={:016x} {:016x} size={}x{}x{}\n",
			                    id == finished ? "finished" : "running ", id, entry.kind,
			                    entry.hash0, entry.hash1, entry.x, entry.y, entry.z);
		}
		if (last - first > 48u) {
			text += fmt::format("  ... {} more running commands\n", last - first - 47u);
		}
	}
	Log::WriteToConsoleAndLog(text);
	std::fflush(stdout);
	Kernel::HangWatchdog::AppendDiagnostic("kyty_gpu_crash.txt", text);
}

} // namespace Libs::Graphics::GpuBreadcrumbs
