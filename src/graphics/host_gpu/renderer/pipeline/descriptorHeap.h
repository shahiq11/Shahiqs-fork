#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORHEAP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORHEAP_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <deque>
#include <unordered_map>

namespace Libs::Graphics {

struct GraphicContext;
class MasterSemaphore;

class DescriptorHeap {
public:
	DescriptorHeap(GraphicContext& graphics, MasterSemaphore& master_semaphore);
	~DescriptorHeap();
	KYTY_CLASS_NO_COPY(DescriptorHeap);

	[[nodiscard]] vk::DescriptorSet Commit(vk::DescriptorSetLayout layout);

private:
	static constexpr uint32_t DescriptorSetBatch = 32;

	struct Batch {
		std::array<vk::DescriptorSet, DescriptorSetBatch> sets {};
		uint32_t                                          size       = 0;
		uint32_t                                          allocation = DescriptorSetBatch;
	};

	struct PendingPool {
		vk::DescriptorPool pool  = nullptr;
		uint64_t           tick  = 0;
		uint32_t           scale = 1;
	};

	[[nodiscard]] bool Allocate(vk::DescriptorSetLayout layout, Batch& batch);
	void               CreateDescriptorPool();

	GraphicContext&                                    m_graphics;
	MasterSemaphore&                                   m_master_semaphore;
	vk::DescriptorPool                                 m_current_pool       = nullptr;
	uint32_t                                           m_current_pool_scale = 1;
	// Pool sizes are multiplied by this; it grows when one set does not fit an empty pool.
	uint32_t                                           m_pool_scale = 1;
	std::deque<PendingPool>                            m_pending_pools;
	std::unordered_map<vk::DescriptorSetLayout, Batch> m_sets;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORHEAP_H_
