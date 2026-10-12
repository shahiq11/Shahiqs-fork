#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"

#include "common/assert.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include <cstdio>

namespace Libs::Graphics {
namespace {

constexpr uint32_t   DescriptorHeapCount = 1024;
constexpr uint32_t   MaxPoolScale        = 64;
constexpr std::array DescriptorPoolSizes = {
    vk::DescriptorPoolSize {vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize {vk::DescriptorType::eSampledImage, 8192},
    vk::DescriptorPoolSize {vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize {vk::DescriptorType::eSampler, 1024},
};

} // namespace

DescriptorHeap::DescriptorHeap(GraphicContext& graphics, MasterSemaphore& master_semaphore)
    : m_graphics(graphics), m_master_semaphore(master_semaphore) {
	CreateDescriptorPool();
}

DescriptorHeap::~DescriptorHeap() {
	m_graphics.device.destroyDescriptorPool(m_current_pool, nullptr);
	for (const auto& pending: m_pending_pools) {
		m_master_semaphore.Wait(pending.tick);
		m_graphics.device.destroyDescriptorPool(pending.pool, nullptr);
	}
}

vk::DescriptorSet DescriptorHeap::Commit(vk::DescriptorSetLayout layout) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(layout == nullptr);

	auto& batch = m_sets[layout];
	if (batch.size != 0) {
		return batch.sets[--batch.size];
	}
	if (Allocate(layout, batch)) {
		return batch.sets[--batch.size];
	}

	m_pending_pools.push_back(
	    {m_current_pool, m_master_semaphore.CurrentTick(), m_current_pool_scale});
	if (const auto front = m_pending_pools.front(); m_master_semaphore.IsFree(front.tick)) {
		m_current_pool       = front.pool;
		m_current_pool_scale = front.scale;
		m_pending_pools.pop_front();
		EXIT_IF(m_graphics.device.resetDescriptorPool(m_current_pool, {}) != vk::Result::eSuccess);
	} else {
		CreateDescriptorPool();
	}

	m_sets.clear();
	auto& fresh_batch = m_sets[layout];
	// The pool is empty here and no set from it is in use, so a failure means one set of this
	// layout needs more descriptors of some type than a whole pool holds. Replace it with a
	// larger pool instead of aborting.
	while (!Allocate(layout, fresh_batch)) {
		const bool largest = m_current_pool_scale == m_pool_scale;
		m_graphics.device.destroyDescriptorPool(m_current_pool, nullptr);
		if (largest) {
			EXIT_IF(m_pool_scale >= MaxPoolScale);
			m_pool_scale *= 2u;
			std::printf("Descriptor pool too small for a pipeline layout; growing pools to %ux\n",
			            m_pool_scale);
		}
		CreateDescriptorPool();
		fresh_batch = {};
	}
	return fresh_batch.sets[--fresh_batch.size];
}

bool DescriptorHeap::Allocate(vk::DescriptorSetLayout layout, Batch& batch) {
	std::array<vk::DescriptorSetLayout, DescriptorSetBatch> layouts;
	layouts.fill(layout);

	vk::DescriptorSetAllocateInfo allocate {};
	allocate.descriptorPool = m_current_pool;
	allocate.pSetLayouts    = layouts.data();

	for (;;) {
		allocate.descriptorSetCount = batch.allocation;
		const auto result = m_graphics.device.allocateDescriptorSets(&allocate, batch.sets.data());
		if (result == vk::Result::eSuccess) {
			batch.size = batch.allocation;
			return true;
		}
		EXIT_IF(result != vk::Result::eErrorOutOfPoolMemory &&
		        result != vk::Result::eErrorFragmentedPool);
		if (batch.allocation == 1) {
			return false;
		}
		batch.allocation /= 2;
	}
}

void DescriptorHeap::CreateDescriptorPool() {
	auto sizes = DescriptorPoolSizes;
	for (auto& size: sizes) {
		size.descriptorCount *= m_pool_scale;
	}
	vk::DescriptorPoolCreateInfo create {};
	create.maxSets       = DescriptorHeapCount;
	create.poolSizeCount = static_cast<uint32_t>(sizes.size());
	create.pPoolSizes    = sizes.data();
	EXIT_IF(m_graphics.device.createDescriptorPool(&create, nullptr, &m_current_pool) !=
	        vk::Result::eSuccess);
	m_current_pool_scale = m_pool_scale;
}

} // namespace Libs::Graphics
