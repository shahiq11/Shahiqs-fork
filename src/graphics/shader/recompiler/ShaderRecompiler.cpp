#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/Tessellation.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/frontend/cfg/ShaderCFG.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/frontend/translate/Translate.h"
#include "graphics/shader/recompiler/frontend/translate/Translator.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ConstantPropagation.h"
#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"
#include "graphics/shader/recompiler/ir/passes/ReadLaneElimination.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/ResourceTracking.h"
#include "graphics/shader/recompiler/ir/passes/ShaderInfoCollection.h"
#include "graphics/shader/recompiler/ir/passes/SsaRewrite.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <fmt/format.h>
#include <functional>
#include <map>
#include <span>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler {

namespace {

const char* GetDumpLabel(const CompileOptions& options) {
	return options.dump_label != nullptr ? options.dump_label : "ShaderRecompiler";
}

std::string MakeIrDump(std::string_view cfg, const IR::Program& ir) {
	std::string dump = "CFG:\n";
	dump += cfg;
	dump += "\nIR:\n";
	dump += fmt::format("mode={} scratch_dwords={}\n",
	                    ir.dispatcher_fallback ? "dispatcher" : "structured", ir.scratch_dwords);
	dump += IR::ProgramToString(ir);
	return dump;
}

const char* StageName(ShaderType stage) {
	switch (stage) {
		case ShaderType::Compute: return "CS";
		case ShaderType::Vertex: return "VS";
		case ShaderType::Local: return "LS";
		case ShaderType::TessellationControl: return "HS";
		case ShaderType::TessellationEvaluation: return "TES";
		case ShaderType::Mesh: return "MS";
		case ShaderType::Pixel: return "PS";
		default: return "unknown";
	}
}

void LogDispatcherFallback(const CompileOptions& options, const CFG::Graph& cfg, const char* phase) {
	const auto* block        = cfg.FindBlock(cfg.failure_block);
	const auto  start        = block != nullptr ? block->start_pc : UINT32_MAX;
	const auto  end          = block != nullptr ? block->end_pc : UINT32_MAX;
	const auto  predecessors = block != nullptr ? block->predecessors.size() : 0u;
	const auto  successors   = block != nullptr ? block->successors.size() : 0u;
	LOGF("%s CFG dispatcher fallback: stage=%s hash=0x%016" PRIx64
	     " phase=%s failure=%s block=%" PRIu32 " pc=0x%08" PRIx32 "..0x%08" PRIx32 " preds=%" PRIu64
	     " succs=%" PRIu64 " blocks=%" PRIu64 " loops=%" PRIu64 " back_edges=%" PRIu64
	     " reason=%s\n",
	     GetDumpLabel(options), StageName(options.stage), options.shader_hash, phase,
	     CFG::FailureKindToString(cfg.failure_kind).c_str(), cfg.failure_block, start, end,
	     static_cast<uint64_t>(predecessors), static_cast<uint64_t>(successors),
	     static_cast<uint64_t>(cfg.blocks.size()), static_cast<uint64_t>(cfg.natural_loops.size()),
	     static_cast<uint64_t>(cfg.back_edges.size()), cfg.unsupported_reason.c_str());
}

enum class EmbeddedFetchValueType {
	Unknown,
	Constant,
	AttribTable,
	Attrib,
	BufferTable,
	Buffer
};

struct EmbeddedFetchSgprInfo {
	EmbeddedFetchValueType type      = EmbeddedFetchValueType::Unknown;
	int                    attrib_id = 0;
	uint32_t               value     = 0;
};

using EmbeddedFetchVectorLanes = std::map<uint64_t, EmbeddedFetchSgprInfo>;

uint64_t EmbeddedFetchVectorLaneKey(uint32_t reg, uint32_t lane) {
	return (static_cast<uint64_t>(reg) << 32u) | lane;
}

uint32_t EmbeddedFetchLane(uint32_t lane, uint32_t wave_size) {
	return wave_size == 32 || wave_size == 64 ? lane % wave_size : lane;
}

void ClearEmbeddedFetchVectorLanes(EmbeddedFetchVectorLanes* lanes, uint32_t reg) {
	const auto first = lanes->lower_bound(EmbeddedFetchVectorLaneKey(reg, 0));
	const auto last  = lanes->lower_bound(EmbeddedFetchVectorLaneKey(reg + 1u, 0));
	lanes->erase(first, last);
}

bool IsDecodedSgpr(const Decoder::Operand& op) {
	return op.kind == Decoder::OperandKind::Sgpr || op.kind == Decoder::OperandKind::VccLo ||
	       op.kind == Decoder::OperandKind::VccHi;
}

uint32_t DecodedSgprReg(const Decoder::Operand& op) {
	switch (op.kind) {
		case Decoder::OperandKind::VccLo: return 106u;
		case Decoder::OperandKind::VccHi: return 107u;
		default: return op.reg;
	}
}

bool IsDecodedVgpr(const Decoder::Operand& op) {
	return op.kind == Decoder::OperandKind::Vgpr;
}

uint32_t DecodedDstSize(const Decoder::Instruction& inst) {
	return std::max(inst.data_dwords, 1u);
}

uint32_t EmbeddedFetchDstSize(const Decoder::Instruction& inst) {
	return inst.opcode == Decoder::Opcode::V_MAD_U64_U32 ? 2u : DecodedDstSize(inst);
}

void ClearEmbeddedFetchSgprs(std::array<EmbeddedFetchSgprInfo, 108>& sgprs,
                             const Decoder::Operand& dst, uint32_t size) {
	if (!IsDecodedSgpr(dst)) {
		return;
	}
	const auto register_id = DecodedSgprReg(dst);
	for (uint32_t i = 0; i < size && register_id + i < sgprs.size(); i++) {
		sgprs[register_id + i] = {};
	}
}

bool TryDecodedOperandConstant(const std::array<EmbeddedFetchSgprInfo, 108>& sgprs,
                               const Decoder::Operand& op, uint32_t& value) {
	switch (op.kind) {
		case Decoder::OperandKind::LiteralConstant:
		case Decoder::OperandKind::IntegerInlineConstant:
		case Decoder::OperandKind::FloatInlineConstant: value = op.value; return true;
		case Decoder::OperandKind::Null: value = 0; return true;
		default: break;
	}
	if (IsDecodedSgpr(op) && DecodedSgprReg(op) < sgprs.size() &&
	    sgprs[DecodedSgprReg(op)].type == EmbeddedFetchValueType::Constant) {
		value = sgprs[DecodedSgprReg(op)].value;
		return true;
	}
	return false;
}

bool TryDecodedSmemOffset(const std::array<EmbeddedFetchSgprInfo, 108>& sgprs,
                          const Decoder::Instruction& inst, uint32_t& raw_offset) {
	uint32_t base = 0;
	if (!TryDecodedOperandConstant(sgprs, inst.src1, base)) {
		return false;
	}
	const auto value = static_cast<uint64_t>(base) + inst.offset;
	if (value > 0xffffffffull) {
		return false;
	}
	raw_offset = static_cast<uint32_t>(value);
	return true;
}

bool IsEmbeddedFetchSLoad(const Decoder::Instruction& inst) {
	switch (inst.opcode) {
		case Decoder::Opcode::S_LOAD_DWORD:
		case Decoder::Opcode::S_LOAD_DWORDX2:
		case Decoder::Opcode::S_LOAD_DWORDX4:
		case Decoder::Opcode::S_LOAD_DWORDX8:
		case Decoder::Opcode::S_LOAD_DWORDX16: return true;
		default: return false;
	}
}

bool IsEmbeddedFetchBufferLoad(const Decoder::Instruction& inst) {
	switch (inst.opcode) {
		case Decoder::Opcode::BUFFER_LOAD_FORMAT_X:
		case Decoder::Opcode::BUFFER_LOAD_FORMAT_XY:
		case Decoder::Opcode::BUFFER_LOAD_FORMAT_XYZ:
		case Decoder::Opcode::BUFFER_LOAD_FORMAT_XYZW: return true;
		default: return false;
	}
}

bool IsEmbeddedFetchAttribPropagationAlu(const Decoder::Instruction& inst) {
	switch (inst.opcode) {
		case Decoder::Opcode::S_BFE_U32:
		case Decoder::Opcode::S_AND_B32:
		case Decoder::Opcode::S_ADD_I32:
		case Decoder::Opcode::S_ADD_U32:
		case Decoder::Opcode::S_LSHL_B32: return true;
		default: return false;
	}
}

int BufferTableAttribFromOffset(uint32_t raw_offset, int dword) {
	return static_cast<int>((raw_offset + static_cast<uint32_t>(dword) * 4u) / 16u);
}

Frontend::EmbeddedFetchPlan DetectEmbeddedVertexFetch(
    const Decoder::Program& decoded, const ShaderVertexInputInfo* input_info,
    uint32_t user_data_base, uint32_t user_data_count, uint32_t wave_size) {
	const uint32_t    vertex_index_reg   = input_info->logical_stage == ShaderType::Local ? 2u : 5u;
	const uint32_t    instance_index_reg = input_info->logical_stage == ShaderType::Local ? 5u : 8u;
	Frontend::EmbeddedFetchPlan data;
	data.loads.reserve(input_info->resources_num);
	int32_t vertex_offset_candidate   = -1;
	int32_t instance_offset_candidate = -1;
	bool    vertex_offset_conflict    = false;
	bool    instance_offset_conflict  = false;

	const int shift_regs = 8;
	const int attrib_reg = input_info->fetch_attrib_reg + shift_regs;
	const int buffer_reg = input_info->fetch_buffer_reg + shift_regs;

	std::array<EmbeddedFetchSgprInfo, 108> sgprs {};
	std::array<bool, 256>                 vgpr_is_index {};
	EmbeddedFetchVectorLanes               vector_lanes;
	const bool                             track_vector_lanes =
	    std::none_of(decoded.instructions.begin(), decoded.instructions.end(),
	                 [](const auto& inst) {
		                 return Decoder::IsDirectBranch(inst.opcode) ||
		                        inst.opcode == Decoder::Opcode::S_SETPC_B64;
	                 });

	if (attrib_reg >= 0 && attrib_reg < static_cast<int>(sgprs.size())) {
		sgprs[attrib_reg].type = EmbeddedFetchValueType::AttribTable;
	}
	if (attrib_reg + 1 >= 0 && attrib_reg + 1 < static_cast<int>(sgprs.size())) {
		sgprs[attrib_reg + 1].type = EmbeddedFetchValueType::AttribTable;
	}
	if (buffer_reg >= 0 && buffer_reg < static_cast<int>(sgprs.size())) {
		sgprs[buffer_reg].type = EmbeddedFetchValueType::BufferTable;
	}
	if (buffer_reg + 1 >= 0 && buffer_reg + 1 < static_cast<int>(sgprs.size())) {
		sgprs[buffer_reg + 1].type = EmbeddedFetchValueType::BufferTable;
	}

	for (const auto& inst: decoded.instructions) {
		// Fetch shaders accumulate the draw's vertex offset in v0. The PS5 NGG ABI
		// seeds S_NGG_VERTEX_INDEX in v5 and S_NGG_INSTANCE_INDEX in v8, then applies the
		// corresponding direct-draw offsets before fetching.
		const bool vertex_index_accumulator =
		    IsDecodedVgpr(inst.dst) &&
		    (inst.dst.reg == 0 || (user_data_base == 8 && inst.dst.reg == vertex_index_reg));
		const bool instance_index_accumulator =
		    IsDecodedVgpr(inst.dst) &&
		    (inst.dst.reg == (user_data_base == 8 ? instance_index_reg : 3u));
		uint32_t   sad_zero = 0;
		const bool index_offset_add =
		    (vertex_index_accumulator || instance_index_accumulator) && IsDecodedSgpr(inst.src0) &&
		    ((inst.opcode == Decoder::Opcode::V_ADD_I32 && IsDecodedVgpr(inst.src1) &&
		      inst.src1.reg == inst.dst.reg) ||
		     (user_data_base == 8 &&
		      (inst.dst.reg == vertex_index_reg || inst.dst.reg == instance_index_reg) &&
		      inst.opcode == Decoder::Opcode::V_SAD_U32 && IsDecodedVgpr(inst.src2) &&
		      inst.src2.reg == inst.dst.reg &&
		      TryDecodedOperandConstant(sgprs, inst.src1, sad_zero) && sad_zero == 0));
		if (data.loads.empty() && index_offset_add) {
			const auto reg = DecodedSgprReg(inst.src0);
			if (reg >= user_data_base && reg - user_data_base < user_data_count) {
				auto& candidate = vertex_index_accumulator ? vertex_offset_candidate
				                                           : instance_offset_candidate;
				auto& conflict  = vertex_index_accumulator ? vertex_offset_conflict
				                                           : instance_offset_conflict;
				if (candidate >= 0 && candidate != static_cast<int32_t>(reg)) {
					conflict = true;
				} else {
					candidate = static_cast<int32_t>(reg);
				}
			}
		}
		switch (inst.opcode) {
			case Decoder::Opcode::V_WRITELANE_B32: {
				uint32_t lane = 0;
				if (IsDecodedVgpr(inst.dst) && inst.dst.reg < vgpr_is_index.size()) {
					vgpr_is_index[inst.dst.reg] = false;
				}
				if (track_vector_lanes && IsDecodedVgpr(inst.dst) && IsDecodedSgpr(inst.src0) &&
				    DecodedSgprReg(inst.src0) < sgprs.size() &&
				    TryDecodedOperandConstant(sgprs, inst.src1, lane)) {
					vector_lanes[EmbeddedFetchVectorLaneKey(inst.dst.reg,
					                                        EmbeddedFetchLane(lane, wave_size))] =
					    sgprs[DecodedSgprReg(inst.src0)];
				} else if (IsDecodedVgpr(inst.dst)) {
					ClearEmbeddedFetchVectorLanes(&vector_lanes, inst.dst.reg);
				}
				break;
			}
			case Decoder::Opcode::V_READLANE_B32: {
				uint32_t lane = 0;
				if (track_vector_lanes && IsDecodedSgpr(inst.dst) &&
				    DecodedSgprReg(inst.dst) < sgprs.size() && IsDecodedVgpr(inst.src0) &&
				    TryDecodedOperandConstant(sgprs, inst.src1, lane)) {
					const auto found = vector_lanes.find(EmbeddedFetchVectorLaneKey(
					    inst.src0.reg, EmbeddedFetchLane(lane, wave_size)));
					sgprs[DecodedSgprReg(inst.dst)] =
					    found != vector_lanes.end() ? found->second : EmbeddedFetchSgprInfo {};
				} else if (IsDecodedSgpr(inst.dst)) {
					ClearEmbeddedFetchSgprs(sgprs, inst.dst, 1);
				}
				break;
			}
			case Decoder::Opcode::S_MOV_B32:
				if (IsDecodedSgpr(inst.dst) && IsDecodedSgpr(inst.src0) &&
				    DecodedSgprReg(inst.src0) < sgprs.size()) {
					sgprs[DecodedSgprReg(inst.dst)] = sgprs[DecodedSgprReg(inst.src0)];
				} else if (IsDecodedSgpr(inst.dst)) {
					uint32_t value = 0;
					if (TryDecodedOperandConstant(sgprs, inst.src0, value)) {
						auto& dst = sgprs[DecodedSgprReg(inst.dst)];
						dst.type  = EmbeddedFetchValueType::Constant;
						dst.value = value;
					} else {
						ClearEmbeddedFetchSgprs(sgprs, inst.dst, 1);
					}
				}
				break;
			case Decoder::Opcode::S_MOVK_I32:
				if (IsDecodedSgpr(inst.dst)) {
					auto& dst = sgprs[DecodedSgprReg(inst.dst)];
					dst.type  = EmbeddedFetchValueType::Constant;
					dst.value = inst.src0.value;
				}
				break;
			default:
				if (IsEmbeddedFetchSLoad(inst)) {
					if (IsDecodedSgpr(inst.src0) && DecodedSgprReg(inst.src0) < sgprs.size() &&
					    sgprs[DecodedSgprReg(inst.src0)].type ==
					        EmbeddedFetchValueType::AttribTable) {
						uint32_t raw_offset = 0;
						if (TryDecodedSmemOffset(sgprs, inst, raw_offset)) {
							const auto register_id = DecodedSgprReg(inst.dst);
							const int  index       = static_cast<int>(raw_offset / 4u);
							for (uint32_t i = 0;
							     i < DecodedDstSize(inst) && register_id + i < sgprs.size(); i++) {
								auto& dst        = sgprs[register_id + i];
								dst.type         = EmbeddedFetchValueType::Attrib;
								dst.attrib_id    = index + static_cast<int>(i);
							}
						} else {
							ClearEmbeddedFetchSgprs(sgprs, inst.dst, DecodedDstSize(inst));
						}
					} else if (IsDecodedSgpr(inst.src0) &&
					           DecodedSgprReg(inst.src0) < sgprs.size() &&
					           sgprs[DecodedSgprReg(inst.src0)].type ==
					               EmbeddedFetchValueType::BufferTable) {
						const auto register_id = DecodedSgprReg(inst.dst);
						uint32_t   raw_offset  = 0;
						if (TryDecodedSmemOffset(sgprs, inst, raw_offset)) {
							for (uint32_t i = 0;
							     i < DecodedDstSize(inst) && register_id + i < sgprs.size(); i++) {
								auto& dst = sgprs[register_id + i];
								dst.type  = EmbeddedFetchValueType::Buffer;
								dst.attrib_id =
								    BufferTableAttribFromOffset(raw_offset, static_cast<int>(i));
							}
						} else if (IsDecodedSgpr(inst.src1) &&
						           DecodedSgprReg(inst.src1) < sgprs.size() &&
						           sgprs[DecodedSgprReg(inst.src1)].type ==
						               EmbeddedFetchValueType::Attrib &&
						           (inst.offset & 0x3u) == 0) {
							for (uint32_t i = 0;
							     i < DecodedDstSize(inst) && register_id + i < sgprs.size(); i++) {
								auto& dst        = sgprs[register_id + i];
								dst.type         = EmbeddedFetchValueType::Buffer;
								dst.attrib_id    = sgprs[DecodedSgprReg(inst.src1)].attrib_id;
							}
						} else {
							ClearEmbeddedFetchSgprs(sgprs, inst.dst, DecodedDstSize(inst));
						}
					} else {
						ClearEmbeddedFetchSgprs(sgprs, inst.dst, DecodedDstSize(inst));
					}
				} else if (inst.opcode == Decoder::Opcode::V_CNDMASK_B32) {
					if (IsDecodedVgpr(inst.dst) && inst.dst.reg < vgpr_is_index.size()) {
						ClearEmbeddedFetchVectorLanes(&vector_lanes, inst.dst.reg);
					}
					if (IsDecodedVgpr(inst.dst) && inst.dst.reg < vgpr_is_index.size() &&
					    IsDecodedVgpr(inst.src0) && inst.src0.reg == instance_index_reg &&
					    IsDecodedVgpr(inst.src1) && inst.src1.reg == vertex_index_reg) {
						vgpr_is_index[inst.dst.reg] = true;
					}
				} else if (IsEmbeddedFetchAttribPropagationAlu(inst)) {
					if (IsDecodedSgpr(inst.dst) && IsDecodedSgpr(inst.src0) &&
					    DecodedSgprReg(inst.src0) < sgprs.size() &&
					    sgprs[DecodedSgprReg(inst.src0)].type == EmbeddedFetchValueType::Attrib) {
						sgprs[DecodedSgprReg(inst.dst)] = sgprs[DecodedSgprReg(inst.src0)];
					} else if (IsDecodedSgpr(inst.dst)) {
						uint32_t src0 = 0;
						uint32_t src1 = 0;
						if (TryDecodedOperandConstant(sgprs, inst.src0, src0) &&
						    TryDecodedOperandConstant(sgprs, inst.src1, src1)) {
							auto& dst = sgprs[DecodedSgprReg(inst.dst)];
							dst.type  = EmbeddedFetchValueType::Constant;
							switch (inst.opcode) {
								case Decoder::Opcode::S_AND_B32: dst.value = src0 & src1; break;
								case Decoder::Opcode::S_LSHL_B32:
									dst.value = src0 << (src1 & 31u);
									break;
								case Decoder::Opcode::S_BFE_U32:
									dst.value = src0 >> (src1 & 31u);
									break;
								default: dst.value = src0 + src1; break;
							}
						} else {
							ClearEmbeddedFetchSgprs(sgprs, inst.dst, 1);
						}
					}
				} else if (IsEmbeddedFetchBufferLoad(inst)) {
					if (IsDecodedVgpr(inst.src0) && inst.src0.reg < vgpr_is_index.size() &&
					    vgpr_is_index[inst.src0.reg] &&
					    IsDecodedSgpr(inst.src1) && DecodedSgprReg(inst.src1) < sgprs.size() &&
					    sgprs[DecodedSgprReg(inst.src1)].type == EmbeddedFetchValueType::Buffer) {
						const auto& buffer = sgprs[DecodedSgprReg(inst.src1)];
						if (data.loads.empty()) {
							if (!vertex_offset_conflict) {
								data.vertex_offset_sgpr = vertex_offset_candidate;
							}
							if (!instance_offset_conflict) {
								data.instance_offset_sgpr = instance_offset_candidate;
							}
						}
						auto& load        = data.loads.emplace_back();
						load.pc           = inst.pc;
						load.attrib_id    = buffer.attrib_id;
						load.components   = DecodedDstSize(inst);
					}
				}
				break;
		}
		if (inst.opcode == Decoder::Opcode::V_MOVRELD_B32) {
			vector_lanes.clear();
		} else if (inst.opcode != Decoder::Opcode::V_WRITELANE_B32 && IsDecodedVgpr(inst.dst)) {
			for (uint32_t i = 0;
			     i < EmbeddedFetchDstSize(inst) && inst.dst.reg + i < vgpr_is_index.size();
			     i++) {
				ClearEmbeddedFetchVectorLanes(&vector_lanes, inst.dst.reg + i);
			}
		}
	}

	return data;
}

Decoder::Program DecodeFusedProgram(std::span<const uint32_t> front, std::span<const uint32_t> back,
                                    std::vector<uint32_t>& joined_code) {
	EXIT_IF(back.empty());
	auto       result      = Decoder::DecodeFrontProgram(front);
	const auto front_words = static_cast<uint32_t>(result.code.size());
	joined_code.assign(result.code.begin(), result.code.end());
	joined_code.insert(joined_code.end(), back.begin(), back.end());
	// The merged-stage ABI passes the back shader in s[6:7]. Give that handoff an
	// ordinary CFG edge, retaining both bodies in one register and LDS lifetime.
	joined_code[front_words - 1u] = 0xbf820000u; // s_branch to the following instruction
	result.instructions.back()    = {};
	Decoder::DecodeInstruction(joined_code, front_words - 1u, result.instructions.back());
	Decoder::Program back_program;
	Decoder::DecodeProgram(back, back_program);
	result.has_swap_pc |= back_program.has_swap_pc;
	const auto back_pc = front_words * sizeof(uint32_t);
	for (auto& inst: back_program.instructions) {
		// A back-stage PC-relative data reference requires its guest code address.
		EXIT_NOT_IMPLEMENTED(inst.opcode == Decoder::Opcode::S_GETPC_B64);
		inst.pc += back_pc;
		inst.branch_target += back_pc;
		result.instructions.push_back(std::move(inst));
	}
	result.code = joined_code;
	return result;
}

// CFG::MayWriteScalarRegister also reports the SGPR after any scalar destination because not
// every 64-bit destination width is decoded. Call-target checks need the exact SGPRs: scalar ALU
// destinations are 64-bit only for *_B64/_U64/_I64 results, SMEM loads have decoded widths, and
// VALU scalar destinations (compare masks, carries) are lane masks of the wave size.
bool WritesScalarRegister(const Decoder::Instruction& inst, uint32_t reg, uint32_t wave_size) {
	if (!CFG::MayWriteScalarRegister(inst, reg)) return false;
	if (inst.dst.kind != Decoder::OperandKind::Sgpr || inst.dst.reg == reg || inst.dst.reg + 1u != reg)
		return true;
	using Family = Decoder::Family;
	const std::string_view name = magic_enum::enum_name(inst.opcode);
	switch (inst.family) {
		case Family::SMEM: return inst.data_dwords > 1u;
		case Family::SOP1:
		case Family::SOP2:
		case Family::SOPK:
		case Family::SOPC: {
			// The first type suffix names the destination (S_BCNT1_I32_B64 writes 32 bits).
			for (size_t at = name.find('_'); at != std::string_view::npos; at = name.find('_', at + 1u)) {
				const auto token = name.substr(at + 1u, 3u);
				if (token == "B32" || token == "U32" || token == "I32") return false;
				if (token == "B64" || token == "U64" || token == "I64") return true;
			}
			return true;
		}
		case Family::VOP1:
		case Family::VOP2:
		case Family::VOP3:
		case Family::VOP3P:
		case Family::VOPC:
			if (name.starts_with("V_READFIRSTLANE") || name.starts_with("V_READLANE")) return false;
			return wave_size == 64u;
		default: return true;
	}
}

void PrepareCallTarget(ShaderSource& source, const CompileOptions& options) {
	using namespace IR;
	std::vector<uint32_t> calls;
	for (uint32_t i = 0; i < source.decoded.instructions.size(); ++i) {
		if (source.decoded.instructions[i].opcode != Decoder::Opcode::S_SWAPPC_B64) continue;
		EXIT_NOT_IMPLEMENTED(options.stage != ShaderType::Compute);
		calls.push_back(i);
	}
	EXIT_IF(calls.empty());
	source.call      = ShaderSource::Call {.instruction = calls.front()};
	source.wave_size = options.wave_size;
	// Each call loads its target pair with S_MOV_B32 in the straight-line code before it, from
	//  - an SGPR pair that still holds its entry (user data) value: a fixed helper, or
	//  - an S_BUFFER_LOAD with a dynamic offset: a function table indexed at run time.
	// A single call may instead follow general reaching definitions (the scalar-leaf case).
	// Helpers are checked when linking not to write registers that hold fixed helper addresses.
	const auto& instructions = source.decoded.instructions;
	std::vector<bool> labels(instructions.size(), false);
	bool              forward_only = true;
	for (uint32_t i = 0; i < instructions.size(); ++i) {
		const auto& inst = instructions[i];
		if (!Decoder::IsDirectBranch(inst.opcode)) continue;
		forward_only = forward_only && inst.branch_target > inst.pc;
		for (uint32_t j = 0; j < instructions.size(); ++j) {
			if (instructions[j].pc == inst.branch_target) labels[j] = true;
		}
	}
	std::array<bool, 128> written {};
	for (const auto& inst: instructions) {
		for (uint32_t reg = 0; reg < written.size(); ++reg) {
			written[reg] = written[reg] || WritesScalarRegister(inst, reg, options.wave_size);
		}
	}
	// The straight-line writer of `reg` before instruction `index`, or UINT32_MAX.
	const auto StraightLineWriter = [&](uint32_t index, uint32_t reg) {
		for (uint32_t i = index; i > 0u;) {
			const auto& inst = instructions[--i];
			if (inst.opcode == Decoder::Opcode::S_SWAPPC_B64 || Decoder::IsDirectBranch(inst.opcode))
				return UINT32_MAX;
			if (WritesScalarRegister(inst, reg, options.wave_size)) return i;
			if (labels[i]) return UINT32_MAX;
		}
		return UINT32_MAX;
	};
	const auto IsUserData = [&](uint32_t reg) {
		return reg >= options.user_data_base && reg - options.user_data_base < options.user_data.size();
	};
	struct CallPlan {
		enum class Kind { Generic, Entry, Table } kind = Kind::Generic;
		std::array<uint32_t, 2> sources {UINT32_MAX, UINT32_MAX};
		uint32_t                load = UINT32_MAX;
	};
	std::vector<CallPlan> plans;
	for (const auto index: calls) {
		const auto& call_inst = instructions[index];
		EXIT_NOT_IMPLEMENTED(call_inst.src0.kind != Decoder::OperandKind::Sgpr ||
		                     call_inst.dst.kind != Decoder::OperandKind::Sgpr ||
		                     call_inst.src0.reg != call_inst.dst.reg || call_inst.dst.reg >= 105u);
		CallPlan                plan;
		std::array<uint32_t, 2> movs {UINT32_MAX, UINT32_MAX};
		for (uint32_t half = 0; half < 2u; ++half) {
			const auto writer = StraightLineWriter(index, call_inst.src0.reg + half);
			if (writer == UINT32_MAX) continue;
			const auto& inst = instructions[writer];
			if (inst.opcode == Decoder::Opcode::S_MOV_B32 && inst.src0.kind == Decoder::OperandKind::Sgpr) {
				movs[half]         = writer;
				plan.sources[half] = inst.src0.reg;
			}
		}
		const bool have_movs = movs[0] != UINT32_MAX && movs[1] != UINT32_MAX;
		const auto entry_value = [&](uint32_t half) {
			const auto reg = plan.sources[half];
			if (!IsUserData(reg) || reg >= written.size()) return false;
			if (!written[reg]) return true;
			if (!forward_only) return false;
			for (uint32_t i = 0; i < movs[half]; ++i) {
				if (WritesScalarRegister(instructions[i], reg, options.wave_size)) return false;
			}
			return true;
		};
		if (have_movs && entry_value(0) && entry_value(1)) {
			plan.kind = CallPlan::Kind::Entry;
		} else if (have_movs && plan.sources[1] == plan.sources[0] + 1u) {
			const auto load = StraightLineWriter(movs[0], plan.sources[0]);
			bool       selector_kept = true;
			for (uint32_t i = movs[0] + 1u; i < index; ++i) {
				selector_kept = selector_kept &&
				                !WritesScalarRegister(instructions[i], plan.sources[0], options.wave_size);
			}
			if (load != UINT32_MAX && selector_kept) {
				const auto& inst    = instructions[load];
				const bool  dynamic = inst.src1.kind == Decoder::OperandKind::Sgpr ||
				                     inst.src1.kind == Decoder::OperandKind::VccLo ||
				                     inst.src1.kind == Decoder::OperandKind::VccHi ||
				                     inst.src1.kind == Decoder::OperandKind::M0;
				if ((inst.opcode == Decoder::Opcode::S_BUFFER_LOAD_DWORDX2 ||
				     inst.opcode == Decoder::Opcode::S_BUFFER_LOAD_DWORDX4 ||
				     inst.opcode == Decoder::Opcode::S_BUFFER_LOAD_DWORDX8 ||
				     inst.opcode == Decoder::Opcode::S_BUFFER_LOAD_DWORDX16) &&
				    dynamic && inst.dst.kind == Decoder::OperandKind::Sgpr &&
				    inst.src0.kind == Decoder::OperandKind::Sgpr && plan.sources[0] >= inst.dst.reg &&
				    plan.sources[1] < inst.dst.reg + inst.data_dwords) {
					plan.kind = CallPlan::Kind::Table;
					plan.load = load;
				}
			}
		}
		EXIT_NOT_IMPLEMENTED(plan.kind == CallPlan::Kind::Generic && calls.size() > 1u);
		if (index != calls.front()) source.extra_calls.push_back({.instruction = index});
		if (plan.kind == CallPlan::Kind::Entry && calls.size() > 1u) {
			for (const auto reg: plan.sources) {
				if (std::ranges::find(source.call_source_registers, reg) == source.call_source_registers.end())
					source.call_source_registers.push_back(reg);
			}
		}
		plans.push_back(plan);
	}
	EXIT_IF(options.input_info.compute == nullptr);
	const auto& call = source.decoded.instructions[source.call->instruction];
	EXIT_NOT_IMPLEMENTED(call.src0.kind != Decoder::OperandKind::Sgpr ||
	                     call.dst.kind != Decoder::OperandKind::Sgpr ||
	                     call.src0.reg != call.dst.reg || call.dst.reg >= 105u);
	const auto graph = CFG::BuildGraph(source.decoded);
	Program query;
	query.stage = options.stage;
	query.user_data_base = options.user_data_base;
	query.user_data_count = static_cast<uint32_t>(options.user_data.size());
	query.wave_size = options.wave_size;
	auto& block = *query.block_storage.emplace_back(std::make_unique<Block>());
	query.blocks.push_back(&block);
	struct Producer { bool ready = false; std::map<uint32_t, Value> values; };
	std::map<uint32_t, Producer> producers;
	std::function<Value(uint32_t, uint32_t)> Register;
	std::function<Value(uint32_t, Value)> Resolve = [&](uint32_t before, Value value) -> Value {
		value = value.Resolve();
		auto* inst = value.TryInstruction();
		if (inst == nullptr) return value;
		uint32_t code = UINT32_MAX;
		switch (inst->GetOpcode()) {
			case ValueOpcode::GetScalarRegister: code = static_cast<uint32_t>(inst->Arg(0).ScalarRegister()); break;
			case ValueOpcode::GetVccLo: code = 106u; break;
			case ValueOpcode::GetVccHi: code = 107u; break;
			default: break;
		}
		if (code != UINT32_MAX) {
			const auto result = Register(before, code);
			inst->ReplaceUsesWith(result);
			return result;
		}
		for (size_t i = 0; i < inst->NumArgs(); ++i) inst->SetArg(i, Resolve(before, inst->Arg(i)));
		return value;
	};
	Register = [&](uint32_t before, uint32_t code) -> Value {
		// An SGPR that no instruction writes holds its entry value everywhere.
		const auto index = code < written.size() && !written[code]
		                       ? UINT32_MAX
		                       : CFG::FindScalarDefinition(source.decoded, graph, before, code);
		if (index == UINT32_MAX) {
			EXIT_IF(code < query.user_data_base || code - query.user_data_base >= query.user_data_count);
			return IREmitter(&block).GetUserData(static_cast<ScalarReg>(code));
		}
		auto [entry, inserted] = producers.try_emplace(index);
		auto& producer = entry->second;
		if (inserted) {
			Block translated;
			Frontend::Translator translator(query, &translated, 1u,
			    (options.input_info.compute->float_mode & 0x10u) == 0u,
			    !options.input_info.compute->async_compute);
			translator.TranslateInstruction(source.decoded.instructions[index]);
			for (const auto& inst: translated) {
				uint32_t destination = UINT32_MAX;
				Value value;
				switch (inst.GetOpcode()) {
					case ValueOpcode::SetScalarRegister:
						destination = static_cast<uint32_t>(inst.Arg(0).ScalarRegister()); value = inst.Arg(1); break;
					case ValueOpcode::SetVccLo: destination = 106u; value = inst.Arg(0); break;
					case ValueOpcode::SetVccHi: destination = 107u; value = inst.Arg(0); break;
					default: break;
				}
				if (destination != UINT32_MAX) producer.values.insert_or_assign(destination, value);
			}
			for (auto& [destination, value]: producer.values) value = Resolve(index, value);
			for (auto& inst: translated) inst.SetParent(&block);
			block.Instructions().splice(block.end(), translated.Instructions());
			producer.ready = true;
		}
		EXIT_IF(!producer.ready || !producer.values.contains(code));
		return producer.values.at(code).Resolve();
	};
	std::vector<ShaderSource::Call*> linked_calls {&*source.call};
	for (auto& extra: source.extra_calls) linked_calls.push_back(&extra);
	const auto Root = [&](Value value) {
		EXIT_IF(!ValidateRuntimeValue(query, value, RuntimeValueType::Integer));
		auto& root = query.value_storage.emplace_back(ValueOpcode::ReferenceU32);
		root.SetArg(0, value);
		return Value(&root);
	};
	for (size_t index = 0; index < linked_calls.size(); ++index) {
		auto&       linked_call = *linked_calls[index];
		const auto& plan        = plans[index];
		if (plan.kind == CallPlan::Kind::Table) {
			const auto& load          = instructions[plan.load];
			linked_call.table         = true;
			linked_call.selector_sgpr = plan.sources[0];
			// The dynamic offset selects whole entries of the loaded size.
			linked_call.entry_bytes   = load.data_dwords * static_cast<uint32_t>(sizeof(uint32_t));
			linked_call.target_offset = static_cast<int64_t>(static_cast<int32_t>(load.offset)) +
			                            int64_t {plan.sources[0] - load.dst.reg} * 4;
			for (uint32_t i = 0; i < 4u; ++i) {
				linked_call.table_descriptor[i] = Root(Register(plan.load, load.src0.reg + i));
			}
			continue;
		}
		for (uint32_t i = 0; i < 2u; ++i) {
			// A generic call follows its reaching definitions; others read entry values.
			linked_call.target[i] = Root(plan.kind == CallPlan::Kind::Generic
			                                 ? Register(linked_call.instruction, call.src0.reg + i)
			                                 : Register(0u, plan.sources[i]));
		}
	}
	RewriteToSsa(query.blocks);
	ConstantPropagationPass(query.blocks, query.wave_size);
	RemoveIdentities(query.blocks);
	EliminateDeadCode(query.blocks);
	for (auto* linked_call: linked_calls) {
		for (auto& value: linked_call->target) {
			if (!value.IsEmpty()) value = value.Instruction()->Arg(0).Resolve();
		}
		for (auto& value: linked_call->table_descriptor) {
			if (!value.IsEmpty()) value = value.Instruction()->Arg(0).Resolve();
		}
	}
	query.value_storage.clear();
	for (auto& inst: block) {
		inst.SetParent(nullptr);
		if (inst.GetOpcode() == ValueOpcode::LoadAddressU32 || inst.GetOpcode() == ValueOpcode::ReadConstBuffer)
			inst.SetFlags(SrtReadFlags {.index = inst.Flags<MemoryFlags>().index});
	}
	query.value_storage.splice(query.value_storage.end(), block.Instructions());
	source.call_targets = std::move(static_cast<ResourcePlan&>(query));
}

} // namespace

ShaderSource PrepareShaderSource(std::span<const uint32_t> code, const CompileOptions& options) {
	EXIT_IF(code.empty());
	ShaderSource source;
	if (!options.back_code.empty()) {
		source.decoded = DecodeFusedProgram(code, options.back_code, source.code);
	} else if (options.stage == ShaderType::Local) {
		source.decoded = Decoder::DecodeFrontProgram(code);
		auto& handoff = source.decoded.instructions.back();
		handoff.opcode = Decoder::Opcode::S_ENDPGM;
		handoff.src_count = 0;
	} else {
		Decoder::DecodeProgram(code, source.decoded);
	}
	if (source.decoded.has_swap_pc) {
		PrepareCallTarget(source, options);
		source.code.assign(code.begin(), code.end());
		source.decoded.code = source.code;
	}
	return source;
}

const Decoder::Program& RefreshShaderSource(ShaderSource& source, const IR::SrtRuntime& runtime) {
	auto& reads = source.reads;
	reads.clear();
	IR::SrtReadCapture capture(runtime, reads);
	const auto clean = IR::CleanRuntime(capture.ObservedRuntime());
	IR::SrtWalker walker(source.call_targets, clean);
	const auto& call = source.decoded.instructions[source.call->instruction];
	std::vector<const ShaderSource::Call*> calls {&*source.call};
	for (const auto& extra: source.extra_calls) calls.push_back(&extra);
	// Raw target values per call (a function-table call lists every distinct table entry), and
	// the distinct helpers in first-use order. SWAPPC ignores the target's low two bits.
	constexpr size_t MaxTableEntries = 4096;
	constexpr size_t MaxTableHelpers = 64;
	std::vector<std::vector<uint64_t>>         call_targets;
	std::vector<std::pair<uint64_t, uint64_t>> table_reads;
	std::vector<uint64_t>                      helper_addresses;
	for (const auto* linked_call: calls) {
		const auto& call_pc = source.decoded.instructions[linked_call->instruction].pc;
		auto&       targets = call_targets.emplace_back();
		if (linked_call->table) {
			std::array<uint32_t, 4> table {};
			for (uint32_t i = 0; i < 4u; ++i) {
				if (!walker.Evaluate(linked_call->table_descriptor[i], table[i]))
					EXIT("shader call at pc 0x%08x has an unavailable function table", call_pc);
			}
			const uint64_t base   = table[0] | (uint64_t {table[1] & 0xffffu} << 32u);
			const uint32_t stride = (table[1] >> 16u) & 0x3fffu;
			const uint64_t bytes  = stride == 0u ? table[2] : uint64_t {table[2]} * stride;
			const auto     count  = std::min<uint64_t>(bytes / linked_call->entry_bytes, MaxTableEntries);
			const auto     first  = static_cast<int64_t>(base) + linked_call->target_offset;
			EXIT_IF(first <= 0 || count == 0u);
			const auto            span_bytes = (count - 1u) * linked_call->entry_bytes + 8u;
			std::vector<uint32_t> words((span_bytes + 3u) / 4u);
			if (!clean.read_memory(clean.userdata, static_cast<uint64_t>(first), words))
				EXIT("shader call at pc 0x%08x cannot read its function table", call_pc);
			table_reads.emplace_back(static_cast<uint64_t>(first), span_bytes);
			for (uint64_t entry = 0; entry < count; ++entry) {
				const auto word = entry * linked_call->entry_bytes / sizeof(uint32_t);
				const auto raw  = (uint64_t {words[word + 1u]} << 32u) | words[word];
				if ((raw & ~uint64_t {3}) == 0u) continue;
				// Cases are selected by the low 32 bits.
				if (std::ranges::any_of(targets, [&](uint64_t known) {
					    return static_cast<uint32_t>(known) == static_cast<uint32_t>(raw);
				    }))
					continue;
				EXIT_NOT_IMPLEMENTED(targets.size() >= MaxTableHelpers);
				targets.push_back(raw);
			}
			if (targets.empty()) EXIT("shader call at pc 0x%08x has an empty function table", call_pc);
		} else {
			uint32_t low = 0, high = 0;
			if (!walker.Evaluate(linked_call->target[0], low) ||
			    !walker.Evaluate(linked_call->target[1], high))
				EXIT("shader call at pc 0x%08x has an unavailable scalar target", call_pc);
			targets.push_back((uint64_t {high} << 32u) | low);
		}
		for (const auto raw: targets) {
			const auto address = raw & ~uint64_t {3};
			EXIT_IF(address == 0u);
			if (std::ranges::find(helper_addresses, address) == helper_addresses.end())
				helper_addresses.push_back(address);
		}
	}
	const auto query_reads = reads.size();
	const auto RecordReads = [&](const ShaderSource::Linked& linked) {
		reads.resize(query_reads);
		for (const auto& [address, words]: linked.helpers) {
			reads.emplace_back(address, uint64_t {words} * sizeof(uint32_t));
		}
		reads.insert(reads.end(), table_reads.begin(), table_reads.end());
	};
	if (source.linked && source.linked->helpers.size() == helper_addresses.size() &&
	    source.linked->call_targets == call_targets) {
		auto& linked = *source.linked;
		const auto function = std::span(linked.code).subspan(source.code.size());
		bool       same     = function.size() == linked.observed_function.size();
		size_t     offset   = 0;
		for (size_t index = 0; same && index < helper_addresses.size(); ++index) {
			const auto words    = linked.helpers[index].second;
			const auto observed = std::span(linked.observed_function).subspan(offset, words);
			same = clean.read_memory(clean.userdata, helper_addresses[index], observed) &&
			       std::ranges::equal(function.subspan(offset, words), observed);
			offset += words;
		}
		if (same) {
			// Identical code may move; record where it was read this time.
			for (size_t index = 0; index < helper_addresses.size(); ++index) {
				linked.helpers[index].first = helper_addresses[index];
			}
			RecordReads(linked);
			return linked.decoded;
		}
		reads.resize(query_reads);
	}
	ShaderSource::Linked linked {.code = source.code,
	                             .decoded = source.decoded};
	const auto& last = source.decoded.instructions.back();
	EXIT_NOT_IMPLEMENTED(last.opcode != Decoder::Opcode::S_ENDPGM);
	// Decode each helper once with helper-relative PCs. A helper ends at the S_SETPC_B64 return
	// that no helper branch jumps past; earlier returns are allowed.
	struct HelperInstruction {
		Decoder::Instruction inst;
		bool                 returns = false;
	};
	std::vector<std::vector<HelperInstruction>> helpers;
	for (const auto address: helper_addresses) {
		auto& helper = helpers.emplace_back();
		constexpr uint32_t MaxHelperWords = 16384u;
		uint32_t           furthest_target = 0;
		const auto         code_begin      = linked.code.size();
		for (uint32_t index = 0;;) {
			EXIT_IF(index >= MaxHelperWords ||
			        address > UINT64_MAX - (uint64_t {index} + Decoder::MaxInstructionRawWords) *
			                                   sizeof(uint32_t));
			std::array<uint32_t, Decoder::MaxInstructionRawWords> words {};
			const auto word_address = address + uint64_t {index} * sizeof(uint32_t);
			// Read as many words as are mapped (a helper may end next to unmapped memory).
			size_t available = words.size();
			while (available > 0u &&
			       !clean.read_memory(clean.userdata, word_address, std::span(words).first(available))) {
				--available;
			}
			if (available == 0u) EXIT("shader call at pc 0x%08x cannot read helper code", call.pc);
			Decoder::Instruction inst;
			Decoder::DecodeInstruction(words, 0u, inst);
			EXIT_IF(inst.word_count > available);
			inst.pc = index * sizeof(uint32_t);
			if (Decoder::IsDirectBranch(inst.opcode)) {
				inst.branch_target += inst.pc;
				furthest_target = std::max(furthest_target, inst.branch_target);
			}
			linked.code.insert(linked.code.end(), inst.raw, inst.raw + inst.word_count);
			index += inst.word_count;
			EXIT_NOT_IMPLEMENTED(inst.opcode == Decoder::Opcode::S_SWAPPC_B64 ||
			                     inst.opcode == Decoder::Opcode::S_GETPC_B64 ||
			                     inst.opcode == Decoder::Opcode::S_ENDPGM);
			EXIT_NOT_IMPLEMENTED(WritesScalarRegister(inst, call.dst.reg, source.wave_size) ||
			                     WritesScalarRegister(inst, call.dst.reg + 1u, source.wave_size));
			for (const auto reg: source.call_source_registers) {
				EXIT_NOT_IMPLEMENTED(WritesScalarRegister(inst, reg, source.wave_size));
			}
			const bool returns = inst.opcode == Decoder::Opcode::S_SETPC_B64;
			if (returns) {
				EXIT_NOT_IMPLEMENTED(inst.src0.kind != Decoder::OperandKind::Sgpr ||
				                     inst.src0.reg != call.dst.reg);
			}
			helper.push_back({std::move(inst), returns});
			if (returns && furthest_target < index * sizeof(uint32_t)) break;
		}
		const auto size = helper.back().inst.pc + helper.back().inst.word_count * sizeof(uint32_t);
		for (const auto& entry: helper) {
			if (!entry.returns && Decoder::IsDirectBranch(entry.inst.opcode))
				EXIT_IF(entry.inst.branch_target >= size);
		}
		linked.helpers.emplace_back(address, static_cast<uint32_t>(linked.code.size() - code_begin));
	}
	// Keep the caller's data/footer bytes at their native offsets. Each call's copy of its helper
	// gets virtual CFG PCs after the decoded instructions; helper code is read and owned once.
	auto function_pc = last.pc + last.word_count * sizeof(uint32_t);
	for (size_t index = 0; index < calls.size(); ++index) {
		const auto call_index = calls[index]->instruction;
		const auto return_pc  = linked.decoded.instructions[call_index].pc +
		                       linked.decoded.instructions[call_index].word_count * sizeof(uint32_t);
		Decoder::IndirectCall indirect {.pc            = linked.decoded.instructions[call_index].pc,
		                                .selector_sgpr = calls[index]->selector_sgpr};
		for (const auto raw: call_targets[index]) {
			const auto helper_index = static_cast<size_t>(
			    std::ranges::find(helper_addresses, raw & ~uint64_t {3}) - helper_addresses.begin());
			const auto& helper = helpers[helper_index];
			const auto  helper_size =
			    helper.back().inst.pc + helper.back().inst.word_count * sizeof(uint32_t);
			EXIT_IF(function_pc > UINT32_MAX - helper_size);
			if (indirect.cases.empty()) linked.decoded.instructions[call_index].branch_target = function_pc;
			indirect.cases.emplace_back(static_cast<uint32_t>(raw), function_pc);
			for (const auto& entry: helper) {
				auto inst = entry.inst;
				if (entry.returns) {
					inst.opcode        = Decoder::Opcode::S_BRANCH;
					inst.branch_target = return_pc;
				} else if (Decoder::IsDirectBranch(inst.opcode)) {
					inst.branch_target += function_pc;
				}
				inst.pc += function_pc;
				linked.decoded.instructions.push_back(std::move(inst));
			}
			function_pc += helper_size;
		}
		if (calls[index]->table) linked.decoded.indirect_calls.push_back(std::move(indirect));
	}
	linked.call_targets = call_targets;
	RecordReads(linked);
	linked.decoded.code = linked.code;
	linked.observed_function.resize(linked.code.size() - source.code.size());
	source.linked = std::move(linked);
	++source.revision;
	return source.linked->decoded;
}

TranslateResult TranslateProgram(std::span<const uint32_t> code, const CompileOptions& options) {
	auto source = PrepareShaderSource(code, options);
	return TranslateProgram(source.decoded, options);
}

TranslateResult TranslateProgram(const Decoder::Program& decoded, const CompileOptions& options) {
	EXIT_IF(decoded.instructions.empty());
	if (decoded.has_swap_pc) {
		for (const auto& inst: decoded.instructions) {
			if (inst.opcode == Decoder::Opcode::S_SWAPPC_B64 && inst.branch_target == UINT32_MAX)
				EXIT("shader call at pc 0x%08x must be linked before translation", inst.pc);
		}
	}
	if (options.stage != ShaderType::Compute && options.stage != ShaderType::Vertex &&
	    options.stage != ShaderType::Pixel && options.stage != ShaderType::Mesh &&
	    options.stage != ShaderType::Local && options.stage != ShaderType::TessellationControl &&
	    options.stage != ShaderType::TessellationEvaluation) {
		EXIT("shader recompiler received unsupported stage %u\n",
		     static_cast<unsigned>(options.stage));
	}

	const auto compile_begin = std::chrono::steady_clock::now();
	const auto phase_ms = [&compile_begin]() {
		return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
		    std::chrono::steady_clock::now() - compile_begin).count());
	};

	std::string decoded_dump;
	if (options.dump_ir) {
		decoded_dump = Decoder::ProgramToString(decoded);
		if (options.early_dump) {
			LOGF("%s decoded RDNA2 (early):\n%s", GetDumpLabel(options), decoded_dump.c_str());
		}
	}

	LOGF("%s phase begin: stage=%s hash=0x%016" PRIx64 " CFG BuildGraph\n", GetDumpLabel(options),
	     StageName(options.stage), options.shader_hash);
	auto native_cfg = CFG::BuildGraph(decoded);
	CFG::Graph structured_cfg;
	auto* selected_cfg = &native_cfg;
	LOGF("%s phase end: stage=%s hash=0x%016" PRIx64 " CFG BuildGraph blocks=%" PRIu64
	     " loops=%" PRIu64 " back_edges=%" PRIu64 " elapsed_ms=%" PRIu64 "\n",
	     GetDumpLabel(options), StageName(options.stage), options.shader_hash,
	     static_cast<uint64_t>(native_cfg.blocks.size()),
	     static_cast<uint64_t>(native_cfg.natural_loops.size()),
	     static_cast<uint64_t>(native_cfg.back_edges.size()), phase_ms());
	if (native_cfg.irreducible) {
		LogDispatcherFallback(options, native_cfg, "build");
	} else {
		LOGF("%s phase begin: stage=%s hash=0x%016" PRIx64 " CFG Structurize\n",
		     GetDumpLabel(options), StageName(options.stage), options.shader_hash);
		structured_cfg = CFG::Structurize(native_cfg);
		if (structured_cfg.unsupported) {
			native_cfg.unsupported = true;
			native_cfg.failure_kind = structured_cfg.failure_kind;
			native_cfg.failure_block = structured_cfg.failure_block;
			native_cfg.unsupported_reason = structured_cfg.unsupported_reason;
			LogDispatcherFallback(options, native_cfg, "structurize");
		} else {
			selected_cfg = &structured_cfg;
			LOGF("%s structured CFG success: blocks=%" PRIu64 "\n", GetDumpLabel(options),
			     static_cast<uint64_t>(selected_cfg->blocks.size()));
		}
		LOGF("%s phase end: stage=%s hash=0x%016" PRIx64 " CFG Structurize blocks=%" PRIu64
		     " loops=%" PRIu64 " elapsed_ms=%" PRIu64 "\n",
		     GetDumpLabel(options), StageName(options.stage), options.shader_hash,
		     static_cast<uint64_t>(selected_cfg->blocks.size()),
		     static_cast<uint64_t>(selected_cfg->natural_loops.size()), phase_ms());
	}

	const auto& cfg = *selected_cfg;
	Frontend::EmbeddedFetchPlan embedded_fetch;
	if ((options.stage == ShaderType::Vertex || options.stage == ShaderType::Local) &&
	    options.input_info.vertex != nullptr && options.input_info.vertex->fetch_embedded) {
		embedded_fetch = DetectEmbeddedVertexFetch(
		    decoded, options.input_info.vertex, options.user_data_base,
		    static_cast<uint32_t>(options.user_data.size()), options.wave_size);
		if (!embedded_fetch.loads.empty()) {
			LOGF("%s embedded vertex fetch plan: detected=%" PRIu64 "\n", GetDumpLabel(options),
			     static_cast<uint64_t>(embedded_fetch.loads.size()));
		}
	}
	Frontend::TranslateOptions translate_options {
	    .stage            = options.stage,
	    .wave_size        = options.wave_size,
	    .shader_hash      = options.shader_hash,
	    .user_data_base   = options.user_data_base,
	    .user_data_count  = static_cast<uint32_t>(options.user_data.size()),
	    .input_info       = options.input_info,
	    .embedded_fetch   = embedded_fetch.loads.empty() ? nullptr : &embedded_fetch,
	};
	LOGF("%s phase begin: stage=%s hash=0x%016" PRIx64 " IR TranslateProgram\n",
	     GetDumpLabel(options), StageName(options.stage), options.shader_hash);
	auto ir = Frontend::TranslateProgram(decoded, cfg, translate_options);
	LOGF("%s phase end: stage=%s hash=0x%016" PRIx64 " IR TranslateProgram blocks=%" PRIu64
	     " elapsed_ms=%" PRIu64 "\n",
	     GetDumpLabel(options), StageName(options.stage), options.shader_hash,
	     static_cast<uint64_t>(ir.blocks.size()), phase_ms());
	IR::RewriteToSsa(ir.blocks);
	IR::ConstantPropagationPass(ir.blocks, ir.wave_size);
	IR::ResolveControlFlowIdentities(ir);
	IR::RemoveIdentities(ir.blocks);
	IR::EliminateDeadCode(ir.blocks);
	const auto read_lane_stats = IR::EliminateReadLane(ir, ir.wave_size);
	if (read_lane_stats.rewritten_reads != 0) {
		LOGF("%s read-lane elimination: reads=%" PRIu32 "\n", GetDumpLabel(options),
		     read_lane_stats.rewritten_reads);
		IR::ConstantPropagationPass(ir.blocks, ir.wave_size);
		IR::ResolveControlFlowIdentities(ir);
		IR::RemoveIdentities(ir.blocks);
		IR::EliminateDeadCode(ir.blocks);
	}
	// Linked programs: helper returns are resolved CFG edges and SWAPPC writes no native PC (see
	// S_GETPC_B64). Helpers may not use S_GETPC_B64, so remaining shader-base values come from the
	// caller, whose instructions keep their native PCs.
	LowerTessellationMemory(ir, options);
	std::string cfg_dump;
	if (options.dump_ir) {
		cfg_dump = CFG::GraphToString(cfg);
		if (options.early_dump) {
			LOGF("%s native IR before resource tracking:\n%s", GetDumpLabel(options),
			     MakeIrDump(cfg_dump, ir).c_str());
		}
	}
	IR::TrackResources(ir, decoded, native_cfg);
	TranslateResult result;
	result.program = std::move(ir);
	if (options.dump_ir) {
		result.decoded_dump = std::move(decoded_dump);
		result.cfg_dump     = std::move(cfg_dump);
	}
	return result;
}

CompileResult CompileProgram(TranslateResult translated, const CompileOptions& options,
                             const IR::ResourceSpecialization& specialization,
                             uint32_t push_data_start_dword) {
	const auto emit_begin = std::chrono::steady_clock::now();
	auto& ir = translated.program;
	IR::ApplyResourceSpecialization(ir, specialization);
	// The resource plan owns host descriptor evaluation now. Keep only dependencies consumed
	// by GPU memory operations; bound descriptor dwords must not retain shader instructions.
	for (auto& inst: ir.value_storage) {
		inst.Invalidate();
	}
	for (auto* block: ir.blocks) {
		for (auto& inst: *block) {
			const auto op = inst.GetOpcode();
			uint32_t first = 0;
			if (op == IR::ValueOpcode::GetBufferResource) {
				if (std::ranges::any_of(inst.Uses(), [&](const IR::Use& use) {
					return ir.memory_info[use.user->Flags<IR::MemoryFlags>().index].kind ==
					       IR::ResourceKind::IndirectBuffer;
				})) {
					continue;
				}
				const auto resource = inst.Flags<uint32_t>();
				first               = resource < ir.info.buffers.size() &&
				                              ir.info.buffers[resource].indirect_root == resource
				                          ? 1u
				                          : 0u;
			} else if (op == IR::ValueOpcode::GetImageResource) {
				const auto resource = inst.Flags<uint32_t>();
				first = resource < ir.info.images.size() &&
				                ir.info.images[resource].indirect_root == resource ? 1u : 0u;
			} else if (op == IR::ValueOpcode::GetSamplerResource) {
				const auto resource = inst.Flags<uint32_t>();
				first = resource < ir.info.samplers.size() &&
				                !ir.info.samplers[resource].indirect_resources.empty() ? 1u : 0u;
			} else {
				continue;
			}
			for (size_t index = first; index < inst.NumArgs(); index++) {
				inst.SetArg(index, IR::Value(0u));
			}
		}
	}
	ir.value_storage.clear();
	IR::RemoveIdentities(ir.blocks);
	IR::EliminateDeadCode(ir.blocks);

	IR::CollectShaderInfo(ir, options.input_info);
	std::string ir_dump;
	if (options.dump_ir) {
		ir_dump = MakeIrDump(translated.cfg_dump, ir);
		if (options.early_dump) {
			LOGF("%s native IR (early):\n%s", GetDumpLabel(options), ir_dump.c_str());
		}
	}

	LOGF("%s phase begin: stage=%s hash=0x%016" PRIx64 " SPIR-V EmitProgram\n",
	     GetDumpLabel(options), StageName(ir.stage), ir.shader_hash);
	auto spirv = Spirv::EmitProgram(ir, options.input_info, push_data_start_dword);
	LOGF("%s phase end: stage=%s hash=0x%016" PRIx64 " SPIR-V EmitProgram words=%" PRIu64
	     " elapsed_ms=%" PRIu64 "\n",
	     GetDumpLabel(options), StageName(ir.stage), ir.shader_hash,
	     static_cast<uint64_t>(spirv.size()),
	     static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                               std::chrono::steady_clock::now() - emit_begin)
	                               .count()));
	CompileResult result;
	result.spirv   = std::move(spirv);
	result.program = std::move(ir);
	if (options.dump_ir) {
		result.decoded_dump = std::move(translated.decoded_dump);
		result.ir_dump      = std::move(ir_dump);
	}
	return result;
}

} // namespace Libs::Graphics::ShaderRecompiler
