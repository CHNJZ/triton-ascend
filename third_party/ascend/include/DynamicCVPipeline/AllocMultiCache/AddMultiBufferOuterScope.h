/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_ADAPTER_ALLOC_MULTI_CACHE_ADD_MULTI_BUFFER_OUTER_SCOPE_PASS_H
#define TRITON_ADAPTER_ALLOC_MULTI_CACHE_ADD_MULTI_BUFFER_OUTER_SCOPE_PASS_H

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir {
namespace triton {

// ============================================================================
// 数据结构
// ============================================================================

/// 传输操作链（用于发端或收端）
struct TransferOpChain {
    Operation *waitOp = nullptr;           // sync_block_wait
    Operation *transferOp = nullptr;        // fixpipe / hir.copy / memory_space_cast / convert_layout
    Operation *toTensorOp = nullptr;       // bufferization.to_tensor (仅 memspacecast 场景有)
    Operation *setOp = nullptr;             // sync_block_set
};

/// Buffer alloc 对
struct BufferAllocPair {
    Operation *allocOp = nullptr;
    Operation *markOp = nullptr;
};

/// 传输组完整信息
struct TransferGroupInfo {
    int tid = -1;
    int originalFlag = -1;      // 原始 flag
    int outputFlag = -1;          // 新分配的 output flag
    bool isCtoV = false;        // true=C→V, false=V→C

    BufferAllocPair senderBuf;   // 发送端 buffer (producer)
    BufferAllocPair receiverBuf; // 接收端 buffer (consumer)

    TransferOpChain senderChain;   // 发送端操作链
    TransferOpChain receiverChain; // 接收端操作链

    // Input/output buffer values (for later steps)
    Value senderInputBuffer;      // sender input buffer (original)
    Value senderOutputBuffer;      // sender output buffer (newly created)
    Value receiverInputBuffer;    // receiver input buffer (original)
    Value receiverOutputBuffer;    // receiver output buffer (newly created)

    // TCB ID: 同一 tid 组的所有 buffer（共 4 个：sender input/output + receiver input/output）共用一个 tcb_id
    int tcbId = -1;

    // Extra sync 位置（用于 output flag 同步）
    Operation *extraSyncSetOp = nullptr;  // extra set op 的位置（插入点）
    Operation *extraSyncWaitOp = nullptr;  // extra wait op 的位置（插入点）
};

/// AddMultiBufferOuterScopePass for adding outer (CV inter-core) multi-buffer optimization
class AddMultiBufferOuterScopePass
    : public PassWrapper<AddMultiBufferOuterScopePass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddMultiBufferOuterScopePass)

    // Constructor
    AddMultiBufferOuterScopePass() = default;

    // Pass argument
    StringRef getArgument() const override { return "add_multi_buffer_outer_scope"; }

    // Run the pass
    void runOnOperation() override;

    // Get dependent dialects
    void getDependentDialects(DialectRegistry &registry) const override;
};

std::unique_ptr<OperationPass<ModuleOp>> createAddMultiBufferOuterScopePass();

void registerAddMultiBufferOuterScopePasses();

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_ALLOC_MULTI_CACHE_ADD_MULTI_BUFFER_OUTER_SCOPE_PASS_H
