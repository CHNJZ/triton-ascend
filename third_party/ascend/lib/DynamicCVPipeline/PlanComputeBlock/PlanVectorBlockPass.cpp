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

#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Passes.h"
#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <memory>

#define DEBUG_TYPE "plan-vector-block"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

using namespace mlir;
using namespace triton;

bool isFusableOp(Operation *op)
{
    return false;
}

void byPassNonFusable(DenseMap<Operation *, int> &indegree, SmallVector<Operation *> &candidates,
                      DenseMap<Operation *, bool> &visited, const CVSplit::MemoryDependenceGraph &memGraph)
{
    // for every non-fusable candidates, bypass it.
}

void updateCandidates(Operation *nextFused, SmallVector<Operation *> &candidates, DenseMap<Operation *, int> &indegree,
                      DenseMap<Operation *, bool> &visited, const CVSplit::MemoryDependenceGraph &memGraph)
{
    // 1. remove nextFused from candidates
    
    // 2. Add new candidates whose indegree becomes 0 after fusing nextFused.
}

void findCandidates(DenseMap<Operation *, int> &indegree, SmallVector<Operation *> &candidates,
                    DenseMap<Operation *, bool> &visited, const CVSplit::MemoryDependenceGraph &memGraph)
{
    // 1. if no candidate, try to bypass non-fusable
    
    // 2. find candidates whose indegree is 0 and not visited, add them to candidates and mark visited
}

void planVectorBlockId(Block *block, const CVSplit::MemoryDependenceGraph &memGraph)
{
    // 1. topo initialize
    llvm::DenseMap<Operation *, int> indegree;
    llvm::SmallVector<Operation *> candidates;
    llvm::DenseMap<Operation *, bool> visited; // has been visited in search
    CVSplit::initializeIndegreeForBlock(block, indegree, memGraph);

    // 2. initialize visited and find initial candidates
    block->walk([&](Operation *op) {
        if (op->getBlock() == block) {
            visited[op] = false;
            if (isFusableOp(op) && indegree[op] == 0) {
                visited[op] = true;
                candidates.push_back(op);
            }
        }
    });
    findCandidates(indegree, candidates, visited, memGraph);  
    
    // 3. BFS to find fuse group
    llvm::SmallVector<Operation *> nowFuseGroup;
    while (!candidates.empty()) {
        auto nextFused = candidates.front();
        if (nextFused) {
            // fused one && update candidates
            nowFuseGroup.push_back(nextFused);
            updateCandidates(nextFused, candidates, indegree, visited, memGraph);
        }
        if (candidates.empty() || nextFused == nullptr) {
            // finish one group, assign block id and start next BFS
            // Cut error operations before assigning block id

            auto &bm = CVSplit::ComputeBlockIdManager::getInstance();
            bm.markOpsWithNewId(nowFuseGroup);
            nowFuseGroup.clear();
            // reset candidates
            findCandidates(indegree, candidates, visited, memGraph);
        }
    }
}

// namespace

namespace mlir {
namespace triton {

void PlanVectorBlockPass::runOnOperation()
{
    ModuleOp module = getOperation();
    // 1. Build memory dependence graph
    auto &aliasAnalysis = getAnalysis<mlir::AliasAnalysis>();
    CVSplit::MemoryDependenceGraph memDepGraph(module, aliasAnalysis);

    // 2. search blocks in topo order and assign block id for each block
    llvm::SmallVector<Block *> blocks;
    module.walk([&](Block *block) { planVectorBlockId(block, memDepGraph); });
}

std::unique_ptr<OperationPass<ModuleOp>> createPlanVectorBlockPass()
{
    return std::make_unique<PlanVectorBlockPass>();
}

} // namespace triton
} // namespace mlir
