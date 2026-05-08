#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"

#include "ascend/include/DynamicCVPipeline/AllocMultiCache/AddMultiBufferOuterScope.h"
#include "ascend/include/DynamicCVPipeline/Common/BufferCountManager.h"
#include "ascend/include/DynamicCVPipeline/Common/FlagIdManager.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

static constexpr const char *DEBUG_TYPE = "AddMultiBufferOuterScope";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << '[' << DEBUG_TYPE << "] " << X << "\n")

using namespace mlir;
using namespace triton;
using namespace hivm;

namespace mlir {
namespace triton {

// ============================================================================
// 辅助函数：类型判断
// ============================================================================

// isSyncOp → isa<hivm::SyncBlockSetOp> || isa<hivm::SyncBlockWaitOp>

// isFixpipeOp → isa<hivm::FixpipeOp>

// isCopyOp → isa<hivm::CopyOp>

// isMemorySpaceCastOp → isa<memref::MemorySpaceCastOp>

// isConvertLayoutOp → isa<hivm::ConvertLayoutOp>

// isToTensorOp → isa<bufferization::ToTensorOp>

// isAllocOp → isa<memref::AllocOp>

// isMarkOp → isa<annotation::MarkOp>

// ============================================================================
// 辅助函数：属性获取
// ============================================================================

static int getFlagFromSyncOp(Operation *op) {
    if (auto attr = op->getAttrOfType<IntegerAttr>("flag_id")) return attr.getInt();
    if (auto attr = op->getAttrOfType<IntegerAttr>("static_flag_id")) return attr.getInt();
    if (auto attr = op->getAttrOfType<IntegerAttr>("flag")) return attr.getInt();
    return -1;
}

static int getBlockId(Operation *op) {
    if (auto attr = op->getAttrOfType<IntegerAttr>("ssbuffer.block_id"))
        return attr.getInt();
    return -1;
}

static int getTransferId(Operation *op) {
    if (auto attr = op->getAttrOfType<IntegerAttr>("ssbuffer.transfer_id"))
        return attr.getInt();
    return -1;
}

// ============================================================================
// 辅助函数：地址空间判断
// ============================================================================

// Check if an op is inside a VECTOR scope (by enclosing scope's tcore_type)
static bool isInVectorScope(Operation *op) {
    auto scopeOp = op->getParentOfType<scope::ScopeOp>();
    if (!scopeOp) return false;
    if (auto tcoreAttr = scopeOp->getAttrOfType<TCoreTypeAttr>("hivm.tcore_type"))
        return tcoreAttr.getTcoretype() == TCoreType::VECTOR;
    return false;
}

// ============================================================================
// 辅助函数：ssbuffer.main_loop 判断
// ============================================================================

/// 检查 forOp（含其 terminator）是否带有 ssbuffer.main_loop 属性
static bool forOpHasMainLoopAttr(scf::ForOp forOp) {
    if (forOp->hasAttr("ssbuffer.main_loop")) {
        return true;
    }
    Operation *terminator = forOp.getBody()->getTerminator();
    return terminator && terminator->hasAttr("ssbuffer.main_loop");
}

/// 判断 sync op 的直接 parentOp 是否带有 ssbuffer.main_loop 属性
static bool parentOpHasMainLoopAttr(Operation *syncOp) {
    if (!syncOp) return false;
    Operation *parent = syncOp->getParentOp();
    if (!parent) return false;
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
        return forOpHasMainLoopAttr(forOp);
    }
    return false;
}

// ============================================================================
// 辅助函数：操作查找
// ============================================================================

/// 在 block 中向前/向后查找指定 flag 的 sync op
static Operation *findSyncOpWithFlag(Block *block, Operation *start, int flag, bool forward, bool wantWait) {
    if (!block) return nullptr;
    auto it = start->getIterator();
    if (forward) {
        for (auto e = block->end(); it != e; ++it) {
            Operation *op = &*it;
            if (!(isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op))) continue;
            if (getFlagFromSyncOp(op) != flag) continue;
            if (wantWait && isa<hivm::SyncBlockWaitOp>(op)) return op;
            if (!wantWait && isa<hivm::SyncBlockSetOp>(op)) return op;
        }
    } else {
        if (it == block->begin()) return nullptr;
        do {
            --it;
            Operation *op = &*it;
            if (!(isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op))) continue;
            if (getFlagFromSyncOp(op) != flag) continue;
            if (wantWait && isa<hivm::SyncBlockWaitOp>(op)) return op;
            if (!wantWait && isa<hivm::SyncBlockSetOp>(op)) return op;
        } while (it != block->begin());
    }
    return nullptr;
}

/// 在 block 中查找 start 之后的 to_tensor op
static Operation *findToTensorAfter(Block *block, Operation *start) {
    if (!block) return nullptr;
    auto it = start->getIterator();
    for (auto e = block->end(); it != e; ++it) {
        if (isa<bufferization::ToTensorOp>(&*it)) return &*it;
    }
    return nullptr;
}

// ============================================================================
// 全局管理器
// ============================================================================

// ============================================================================
// 数据结构定义（用于存储收集到的信息）
// ============================================================================

/// Buffer alloc 对: {alloc, mark}
struct BufferAllocInfo {
    BufferAllocPair sender;    // 发送端 buffer
    BufferAllocPair receiver; // 接收端 buffer
};

/// Extra sync 信息: extra_sync 是 parentOp 没有 main_loop 标签的 sync op
struct ExtraSyncInfo {
    Operation *setOp = nullptr;  // sync_block_set
    Operation *waitOp = nullptr; // sync_block_wait
};

/// 传输链信息
struct TransferChainInfo {
    TransferOpChain sender;   // 发送端操作链
    TransferOpChain receiver; // 接收端操作链
};

// ============================================================================
// Step 1: 收集所有带 ssbuffer.transfer_id 的操作，按 tid 分组
// ============================================================================

static int collectOpsByTransferId(ModuleOp module,
                                  DenseMap<int, SmallVector<Operation *>> &opsByTid) {
    module.walk([&](Operation *op) {
        if (!op->hasAttr("ssbuffer.transfer_id")) return;
        int tid = getTransferId(op);
        if (tid >= 0) opsByTid[tid].push_back(op);
    });
    LDBG("Collected " << opsByTid.size() << " transfer groups");

    for (auto &p : opsByTid) {
        LDBG("  tid=" << p.first << " has " << p.second.size() << " ops");
        DenseMap<int, int> blockIdCount;
        for (auto *op : p.second) {
            int bid = getBlockId(op);
            blockIdCount[bid]++;
        }
        for (auto &bp : blockIdCount) {
            LDBG("    block_id=" << bp.first << ": " << bp.second << " ops");
        }
    }
    return 0;
}

// ============================================================================
// 内部函数：为单个 tid 收集三类信息
// ============================================================================

/// 收集 alloc/mark 对（与 block_id 和 main_loop 无关）
static int collectBufferAllocs(const SmallVector<Operation *> &ops, BufferAllocInfo &info) {
    SmallVector<Operation *> allocs;
    SmallVector<Operation *> marks;

    for (Operation *op : ops) {
        if (isa<memref::AllocOp>(op)) allocs.push_back(op);
        else if (isa<annotation::MarkOp>(op)) marks.push_back(op);
    }

    LDBG("collectBufferAllocs: allocs=" << allocs.size() << ", marks=" << marks.size());

    // 按顺序配对: sender在前，receiver在后
    if (!allocs.empty()) info.sender.allocOp = allocs[0];
    if (allocs.size() > 1) info.receiver.allocOp = allocs[1];
    if (!marks.empty()) info.sender.markOp = marks[0];
    if (marks.size() > 1) info.receiver.markOp = marks[1];

    return 0;
}

/// 收集 extra_sync: parentOp 没有 main_loop 标签的 sync op，按 flag 配对
static int collectExtraSync(const SmallVector<Operation *> &ops, int originalFlag, ExtraSyncInfo &info) {
    SmallVector<Operation *> extraSets;
    SmallVector<Operation *> extraWaits;

    for (Operation *op : ops) {
        if (!(isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op))) continue;

        bool hasMainLoop = parentOpHasMainLoopAttr(op);
        LDBG("sync op: flag=" << getFlagFromSyncOp(op) << ", block_id=" << getBlockId(op)
                     << ", parentHasMainLoop=" << hasMainLoop);

        if (!hasMainLoop) {
            if (isa<hivm::SyncBlockSetOp>(op)) {
                extraSets.push_back(op);
            } else if (isa<hivm::SyncBlockWaitOp>(op)) {
                extraWaits.push_back(op);
            }
        }
    }

    // 按 flag 配对
    for (auto *setOp : extraSets) {
        if (getFlagFromSyncOp(setOp) != originalFlag) continue;
        for (auto *waitOp : extraWaits) {
            if (getFlagFromSyncOp(waitOp) != originalFlag) continue;
            info.setOp = setOp;
            info.waitOp = waitOp;
            LDBG("Extra sync pair: set(flag=" << originalFlag
                         << ", block_id=" << getBlockId(setOp)
                         << "), wait(flag=" << originalFlag
                         << ", block_id=" << getBlockId(waitOp));
            return 0;
        }
    }

    // 没找到精确配对则用第一对
    if (!extraSets.empty() && !extraWaits.empty()) {
        info.setOp = extraSets.front();
        info.waitOp = extraWaits.front();
    }

    return 0;
}

/// 收集传输链: parentOp 有 main_loop 标签的 transfer ops
static int collectTransferChains(const SmallVector<Operation *> &ops,
                                                int originalFlag, TransferChainInfo &info) {

    for (Operation *op : ops) {
        if ((isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op)) || !op->getBlock()) continue;
        if (!parentOpHasMainLoopAttr(op)) continue;

        Block *block = op->getBlock();

        if (isa<hivm::FixpipeOp>(op)) {
            // CUBE sender
            info.sender.transferOp = op;
            info.sender.waitOp = findSyncOpWithFlag(block, op, originalFlag, false, true);
            info.sender.setOp = findSyncOpWithFlag(block, op, originalFlag, true, false);
            LDBG("Sender chain (CUBE): fixpipe, flag=" << originalFlag);
        } else if (isa<hivm::CopyOp>(op)) {
            // VECTOR sender
            info.sender.transferOp = op;
            info.sender.waitOp = findSyncOpWithFlag(block, op, originalFlag, false, true);
            info.sender.setOp = findSyncOpWithFlag(block, op, originalFlag, true, false);
            LDBG("Sender chain (VECTOR): hir.copy, flag=" << originalFlag);
        } else if (isa<memref::MemorySpaceCastOp>(op) && isInVectorScope(op)) {
            // VECTOR receiver
            info.receiver.transferOp = op;
            info.receiver.waitOp = findSyncOpWithFlag(block, op, originalFlag, false, true);
            info.receiver.setOp = findSyncOpWithFlag(block, op, originalFlag, true, false);
            info.receiver.toTensorOp = findToTensorAfter(block, op);
            LDBG("Receiver chain (VECTOR): memory_space_cast, flag=" << originalFlag);
        } else if (isa<hivm::ConvertLayoutOp>(op)) {
            // CUBE receiver
            info.receiver.transferOp = op;
            info.receiver.waitOp = findSyncOpWithFlag(block, op, originalFlag, false, true);
            info.receiver.setOp = findSyncOpWithFlag(block, op, originalFlag, true, false);
            LDBG("Receiver chain (CUBE): convert_layout, flag=" << originalFlag);
        }
    }

    return 0;
}

/// 构建单个 tid 的 TransferGroupInfo
static int buildTransferGroupData(int tid, const SmallVector<Operation *> &ops,
                                                FlagIdManager &flagIdMgr, TransferGroupInfo &info) {
    info.tid = tid;

    LDBG("Building group tid=" << tid << ", ops=" << ops.size());

    // 1. 收集 buffer alloc/mark 对
    BufferAllocInfo bufInfo;
    if (collectBufferAllocs(ops, bufInfo)) return -1;
    info.senderBuf = bufInfo.sender;
    info.receiverBuf = bufInfo.receiver;
    LDBG("Sender buffer: " << (info.senderBuf.allocOp ? "alloc" : "none")
                 << " + " << (info.senderBuf.markOp ? "mark" : "none"));
    LDBG("Receiver buffer: " << (info.receiverBuf.allocOp ? "alloc" : "none")
                 << " + " << (info.receiverBuf.markOp ? "mark" : "none"));

    // 2. 确定 original flag
    for (Operation *op : ops) {
        if ((isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op))) {
            int f = getFlagFromSyncOp(op);
            if (f >= 0) { info.originalFlag = f; break; }
        }
    }

    // 3. 收集 extra sync（parentOp 没有 main_loop）
    ExtraSyncInfo extraInfo;
    if (collectExtraSync(ops, info.originalFlag, extraInfo)) return -1;
    info.extraSyncSetOp = extraInfo.setOp;
    info.extraSyncWaitOp = extraInfo.waitOp;
    if (extraInfo.setOp && extraInfo.waitOp) {
        LDBG("Extra sync: set(block_id=" << getBlockId(extraInfo.setOp)
                     << "), wait(block_id=" << getBlockId(extraInfo.waitOp));
    } else {
        LDBG("Extra sync: not found");
    }

    // 4. 收集传输链（parentOp 有 main_loop）
    TransferChainInfo chainInfo;
    if (collectTransferChains(ops, info.originalFlag, chainInfo)) return -1;
    info.senderChain = chainInfo.sender;
    info.receiverChain = chainInfo.receiver;

    // 5. 确定方向
    if (info.senderChain.transferOp) {
        if (isa<hivm::FixpipeOp>(info.senderChain.transferOp)) {
            info.isCtoV = true;
        } else if (isa<hivm::CopyOp>(info.senderChain.transferOp)) {
            info.isCtoV = false;
        }
    }

    // 5.5. 对于 C→V 传输，sender 使用 receiver 的 buffer（allocs 中第二个）
    // 因为 fixpipe 输出到 receiver 的输入 buffer，这个 buffer 在 IR 中出现在第二位
    if (info.isCtoV && info.senderBuf.allocOp && info.receiverBuf.allocOp) {
        LDBG("C→V transfer: swapping sender/receiver buffers");
        std::swap(info.senderBuf, info.receiverBuf);
    }

    // 6. 获取 output flag
    for (int attempt = 0; attempt < 16; ++attempt) {
        int64_t pf = flagIdMgr.acquireId(nullptr);
        if (pf == FlagIdManager::INVALID_FLAG_ID) break;
        if (pf != info.originalFlag) {
            info.outputFlag = static_cast<int>(pf);
            break;
        }
    }

    if (info.senderChain.transferOp || info.receiverChain.transferOp) {
        LDBG("Direction: " << (info.isCtoV ? "C→V" : "V→C")
                     << ", flag=" << info.originalFlag << ", outputFlag=" << info.outputFlag);
    }

    return 0;
}

/// 收集所有传输组的 TransferGroupInfo
static int collectTransferGroupData(
    ModuleOp module,
    DenseMap<int, SmallVector<Operation *>> &opsByTid,
    FlagIdManager &flagIdMgr, DenseMap<int, TransferGroupInfo> &groups) {

    for (auto &p : opsByTid) {
        TransferGroupInfo info;
        if (buildTransferGroupData(p.first, p.second, flagIdMgr, info)) continue;
        if ((info.senderChain.transferOp || info.receiverChain.transferOp)
            && info.outputFlag >= 0) {
            groups[p.first] = info;
        }
    }

    // output flag reuse: groups with same originalFlag & direction share a output flag
    // This mirrors upstream's flag reuse across different transfer_ids.
    std::map<std::pair<int, bool>, int> outputFlagByKey; // (originalFlag, isCtoV) → outputFlag
    int nextSharedOutputFlag = -1;
    for (auto &p : groups) {
        auto &g = p.second;
        auto key = std::make_pair(g.originalFlag, g.isCtoV);
        auto it = outputFlagByKey.find(key);
        if (it != outputFlagByKey.end()) {
            // Reuse existing output flag for this shared group
            g.outputFlag = it->second;
            LDBG("Group tid=" << g.tid << " reuses outputFlag=" << g.outputFlag
                 << " (shared originalFlag=" << g.originalFlag << ")");
        } else {
            // First group with this key: allocate a new shared output flag
            outputFlagByKey[key] = g.outputFlag;
            nextSharedOutputFlag = g.outputFlag;
            LDBG("Group tid=" << g.tid << " gets new shared outputFlag=" << g.outputFlag
                 << " for originalFlag=" << g.originalFlag);
        }
    }

    // 打印汇总
    LDBG("=== Step 1 Summary ===");
    LDBG("Transfer groups: " << groups.size());
    for (auto &p : groups) {
        LDBG("Group tid=" << p.first
                     << ", dir=" << (p.second.isCtoV ? "C→V" : "V→C")
                     << ", flag=" << p.second.originalFlag
                     << ", outputFlag=" << p.second.outputFlag);
        if (p.second.senderChain.transferOp)
            LDBG("  Sender: " << p.second.senderChain.transferOp->getName().getStringRef());
        if (p.second.receiverChain.transferOp)
            LDBG("  Receiver: " << p.second.receiverChain.transferOp->getName().getStringRef());
    }

    return 0;
}

// ============================================================================
void AddMultiBufferOuterScopePass::runOnOperation()
{
    ModuleOp module = getOperation();
    LDBG("Enter pass.");

    // Step 1: Collect transfer group information
    FlagIdManager flagIdMgr(module);
    DenseMap<int, SmallVector<Operation *>> opsByTid;
    collectOpsByTransferId(module, opsByTid);
    DenseMap<int, TransferGroupInfo> groups;
    if (collectTransferGroupData(module, opsByTid, flagIdMgr, groups)) {
        LDBG("[Step 1] FAILED: no valid transfer groups found");
        signalPassFailure();
        return;
    }
    LDBG("Collected " << groups.size() << " transfer groups");

    // Step 2: Create output buffers (TODO)

    // Step 3: Add multi-buffer control flow (TODO)

    LDBG("Process successfully");
}

std::unique_ptr<OperationPass<ModuleOp>> createAddMultiBufferOuterScopePass() {
    return std::make_unique<AddMultiBufferOuterScopePass>();
}

void AddMultiBufferOuterScopePass::getDependentDialects(DialectRegistry &registry) const {
    registry.insert<mlir::annotation::AnnotationDialect,
                    mlir::memref::MemRefDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::arith::ArithDialect,
                    mlir::scf::SCFDialect,
                    mlir::hivm::HIVMDialect,
                    mlir::scope::ScopeDialect>();
}

void registerAddMultiBufferOuterScopePasses() {
    registerPass([]() -> std::unique_ptr<mlir::Pass> { return createAddMultiBufferOuterScopePass(); });
}

} // namespace triton
} // namespace mlir
