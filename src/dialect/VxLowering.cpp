#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include "VxDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FileSystem.h"

#include <atomic>

using namespace mlir;
using namespace mlir::vx;

#define DEBUG_TYPE "vx-lowering"

namespace {

// Classify what an outlined region computes, for plugins that route to vendor
// kernels (#325). Returns "" when the region is not a single recognised
// operation, which is the common case and carries no attribute.
//
// Deliberately conservative: it counts the payload operations in the region and
// classifies only when exactly one is present, because a library call replaces
// the whole kernel. A matmul with a bias add fused around it is not something a
// plain GEMM call can stand in for.
static bool isZeroConstant(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def || def->getName().getStringRef() != "arith.constant")
    return false;
  auto attr = def->getAttrOfType<FloatAttr>("value");
  return attr && attr.getValue().isZero();
}

// Whether a transfer moves data into memory the host cannot reach.
//
// `scope` is the machine model's own word for it: a space declared
// `scope: device` in fleet/*.vx is on the far side of a bus, and one declared
// `sm` or `cta` is inside a device rather than on the host. That is the fact
// this lowering turns on, so it is the one to ask about.
//
// `target_topology` is the fallback for a space that declares no scope at all.
// The host is 0 by construction in `topology_dispatch_id`, so a non-zero target
// is another topology; it is a weaker signal because it says where the data is
// going rather than what kind of memory it lands in.
static bool isDeviceTransfer(vx::TransferOp op) {
  if (auto scope = op->getAttrOfType<StringAttr>("scope")) {
    StringRef s = scope.getValue();
    return s == "device" || s == "sm" || s == "cta";
  }
  if (auto topo = op->getAttrOfType<IntegerAttr>("target_topology"))
    return topo.getInt() != 0;
  return false;
}

/// What a variable's slot holds, when one store settles it.
///
/// A `let` binding is an `alloca` of a memref with a store into it, and every
/// mention of the variable afterwards is a load. So `let d = transfer(x, HBM)`
/// puts the device memref in a slot, and the later `transfer(d, CPU_DRAM)`
/// receives a *load*, not the transfer's result.
///
/// Walking back from that load through its single operand lands on the alloca,
/// which no transfer produced -- so provenance is lost at the first variable a
/// value is stored in, and the caller concludes the data came from nowhere.
/// Every binding in a Vx program has this shape, which is why the way home was
/// still compiled into a memcpy after being wired to the fetch: the wiring was
/// only ever reached by a transfer applied directly to another transfer.
///
/// One store, because that is what a binding that is never reassigned has, and
/// because two stores are two answers -- returning either would be a guess
/// about which one ran. Null for anything else, which leaves the caller exactly
/// where it was.
static Value slotContents(Value slot) {
  if (!slot.getDefiningOp<memref::AllocaOp>())
    return nullptr;
  Value stored = nullptr;
  for (Operation *user : slot.getUsers()) {
    auto store = dyn_cast<memref::StoreOp>(user);
    if (!store)
      continue;
    if (!store.getIndices().empty() || store.getMemRef() != slot)
      return nullptr;
    if (stored)
      return nullptr;
    stored = store.getValueToStore();
  }
  return stored;
}

/// The `vx.transfer` that put `v` where it is, if one did.
///
/// Walks back through single-operand ops, because a capture usually reaches a
/// spawn having been cast or reshaped since the transfer produced it, and
/// through the slot of any variable it was bound to on the way. Bounded: a
/// chain longer than this is one this cannot reason about anyway, and answering
/// "no transfer" for it is the safe direction -- it declines to reject rather
/// than rejecting something it has not understood.
static vx::TransferOp definingTransfer(Value v) {
  for (int hops = 0; v && hops < 8; ++hops) {
    Operation *def = v.getDefiningOp();
    if (!def)
      return nullptr;
    if (auto xfer = dyn_cast<vx::TransferOp>(def))
      return xfer;
    if (auto load = dyn_cast<memref::LoadOp>(def)) {
      if (!load.getIndices().empty())
        return nullptr;
      v = slotContents(load.getMemRef());
      continue;
    }
    if (def->getNumOperands() != 1)
      return nullptr;
    v = def->getOperand(0);
  }
  return nullptr;
}

/// The topology a transfer's source is on, or 0 for "here".
///
/// Two ways of knowing, asked in order of how much they know.
///
/// `source_topology` is the checker's answer, stamped on the op by the AST
/// emitter: a placed tensor's type is `Pinned(_, topology)`, so the placement
/// is decided long before this file sees anything, and no analysis is involved.
///
/// The walk is the fallback for producers that do not stamp it. It is weaker in
/// a way that mattered: it can only follow SSA edges, and a value bound to a
/// variable reaches its use through a slot -- so it answers "nowhere" for every
/// program that names its data, which is every program. `slotContents` closes
/// the common case of that, and the attribute closes the rest.
static int32_t sourceTopologyOf(vx::TransferOp op) {
  if (auto a = op->getAttrOfType<IntegerAttr>("source_topology"))
    return static_cast<int32_t>(a.getInt());
  if (vx::TransferOp srcXfer = definingTransfer(op.getOperand())) {
    if (isDeviceTransfer(srcXfer)) {
      if (auto a = srcXfer->getAttrOfType<IntegerAttr>("target_topology"))
        return static_cast<int32_t>(a.getInt());
    }
  }
  return 0;
}

/// Whether a region would actually dereference its operands.
///
/// A `spawn` whose body only yields what it captured moves no data and reads
/// nothing, so running it on the host is harmless however the operands are
/// managed -- `tests/frontend/pass/rubin_disaggregated.vx` is exactly that: two
/// placements and an explicit KV handoff between them, with no arithmetic in
/// either region. Rejecting it would be rejecting a program that cannot fault.
///
/// So the question is not "is this placed on a device" but "would the host
/// fallback load through a device pointer", and that is what a load or store
/// in the body means. Walks nested regions, since the loads that matter are
/// inside loops. A body that reaches memory only through a call is missed,
/// which is the safe direction: this predicate exists to reject, and declining
/// to reject something it has not understood is the error worth making.
static bool regionTouchesMemory(Region &body) {
  bool touches = false;
  body.walk([&](Operation *op) {
    StringRef name = op->getName().getStringRef();
    if (name == "memref.load" || name == "memref.store" ||
        name == "affine.load" || name == "affine.store" ||
        name == "vector.load" || name == "vector.store" ||
        name.starts_with("linalg.")) {
      touches = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return touches;
}

/// Whether the host can read what a transfer produced.
///
/// `managed: explicit` is the declaration saying movement in and out requires
/// an explicit transfer -- which is how a machine model says the CPU cannot
/// simply load from this space. `cached` says the hardware moves it implicitly,
/// so a host read is fine. A space that declares neither makes no claim, and
/// the answer is yes: this predicate exists to reject programs, and it should
/// only do so on the strength of something the program actually said.
static bool hostCanRead(vx::TransferOp op) {
  if (auto managed = op->getAttrOfType<StringAttr>("managed"))
    return managed.getValue() != "explicit";
  return true;
}

static StringRef kernelKindOf(Region &body, Operation **payloadOut = nullptr) {
  if (payloadOut)
    *payloadOut = nullptr;
  Operation *matmul = nullptr;
  Operation *fill = nullptr;
  unsigned otherPayload = 0;

  // Only the region's own operations, not a deep walk: a linalg op carries its
  // arithmetic in a nested region, and descending into it would count that
  // op's `linalg.yield` as separate payload and classify nothing.
  for (Block &block : body) {
    for (Operation &opRef : block) {
      Operation *op = &opRef;
      StringRef name = op->getName().getStringRef();
      // Structural and bookkeeping operations are not the payload.
      if (name == "vx.yield" || name == "vx.return" ||
          name.starts_with("arith.") || name.starts_with("memref.") ||
          name.starts_with("llvm.") || name.starts_with("scf.") ||
          name.starts_with("cf.") || name.starts_with("builtin.")) {
        continue;
      }
      if (name == "linalg.matmul" && !matmul) {
        matmul = op;
        continue;
      }
      if (name == "linalg.fill" && !fill) {
        fill = op;
        continue;
      }
      ++otherPayload;
    }
  }

  if (!matmul || otherPayload > 0)
    return StringRef();

  // `c = a @ b` emits a zero-fill of the accumulator followed by the matmul,
  // which is precisely GEMM with beta = 0 -- a library call subsumes both. The
  // fill has to be zeroing *this* matmul's output for that to hold: filling
  // with a non-zero v computes A*B + v, which no beta = 0 GEMM reproduces.
  if (fill) {
    if (fill->getNumOperands() < 2 || matmul->getNumOperands() < 3)
      return StringRef();
    if (fill->getOperand(1) != matmul->getOperand(2))
      return StringRef();
    if (!isZeroConstant(fill->getOperand(0)))
      return StringRef();
  }

  // Only on success, so a caller cannot read a payload op out of a region that
  // was ultimately rejected.
  if (payloadOut)
    *payloadOut = matmul;

  return "matmul";
}

// Describe which launch operand plays which role in a recognised matmul, as
// `a:<i>,b:<j>,out:<k>`, indexing the operand list the launch is built from.
//
// This is the half that actually resolves the ambiguity #325 exists for. The
// operation name alone tells a plugin it has a GEMM; it does not say which
// buffer is which, and for square operands nothing about the shapes can, since
// every assignment conforms. linalg names them -- `ins` in order, `outs` -- so
// the mapping is read off rather than guessed.
//
// The result buffer is normally *not* a capture: `c = a @ b` allocates inside
// the kernel and stores the pointer through a captured slot, so `out:` names
// that slot and `outkind=slot` says so. A plugin then knows it must publish the
// result by writing a descriptor there rather than filling a buffer it was
// handed. When the buffer is captured directly, `outkind=buffer`.
//
// Returns "" unless every role resolves. A partial mapping is worse than none:
// it invites a plugin to fill in the rest by convention, which is the guessing
// this is meant to replace.
static std::string matmulRolesOf(Operation *matmul, Region &body,
                                 const SetVector<Value> &captures,
                                 StringRef &outKind) {
  if (!matmul || matmul->getNumOperands() < 3)
    return std::string();

  auto indexOf = [&](Value v) -> int {
    for (auto en : llvm::enumerate(captures)) {
      if (en.value() == v)
        return static_cast<int>(en.index());
    }
    return -1;
  };

  // Where the operand came from, following the one indirection a local tensor
  // introduces. A tensor declared inside the enclosing function lives in a slot
  // -- `memref<memref<?x?xf32>>` -- and the region loads the descriptor out of
  // it before use, so the captured value is the slot and not the buffer. Only
  // function parameters arrive as buffers directly.
  //
  // Without this the roles resolve for parameters and silently do not for
  // locals, which is the common case: the attribute would simply be absent and
  // every plugin would fall back, with nothing to indicate why.
  // The single value stored into `slot` within the region, if there is exactly
  // one. More than one and the slot's contents at the matmul are not decidable
  // by inspection, so no role is claimed.
  auto soleStoreInto = [&](Value slot) -> Value {
    Value stored;
    unsigned count = 0;
    for (Block &block : body) {
      for (Operation &opRef : block) {
        if (opRef.getName().getStringRef() != "memref.store" ||
            opRef.getNumOperands() < 2) {
          continue;
        }
        if (opRef.getOperand(1) == slot) {
          stored = opRef.getOperand(0);
          ++count;
        }
      }
    }
    return count == 1 ? stored : Value();
  };

  auto indexOfSource = [&](Value v) -> int {
    int idx = indexOf(v);
    if (idx >= 0)
      return idx;

    Operation *def = v.getDefiningOp();
    if (!def || def->getName().getStringRef() != "memref.load" ||
        def->getNumOperands() < 1) {
      return -1;
    }

    // A tensor declared in the enclosing function lives in a slot --
    // `memref<memref<?x?xf32>>` -- and the region loads the descriptor out of
    // it, so the captured value is the slot rather than the buffer.
    Value slot = def->getOperand(0);
    idx = indexOf(slot);
    if (idx >= 0)
      return idx;

    // Or the buffer itself was captured and the outliner re-homed it into a
    // slot of the kernel's own: `store %capture, %local ; load %local`. The
    // slot is then local and names nothing, so follow the store that filled it.
    // Both shapes occur -- which one depends on how the tensor was bound in the
    // caller -- and resolving only the first left roles absent on the other,
    // silently, with every plugin falling back and nothing to say why.
    if (Value stored = soleStoreInto(slot))
      return indexOf(stored);

    return -1;
  };

  int aIdx = indexOfSource(matmul->getOperand(0));
  int bIdx = indexOfSource(matmul->getOperand(1));
  int outIdx = indexOfSource(matmul->getOperand(2));
  outKind = "buffer";

  if (outIdx < 0) {
    // Follow the store that publishes the locally allocated result.
    Value resultBuf = matmul->getOperand(2);
    for (Block &block : body) {
      for (Operation &opRef : block) {
        if (opRef.getName().getStringRef() != "memref.store" ||
            opRef.getNumOperands() < 2) {
          continue;
        }
        if (opRef.getOperand(0) == resultBuf) {
          outIdx = indexOf(opRef.getOperand(1));
          outKind = "slot";
        }
      }
    }
  }

  if (aIdx < 0 || bIdx < 0 || outIdx < 0)
    return std::string();

  // A plugin reads the operands out of the slots as they stand when the launch
  // is called, which is before any of the region runs. That matches what the
  // loads above would have produced only if nothing in the region writes to an
  // input's slot first. Nothing does today -- the region holds one matmul and
  // at most a zero fill -- but the reading is only valid while that is true.
  for (Block &block : body) {
    for (Operation &opRef : block) {
      if (opRef.getName().getStringRef() != "memref.store" ||
          opRef.getNumOperands() < 2) {
        continue;
      }
      int target = indexOf(opRef.getOperand(1));
      if (target == aIdx || target == bIdx)
        return std::string();
    }
  }

  return ("a:" + Twine(aIdx) + ",b:" + Twine(bIdx) + ",out:" + Twine(outIdx))
      .str();
}

// The `math` dialect op for a libm symbol, or empty for one that is not a
// transcendental Vx routes through the stdlib.
//
// `math` is the portable spelling: it lowers to libm on the host (the pipeline
// already runs `convert-math-to-libm`) and to device intrinsics under the NVVM
// pipeline. A `func.call @expf` lowers to neither -- it is a host symbol, and a
// kernel that calls it cannot be loaded.
static StringRef mathOpForLibm(StringRef libm) {
  return llvm::StringSwitch<StringRef>(libm)
      .Cases("expf", "exp", "math.exp")
      .Cases("logf", "log", "math.log")
      .Cases("log2f", "log2", "math.log2")
      .Cases("log10f", "log10", "math.log10")
      .Cases("sqrtf", "sqrt", "math.sqrt")
      .Cases("sinf", "sin", "math.sin")
      .Cases("cosf", "cos", "math.cos")
      .Cases("tanf", "tan", "math.tan")
      .Cases("asinf", "asin", "math.asin")
      .Cases("acosf", "acos", "math.acos")
      .Cases("atanf", "atan", "math.atan")
      .Cases("fabsf", "fabs", "math.absf")
      .Default(StringRef());
}

// The libm symbol a call ultimately reaches, seeing through one layer of Vx
// stdlib wrapper.
//
// `x.exp()` does not call `expf` directly. It calls `@f32$exp`, whose whole
// body is `%0 = call @expf(%arg0); return %0` -- the `impl Math for f32` in
// stdlib/std/math.vx. Recognising the wrapper by that shape rather than by its
// generated name means monomorphization can spell it however it likes, and a
// wrapper that grows a second statement stops being treated as an intrinsic
// instead of being silently mistaken for one.
static StringRef libmSymbolBehind(StringRef callee, ModuleOp module) {
  auto fn = module.lookupSymbol<func::FuncOp>(callee);
  if (!fn)
    return callee;
  if (fn.isExternal())
    return callee;
  Region &body = fn.getBody();
  if (!body.hasOneBlock())
    return StringRef();
  Block &blk = body.front();
  auto ops = blk.without_terminator();
  if (!llvm::hasSingleElement(ops))
    return StringRef();
  auto inner = dyn_cast<func::CallOp>(&*ops.begin());
  auto ret = dyn_cast<func::ReturnOp>(blk.getTerminator());
  if (!inner || !ret)
    return StringRef();
  if (ret->getOperands() != inner->getResults())
    return StringRef();
  if (inner.getArgOperands() != ValueRange(blk.getArguments()))
    return StringRef();
  return inner.getCallee();
}

// Rewrite host libm calls in an outlined kernel to `math` ops.
static void useDeviceMathIn(Region &kernel, ModuleOp module,
                            PatternRewriter &rewriter) {
  SmallVector<func::CallOp> calls;
  kernel.walk([&](func::CallOp call) { calls.push_back(call); });
  for (func::CallOp call : calls) {
    StringRef libm = libmSymbolBehind(call.getCallee(), module);
    if (libm.empty())
      continue;
    StringRef mathName = mathOpForLibm(libm);
    if (mathName.empty())
      continue;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(call);
    OperationState state(call.getLoc(), mathName);
    state.addOperands(call.getOperands());
    state.addTypes(call.getResultTypes());
    Operation *mathOp = rewriter.create(state);
    rewriter.replaceOp(call, mathOp->getResults());
  }
}

// Put a kernel's own scratch on the stack.
//
// A `Tensor` declared inside the region -- FlashAttention's `ts`, the one score
// tile it keeps -- lowers to `memref.alloc`. That is a heap allocation, and in
// a device kernel it becomes a call to device-side `malloc`: a heap the launch
// has to be configured with, and a call on every invocation, for what is a
// 64-byte scratch buffer. Nothing frees it either, so on the host path it leaks
// once per dispatch.
//
// Only allocations in the entry block, with a static shape, and with no
// `dealloc` of their own. An `alloca` inside a loop grows the stack every
// iteration, which is a worse bug than the one being fixed; a dynamic extent is
// not a stack slot on a device at all; and something that is explicitly freed
// is not scratch whose lifetime this may shorten.
static void useStackScratchIn(Region &kernel, PatternRewriter &rewriter) {
  if (kernel.empty())
    return;
  SmallVector<memref::AllocOp> allocs;
  for (Operation &op : kernel.front())
    if (auto alloc = dyn_cast<memref::AllocOp>(&op))
      allocs.push_back(alloc);

  for (memref::AllocOp alloc : allocs) {
    MemRefType ty = alloc.getType();
    if (!ty.hasStaticShape() || !alloc.getDynamicSizes().empty())
      continue;
    if (llvm::any_of(alloc->getUsers(),
                     [](Operation *u) { return isa<memref::DeallocOp>(u); }))
      continue;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(alloc);
    auto stack = rewriter.create<memref::AllocaOp>(alloc.getLoc(), ty);
    rewriter.replaceOp(alloc, stack.getResult());
  }
}

// Lower `vx.spawn` to `async.execute` for CPU topologies, or an outlined
// kernel + `vx.launch` for NPU/AccCore topologies.
struct SpawnOpLowering : public OpRewritePattern<SpawnOp> {
  using OpRewritePattern<SpawnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SpawnOp op,
                                PatternRewriter &rewriter) const override {
    int32_t topology = op.getTopology();

    if (topology == 0) {
      Region &spawnBody = op.getBody();
      if (spawnBody.empty()) {
        rewriter.eraseOp(op);
        return success();
      }

      auto asyncExecuteOp =
          rewriter.create<async::ExecuteOp>(op.getLoc(),
                                            /*resultTypes=*/TypeRange{},
                                            /*dependencies=*/ValueRange{},
                                            /*operands=*/ValueRange{});
      Region &asyncBody = asyncExecuteOp.getRegion();

      // `vx.yield` terminates the spawn region and `async.yield` terminates
      // this one. Rewritten before the blocks move and found by walking the
      // region, because a region with control flow in it yields from its merge
      // block rather than from the block it started in.
      SmallVector<Value> yieldedValues;
      SmallVector<vx::YieldOp> yields;
      spawnBody.walk([&](vx::YieldOp y) { yields.push_back(y); });
      for (vx::YieldOp y : yields) {
        if (yields.front() == y)
          llvm::append_range(yieldedValues, y.getOperands());
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(y);
        rewriter.create<async::YieldOp>(y.getLoc(), ValueRange{});
        rewriter.eraseOp(y);
      }

      // Every block of the spawn region becomes a block of the async region,
      // its entry block included. Moving only the first block's *operations* --
      // which is what this did -- dropped every other block, and carried the
      // branch to them into the middle of the async body. A region containing
      // any `for` or `if` is more than one block, so
      // `spawn on(Topology::CPU) { for ... }` failed to compile with
      // "operation with block successors must terminate its parent block".
      while (!asyncBody.empty())
        rewriter.eraseBlock(&asyncBody.front());
      rewriter.inlineRegionBefore(spawnBody, asyncBody, asyncBody.end());

      rewriter.replaceOp(op, yieldedValues);
      return success();
    }

    // For NPU (100) or AccCore (200), we lower to an outlined kernel.
    Region &spawnBody = op.getBody();
    if (spawnBody.empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    SetVector<Value> captures;
    getUsedValuesDefinedAbove(spawnBody, captures);

    // A constant used inside the region is rematerialized in it rather than
    // passed to it.
    //
    // Vx loop bounds are literals -- `for i in 0..32` -- and they are defined
    // outside the region, so capturing them by value turns every trip count
    // into a runtime argument. That costs twice. It widens the dispatch: the
    // FlashAttention kernel took eight scalar arguments that are all constants,
    // 8 of its 36 `.param`s. And it hides the shape from the backend, which is
    // the expensive half -- with the bounds opaque, NVVM cannot unroll the
    // 16-wide inner loops, and the same kernel goes from 306 instructions of
    // straight-line arithmetic to 149 with 19 branches.
    //
    // Constant-like with no operands, so the clone is self-contained and
    // sinking it cannot reorder anything observable.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&spawnBody.front());
      for (Value cap : captures) {
        Operation *def = cap.getDefiningOp();
        if (!def || def->getNumOperands() != 0 ||
            !def->hasTrait<OpTrait::ConstantLike>())
          continue;
        Operation *inRegion = rewriter.clone(*def);
        rewriter.replaceUsesWithIf(
            cap, inRegion->getResult(0), [&](OpOperand &use) {
              return spawnBody.isAncestor(use.getOwner()->getParentRegion());
            });
      }
      captures.clear();
      getUsedValuesDefinedAbove(spawnBody, captures);
    }

    // Create the outlined function at the module level
    auto module = op->getParentOfType<ModuleOp>();
    auto ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToEnd(module.getBody());

    SmallVector<Type> argTypes;
    for (auto val : captures) {
      argTypes.push_back(val.getType());
    }

    // Determine result types from the yield operation. The yield may live in a
    // nested block (e.g. inside control flow) now that the entire region is
    // inlined, so search the whole region rather than only the entry block's
    // terminator.
    SmallVector<Type> resultTypes;
    vx::YieldOp resultYield;
    spawnBody.walk([&](vx::YieldOp y) {
      if (!resultYield)
        resultYield = y;
    });
    if (resultYield) {
      for (auto val : resultYield.getOperands()) {
        resultTypes.push_back(val.getType());
      }
    }

    // Create the kernel op. The counter is process-global (kernel names must be
    // unique across the module) and the test harness compiles files in
    // parallel, so it must be atomic to avoid a data race / duplicate names.
    auto funcType = rewriter.getFunctionType(argTypes, resultTypes);
    static std::atomic<int> kernelIdx{0};
    std::string funcName =
        "vx_npu_kernel_" + std::to_string(kernelIdx.fetch_add(1));
    auto kernelOp = rewriter.create<vx::KernelOp>(op.getLoc(), funcName,
                                                  funcType, topology);
    // The declared arch travels from the machine file, via the spawn, onto the outlined kernel --
    // a discardable attribute, so no dialect change. It is what lets `materializeGpuKernels` gate
    // device compilation on the DECLARATION instead of the dispatch-id band, which a custom
    // topology arithmetically cannot enter (custom ids are 1000 + fnv %% 1000; the band is
    // [500, 600)) (Vx#352).
    if (auto arch = op->getAttrOfType<StringAttr>("arch"))
      kernelOp->setAttr("arch", arch);

    // Name what the region computes, so a plugin can route it to a vendor
    // kernel instead of inferring the operation from buffer shapes (#325).
    // Shape conformance alone cannot do it: for square operands every operand
    // assignment conforms, and swapping them yields a plausible matrix of wrong
    // numbers rather than a failure.
    //
    // Matched on the operation name rather than through the Linalg headers, to
    // avoid taking a dialect dependency here for what is a single string
    // comparison. Only a region whose whole job is the one op is classified: a
    // matmul with other computation around it is not a matmul the runtime can
    // hand to cuBLAS wholesale.
    //
    // A kernel that is not recognised carries no attribute and dispatches
    // exactly as before -- routing is an optimisation, and a kernel the
    // compiler cannot classify is not an error.
    Operation *payloadOp = nullptr;
    StringRef kernelKind = kernelKindOf(spawnBody, &payloadOp);
    StringRef outKind;
    std::string kernelRoles =
        matmulRolesOf(payloadOp, spawnBody, captures, outKind);
    if (!kernelKind.empty()) {
      kernelOp->setAttr("vx.kernel_kind", rewriter.getStringAttr(kernelKind));
      if (!kernelRoles.empty()) {
        kernelOp->setAttr("vx.kernel_roles",
                          rewriter.getStringAttr(kernelRoles));
        kernelOp->setAttr("vx.kernel_out_kind",
                          rewriter.getStringAttr(outKind));
      }
    }

    // Clone the entire region to avoid leaving SpawnOp with an invalid empty
    // region
    Region &kernelRegion = kernelOp.getBody();
    rewriter.cloneRegionBefore(spawnBody, kernelRegion, kernelRegion.end());

    // Fix up the entry block arguments to accept the captured variables
    Block &entryBlock = kernelRegion.front();
    for (auto type : argTypes) {
      entryBlock.addArgument(type, op.getLoc());
    }

    // Replace usages of captured variables inside the region with the block
    // arguments
    for (auto [cap, arg] : llvm::zip(captures, entryBlock.getArguments())) {
      Value capVal = cap;
      rewriter.replaceUsesWithIf(capVal, arg, [&](OpOperand &use) {
        return kernelRegion.isAncestor(use.getOwner()->getParentRegion());
      });
    }

    // The kernel is dispatched, not called from here, so the host's libm is not
    // reachable from inside it. `math` ops are, on either side.
    useDeviceMathIn(kernelRegion, module, rewriter);
    useStackScratchIn(kernelRegion, rewriter);

    // Replace vx.yield with vx.return
    SmallVector<vx::YieldOp> yieldsToErase;
    kernelRegion.walk(
        [&](vx::YieldOp yieldOp) { yieldsToErase.push_back(yieldOp); });
    for (auto y : yieldsToErase) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(y);
      rewriter.create<vx::ReturnOp>(y.getLoc(), y.getOperands());
      rewriter.eraseOp(y);
    }

    // If the region has no terminator in the last block (e.g. empty spawn), add
    // vx.return
    for (Block &block : kernelRegion) {
      if (block.empty() || !block.back().hasTrait<OpTrait::IsTerminator>()) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToEnd(&block);
        rewriter.create<vx::ReturnOp>(op.getLoc(), ValueRange{});
      }
    }

    // Restore insertion point to replace vx.spawn with vx.launch
    rewriter.restoreInsertionPoint(ip);

    SmallVector<Value> launchOperands(captures.begin(), captures.end());
    auto launchOp = rewriter.create<vx::LaunchOp>(
        op.getLoc(), resultTypes,
        SymbolRefAttr::get(rewriter.getContext(), funcName),
        rewriter.getI32IntegerAttr(topology), launchOperands);

    // Carried on the launch as well as the kernel: the launch is what lowers to
    // the dispatch call, so this is where the fact has to be to reach a plugin.
    if (!kernelKind.empty()) {
      launchOp->setAttr("vx.kernel_kind", rewriter.getStringAttr(kernelKind));
      if (!kernelRoles.empty()) {
        launchOp->setAttr("vx.kernel_roles",
                          rewriter.getStringAttr(kernelRoles));
        launchOp->setAttr("vx.kernel_out_kind",
                          rewriter.getStringAttr(outKind));
      }
    }

    // The topology's declared name, forwarded from the spawn. Optional by
    // design: the flat path emits `vx.spawn` from an instruction stream that
    // carries the dispatch id and not the name, so a launch may legitimately
    // arrive without one. A plugin then has the id and nothing else, which is
    // exactly the situation before this existed -- the same way `kind=` and
    // `roles=` degrade rather than fail (#348).
    if (auto topoName = op->getAttrOfType<StringAttr>("topology_name")) {
      launchOp->setAttr("vx.topology_name", topoName);
    }

    rewriter.replaceOp(op, launchOp.getResults());
    return success();
  }
};

/// Re-type the users of a value that has just moved into a different memory space.
///
/// Only `memref.reinterpret_cast` needs this: it is the one op in the chain whose RESULT type
/// restates the memory space, so leaving it alone makes it a cast between spaces and the verifier
/// rejects it with "different memory spaces specified for source type ... and result memref type".
/// `memref.load` and `memref.store` accept any space and need no change.
///
/// The casts come from the flat code generator, which emits a row view for `t[i][d]` before
/// anything knows the tile will live in shared memory -- so the space cannot be filled in there and
/// has to be threaded through here. Recursive because a view of a view is a chain (#352).
static void propagateMemorySpace(Value v, Attribute space) {
  SmallVector<Operation *> users(v.getUsers().begin(), v.getUsers().end());
  for (Operation *user : users) {
    auto view = dyn_cast<memref::ReinterpretCastOp>(user);
    if (!view)
      continue;
    auto old = llvm::cast<MemRefType>(view.getResult().getType());
    if (old.getMemorySpace() == space)
      continue;
    auto retyped = MemRefType::get(old.getShape(), old.getElementType(),
                                   old.getLayout(), space);
    view.getResult().setType(retyped);
    propagateMemorySpace(view.getResult(), space);
  }
}

struct TransferOpLowering : public OpRewritePattern<TransferOp> {
  using OpRewritePattern<TransferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransferOp op,
                                PatternRewriter &rewriter) const override {
    auto src = op.getOperand();
    auto srcType = dyn_cast<MemRefType>(src.getType());

    // If it's not a MemRef (e.g., primitive i32/f32), we fail compilation.
    // The user explicitly mandated that implicit conversions/pass-throughs
    // are disabled to enforce strict data layout transitions.
    if (!srcType) {
      llvm::errs() << "[VxLowering] TransferOp srcType is not MemRefType\n";
      op.emitError("vx.transfer currently only supports MemRef types. "
                   "Attempted to transfer a scalar/primitive.");
      return failure();
    }

    auto targetType = cast<MemRefType>(op.getResult().getType());
    LLVM_DEBUG(llvm::errs() << "[VxLowering] TransferOp lowering from "
                            << srcType << " to " << targetType << "\n");

    // A placement into an SM-scoped space is shared memory, and it is the one
    // device transfer this stage CAN express. Everything needed is already on the
    // op -- `scope = "sm"`, plus the space name, granule and slot offset the
    // checker computed -- so it becomes a workgroup-space allocation and a copy
    // into it, right here, in dialects the device pipeline accepts.
    //
    // Doing it here rather than later is what makes the kernel compilable at all.
    // `isDeviceLowerableDialect` allows arith/cf/gpu/math/memref/scf; a surviving
    // `vx.transfer` is none of those, so a kernel containing one is classified
    // not-device-ready and dropped from GPU compilation entirely -- silently, and
    // the whole region falls back. That is why a tile placed in SMEM produced no
    // `.shared` in the emitted PTX and no `image=` in the payload: the placement
    // disqualified the very kernel that was supposed to use it (#352).
    if (auto scope = op->getAttrOfType<StringAttr>("scope")) {
      if (scope.getValue() == "sm") {
        // Integer address space 3, not `#gpu.address_space<workgroup>`. The two mean the same
        // thing to NVVM, but the symbolic attribute needs a memory-space conversion registered on
        // the type converter, and the device pipeline here does not install one -- it fails with
        // "conversion of memref memory space #gpu.address_space<workgroup> to integer address
        // space failed". 3 is what NVPTX calls shared, and it converts with no extra plumbing.
        Attribute workgroup = rewriter.getI64IntegerAttr(3);
        auto sharedType =
            MemRefType::get(targetType.getShape(), targetType.getElementType(),
                            targetType.getLayout(), workgroup);
        // The AST codegen path types this transfer as a dynamic memref (`?x?`),
        // so the alloca needs the sizes as SSA values, read off the source --
        // the flat path's shapes are static and contribute none. Measured, not
        // assumed: with the sizes supplied, the AST path's kernel compiles to
        // the same `.shared` PTX as the flat path's (the dims fold to constants
        // by the time NVVM sees them). A tile whose shape stays GENUINELY
        // dynamic at device compilation is untested territory, owned by #353 A3
        // alongside the barrier -- if it fails there, it fails in
        // `deviceImageOf` with a diagnostic, not silently.
        SmallVector<Value> sharedDynSizes;
        for (int i = 0; i < sharedType.getRank(); ++i) {
          if (sharedType.isDynamicDim(i)) {
            auto idx = rewriter.create<arith::ConstantOp>(
                op.getLoc(), rewriter.getIndexAttr(i));
            sharedDynSizes.push_back(
                rewriter.create<memref::DimOp>(op.getLoc(), src, idx));
          }
        }
        // `memref.alloca`, not `alloc`: shared memory is scratch for the
        // lifetime of the kernel, not something anyone frees, and
        // convert-gpu-to-nvvm turns a workgroup-space alloca into a `.shared`
        // global rather than a call into a device allocator.
        Value shared = rewriter.create<memref::AllocaOp>(op.getLoc(), sharedType,
                                                         sharedDynSizes);
        rewriter.create<memref::CopyOp>(op.getLoc(), src, shared);
        rewriter.replaceOp(op, shared);
        propagateMemorySpace(shared, workgroup);
        return success();
      }
    }

    // A transfer into a device space is not a host allocation and a copy. It
    // is the plugin's allocate-and-transfer, so leave the op alone here and let
    // the LLVM stage emit that call -- getting a raw pointer out of a memref is
    // not something this stage can express.
    if (isDeviceTransfer(op))
      return failure();

    // Coming home. `transfer(x, Memory::CPU_DRAM)` where x is on a device is
    // the way back, and it is not a host allocation and a `memref.copy`: the
    // source is not host memory. Locally that copy reads a device pointer;
    // across a fleet it reads a non-canonical handle, which is an address no
    // process owns. Left to the LLVM stage below, which can get a raw pointer
    // out of a memref and call the plugin's fetch.
    //
    // This is why the way home did nothing. The op was lowered here into an
    // allocation and a copy, so no `vx.transfer` survived and nothing ever
    // called `vx_plugin_transfer_device_to_host` -- an entry point that exists,
    // is wired to the fleet routing, and had no caller in generated code.
    if (sourceTopologyOf(op) != 0)
      return failure();

    // Extract dynamic sizes from the source memref
    SmallVector<Value> dynamicSizes;
    for (int i = 0; i < srcType.getRank(); ++i) {
      if (srcType.isDynamicDim(i)) {
        auto indexAttr = rewriter.getIndexAttr(i);
        auto indexVal =
            rewriter.create<arith::ConstantOp>(op.getLoc(), indexAttr);
        auto dimVal =
            rewriter.create<memref::DimOp>(op.getLoc(), src, indexVal);
        dynamicSizes.push_back(dimVal);
      }
    }

    // Emit memref.alloc on target topology with the dynamic sizes
    auto allocOp =
        rewriter.create<memref::AllocOp>(op.getLoc(), targetType, dynamicSizes);

    // Emit memref.copy from src to alloc
    rewriter.create<memref::CopyOp>(op.getLoc(), src, allocOp);

    // Enforce Zero Memory Leaks:
    // We must emit a memref.dealloc at the end of the current scope (block)
    // so that the allocated memory behaves like a C++ RAII object.
    Block *currentBlock = op->getBlock();
    if (!currentBlock->empty() &&
        currentBlock->back().hasTrait<OpTrait::IsTerminator>()) {
      // Temporarily move insertion point to just before the terminator
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(&currentBlock->back());
      // Cast to memory space 0 so memref-to-llvm can use standard `free`
      auto defaultType =
          MemRefType::get(targetType.getShape(), targetType.getElementType());
      auto castOp = rewriter.create<memref::MemorySpaceCastOp>(
          op.getLoc(), defaultType, allocOp);
      rewriter.create<memref::DeallocOp>(op.getLoc(), castOp);
    } else {
      // If there is no terminator yet, just append it to the block
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(currentBlock);
      // Cast to memory space 0 so memref-to-llvm can use standard `free`
      auto defaultType =
          MemRefType::get(targetType.getShape(), targetType.getElementType());
      auto castOp = rewriter.create<memref::MemorySpaceCastOp>(
          op.getLoc(), defaultType, allocOp);
      rewriter.create<memref::DeallocOp>(op.getLoc(), castOp);
    }

    // Replace transfer with the allocated memref
    rewriter.replaceOp(op, allocOp.getResult());
    return success();
  }
};

/// Reject a placed region the target cannot run and the host cannot either.
///
/// `kernelKindOf` deciding it cannot classify a region is not an error on its
/// own: routing is an optimisation, and anything unrecognised falls back to the
/// host. That reasoning holds exactly while the operands are host memory, and
/// stops the moment one has been transferred into a space declared
/// `managed: explicit` -- because the fallback then *is* a host read of memory
/// the declaration says the host cannot read. It is the same access the checker
/// already rejects when a program writes it out; `o[0][0]` after a transfer to
/// GPU_HBM is refused today. The difference is that this one is implicit,
/// created by the fallback rather than by the programmer, so nothing looked at
/// it.
///
/// Unchecked it is not a diagnostic at all. On an A100 the outlined kernel
/// dereferences device memory and the process dies inside vx_npu_kernel_0,
/// three frames below anything naming a cause (#251, #348). On a machine with
/// no device the transfer is a no-op, the pointers stay host pointers and it
/// passes -- which is how tests/backend/pass/flash_attention_placed.vx sat in
/// the passing set while faulting on every GPU it was written for.
///
/// Run before the conversion rather than inside the rewrite: a pattern that
/// fails is retried, so the same message arrived several times and left behind
/// a half-rewritten region that then produced a dominance error of its own.
/// Deliberately not conditioned on the compiling machine having a GPU. The
/// question is whether this *program* can execute as written, and one asking
/// for a kernel the compiler cannot emit for the device it named cannot, on any
/// machine. Answering it differently per build box is how the fault hid.
static LogicalResult diagnoseUnrunnableSpawns(Operation *root) {
  bool failed = false;
  root->walk([&](vx::SpawnOp spawn) {
    int32_t topology = spawn.getTopology();

    // A host region lowers to `async.execute`, whose body must be a single
    // block -- and any `if`, `for` or `match` in the region makes it several.
    // Said here, naming the construct, rather than left to surface three passes
    // later as `'async.execute' op expects region #0 to have 0 or 1 blocks`,
    // which describes an operation the program never mentions.
    if (topology == 0) {
      Region &body = spawn.getBody();
      if (!body.empty() && !body.hasOneBlock()) {
        spawn.emitError()
            << "a region placed on the host cannot carry control flow: it "
               "lowers to `async.execute`, whose body is a single block, and "
               "an `if`, `for` or `match` here makes it several.\n"
            << "  Either lift the control flow out of the region, or place the "
               "region on a device topology, where it is outlined into a "
               "kernel and keeps its blocks.";
        failed = true;
      }
      return;
    }

    Region &body = spawn.getBody();
    if (body.empty())
      return;
    if (!kernelKindOf(body).empty())
      return;
    if (!regionTouchesMemory(body))
      return;

    SetVector<Value> captures;
    getUsedValuesDefinedAbove(body, captures);
    for (Value capture : captures) {
      vx::TransferOp xfer = definingTransfer(capture);
      if (!xfer || !isDeviceTransfer(xfer) || hostCanRead(xfer))
        continue;

      StringRef space = "a device memory space";
      if (auto s = xfer->getAttrOfType<StringAttr>("space"))
        space = s.getValue();

      spawn.emitError()
          << "this region is placed on topology " << topology
          << ", but the compiler cannot emit a device kernel for it and its "
             "operands are in '"
          << space
          << "', declared `managed: explicit` -- memory the host cannot read.\n"
          << "  Falling back to the host would dereference device memory "
             "there.\n"
          << "  Either the computation must be one the target can run (a "
             "matmul is routed today; general kernel emission is #251), or its "
             "operands must stay in a space the host can read.";
      failed = true;
      break;
    }
  });
  return success(!failed);
}

/// Dialects the NVPTX pipeline in this file can lower to a device.
///
/// An allow-list, not a list of known-bad ops, and deliberately so: a kernel
/// body that grows something new should stop getting a device twin until
/// someone has decided what that op means on a GPU. The other direction fails
/// late and quietly -- an image that compiles but cannot load, blamed at the
/// far end on whatever the worker says last.
static bool isDeviceLowerableDialect(StringRef ns) {
  return ns == "arith" || ns == "cf" || ns == "gpu" || ns == "math" ||
         ns == "memref" || ns == "scf";
}

struct ConvertVxToStandardPass
    : public PassWrapper<ConvertVxToStandardPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertVxToStandardPass)

  llvm::StringRef getArgument() const override {
    return "convert-vx-to-standard";
  }

  llvm::StringRef getDescription() const override {
    return "Lowers Vx dialect operations to standard MLIR dialects";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<async::AsyncDialect, func::FuncDialect, memref::MemRefDialect,
                arith::ArithDialect, gpu::GPUDialect, math::MathDialect>();
  }

  /// Give every device kernel a `gpu.func` beside its `vx.kernel`.
  ///
  /// `vx.kernel` is what the outliner produces and what the plugin ABI
  /// dispatches: a function the host can call, by symbol, in this process. It
  /// is not something a GPU can be handed. The upstream path to a cubin --
  /// `nvvm-attach-target`, `convert-gpu-to-nvvm`, `gpu-module-to-binary` --
  /// starts at `gpu.func` inside a `gpu.module`, and nothing emitted one.
  ///
  /// scripts/flash_kernel_to_ptx.sh has been doing this edit with a text
  /// transform for as long as #251 has been open, and its header says so: "the
  /// only edit it makes is structural ... that step *is* the remaining work".
  /// It is structural because the outliner already did the hard part. The
  /// captures are the entry block's arguments, so the signature is a
  /// transcription; the body is a clone; `vx.return` becomes `gpu.return`.
  ///
  /// Both survive. The `vx.kernel` still carries the host path, so a program
  /// that runs on this machine is unaffected and the classified-matmul route is
  /// untouched. The `gpu.func` is what a device backend compiles: `vx-to-llvm`
  /// compiles it to PTX and puts the result in the dispatch payload, then drops
  /// it. So it is not dead weight any more, but it is still not on the critical
  /// path of a program that runs at home.
  ///
  /// Only device topologies. A region placed on the host has no business
  /// growing a GPU twin.
  void materializeGpuKernels(ModuleOp module) {
    SmallVector<vx::KernelOp> kernels;
    module.walk([&](vx::KernelOp k) {
      const int32_t topo = static_cast<int32_t>(k.getTopology());
      // Eligibility: the DECLARED arch when the kernel carries one, the dispatch-id band when it
      // does not. A machine file that says `arch: nvptx64` has answered "what code do we emit for
      // this topology" -- that is the field's documented meaning -- and this is the place that
      // needed the answer; before the attribute existed the gate was the band alone, which a
      // custom topology can never enter (custom ids own 1000..1999 by construction), so no
      // declaration could produce a device image (Vx#352).
      //
      // The band stays as the fallback for kernels with no arch attribute: built-in topologies
      // (whose descriptors declare no arch) and the AST-codegen path (which does not stamp).
      // A declared arch we have no device pipeline for -- `applegpu` reaches its device through
      // the plugin ABI, not NVVM -- is excluded here exactly like NPU/AccCore below.
      if (auto arch = k->getAttrOfType<StringAttr>("arch")) {
        if (arch.getValue() != "nvptx64")
          return;
      } else if (topo < 500 || topo >= 600) {
        // The GPU band from `topology_dispatch_id`. NPU and AccCore outline the
        // same way but reach their devices through a different backend, and
        // giving them an NVVM twin would be claiming something untrue.
        return;
      }
      // Only a body the device pipeline can actually compile.
      //
      // Two things get excluded, for two different reasons, and both are
      // reasons the region is not ready rather than reasons this transcription
      // is hard.
      //
      // `func`, because a `gpu.module` is its own symbol table: a body that
      // calls a host helper -- `flash_attention_v4.vx` calls `exp_poly` --
      // clones into a kernel whose callee is not visible from it, and the
      // verifier rejects the module before anything downstream sees it. Making
      // it a kernel means bringing the callee along or inlining it.
      //
      // `linalg`, because a `linalg.matmul` has not been lowered for anything
      // yet. It survives to here on purpose: it is the classified-matmul route
      // (`kernelKindOf` above), and the plugin turns it into a cuBLAS call. A
      // device twin of it would have to be tiled and mapped to threads first,
      // which is the parallelism work #251 set aside -- and it would lose:
      // cuBLAS runs this at 14389 GFLOP/s on an A100 and the dispatch path
      // already reaches 13888 of that (#321). The kernel worth shipping is the
      // one no library has, which is exactly the one with no `linalg` left in
      // it.
      //
      // Skipping leaves such a program exactly as it was. Nothing downstream
      // requires a twin; a launch without an image simply carries no `image=`
      // field, as every launch did before this.
      bool deviceReady = true;
      k.getBody().walk([&](Operation *op) {
        // Rewritten to `gpu.return` below, so it is not a foreign dialect here.
        if (isa<vx::ReturnOp>(op))
          return WalkResult::advance();
        // A DYNAMIC shared-memory tile cannot become a `.shared` global (those
        // need a static shape), and leaving the alloca as-is ships a kernel
        // that faults: measured on an A100 as ILLEGAL_ADDRESS, because the
        // stack slot the alloca lowers to is then accessed through
        // shared-typed pointers. Refusing materialization keeps the program on
        // the host path, which computes the right answer.
        if (auto alloca = dyn_cast<memref::AllocaOp>(op)) {
          auto t = dyn_cast<MemRefType>(alloca.getType());
          auto space =
              t ? dyn_cast_or_null<IntegerAttr>(t.getMemorySpace()) : nullptr;
          if (space && space.getInt() == 3 && !t.hasStaticShape()) {
            deviceReady = false;
            return WalkResult::interrupt();
          }
        }
        Dialect *dialect = op->getDialect();
        if (dialect && isDeviceLowerableDialect(dialect->getNamespace()))
          return WalkResult::advance();
        deviceReady = false;
        return WalkResult::interrupt();
      });
      if (deviceReady)
        kernels.push_back(k);
    });
    if (kernels.empty())
      return;

    OpBuilder builder(module.getBodyRegion());
    builder.setInsertionPointToEnd(module.getBody());
    auto gpuModule =
        builder.create<gpu::GPUModuleOp>(module.getLoc(), "vx_kernels");

    for (vx::KernelOp kernel : kernels) {
      Region &body = kernel.getBody();
      if (body.empty())
        continue;

      OpBuilder inner(gpuModule.getBody(), gpuModule.getBody()->end());
      auto funcType = FunctionType::get(
          &getContext(), body.front().getArgumentTypes(), /*results=*/{});
      auto gpuFunc = inner.create<gpu::GPUFuncOp>(
          kernel.getLoc(), kernel.getSymName(), funcType);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       inner.getUnitAttr());

      // The whole region, not its entry block. What the outliner produces is
      // raw CFG -- `cf.br` and `cf.cond_br` over a dozen blocks, every
      // induction variable in an `alloca` -- so copying only the first block
      // leaves every branch pointing at a block that was never cloned, and the
      // verifier says "reference to block defined in another region".
      //
      // `GPUFuncOp` builds its own entry block from the signature, so the clone
      // lands after it and the two are then stitched: the cloned entry's
      // arguments are replaced by the real ones and its operations move up.
      // An entry block cannot be a branch target, so nothing is left pointing
      // at the husk that gets erased.
      IRMapping map;
      Region &target = gpuFunc.getBody();
      Block &gpuEntry = target.front();
      body.cloneInto(&target, map);

      Block *clonedEntry = &*std::next(target.begin());
      for (auto [from, to] :
           llvm::zip(clonedEntry->getArguments(), gpuEntry.getArguments()))
        from.replaceAllUsesWith(to);
      gpuEntry.getOperations().splice(gpuEntry.end(),
                                      clonedEntry->getOperations());
      clonedEntry->erase();

      // Shared-memory STORAGE. A space-3 `memref.alloca` is correct on the
      // host path (it lowers to a stack slot and the host runs the region
      // correctly -- verified by the corpus tests), but in the device clone
      // that same lowering is a fault: the PTX gets `ld.shared`/`st.shared`
      // against `cvta.shared` of the LOCAL depot, with no `.shared` storage
      // declared anywhere. Measured on an A100: CUDA_ERROR_ILLEGAL_ADDRESS.
      // The instructions were shared-typed; the storage never was. (The
      // `.shared`-substring check on the PTX passed on the instructions alone,
      // which is the mislabelled-artifact failure one layer deeper.)
      //
      // So in the device clone the storage must BE shared: each static
      // space-3 alloca becomes a module-level `memref.global` in space 3 --
      // NVPTX renders exactly that as a `.shared` declaration -- and the
      // types line up with the alloca's uses with no casts. Dynamic ones were
      // refused above.
      {
        SmallVector<memref::AllocaOp> smemAllocas;
        gpuFunc.walk([&](memref::AllocaOp a) {
          auto t = dyn_cast<MemRefType>(a.getType());
          auto space =
              t ? dyn_cast_or_null<IntegerAttr>(t.getMemorySpace()) : nullptr;
          if (space && space.getInt() == 3 && t.hasStaticShape())
            smemAllocas.push_back(a);
        });
        int smemIdx = 0;
        for (memref::AllocaOp a : smemAllocas) {
          std::string gname = (kernel.getSymName() + "_smem_" +
                               std::to_string(smemIdx++))
                                  .str();
          OpBuilder atModule(gpuModule.getBody(), gpuModule.getBody()->begin());
          atModule.create<memref::GlobalOp>(
              a.getLoc(), gname,
              /*sym_visibility=*/atModule.getStringAttr("private"),
              /*type=*/cast<MemRefType>(a.getType()),
              /*initial_value=*/Attribute(), /*constant=*/false,
              /*alignment=*/IntegerAttr());
          OpBuilder at(a);
          auto gg =
              at.create<memref::GetGlobalOp>(a.getLoc(), a.getType(), gname);
          a.replaceAllUsesWith(gg.getResult());
          a.erase();
        }
      }

      // `vx.return` is not a terminator a GPU module may contain.
      SmallVector<vx::ReturnOp> returns;
      gpuFunc.walk([&](vx::ReturnOp r) { returns.push_back(r); });
      for (vx::ReturnOp r : returns) {
        OpBuilder at(r);
        at.create<gpu::ReturnOp>(r.getLoc());
        r.erase();
      }

      // A region whose last statement fell through carried no terminator.
      for (Block &b : target) {
        if (b.empty() || !b.back().hasTrait<OpTrait::IsTerminator>()) {
          OpBuilder end(&b, b.end());
          end.create<gpu::ReturnOp>(kernel.getLoc());
        }
      }
    }
  }

  void runOnOperation() override {
    getOperation()->emitRemark("Lowering Vx to Standard dialects");

    // Before anything is rewritten, so the diagnostic describes the program as
    // written rather than a partially converted version of it.
    if (::mlir::failed(diagnoseUnrunnableSpawns(getOperation()))) {
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<SpawnOpLowering, TransferOpLowering>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // After the outlining, because that is what produces the kernels this
    // reads.
    materializeGpuKernels(getOperation());
  }
};

// ABI type tags for kernel arguments, shared with runtime/npu_dispatch.mm.
// The runtime maps these to libffi types to reconstruct the C calling
// convention. Keep the encoding in sync with vx_abi_ffi_type() there.
//   0=ptr, 1=i1, 2=i8, 3=i16, 4=i32, 5=i64, 6=f32, 7=f64
static int32_t abiTagForType(Type t) {
  if (isa<Float32Type>(t))
    return 6;
  if (isa<Float64Type>(t))
    return 7;
  if (auto it = dyn_cast<IntegerType>(t)) {
    switch (it.getWidth()) {
    case 1:
      return 1;
    case 8:
      return 2;
    case 16:
      return 3;
    case 32:
      return 4;
    default:
      return 5; // i64 (and any wider integer, widened to i64 on the slot)
    }
  }
  return 0; // pointer (memref descriptor) and fallback
}

// Element type code for a memref's element, packed into the high bytes of the
// argument tag. Keep in sync with the VX_DTYPE_* enum in
// include/vx_hardware_runtime.h.
//
// This is what a plugin needs in order to interpret the descriptor it receives:
// without it, a runtime can only assume a layout, which is why the Apple path
// hardcodes float and rank 2 and would silently misread an f16 buffer.
static int32_t elemDtypeCode(Type t) {
  if (isa<Float32Type>(t))
    return 6;
  if (isa<Float64Type>(t))
    return 7;
  if (isa<Float16Type>(t))
    return 8;
  if (isa<BFloat16Type>(t))
    return 9;
  if (auto it = dyn_cast<IntegerType>(t)) {
    switch (it.getWidth()) {
    case 1:
      return 1;
    case 8:
      return 2;
    case 16:
      return 3;
    case 32:
      return 4;
    default:
      return 5;
    }
  }
  return 0; // unknown
}

// Lower a transfer into a device space to the plugin's allocate-and-transfer.
//
// This is what makes a `transfer` mean something physical. It used to lower to
// `memref.alloc` + `memref.copy` -- a host allocation, whatever placement the
// program declared -- so the compiler planned data movement that never
// happened, and every dispatch then staged its operands across the bus again
// because nothing was ever resident (#319).
//
// The compiler emits a call to the vendor-neutral plugin ABI and never names
// CUDA: `cuda_dispatch.cpp` answers it with cudaMalloc and an H2D copy, the
// Apple path with its own, the portable shim with malloc and memcpy. The same
// program is correct on a laptop and resident on a GPU.
//
// The result is a descriptor over the returned pointer, carrying the source's
// extents and strides. It points at device memory, which the host must not
// dereference -- that is what the placement rules in the checker are for
// (E6003 refuses a device buffer read from the wrong topology).
struct TransferToPluginLowering : public OpRewritePattern<vx::TransferOp> {
  const LLVMTypeConverter &typeConverter;

  TransferToPluginLowering(const LLVMTypeConverter &typeConverter,
                           MLIRContext *context)
      : OpRewritePattern<vx::TransferOp>(context),
        typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(vx::TransferOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value src = op.getOperand();

    // Two directions, both of them this pattern's business because both need a
    // raw pointer out of a memref.
    //
    //   out    `transfer(x, Memory::GPU_HBM)` -- allocate there, copy into it.
    //   home   `transfer(d, Memory::CPU_DRAM)` where d is on a device --
    //          allocate here, and fetch.
    //
    // The way home used to be handled by TransferOpLowering as an allocation
    // and a `memref.copy`, which reads the source directly: a device pointer
    // locally, and across a fleet a non-canonical handle naming memory in
    // another process. So `vx_plugin_transfer_device_to_host` -- which exists
    // and is wired to the fleet routing -- had no caller in generated code, and
    // a program asking for its data back silently did not get it.
    const int32_t srcTopology = sourceTopologyOf(op);
    const bool goingOut = isDeviceTransfer(op);
    const bool comingHome = !goingOut && srcTopology != 0;
    // And the third direction: device to device, which is neither of the above
    // and was being lowered as though it were the first. `transfer(kv,
    // Memory::HBM4)` where kv is already on another device became
    // `vx_plugin_alloc_and_transfer(bytes, src, dst)`, whose second argument is
    // read as host memory -- so the source's address was dereferenced here. On
    // one machine that is a device pointer and the copy is merely wrong about
    // which memory it names; across a fleet it is a handle, and the process
    // faults inside the staging memcpy.
    //
    // `vx_plugin_transfer_peer` is the entry point for this. It exists in every
    // backend and is wired to the routing, and had no caller in generated code
    // -- the same shape of gap the way home had (#347, #348).
    int32_t targetTopology = 0;
    if (auto a = op->getAttrOfType<IntegerAttr>("target_topology"))
      targetTopology = static_cast<int32_t>(a.getInt());
    const bool peerHandoff =
        goingOut && srcTopology != 0 && srcTopology != targetTopology;
    if (!goingOut && !comingHome)
      return failure();

    auto srcType = dyn_cast<MemRefType>(src.getType());
    if (!srcType)
      return failure();

    int64_t rank = srcType.getRank();
    auto llvmPtrType = LLVM::LLVMPointerType::get(getContext());
    auto llvmI64Type = IntegerType::get(getContext(), 64);
    auto llvmI32Type = IntegerType::get(getContext(), 32);

    Type convertedSrc = typeConverter.convertType(srcType);
    if (!convertedSrc)
      return failure();
    Value desc =
        rewriter.create<UnrealizedConversionCastOp>(loc, convertedSrc, src)
            .getResult(0);

    // Field 1 of {allocated, aligned, offset, sizes, strides} is where the
    // elements are; the plugin copies from there.
    Value alignedPtr = rewriter.create<LLVM::ExtractValueOp>(
        loc, llvmPtrType, desc, ArrayRef<int64_t>{1});

    // Bytes, read off the descriptor rather than the type, so a dynamic extent
    // costs nothing extra and no shape is assumed.
    unsigned elemBits = srcType.getElementType().getIntOrFloatBitWidth();
    Value bytes = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI64Type, rewriter.getI64IntegerAttr((elemBits + 7) / 8));
    for (int64_t d = 0; d < rank; ++d) {
      Value dim = rewriter.create<LLVM::ExtractValueOp>(
          loc, llvmI64Type, desc, ArrayRef<int64_t>{3, d});
      bytes = rewriter.create<LLVM::MulOp>(loc, bytes, dim);
    }

    // Which device, as the machine model named it.
    int32_t topology = 0;
    if (auto topoAttr = op->getAttrOfType<IntegerAttr>("target_topology"))
      topology = static_cast<int32_t>(topoAttr.getInt());
    Value topoVal = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI32Type, rewriter.getI32IntegerAttr(topology));

    ModuleOp module = op->getParentOfType<ModuleOp>();
    StringRef allocName = "vx_plugin_alloc_and_transfer";
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>(allocName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto fnTy = LLVM::LLVMFunctionType::get(
          llvmPtrType, {llvmI64Type, llvmPtrType, llvmI32Type}, false);
      rewriter.create<LLVM::LLVMFuncOp>(loc, allocName, fnTy);
    }
    Value devicePtr;
    if (comingHome) {
      // Host memory first, with no copy -- there is nothing here to copy from.
      // A null source is what tells the plugin to allocate and stop.
      Value nullSrc = rewriter.create<LLVM::ZeroOp>(loc, llvmPtrType);
      Value hostTopo = rewriter.create<LLVM::ConstantOp>(
          loc, llvmI32Type, rewriter.getI32IntegerAttr(0));
      devicePtr = rewriter
                      .create<LLVM::CallOp>(
                          loc, TypeRange{llvmPtrType},
                          SymbolRefAttr::get(rewriter.getContext(), allocName),
                          ValueRange{bytes, nullSrc, hostTopo})
                      .getResult();

      // Then the fetch, addressed to the topology the data is actually on --
      // this op names only where it is going.
      Value srcTopoVal = rewriter.create<LLVM::ConstantOp>(
          loc, llvmI32Type, rewriter.getI32IntegerAttr(srcTopology));

      StringRef fetchName = "vx_plugin_transfer_device_to_host";
      if (!module.lookupSymbol<LLVM::LLVMFuncOp>(fetchName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        auto fnTy = LLVM::LLVMFunctionType::get(
            llvmI32Type, {llvmPtrType, llvmPtrType, llvmI64Type, llvmI32Type},
            false);
        rewriter.create<LLVM::LLVMFuncOp>(loc, fetchName, fnTy);
      }
      rewriter.create<LLVM::CallOp>(
          loc, TypeRange{llvmI32Type},
          SymbolRefAttr::get(rewriter.getContext(), fetchName),
          ValueRange{alignedPtr, devicePtr, bytes, srcTopoVal});
    } else if (peerHandoff) {
      // One call, because only the plugin knows whether the two devices can
      // reach each other: a peer copy where they can, and a read followed by a
      // write where they cannot. Staging it here would force the second on
      // every backend and put the bytes through this process, which for two
      // GPUs on one machine is the thing worth avoiding.
      Value srcTopoVal = rewriter.create<LLVM::ConstantOp>(
          loc, llvmI32Type, rewriter.getI32IntegerAttr(srcTopology));

      StringRef peerName = "vx_plugin_transfer_peer";
      if (!module.lookupSymbol<LLVM::LLVMFuncOp>(peerName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        auto fnTy = LLVM::LLVMFunctionType::get(
            llvmPtrType, {llvmPtrType, llvmI32Type, llvmI32Type, llvmI64Type},
            false);
        rewriter.create<LLVM::LLVMFuncOp>(loc, peerName, fnTy);
      }
      devicePtr = rewriter
                      .create<LLVM::CallOp>(
                          loc, TypeRange{llvmPtrType},
                          SymbolRefAttr::get(rewriter.getContext(), peerName),
                          ValueRange{alignedPtr, srcTopoVal, topoVal, bytes})
                      .getResult();
    } else {
      devicePtr = rewriter
                      .create<LLVM::CallOp>(
                          loc, TypeRange{llvmPtrType},
                          SymbolRefAttr::get(rewriter.getContext(), allocName),
                          ValueRange{bytes, alignedPtr, topoVal})
                      .getResult();
    }

    // Rebuild a descriptor over the device pointer, keeping the source's shape.
    Value newDesc = rewriter.create<LLVM::UndefOp>(loc, convertedSrc);
    newDesc = rewriter.create<LLVM::InsertValueOp>(loc, newDesc, devicePtr,
                                                   ArrayRef<int64_t>{0});
    newDesc = rewriter.create<LLVM::InsertValueOp>(loc, newDesc, devicePtr,
                                                   ArrayRef<int64_t>{1});
    Value zero = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI64Type, rewriter.getI64IntegerAttr(0));
    newDesc = rewriter.create<LLVM::InsertValueOp>(loc, newDesc, zero,
                                                   ArrayRef<int64_t>{2});
    for (int64_t d = 0; d < rank; ++d) {
      Value size = rewriter.create<LLVM::ExtractValueOp>(
          loc, llvmI64Type, desc, ArrayRef<int64_t>{3, d});
      Value stride = rewriter.create<LLVM::ExtractValueOp>(
          loc, llvmI64Type, desc, ArrayRef<int64_t>{4, d});
      newDesc = rewriter.create<LLVM::InsertValueOp>(loc, newDesc, size,
                                                     ArrayRef<int64_t>{3, d});
      newDesc = rewriter.create<LLVM::InsertValueOp>(loc, newDesc, stride,
                                                     ArrayRef<int64_t>{4, d});
    }

    Value result = rewriter
                       .create<UnrealizedConversionCastOp>(
                           loc, op.getResult().getType(), newDesc)
                       .getResult(0);

    // Release it through the same ABI that allocated it. The host transfer path
    // emits `memref.dealloc`, which becomes a libc `free` -- correct for a host
    // allocation and heap corruption for one that came from cudaMalloc. An
    // allocator and its free have to be the same backend, so the compiler names
    // neither and calls the plugin.
    StringRef freeName = "vx_plugin_free";
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>(freeName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto freeTy =
          LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(getContext()),
                                      {llvmPtrType, llvmI32Type}, false);
      rewriter.create<LLVM::LLVMFuncOp>(loc, freeName, freeTy);
    }
    // *Where* the free goes is the whole question, and the end of the defining
    // block is the wrong answer whenever any control flow follows.
    //
    //   let a = transfer(a_h, Memory::GPU_HBM);
    //   for step in 0..8 { spawn on(Topology::GPU) { ... a ... } }
    //
    // The loop opens a new block, so the transfer's block ends at the branch
    // into it and the free was emitted there -- before a single iteration ran.
    // Against a fleet worker that is visible and fatal: the trace is three
    // TRANSFERs, three FREEs, and then a dispatch refused because "an argument
    // named no live region". Locally it is neither, because freed host memory
    // still reads, so every placed test passed while doing this.
    //
    // A transfer whose block dominates the function's exits is live until one
    // of them, so the free belongs there. One that does not -- a transfer
    // inside a loop -- keeps the old placement: its result is a different
    // allocation each iteration, the SSA value does not reach the return, and
    // the end of its own block is where that allocation's life actually ends.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      Block *defBlock = op->getBlock();
      Operation *parentFn = op->getParentOfType<LLVM::LLVMFuncOp>();
      if (!parentFn)
        parentFn = op->getParentOfType<func::FuncOp>();

      SmallVector<Block *> exits;
      bool dominatesAllExits = parentFn != nullptr;
      if (parentFn) {
        DominanceInfo dom(parentFn);
        for (Region &region : parentFn->getRegions()) {
          for (Block &block : region) {
            if (block.empty())
              continue;
            Operation &term = block.back();
            if (!isa<func::ReturnOp, LLVM::ReturnOp>(term))
              continue;
            exits.push_back(&block);
            if (!dom.dominates(defBlock, &block))
              dominatesAllExits = false;
          }
        }
      }

      if (dominatesAllExits && !exits.empty()) {
        for (Block *exit : exits) {
          rewriter.setInsertionPoint(&exit->back());
          rewriter.create<LLVM::CallOp>(
              loc, TypeRange{},
              SymbolRefAttr::get(rewriter.getContext(), freeName),
              ValueRange{devicePtr, topoVal});
        }
      } else {
        if (!defBlock->empty() &&
            defBlock->back().hasTrait<OpTrait::IsTerminator>())
          rewriter.setInsertionPoint(&defBlock->back());
        else
          rewriter.setInsertionPointToEnd(defBlock);
        rewriter.create<LLVM::CallOp>(
            loc, TypeRange{},
            SymbolRefAttr::get(rewriter.getContext(), freeName),
            ValueRange{devicePtr, topoVal});
      }
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LaunchOpLowering : public OpRewritePattern<vx::LaunchOp> {
  const LLVMTypeConverter &typeConverter;
  /// Kernel name -> device image, for the kernels that have one. Owned by the
  /// pass; empty for a host launch, and empty everywhere until #251.
  const llvm::StringMap<std::string> *deviceImages;

  LaunchOpLowering(const LLVMTypeConverter &typeConverter, MLIRContext *context,
                   const llvm::StringMap<std::string> *deviceImages)
      : OpRewritePattern<vx::LaunchOp>(context), typeConverter(typeConverter),
        deviceImages(deviceImages) {}

  LogicalResult matchAndRewrite(vx::LaunchOp op,
                                PatternRewriter &rewriter) const override {
    auto &convRewriter = static_cast<ConversionPatternRewriter &>(rewriter);
    Location loc = op.getLoc();
    auto callee = op.getCalleeAttr().getValue();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto llvmI8Type = IntegerType::get(getContext(), 8);
    auto llvmPtrType = LLVM::LLVMPointerType::get(getContext());
    auto llvmI64Type = IntegerType::get(getContext(), 64);
    auto llvmI32Type = IntegerType::get(getContext(), 32);

    // 1. Create the payload: a NUL-separated blob whose first entry is the
    //    kernel name, followed by zero or more `key=value` entries, bounded by
    //    the payload_size argument. Reading it as a `const char *` still yields
    //    the kernel name exactly as before, so a consumer that predates the
    //    extra entries is unaffected -- which is why the name leads rather than
    //    a header. See vx_payload_field() in include/vx_hardware_runtime.h.
    std::string payload = callee.str();
    payload.push_back('\0');
    if (auto kindAttr = op->getAttrOfType<StringAttr>("vx.kernel_kind")) {
      payload += "kind=";
      payload += kindAttr.getValue().str();
      payload.push_back('\0');
    }
    if (auto rolesAttr = op->getAttrOfType<StringAttr>("vx.kernel_roles")) {
      payload += "roles=";
      payload += rolesAttr.getValue().str();
      payload.push_back('\0');
    }
    if (auto outAttr = op->getAttrOfType<StringAttr>("vx.kernel_out_kind")) {
      payload += "outkind=";
      payload += outAttr.getValue().str();
      payload.push_back('\0');
    }

    // The device this launch targets. Without it every dispatch reaches a
    // plugin identically and a program cannot say "prefill here, decode there"
    // -- the runtime half of #331. The value is the same topology id
    // `vx.kernel` carries (src/arch.rs `topology_dispatch_id`); a plugin turns
    // it into a device ordinal with vx_topology_device_index().
    payload += "topo=";
    payload += std::to_string(op.getTopology());
    payload.push_back('\0');

    // The name that id was derived from, when the producer knew it.
    //
    // For a topology declared in a machine file the id is `1000 + fnv32(name) %
    // 1000` -- one-way, and only a thousand wide. A plugin given `topo=1113`
    // cannot recover `DecodeWorker`, so it cannot resolve the worker against a
    // fleet manifest to find out which machine it is; and two names can collide
    // onto one id with nothing to notice. The name is the identity, the id a
    // shortcut, and this is what a remote placement will key on (#348).
    if (auto nameAttr = op->getAttrOfType<StringAttr>("vx.topology_name")) {
      payload += "toponame=";
      payload += nameAttr.getValue().str();
      payload.push_back('\0');
    }

    // The kernel itself, last.
    //
    // A dispatch names a kernel; a plugin that does not recognise the name has
    // nothing to run, which is why a region that is not a classified matmul is
    // refused rather than executed. This is the kernel, as PTX, so that a
    // plugin which cannot recognise it can still load and launch it (#251).
    //
    // In the payload because that is the one channel that already reaches every
    // plugin. The alternative -- another argument on vx_plugin_dispatch_async
    // -- changes the ABI in five backends for a field four of them ignore, and
    // changes it again the first time something else needs carrying.
    //
    // The blob's encoding survives it: PTX is text with no NUL, so it is one
    // entry like any other and vx_payload_field walks past it unchanged.
    // deviceImageOf checks that, rather than trusting it.
    //
    // Last, because a reader dumping a payload should meet the small fields
    // first, and because this is the only entry measured in kilobytes.
    //
    // It costs a dispatch nothing: the blob is a constant global, one per
    // kernel, so a launch passes a pointer and a length however large the image
    // is. It costs the object file one copy of the PTX in .rodata.
    if (deviceImages) {
      auto image = deviceImages->find(callee);
      if (image != deviceImages->end()) {
        payload += "image=";
        payload += image->second;
        payload.push_back('\0');
      }
    }

    std::string globalName = (callee + "_str").str();
    LLVM::GlobalOp globalOp = module.lookupSymbol<LLVM::GlobalOp>(globalName);
    if (!globalOp) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      std::string strWithNull = payload;
      auto arrayTy = LLVM::LLVMArrayType::get(llvmI8Type, strWithNull.size());
      globalOp = rewriter.create<LLVM::GlobalOp>(
          loc, arrayTy, /*isConstant=*/true, LLVM::Linkage::Internal,
          globalName,
          rewriter.getStringAttr(
              StringRef(strWithNull.data(), strWithNull.size())));
    }
    Value globalPtr =
        rewriter.create<LLVM::AddressOfOp>(loc, llvmPtrType, globalName);

    // 2. Allocate the device_args array (one pointer per argument) and a
    //    parallel array of ABI type tags. Every device_args[i] points to the
    //    value of the i-th C-interface argument, so the runtime can rebuild the
    //    platform calling convention via libffi (see docs/lang/abi.md):
    //      - scalar param -> pointer to the scalar value
    //      - memref param -> pointer to the (pointer-to-descriptor)
    int numArgs = op.getNumOperands();
    Value countVal = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI32Type,
        rewriter.getI32IntegerAttr(numArgs > 0 ? numArgs : 1));
    Value argsArray = rewriter.create<LLVM::AllocaOp>(
        loc, llvmPtrType, llvmPtrType, countVal, /*alignment=*/0);
    Value tagsArray = rewriter.create<LLVM::AllocaOp>(
        loc, llvmPtrType, llvmI32Type, countVal, /*alignment=*/0);

    Value one = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI32Type, rewriter.getI32IntegerAttr(1));

    for (auto en : llvm::enumerate(op.getOperands())) {
      Value originalArg = en.value();
      Type originalTy = originalArg.getType();
      Type argTy = typeConverter.convertType(originalTy);

      Value arg = originalArg;
      if (originalTy.isIndex()) {
        arg = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(),
                                                  originalArg);
        argTy = rewriter.getI64Type();
      } else if (argTy && argTy != originalTy) {
        arg =
            rewriter.create<UnrealizedConversionCastOp>(loc, argTy, originalArg)
                .getResult(0);
      }

      Value valuePtr;
      int32_t tag;
      if (isa<MemRefType>(originalTy)) {
        // The C-interface passes a memref as a pointer to its descriptor, so
        // the arg value is that descriptor pointer; device_args[i] points to
        // it.
        Value descAlloc = rewriter.create<LLVM::AllocaOp>(
            loc, llvmPtrType, argTy, one, /*alignment=*/0);
        rewriter.create<LLVM::StoreOp>(loc, arg, descAlloc);
        Value descPtrAlloc = rewriter.create<LLVM::AllocaOp>(
            loc, llvmPtrType, llvmPtrType, one, /*alignment=*/0);
        rewriter.create<LLVM::StoreOp>(loc, descAlloc, descPtrAlloc);
        valuePtr = descPtrAlloc;
        // Kind stays 0 so the calling convention is unchanged; rank and element
        // type ride in the high bytes for plugins that route to vendor kernels.
        auto memrefTy = cast<MemRefType>(originalTy);
        Type elemTy = memrefTy.getElementType();
        int32_t rank = static_cast<int32_t>(memrefTy.getRank());
        int32_t slotBit = 0;

        // A memref whose element is itself a memref is a *slot*: storage
        // holding a descriptor rather than elements. `c = a @ b` allocates its
        // result inside the kernel and stores the descriptor through such a
        // slot, so a plugin standing in for the kernel has to write one there
        // itself.
        //
        // Described verbatim the slot is rank 0 with no scalar element type,
        // which is exactly the encoding of an opaque pointer -- the compiler
        // would be discarding the one fact that makes the argument writable.
        // Describe the pointee instead and set the slot bit to say so. One
        // level only: a slot of slots leaves the element unknown, which is
        // honest.
        if (auto innerTy = dyn_cast<MemRefType>(elemTy)) {
          slotBit = 1 << 24;
          rank = static_cast<int32_t>(innerTy.getRank());
          elemTy = innerTy.getElementType();
        }

        tag = slotBit | (rank << 16) | (elemDtypeCode(elemTy) << 8);
      } else {
        // Scalar: device_args[i] points directly to the value.
        Value scalarAlloc = rewriter.create<LLVM::AllocaOp>(
            loc, llvmPtrType, argTy, one, /*alignment=*/0);
        rewriter.create<LLVM::StoreOp>(loc, arg, scalarAlloc);
        valuePtr = scalarAlloc;
        tag = abiTagForType(argTy);
      }

      Value argSlot =
          rewriter.create<LLVM::GEPOp>(loc, llvmPtrType, llvmPtrType, argsArray,
                                       ArrayRef<LLVM::GEPArg>{en.index()});
      rewriter.create<LLVM::StoreOp>(loc, valuePtr, argSlot);

      Value tagVal = rewriter.create<LLVM::ConstantOp>(
          loc, llvmI32Type, rewriter.getI32IntegerAttr(tag));
      Value tagSlot =
          rewriter.create<LLVM::GEPOp>(loc, llvmPtrType, llvmI32Type, tagsArray,
                                       ArrayRef<LLVM::GEPArg>{en.index()});
      rewriter.create<LLVM::StoreOp>(loc, tagVal, tagSlot);
    }

    // 3. Declare vx_plugin_dispatch_async(name, payload_size, device_args,
    //    arg_tags, num_args)
    StringRef dispatchFuncName = "vx_plugin_dispatch_async";
    LLVM::LLVMFuncOp dispatchFunc =
        module.lookupSymbol<LLVM::LLVMFuncOp>(dispatchFuncName);
    if (!dispatchFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto funcType = LLVM::LLVMFunctionType::get(
          llvmI64Type,
          {llvmPtrType, llvmI64Type, llvmPtrType, llvmPtrType, llvmI64Type},
          false);
      dispatchFunc =
          rewriter.create<LLVM::LLVMFuncOp>(loc, dispatchFuncName, funcType);
    }

    // 4. Emit the call. payload_size bounds the blob so a consumer can walk the
    //    entries past the kernel name without running off the end; it was
    //    previously passed as 0.
    Value payloadSize = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI64Type,
        rewriter.getI64IntegerAttr(static_cast<int64_t>(payload.size())));
    Value numArgsVal = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI64Type, rewriter.getI64IntegerAttr(numArgs));
    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, dispatchFunc,
        ValueRange{globalPtr, payloadSize, argsArray, tagsArray, numArgsVal});

    // 5. Handle return type
    // If the original operation had a result, we must provide it.
    // Since this is a stub for now, we provide a dummy value (e.g., 0) casted
    // to the expected LLVM type.
    if (op.getNumResults() > 0) {
      Type resultTy = typeConverter.convertType(op.getResultTypes()[0]);
      if (resultTy.isInteger(32)) {
        Value zero = rewriter.create<LLVM::ConstantOp>(
            loc, resultTy, rewriter.getI32IntegerAttr(0));
        rewriter.replaceOp(op, zero);
      } else {
        // Fallback to undef for other types (e.g. memref returns)
        Value undef = rewriter.create<LLVM::UndefOp>(loc, resultTy);
        rewriter.replaceOp(op, undef);
      }
    } else {
      rewriter.eraseOp(op);
    }

    return success();
  }
};

struct KernelOpLowering : public OpRewritePattern<vx::KernelOp> {
  using OpRewritePattern<vx::KernelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vx::KernelOp op,
                                PatternRewriter &rewriter) const override {
    auto funcOp = rewriter.create<func::FuncOp>(
        op.getLoc(), op.getSymName(), cast<FunctionType>(op.getFunctionType()));
    funcOp->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());

    // Move the region over
    rewriter.inlineRegionBefore(op.getBody(), funcOp.getBody(), funcOp.end());

    rewriter.eraseOp(op);
    return success();
  }
};

struct ReturnOpLowering : public OpRewritePattern<vx::ReturnOp> {
  using OpRewritePattern<vx::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vx::ReturnOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, op.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Device images
//===----------------------------------------------------------------------===//

/// The GPU generation to compile a device kernel for.
///
/// The machine model does not carry one. `fleet/a100-40.vx` says `arch:
/// nvptx64` -- the target, not the part -- and until now nothing needed the
/// difference, because a GEMM routed to cuBLAS is compiled by whoever built
/// cuBLAS. A kernel of our own does need it: `sm_80` code will not load on an
/// `sm_70` device.
///
/// The topology id cannot supply it. It is a band plus a device ordinal
/// (`topology_dispatch_id` in src/arch.rs), so `500` means "the first GPU" and
/// says nothing about which GPU. So the default is the part this work is being
/// measured on and the override is an environment variable -- which is honest
/// about what it is, a stand-in for a `capability:` field on the Topology
/// declaration that the machine files should eventually carry.
static std::string deviceChip() {
  if (const char *chip = ::getenv("VX_GPU_CHIP"))
    if (*chip)
      return chip;
  return "sm_80";
}

/// libdevice.10.bc, if this machine has it.
///
/// Every `math` op lowers to a libdevice call under the NVVM conversion --
/// `__nv_expf` for `.exp()`, and `__nv_sqrtf` and `__nv_fabsf` too, which have
/// native PTX instructions. Linking the bitcode resolves and inlines them:
/// `math.exp` becomes a single `ex2.approx.f32` and the module is left with no
/// externals at all, which is what makes it loadable.
///
/// The file is architecture-independent bitcode, so it need not come from the
/// machine that will run the kernel; copying it to a laptop is enough, which is
/// the case that matters here because the host program compiles where the
/// developer is and runs where the GPU is.
///
/// Absent, the kernel is still compiled and still emitted, carrying an
/// unresolved `__nv_expf` that a loader will name. Refusing to emit would
/// suppress the one diagnostic worth having.
static std::string deviceLibdevice() {
  if (const char *path = ::getenv("VX_LIBDEVICE"))
    if (*path)
      return path;
  SmallVector<std::string, 2> candidates;
  if (const char *home = ::getenv("CUDA_HOME"))
    if (*home)
      candidates.push_back(std::string(home) +
                           "/nvvm/libdevice/libdevice.10.bc");
  candidates.push_back("/usr/local/cuda/nvvm/libdevice/libdevice.10.bc");
  for (const std::string &candidate : candidates)
    if (llvm::sys::fs::exists(candidate))
      return candidate;
  return {};
}

/// Compile a `gpu.module` to a device image, as PTX text.
///
/// This is the pipeline scripts/flash_kernel_to_ptx.sh established, moved into
/// the compiler. That script drove `convert-vx-to-standard`'s own kernel to
/// sm_80 by hand and reported what was left; running the same passes here is
/// what turns "this kernel can reach PTX" into "every compile produces one".
/// The script still exists and still runs the pipeline externally, so the two
/// transcriptions can be compared -- which is how the first one was checked.
///
/// PTX rather than a cubin, for two reasons. The driver JITs PTX inside
/// `cuModuleLoadData`, so a worker needs no `ptxas` and the compile host needs
/// no CUDA toolkit; and PTX stays loadable on a device newer than the one it
/// was compiled for, which a cubin is not. It costs a JIT on first load, once
/// per worker per kernel, against a dispatch path whose overhead is already
/// tens of microseconds.
///
/// Serialized from a copy in a module of its own, for two reasons of its own.
/// `gpu-to-llvm` and `convert-arith-to-llvm` are not scoped to device code: run
/// against the real module they would rewrite the host program that this pass
/// is itself in the middle of converting. And a failure in here cannot then
/// leave the host module half-lowered -- the copy is what gets damaged.
///
/// Returns the empty string on failure, with `error` naming the stage.
static std::string deviceImageOf(gpu::GPUModuleOp gpuModule,
                                 std::string &error) {
  MLIRContext *context = gpuModule.getContext();

  // `#nvvm.target` cannot serialize itself until the external model is
  // attached, and translating the kernel to LLVM IR needs the NVVM and GPU
  // dialects' translation interfaces. melior's `register_all_dialects` supplies
  // neither -- in this MLIR neither `registerAllDialects` nor
  // `registerAllExtensions` mentions NVVM -- so this is where they arrive.
  // Appending is idempotent and reaches dialects that are already loaded.
  DialectRegistry registry;
  registerAllToLLVMIRTranslations(registry);
  NVVM::registerNVVMTargetInterfaceExternalModels(registry);
  context->appendDialectRegistry(registry);

  OwningOpRef<ModuleOp> device = ModuleOp::create(gpuModule.getLoc());
  device->getBody()->push_back(gpuModule->clone());

  PassManager pm(context, ModuleOp::getOperationName());

  GpuNVVMAttachTargetOptions attach;
  attach.chip = deviceChip();
  // The PTX ISA version, not the device. 7.6 is what CUDA 11.6 and later
  // accept, and it is what the script has been assembling with.
  attach.features = "+ptx76";
  std::string libdevice = deviceLibdevice();
  if (!libdevice.empty())
    attach.linkLibs.push_back(libdevice);
  pm.addPass(createGpuNVVMAttachTarget(attach));

  pm.nest<gpu::GPUModuleOp>().addPass(createConvertGpuOpsToNVVMOps());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createGpuToLLVMConversionPass());
  pm.addPass(createReconcileUnrealizedCastsPass());

  GpuModuleToBinaryPassOptions binary;
  // "isa" stops at PTX text. "fatbin" and "cubin" would both invoke `ptxas`,
  // which is a toolkit the compile host is not required to have.
  binary.compilationTarget = "isa";
  pm.addPass(createGpuModuleToBinaryPass(binary));

  if (failed(pm.run(*device))) {
    error = "the NVPTX pipeline failed";
    return {};
  }

  // `gpu-module-to-binary` replaces the module with a `gpu.binary` holding one
  // object per target. One target was attached, so one object is expected; more
  // than one would mean the attach pass grew a second and the caller's
  // one-image-per-module assumption needs revisiting rather than papering over.
  std::string image;
  unsigned objects = 0;
  device->walk([&](gpu::BinaryOp bin) {
    for (Attribute attr : bin.getObjects()) {
      auto object = dyn_cast<gpu::ObjectAttr>(attr);
      if (!object)
        continue;
      ++objects;
      image = object.getObject().getValue().str();
    }
  });

  if (objects == 0) {
    error = "the pipeline ran but produced no device object";
    return {};
  }
  if (objects > 1) {
    error = "the pipeline produced " + std::to_string(objects) +
            " device objects for one target";
    return {};
  }
  // A NUL would truncate the payload entry this ends up in, and PTX is text, so
  // one cannot appear unless the serializer stopped producing assembly.
  if (image.find('\0') != std::string::npos) {
    error = "the device image is not text (it contains a NUL)";
    return {};
  }
  return image;
}

struct ConvertVxToLLVMPass
    : public PassWrapper<ConvertVxToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertVxToLLVMPass)

  llvm::StringRef getArgument() const override { return "vx-to-llvm"; }

  llvm::StringRef getDescription() const override {
    return "Lowers remaining Vx dialect operations to standard/LLVM dialects";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // The device twin does not come to the host party -- it leaves its image at
    // the door.
    //
    // `convert-vx-to-standard` gives every GPU kernel a `gpu.func` beside its
    // `vx.kernel`, for a device backend to compile. An unconsumed `gpu.module`
    // reaching this pass is not inert: there is no conversion for it, so the
    // whole module fails to lower and every program with a GPU placement stops
    // compiling. Which is what happened when the emission first landed: seven
    // backend tests, llama2 among them, on a change that was supposed to add
    // something unused.
    //
    // So it is compiled here and then dropped. Compiled here because this is
    // the last point at which it exists -- and dropped rather than never
    // emitted, so the kernel is still visible in the IR between the two passes
    // (`--pass-pipeline=builtin.module(convert-vx-to-standard)` shows it, which
    // is what scripts/flash_kernel_to_ptx.sh reads).
    SmallVector<gpu::GPUModuleOp> deviceModules;
    getOperation().walk(
        [&](gpu::GPUModuleOp m) { deviceModules.push_back(m); });

    llvm::StringMap<std::string> deviceImages;
    for (gpu::GPUModuleOp m : deviceModules) {
      std::string error;
      std::string image = deviceImageOf(m, error);
      // Fatal rather than "emit nothing and carry on". Carrying on produces a
      // program that compiles, dispatches, and is then refused at the far end
      // for a reason that has nothing to do with the refusal -- the kernel is
      // missing, and the message says the region is unroutable. The compile is
      // the place where the cause is still legible.
      if (image.empty()) {
        m.emitError("cannot compile this device kernel: ") << error;
        signalPassFailure();
        return;
      }
      // One image per module, and every kernel in it is an entry point: a
      // loader takes the module and then asks for a function by name. So each
      // `gpu.func` maps to the same image, and the kernel name -- already the
      // payload blob's first field -- is what selects the entry.
      m.walk([&](gpu::GPUFuncOp f) { deviceImages[f.getName()] = image; });
    }

    for (gpu::GPUModuleOp m : deviceModules)
      m.erase();

    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<vx::VxDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<cf::ControlFlowDialect>();
    target.addLegalDialect<async::AsyncDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    // A device transfer reaches this stage unlowered by design (the standard
    // stage defers it); marking it illegal is what makes the pattern run. Host
    // transfers were already lowered there, so none should be left.
    target.addIllegalOp<vx::TransferOp>();
    target.addIllegalOp<vx::LaunchOp>();
    target.addIllegalOp<vx::KernelOp>();
    target.addIllegalOp<vx::ReturnOp>();

    LLVMTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<LaunchOpLowering>(typeConverter, &getContext(), &deviceImages);
    patterns.add<TransferToPluginLowering>(typeConverter, &getContext());
    patterns.add<KernelOpLowering>(&getContext());
    patterns.add<ReturnOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

extern "C" {
void registerVxLoweringPass(MlirContext ctx) {
  // Exposed via registerVxPasses instead.
}

void addVxLoweringPass(MlirPassManager pm) {
  unwrap(pm)->addPass(std::make_unique<ConvertVxToStandardPass>());
}

void addVxToLLVMPass(MlirPassManager pm) {
  unwrap(pm)->addPass(std::make_unique<ConvertVxToLLVMPass>());
}
} // extern "C"

namespace mlir {
namespace vx {
void registerVxPasses() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<ConvertVxToStandardPass>();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<ConvertVxToLLVMPass>();
  });
}
} // namespace vx
} // namespace mlir
