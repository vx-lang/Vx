#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include "VxDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

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

/// The `vx.transfer` that put `v` where it is, if one did.
///
/// Walks back through single-operand ops, because a capture usually reaches a
/// spawn having been cast or reshaped since the transfer produced it. Bounded:
/// a chain longer than this is one this cannot reason about anyway, and
/// answering "no transfer" for it is the safe direction -- it declines to
/// reject rather than rejecting something it has not understood.
static vx::TransferOp definingTransfer(Value v) {
  for (int hops = 0; v && hops < 8; ++hops) {
    Operation *def = v.getDefiningOp();
    if (!def)
      return nullptr;
    if (auto xfer = dyn_cast<vx::TransferOp>(def))
      return xfer;
    if (def->getNumOperands() != 1)
      return nullptr;
    v = def->getOperand(0);
  }
  return nullptr;
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

    // A transfer into a device space is not a host allocation and a copy. It
    // is the plugin's allocate-and-transfer, so leave the op alone here and let
    // the LLVM stage emit that call -- getting a raw pointer out of a memref is
    // not something this stage can express.
    if (isDeviceTransfer(op))
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
    }
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
    if (!isDeviceTransfer(op))
      return failure();

    Location loc = op.getLoc();
    Value src = op.getOperand();
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
    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, TypeRange{llvmPtrType},
        SymbolRefAttr::get(rewriter.getContext(), allocName),
        ValueRange{bytes, alignedPtr, topoVal});
    Value devicePtr = callOp.getResult();

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
    {
      OpBuilder::InsertionGuard guard(rewriter);
      Block *block = op->getBlock();
      if (!block->empty() && block->back().hasTrait<OpTrait::IsTerminator>())
        rewriter.setInsertionPoint(&block->back());
      else
        rewriter.setInsertionPointToEnd(block);
      rewriter.create<LLVM::CallOp>(
          loc, TypeRange{}, SymbolRefAttr::get(rewriter.getContext(), freeName),
          ValueRange{devicePtr, topoVal});
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LaunchOpLowering : public OpRewritePattern<vx::LaunchOp> {
  const LLVMTypeConverter &typeConverter;

  LaunchOpLowering(const LLVMTypeConverter &typeConverter, MLIRContext *context)
      : OpRewritePattern<vx::LaunchOp>(context), typeConverter(typeConverter) {}

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
    patterns.add<LaunchOpLowering>(typeConverter, &getContext());
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
