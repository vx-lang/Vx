//===- SpirvKernelArgs.cpp - flatten a kernel's arguments for SPIR-V ------===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPIR-V refuses a memref value on the device side. A memref argument arrives
// at the gpu -> llvm-spv conversion as the five-field descriptor struct, and
// the backend rebuilds the descriptor's pointer fields as `i8` while the
// pointers stored in them are `float*` or `i32*`. SPIR-V's typed pointers
// reject the composite that puts one into the other. Measured on this machine
// with scripts/tools/spirv_module_check.sh, for every kernel that keeps a
// memref value -- so the device kernel has to take the flat list instead.
//
// The flat list is not new. It is exactly what the launch ABI already passes
// and what runtime/vx_kernel_launch.h counts: `allocated`, `aligned`, the
// offset, then a size and a stride per dimension, all `i64` except the two
// pointers. This file rewrites each memref argument into those values and each
// memref access into address arithmetic on them, so no descriptor is built at
// all. Hand-flattening the smallest kernel this way produces a module
// spirv-val accepts (516 bytes, against a rejection for the memref form).
//
//===----------------------------------------------------------------------===//

#include "SpirvKernelArgs.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

using namespace mlir;

namespace mlir {
namespace vx {
namespace {

/// The LLVM address space a kernel's tensors live in. LLVM's SPIR-V backend
/// reads it as SPIR-V's CrossWorkgroup storage class, which is where the
/// runtime's device allocations are, and it is the number
/// `convert-gpu-to-llvm-spv` gives an un-annotated memref.
constexpr unsigned kDeviceAddressSpace = 1;

/// What one memref value on the device side becomes: the pointer the elements
/// are addressed through, the linear element offset from that pointer, one
/// size per dimension, one stride per dimension, and the element type.
///
/// The sizes are kept even though nothing here addresses by a size: a body that
/// reads one back (`memref.dim`, which the flat emitter writes for a `?`-shaped
/// row) needs the value the ABI passed for it.
struct FlatView {
  Value base;
  Value offset;
  SmallVector<Value, 4> sizes;
  SmallVector<Value, 4> strides;
  Type elementType;
};

/// `v`, as an `i64`. An `index` index becomes an `llvm`-lowerable cast.
Value asI64(OpBuilder &builder, Location loc, Value v) {
  if (v.getType().isInteger(64))
    return v;
  return arith::IndexCastOp::create(builder, loc, builder.getI64Type(), v);
}

/// `v`, as an `index`, for the arithmetic a `memref.dim` or an
/// `extract_strided_metadata` result takes part in.
Value asIndex(OpBuilder &builder, Location loc, Value v) {
  if (v.getType().isIndex())
    return v;
  return arith::IndexCastOp::create(builder, loc, builder.getIndexType(), v);
}

Value constantI64(OpBuilder &builder, Location loc, int64_t v) {
  return LLVM::ConstantOp::create(builder, loc, builder.getI64Type(),
                                  builder.getI64IntegerAttr(v));
}

/// `a + b`, without an `add` for a value that adds nothing.
Value addI64(OpBuilder &builder, Location loc, Value a, Value b) {
  if (auto c = b.getDefiningOp<LLVM::ConstantOp>())
    if (auto attr = dyn_cast<IntegerAttr>(c.getValue());
        attr && attr.getInt() == 0)
      return a;
  return LLVM::AddOp::create(builder, loc, a, b);
}

/// `a * b`, without a `mul` for a stride of one or zero.
Value mulI64(OpBuilder &builder, Location loc, Value a, Value b) {
  if (auto c = b.getDefiningOp<LLVM::ConstantOp>())
    if (auto attr = dyn_cast<IntegerAttr>(c.getValue())) {
      if (attr.getInt() == 1)
        return a;
      if (attr.getInt() == 0)
        return constantI64(builder, loc, 0);
    }
  return LLVM::MulOp::create(builder, loc, a, b);
}

/// The static strides of a memref type, or nothing when its layout is not
/// "strides and offset" (an affine layout, say), which this rewrite cannot
/// express as arithmetic on the ABI's values.
FailureOr<SmallVector<int64_t, 4>> staticStrides(MemRefType type) {
  SmallVector<int64_t, 4> strides;
  int64_t offset = 0;
  if (failed(type.getStridesAndOffset(strides, offset)))
    return failure();
  return strides;
}

/// The types one memref argument expands into, in the order the launch ABI
/// passes them: allocated, aligned, offset, one size per dimension, one stride
/// per dimension. `vx_launch_param_width` counts the same 3 + 2 * rank.
void flatArgTypes(MLIRContext *ctx, MemRefType type,
                  SmallVectorImpl<Type> &out) {
  auto pointer = LLVM::LLVMPointerType::get(ctx, kDeviceAddressSpace);
  auto i64 = IntegerType::get(ctx, 64);
  out.push_back(pointer); // allocated
  out.push_back(pointer); // aligned
  out.push_back(i64);     // offset
  for (int64_t d = 0; d < type.getRank(); ++d)
    out.push_back(i64); // a size per dimension
  for (int64_t d = 0; d < type.getRank(); ++d)
    out.push_back(i64); // a stride per dimension
}

/// Whether `op` takes, returns or holds a memref value, which the flat
/// argument list has no form for.
bool mentionsMemref(Operation *op) {
  auto isMemref = [](Type t) { return isa<BaseMemRefType>(t); };
  if (llvm::any_of(op->getOperandTypes(), isMemref) ||
      llvm::any_of(op->getResultTypes(), isMemref))
    return true;
  for (Region &region : op->getRegions())
    for (Block &block : region)
      if (llvm::any_of(block.getArgumentTypes(), isMemref))
        return true;
  return false;
}

/// Why a tensor on `op` stops the image, in words that say what the program
/// does. A call is what a Vx program has: it passes its tensor to a helper, and
/// the image has no way to pass one on.
std::string memrefRefusal(Operation *op) {
  if (auto call = dyn_cast<CallOpInterface>(op))
    if (auto callee = dyn_cast<SymbolRefAttr>(call.getCallableForCallee()))
      return "it passes a tensor to `" + callee.getLeafReference().str() +
             "`, and a SPIR-V image cannot pass a tensor to a call yet";
  return "it hands a tensor to `" + op->getName().getStringRef().str() +
         "`, and a SPIR-V image cannot do that yet";
}

/// Whether `op` mentions a memref whose elements are themselves memrefs -- a
/// buffer that holds a descriptor, which is the shape the AST code generator
/// gives a kernel's result slot. The flat argument list is pointers and scalars
/// and has no form for one, and a descriptor read out of device memory would
/// have to be rebuilt field by field.
bool mentionsMemrefCell(Operation *op) {
  auto isCell = [](Type type) {
    auto memref = dyn_cast<MemRefType>(type);
    return memref && isa<BaseMemRefType>(memref.getElementType());
  };
  return llvm::any_of(op->getOperandTypes(), isCell) ||
         llvm::any_of(op->getResultTypes(), isCell);
}

/// Whether `op` is a branch that hands an `index` value to its destination
/// block, which is how the AST code generator writes a loop counter.
///
/// The gpu -> llvm-spv conversion turns `index` into a 64-bit integer in the
/// operations it rewrites, but not in a branch's block arguments, and refuses
/// the module with "type mismatch for bb argument #0 of successor #0". The flat
/// code generator counts its loops in `i32` and is not affected.
bool carriesIndexBlockArgument(Operation *op) {
  auto branch = dyn_cast<BranchOpInterface>(op);
  if (!branch)
    return false;
  for (Block *successor : branch->getSuccessors())
    for (BlockArgument arg : successor->getArguments())
      if (isa<IndexType>(arg.getType()))
        return true;
  return false;
}

/// The address space of a tile in shared memory, which `Memory::SMEM` becomes.
/// LLVM's SPIR-V backend reads it as Workgroup storage: one copy for each
/// block.
constexpr unsigned kSharedAddressSpace = 3;

/// Whether `type` is a tile in shared memory.
bool isShared(MemRefType type) {
  auto space = dyn_cast_or_null<IntegerAttr>(type.getMemorySpace());
  return space && space.getInt() == kSharedAddressSpace;
}

/// The linear element offset of an access to `view` at `indices`, as an
/// `i64`: the view's offset plus `index * stride` per dimension.
Value linearIndex(OpBuilder &builder, Location loc, const FlatView &view,
                  ValueRange indices) {
  Value linear = view.offset;
  for (auto [index, stride] : llvm::zip(indices, view.strides))
    linear = addI64(builder, loc, linear,
                    mulI64(builder, loc, asI64(builder, loc, index), stride));
  return linear;
}

/// The type an element is stored as. LLVM has no type for an 8-bit or 4-bit
/// float, so such an element is stored as the integer of its width, as the
/// host stores it.
Type storageType(Type element) {
  auto real = dyn_cast<FloatType>(element);
  if (!real || LLVM::isCompatibleType(real))
    return element;
  return IntegerType::get(element.getContext(), real.getWidth());
}

/// `memref.store` as a store through a computed address.
LogicalResult lowerStore(memref::StoreOp store, const FlatView &view,
                         std::string &error) {
  if (store.getValueToStore().getType() != view.elementType) {
    error = "a store of a value that is not the memref's element type";
    return failure();
  }
  if (store.getIndices().size() != view.strides.size()) {
    error = "an access whose number of indices is not the memref's rank";
    return failure();
  }
  OpBuilder builder(store);
  Location loc = store.getLoc();
  Value linear = linearIndex(builder, loc, view, store.getIndices());
  Type storage = storageType(view.elementType);
  auto gep =
      LLVM::GEPOp::create(builder, loc, view.base.getType(), storage, view.base,
                          ValueRange{linear}, LLVM::GEPNoWrapFlags::inbounds);
  Value value = store.getValueToStore();
  if (storage != view.elementType)
    value = arith::BitcastOp::create(builder, loc, storage, value);
  LLVM::StoreOp::create(builder, loc, value, gep);
  store.erase();
  return success();
}

/// `memref.load` as a load through a computed address.
LogicalResult lowerLoad(memref::LoadOp load, const FlatView &view,
                        std::string &error) {
  if (load.getResult().getType() != view.elementType) {
    error = "a load whose type is not the memref's element type";
    return failure();
  }
  if (load.getIndices().size() != view.strides.size()) {
    error = "an access whose number of indices is not the memref's rank";
    return failure();
  }
  OpBuilder builder(load);
  Location loc = load.getLoc();
  Value linear = linearIndex(builder, loc, view, load.getIndices());
  Type storage = storageType(view.elementType);
  auto gep =
      LLVM::GEPOp::create(builder, loc, view.base.getType(), storage, view.base,
                          ValueRange{linear}, LLVM::GEPNoWrapFlags::inbounds);
  Value value = LLVM::LoadOp::create(builder, loc, storage, gep);
  if (storage != view.elementType)
    value = arith::BitcastOp::create(builder, loc, view.elementType, value);
  load.getResult().replaceAllUsesWith(value);
  load.erase();
  return success();
}

/// Rewrite one `memref.reinterpret_cast` into the arithmetic its accesses will
/// use.
///
/// The cast's offset replaces the source's, and its sizes and strides replace
/// the source's. That is what the op means: `MemRefOps.td` says the result
/// descriptor is the source's base with `%dst.offset = %offset`, so "it does
/// not matter if you use %result1 or %arg0 as a source" -- a cast never builds
/// on the offset of the view it is made from (`memref.subview` does, and the
/// two differ there for that reason).
///
/// The frontend keeps to that: it writes an offset relative to the base buffer.
/// A row of a view is built from `memref.extract_strided_metadata`'s base
/// buffer, with the source's offset folded into the cast's, which is why the
/// base-buffer view below starts at offset 0.
///
/// The op is left in place for the caller to erase: its result still has uses
/// until they have been rewritten, and erasing an op whose result is used
/// leaves those uses pointing at freed storage.
void lowerReinterpretCast(memref::ReinterpretCastOp cast, const FlatView &view,
                          DenseMap<Value, FlatView> &views) {
  OpBuilder builder(cast);
  Location loc = cast.getLoc();

  Value offset;
  for (OpFoldResult part : cast.getMixedOffsets()) {
    Value partValue = asI64(
        builder, loc, getValueOrCreateConstantIndexOp(builder, loc, part));
    offset = offset ? addI64(builder, loc, offset, partValue) : partValue;
  }
  if (!offset)
    offset = constantI64(builder, loc, 0);
  SmallVector<Value, 4> sizes;
  for (OpFoldResult part : cast.getMixedSizes())
    sizes.push_back(asI64(builder, loc,
                          getValueOrCreateConstantIndexOp(builder, loc, part)));
  SmallVector<Value, 4> strides;
  for (OpFoldResult part : cast.getMixedStrides())
    strides.push_back(asI64(
        builder, loc, getValueOrCreateConstantIndexOp(builder, loc, part)));
  views[cast.getResult()] = FlatView{view.base, offset, sizes, strides,
                                     cast.getType().getElementType()};
}

/// The address space of the tensor whose address `v` holds, through casts and
/// sums, or nothing when `v` does not hold one.
std::optional<unsigned>
addressSpaceOf(Value v, const DenseMap<Value, unsigned> &spaces) {
  if (auto it = spaces.find(v); it != spaces.end())
    return it->second;
  Operation *def = v.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtSIOp,
          arith::ExtUIOp>(def))
    return addressSpaceOf(def->getOperand(0), spaces);
  if (isa<arith::AddIOp, LLVM::AddOp>(def)) {
    std::optional<unsigned> a = addressSpaceOf(def->getOperand(0), spaces);
    std::optional<unsigned> b = addressSpaceOf(def->getOperand(1), spaces);
    if (a && b)
      return std::nullopt;
    return a ? a : b;
  }
  return std::nullopt;
}

/// Give `pointer`, a plain `!llvm.ptr`, the address space `space`, and each
/// pointer that a `getelementptr` makes from it too. An atomic, a load or a
/// store through it takes any address space. Another use fails.
LogicalResult giveAddressSpace(Value pointer, unsigned space,
                               std::string &error) {
  pointer.setType(LLVM::LLVMPointerType::get(pointer.getContext(), space));
  for (OpOperand &use : pointer.getUses()) {
    Operation *user = use.getOwner();
    if (auto gep = dyn_cast<LLVM::GEPOp>(user)) {
      if (use.getOperandNumber() != 0 ||
          failed(giveAddressSpace(gep.getResult(), space, error)))
        return failure();
      continue;
    }
    // SPIR-V has an instruction for these atomics, and LLVM's SPIR-V backend
    // stops the process on another one, such as `uinc_wrap`. A float atomic
    // needs an extension of SPIR-V.
    if (auto rmw = dyn_cast<LLVM::AtomicRMWOp>(user)) {
      using K = LLVM::AtomicBinOp;
      K kind = rmw.getBinOp();
      if (!llvm::is_contained({K::xchg, K::add, K::sub, K::_and, K::_or,
                               K::_xor, K::max, K::min, K::umax, K::umin},
                              kind)) {
        error = "it uses the atomic `" +
                LLVM::stringifyAtomicBinOp(kind).str() +
                "`, which a SPIR-V image cannot hold yet";
        return failure();
      }
    }
    bool access =
        (isa<LLVM::AtomicRMWOp, LLVM::AtomicCmpXchgOp, LLVM::LoadOp>(user) &&
         use.getOperandNumber() == 0) ||
        (isa<LLVM::StoreOp>(user) && use.getOperandNumber() == 1);
    if (!access) {
      error = "it hands a pointer to `" + user->getName().getStringRef().str() +
              "`, and a SPIR-V image cannot do that yet";
      return failure();
    }
  }
  return success();
}

/// Rewrite one kernel, in three steps: expand its memref arguments in place,
/// rewrite the body's accesses into address arithmetic, then drop the memref
/// arguments the body no longer mentions.
LogicalResult flattenKernel(gpu::GPUFuncOp func, std::string &error) {
  MLIRContext *ctx = func.getContext();
  OpBuilder builder(ctx);
  Block &entry = func.getBody().front();
  DenseMap<Value, FlatView> views;
  // The address space of each integer that holds the address of a tensor.
  DenseMap<Value, unsigned> spaces;

  // The signature first. Each memref argument is expanded where it stands, so
  // a scalar argument keeps its position and the flat list stays in the order
  // the runtime's own `vx_launch_build_params` builds for the same arguments.
  // Walking from the back keeps the indices of the arguments not yet visited.
  builder.setInsertionPointToStart(&entry);
  for (unsigned i = entry.getNumArguments(); i-- > 0;) {
    BlockArgument arg = entry.getArgument(i);
    auto memref = dyn_cast<MemRefType>(arg.getType());
    if (!memref)
      continue;
    if (memref.getMemorySpace()) {
      error = "a memref argument whose memory space is not the device's";
      return failure();
    }
    auto strides = staticStrides(memref);
    if (failed(strides)) {
      error = "a memref argument whose layout is not strides and offset";
      return failure();
    }

    SmallVector<Type, 8> types;
    flatArgTypes(ctx, memref, types);
    for (unsigned k = 0; k < types.size(); ++k)
      entry.insertArgument(i + k, types[k], arg.getLoc());

    // The offset is the runtime's, always: it describes the buffer this launch
    // was handed, which no type can know. A size or a stride comes from the
    // type when it has one and from the argument when it says `?`.
    SmallVector<Value, 4> sizeValues;
    for (int64_t d = 0; d < memref.getRank(); ++d) {
      if (memref.isDynamicDim(d))
        sizeValues.push_back(entry.getArgument(i + 3 + d));
      else
        sizeValues.push_back(
            constantI64(builder, func.getLoc(), memref.getShape()[d]));
    }
    SmallVector<Value, 4> strideValues;
    for (int64_t d = 0; d < memref.getRank(); ++d) {
      int64_t stride = (*strides)[d];
      if (stride == ShapedType::kDynamic)
        strideValues.push_back(entry.getArgument(i + 3 + memref.getRank() + d));
      else
        strideValues.push_back(constantI64(builder, func.getLoc(), stride));
    }
    views.try_emplace(arg, FlatView{entry.getArgument(i + 1),
                                    entry.getArgument(i + 2), sizeValues,
                                    strideValues, memref.getElementType()});
  }

  // The body. The ops are collected first, in order, so that a value is in the
  // map by the time something uses it, and so that creating the replacements
  // does not disturb the walk. `walk` visits the function itself last -- it is
  // not part of its own body, and the memref arguments it still carries at this
  // point are the ones this function is about to replace.
  SmallVector<Operation *> ops;
  func.walk([&](Operation *op) {
    if (op != func.getOperation())
      ops.push_back(op);
  });
  // The casts and the metadata reads are erased last: their results are what
  // the accesses after them were written against, so erasing one before they
  // are rewritten would leave them pointing at freed storage. They go in
  // reverse order, so a view built on another one (`strided_row` of a
  // `strided_row`) loses its consumer before it is erased.
  SmallVector<Operation *> deferred;
  for (Operation *op : ops) {
    auto found = [&](Value memref) -> const FlatView * {
      auto it = views.find(memref);
      return it == views.end() ? nullptr : &it->second;
    };
    if (auto cast = dyn_cast<memref::ReinterpretCastOp>(op)) {
      const FlatView *view = found(cast.getSource());
      if (!view) {
        error = "a view of a memref the flattening does not know";
        return failure();
      }
      lowerReinterpretCast(cast, *view, views);
      deferred.push_back(cast);
      continue;
    }
    // A tile in shared memory: the address of its global, which
    // `flattenSpirvKernelArgs` made an LLVM global in Workgroup storage.
    if (auto get = dyn_cast<memref::GetGlobalOp>(op);
        get && isShared(get.getType())) {
      MemRefType type = get.getType();
      auto strides = staticStrides(type);
      if (!type.hasStaticShape() || failed(strides)) {
        error = "a tile in shared memory whose shape is not known";
        return failure();
      }
      OpBuilder builder(get);
      Location loc = get.getLoc();
      Value base = LLVM::AddressOfOp::create(
          builder, loc, LLVM::LLVMPointerType::get(ctx, kSharedAddressSpace),
          get.getName());
      SmallVector<Value, 4> sizes;
      for (int64_t size : type.getShape())
        sizes.push_back(constantI64(builder, loc, size));
      SmallVector<Value, 4> strideValues;
      for (int64_t stride : *strides)
        strideValues.push_back(constantI64(builder, loc, stride));
      views[get.getResult()] =
          FlatView{base, constantI64(builder, loc, 0), sizes, strideValues,
                   type.getElementType()};
      deferred.push_back(get);
      continue;
    }
    // The address of a tensor as an integer, which an atomic turns back into a
    // pointer to an element. It is the integer of the view's pointer.
    if (auto aligned = dyn_cast<memref::ExtractAlignedPointerAsIndexOp>(op)) {
      const FlatView *view = found(aligned.getSource());
      if (!view) {
        error = "the address of a memref the flattening does not know";
        return failure();
      }
      OpBuilder builder(aligned);
      Location loc = aligned.getLoc();
      Value address = LLVM::PtrToIntOp::create(
          builder, loc, builder.getI64Type(), view->base);
      Value index = asIndex(builder, loc, address);
      spaces[index] =
          cast<LLVM::LLVMPointerType>(view->base.getType()).getAddressSpace();
      aligned.getResult().replaceAllUsesWith(index);
      aligned.erase();
      continue;
    }
    // An op that only an NVIDIA GPU has, which an `mlir!` block can hold: an
    // `nvvm` op, or a call of an NVVM intrinsic. LLVM's SPIR-V backend stops
    // the process on it.
    std::string nvidia;
    if (op->getDialect() && op->getDialect()->getNamespace() == "nvvm")
      nvidia = op->getName().getStringRef().str();
    if (auto call = dyn_cast<LLVM::CallIntrinsicOp>(op);
        call && call.getIntrin().starts_with("llvm.nvvm."))
      nvidia = call.getIntrin().str();
    if (!nvidia.empty()) {
      error = "it uses `" + nvidia + "`, which only an NVIDIA GPU has";
      return failure();
    }
    // A shuffle, which a function the kernel calls can hold. It needs
    // subgroups of 32 lanes.
    if (isa<gpu::ShuffleOp>(op)) {
      error =
          "it shuffles values between the lanes of a subgroup "
          "(`gpu.shuffle`), which needs subgroups of 32 lanes, and a SPIR-V "
          "kernel cannot ask for them yet";
      return failure();
    }
    // A cast changes only what the type says about the sizes and strides, or
    // about the memory, as a call site does to give a function the type it
    // takes. The values are the source's, so the result has the source's
    // view. A tile in shared memory keeps its Workgroup pointer.
    if (isa<memref::CastOp, memref::MemorySpaceCastOp>(op)) {
      const FlatView *view = found(op->getOperand(0));
      if (!view) {
        error = "a cast of a memref the flattening does not know";
        return failure();
      }
      FlatView same = *view;
      same.elementType =
          cast<MemRefType>(op->getResult(0).getType()).getElementType();
      views[op->getResult(0)] = same;
      deferred.push_back(op);
      continue;
    }
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      const FlatView *view = found(load.getMemref());
      if (!view) {
        error = "a load from a memref the flattening does not know";
        return failure();
      }
      if (failed(lowerLoad(load, *view, error)))
        return failure();
      continue;
    }
    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      const FlatView *view = found(store.getMemref());
      if (!view) {
        error = "a store into a memref the flattening does not know";
        return failure();
      }
      if (failed(lowerStore(store, *view, error)))
        return failure();
      continue;
    }
    // A dimension read back by name (`memref.dim`, which the flat emitter
    // writes for a `?`-shaped row): the size the ABI passed for it, or the
    // size the view was made with.
    if (auto dim = dyn_cast<memref::DimOp>(op)) {
      const FlatView *view = found(dim.getSource());
      if (!view) {
        error = "a `memref.dim` of a memref the flattening does not know";
        return failure();
      }
      std::optional<int64_t> which = dim.getConstantIndex();
      if (!which || *which < 0 ||
          static_cast<size_t>(*which) >= view->sizes.size()) {
        error = "a `memref.dim` whose dimension is not one of the memref's";
        return failure();
      }
      OpBuilder builder(dim);
      dim.getResult().replaceAllUsesWith(
          asIndex(builder, dim.getLoc(), view->sizes[*which]));
      dim.erase();
      continue;
    }
    // A view's offset, sizes and strides read back by name. The frontend
    // writes one to build a row of a row (`strided_row`): the base buffer
    // result is the same buffer as a zero-offset view, and the three value
    // results are the source's own parts, so the cast that follows lands on
    // the address it would have.
    if (auto md = dyn_cast<memref::ExtractStridedMetadataOp>(op)) {
      const FlatView *found_view = found(md.getSource());
      if (!found_view) {
        error = "an `extract_strided_metadata` of a memref the flattening "
                "does not know";
        return failure();
      }
      // Only the cast that follows reads the base buffer. Anything else would
      // be left pointing at a memref this rewrite has no value for.
      for (Operation *user : md.getBaseBuffer().getUsers())
        if (!isa<memref::ReinterpretCastOp>(user)) {
          error = "an `extract_strided_metadata` whose base buffer reaches `" +
                  user->getName().getStringRef().str() +
                  "`, which the SPIR-V argument flattening does not lower";
          return failure();
        }
      // Copied, because inserting the base buffer's view below may rehash the
      // map and move the entry `found_view` points into.
      FlatView source = *found_view;
      OpBuilder builder(md);
      Location loc = md.getLoc();
      views.try_emplace(md.getBaseBuffer(),
                        FlatView{source.base, constantI64(builder, loc, 0),
                                 source.sizes, source.strides,
                                 source.elementType});
      md.getOffset().replaceAllUsesWith(asIndex(builder, loc, source.offset));
      for (auto [result, size] : llvm::zip(md.getSizes(), source.sizes))
        result.replaceAllUsesWith(asIndex(builder, loc, size));
      for (auto [result, stride] : llvm::zip(md.getStrides(), source.strides))
        result.replaceAllUsesWith(asIndex(builder, loc, stride));
      deferred.push_back(md);
      continue;
    }
    if (mentionsMemref(op)) {
      error = memrefRefusal(op);
      return failure();
    }
  }
  // Consumers before producers, so nothing is left holding a freed result.
  for (Operation *op : llvm::reverse(deferred))
    op->erase();

  // A pointer that `llvm.inttoptr` makes from the address of a tensor gets the
  // address space of that tensor, and so does each pointer an access makes
  // from it. A plain `!llvm.ptr` is private memory in SPIR-V.
  SmallVector<LLVM::IntToPtrOp> pointers;
  func.walk([&](LLVM::IntToPtrOp p) { pointers.push_back(p); });
  for (LLVM::IntToPtrOp p : pointers) {
    std::optional<unsigned> space = addressSpaceOf(p.getArg(), spaces);
    if (!space) {
      error = "it makes a pointer from an integer that is not the address of "
              "a tensor";
      return failure();
    }
    if (failed(giveAddressSpace(p.getResult(), *space, error)))
      return failure();
  }

  // The memref arguments are unused now. Dropping them leaves the flat list in
  // the order built above, and the function type follows the block.
  entry.eraseArguments(
      [](BlockArgument a) { return isa<MemRefType>(a.getType()); });
  FunctionType type = func.getFunctionType();
  function_interface_impl::setFunctionType(
      func,
      FunctionType::get(ctx, entry.getArgumentTypes(), type.getResults()));
  return success();
}

/// An inliner that also takes a call in a `gpu.func`, which has no inliner
/// interface of its own.
struct KernelInliner : public InlinerInterface {
  KernelInliner(MLIRContext *context) : InlinerInterface(context) {}
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &mapping) const override {
    return isa<gpu::GPUFuncOp>(dest->getParentOp()) ||
           InlinerInterface::isLegalToInline(dest, src, wouldBeCloned, mapping);
  }
};

} // namespace

LogicalResult inlineSpirvKernelCalls(gpu::GPUModuleOp gpuModule,
                                     std::string &error) {
  MLIRContext *context = gpuModule.getContext();
  DialectRegistry registry;
  func::registerInlinerExtension(registry);
  context->appendDialectRegistry(registry);
  // A function can hold an `affine.for`, which nothing after this lowers. Then
  // a call from one function to another. The calls in a kernel come next.
  {
    PassManager pm(context, gpu::GPUModuleOp::getOperationName());
    pm.addPass(createLowerAffinePass());
    pm.addPass(createInlinerPass());
    if (failed(pm.run(gpuModule))) {
      error = "inlining the functions that a kernel calls failed";
      return failure();
    }
  }
  SmallVector<func::CallOp> calls;
  for (gpu::GPUFuncOp f : gpuModule.getOps<gpu::GPUFuncOp>())
    f.walk([&](func::CallOp call) { calls.push_back(call); });
  SymbolTable symbols(gpuModule);
  KernelInliner inliner(context);
  InlinerConfig config;
  for (func::CallOp call : calls) {
    auto callee = symbols.lookup<func::FuncOp>(call.getCallee());
    if (!callee || callee.isExternal())
      continue;
    Region &body = callee.getBody();
    if (!body.hasOneBlock() &&
        call->getParentOp()->mightHaveTrait<OpTrait::SingleBlock>())
      continue;
    if (succeeded(inlineCall(inliner, config.getCloneCallback(), call, callee,
                             &body)))
      call.erase();
  }
  for (auto f : llvm::make_early_inc_range(gpuModule.getOps<func::FuncOp>()))
    if (!f.isExternal() && SymbolTable::symbolKnownUseEmpty(f, gpuModule))
      f.erase();
  return success();
}

void widenIndexBlockArguments(gpu::GPUModuleOp gpuModule) {
  gpuModule.walk([](gpu::GPUFuncOp func) {
    Type i64 = IntegerType::get(func.getContext(), 64);
    for (Block &block : func.getBody()) {
      if (block.isEntryBlock())
        continue;
      for (BlockArgument arg : block.getArguments()) {
        if (!isa<IndexType>(arg.getType()))
          continue;
        llvm::SetVector<Block *> preds(block.pred_begin(), block.pred_end());
        for (Block *pred : preds) {
          auto branch = cast<BranchOpInterface>(pred->getTerminator());
          for (unsigned s = 0; s < branch->getNumSuccessors(); ++s) {
            if (branch->getSuccessor(s) != &block)
              continue;
            SuccessorOperands operands = branch.getSuccessorOperands(s);
            OpOperand &operand =
                operands.getMutableForwardedOperands()
                    [arg.getArgNumber() - operands.getProducedOperandCount()];
            OpBuilder builder(branch);
            operand.set(arith::IndexCastOp::create(builder, branch.getLoc(),
                                                   i64, operand.get()));
          }
        }
        arg.setType(i64);
        OpBuilder builder(&block, block.begin());
        auto back = arith::IndexCastOp::create(builder, arg.getLoc(),
                                               builder.getIndexType(), arg);
        arg.replaceAllUsesExcept(back, back);
      }
    }
  });
}

std::optional<std::string> spirvRefusalOf(Operation *op) {
  // A memref of memrefs, which the AST code generator writes for a kernel's
  // result slot: the flat argument list is pointers and scalars.
  if (mentionsMemrefCell(op))
    return "its body holds a memref of memrefs (`memref<memref<...>>`), which "
           "the flat argument list has no form for";
  // A loop counter handed to the next block as an `index` block argument.
  if (carriesIndexBlockArgument(op))
    return "a branch in its body carries an `index` value between blocks, "
           "which the SPIR-V conversion cannot take";
  // A lane routine, such as `lanes::sum`. Most of them use shuffles across a
  // subgroup of 32 lanes, which a SPIR-V kernel cannot ask for yet.
  if (op->getName().getStringRef().starts_with("vx.lane") ||
      op->getName().getStringRef().starts_with("vx.subgroup"))
    return "it calls a routine of `lanes::`, which a SPIR-V image cannot run "
           "yet. Most of them need subgroups of 32 lanes";
  // A tile in shared memory becomes a `memref.global`, which the flattening
  // gives Workgroup storage. A tile with a dynamic shape gets no global, and
  // `runsOnDevice` refuses it for every pipeline.
  if (auto alloca = dyn_cast<memref::AllocaOp>(op)) {
    auto type = dyn_cast<MemRefType>(alloca.getType());
    if (type && isShared(type))
      return std::nullopt;
    // A slot `mem2reg` promotes is a loop counter, and the flattening never
    // sees it. A buffer the region made is a memref value nothing built, which
    // the flattening would refuse.
    if (type && type.getRank() != 0)
      return "it keeps a scratch buffer of its own (`memref.alloca`), and the "
             "device image addresses only the kernel's arguments";
  }
  if (isa<memref::AllocOp>(op))
    return "it allocates a buffer on the device (`memref.alloc`), and the "
           "device image addresses only the kernel's arguments";
  return std::nullopt;
}

LogicalResult
flattenSpirvKernelArgs(gpu::GPUModuleOp gpuModule, std::string &error,
                       SmallVectorImpl<SpirvKernelRefusal> &refused) {
  SmallVector<gpu::GPUFuncOp> kernels;
  gpuModule.walk([&](gpu::GPUFuncOp f) { kernels.push_back(f); });
  if (kernels.empty()) {
    error = "the device module holds no kernel";
    return failure();
  }
  // Each tile in shared memory becomes an LLVM global of the same name in
  // Workgroup storage, with no initial value, as OpenCL's `__local` arrays are.
  // The kernels below take its address.
  for (auto global :
       llvm::make_early_inc_range(gpuModule.getOps<memref::GlobalOp>())) {
    MemRefType type = global.getType();
    if (!isShared(type) || !type.hasStaticShape())
      continue;
    auto array = LLVM::LLVMArrayType::get(storageType(type.getElementType()),
                                          type.getNumElements());
    std::string name = global.getSymName().str();
    uint64_t alignment = global.getAlignment().value_or(16);
    Location loc = global.getLoc();
    global.erase();
    OpBuilder builder(gpuModule.getBody(), gpuModule.getBody()->begin());
    LLVM::GlobalOp::create(builder, loc, array, /*isConstant=*/false,
                           LLVM::Linkage::Internal, name, Attribute(),
                           alignment, kSharedAddressSpace);
  }
  // Each kernel is rewritten on its own, so one that fails does not stop the
  // next. Every region that fails then gets its own reason.
  for (gpu::GPUFuncOp kernel : kernels) {
    std::string reason;
    if (failed(flattenKernel(kernel, reason)))
      refused.push_back({kernel.getName().str(), reason});
  }
  // A kernel that failed can be half rewritten, and a helper it calls still
  // takes a tensor. So the check below runs only when every kernel passed.
  if (!refused.empty())
    return failure();

  // Nothing left anywhere in the module may mention a memref, including in a
  // function the kernel calls and in a global a kernel reads: a memref that
  // reaches the gpu -> llvm-spv conversion outside a rewritten kernel is the
  // descriptor that SPIR-V rejects, and it would fail as an invalid module
  // rather than as this message.
  WalkResult left = gpuModule.walk([&](Operation *op) {
    if (auto global = dyn_cast<memref::GlobalOp>(op)) {
      /* A call in the kernel's body put this here, and a libm call is the one
         that does: `f32::exp`, `tanh`, `sin` and `cos` read a lookup table from
         the stdlib, and the table becomes a `memref.global`. The symbol names
         the table, so the message names the function without this code knowing
         which one it was. */
      error = "it reads `" + global.getSymName().str() +
              "` from the device module, and the SPIR-V argument flattening "
              "lowers only the kernel's arguments, not a `memref.global`. A "
              "call to a libm function such as `f32::exp`, `tanh`, `sin` or "
              "`cos` puts its constant table there";
      return WalkResult::interrupt();
    }
    if (mentionsMemref(op)) {
      if (auto kernel = op->getParentOfType<gpu::GPUFuncOp>())
        refused.push_back({kernel.getName().str(), memrefRefusal(op)});
      else
        error = memrefRefusal(op);
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return left.wasInterrupted() ? failure() : success();
}

namespace {
/// The flattening as a pass, so `vx-opt` can show the arithmetic it builds.
///
/// The image is the only other place it appears, and reading an address out of
/// a SPIR-V module is not a test anyone can write. Here the rewrite's output is
/// MLIR with the addresses written out.
struct FlattenSpirvKernelArgsPass
    : public PassWrapper<FlattenSpirvKernelArgsPass,
                         OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FlattenSpirvKernelArgsPass)

  StringRef getArgument() const override {
    return "vx-flatten-spirv-kernel-args";
  }

  StringRef getDescription() const override {
    return "Rewrite a kernel's memref arguments into the launch ABI's flat "
           "argument list";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    std::string error;
    SmallVector<SpirvKernelRefusal> refused;
    if (succeeded(flattenSpirvKernelArgs(getOperation(), error, refused)))
      return;
    if (refused.empty())
      getOperation().emitError() << error;
    for (const SpirvKernelRefusal &r : refused)
      getOperation().lookupSymbol(r.kernel)->emitError() << r.reason;
    signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> createFlattenSpirvKernelArgsPass() {
  return std::make_unique<FlattenSpirvKernelArgsPass>();
}

} // namespace vx
} // namespace mlir
