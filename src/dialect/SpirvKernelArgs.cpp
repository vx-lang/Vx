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
// pointers stored in them are `float*` or `i32*`; SPIR-V's typed pointers
// reject the composite that puts one into the other. Measured on this machine
// with scripts/tools/spirv_module_check.sh, for every kernel that keeps a
// memref value -- so the device twin has to take the flat list instead.
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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/STLExtras.h"
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

/// Whether `op` is a construct this rewrite cannot put in arithmetic. The
/// caller names the op in the diagnostic, so a memref handed to a call reads
/// as `func.call` rather than as a rule about memrefs.
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
  auto gep = LLVM::GEPOp::create(
      builder, loc, view.base.getType(), view.elementType, view.base,
      ValueRange{linear}, LLVM::GEPNoWrapFlags::inbounds);
  LLVM::StoreOp::create(builder, loc, store.getValueToStore(), gep);
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
  auto gep = LLVM::GEPOp::create(
      builder, loc, view.base.getType(), view.elementType, view.base,
      ValueRange{linear}, LLVM::GEPNoWrapFlags::inbounds);
  auto ld = LLVM::LoadOp::create(builder, loc, view.elementType, gep);
  load.getResult().replaceAllUsesWith(ld);
  load.erase();
  return success();
}

/// Rewrite one `memref.reinterpret_cast` into the arithmetic its accesses will
/// use.
///
/// The cast's offset is added to the source's, and its sizes and strides
/// replace the source's. That is what the frontend means by these casts: a row
/// view names `index * row_size` relative to the memref it is made from. When
/// the source is already a view of its own (`strided_row`), the frontend reads
/// the source's offset, sizes and strides back out with
/// `memref.extract_strided_metadata`, folds the row index into the offset, and
/// makes the cast from the base buffer -- so the offset a launch passes for
/// that tensor still reaches the address, which is what the `offset` argument
/// in the ABI is for.
///
/// The op is left in place for the caller to erase: its result still has uses
/// until they have been rewritten, and erasing an op whose result is used
/// leaves those uses pointing at freed storage.
void lowerReinterpretCast(memref::ReinterpretCastOp cast, const FlatView &view,
                          DenseMap<Value, FlatView> &views) {
  OpBuilder builder(cast);
  Location loc = cast.getLoc();

  Value offset = view.offset;
  for (OpFoldResult part : cast.getMixedOffsets())
    offset = addI64(builder, loc, offset,
                    asI64(builder, loc,
                          getValueOrCreateConstantIndexOp(builder, loc, part)));
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

/// Rewrite one kernel, in three steps: expand its memref arguments in place,
/// rewrite the body's accesses into address arithmetic, then drop the memref
/// arguments the body no longer mentions.
LogicalResult flattenKernel(gpu::GPUFuncOp func, std::string &error) {
  MLIRContext *ctx = func.getContext();
  OpBuilder builder(ctx);
  Block &entry = func.getBody().front();
  DenseMap<Value, FlatView> views;

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
      // Only the cast that follows reads the base buffer; anything else would
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
      error = "a memref reaches `" + op->getName().getStringRef().str() +
              "`, which the SPIR-V argument flattening does not lower";
      return failure();
    }
  }
  // Consumers before producers, so nothing is left holding a freed result.
  for (Operation *op : llvm::reverse(deferred))
    op->erase();

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

} // namespace

LogicalResult flattenSpirvKernelArgs(gpu::GPUModuleOp gpuModule,
                                     std::string &error) {
  SmallVector<gpu::GPUFuncOp> kernels;
  gpuModule.walk([&](gpu::GPUFuncOp f) { kernels.push_back(f); });
  if (kernels.empty()) {
    error = "the device module holds no kernel";
    return failure();
  }
  for (gpu::GPUFuncOp kernel : kernels)
    if (failed(flattenKernel(kernel, error)))
      return failure();

  // Nothing left anywhere in the module may mention a memref, including in a
  // function the kernel calls and in a global a kernel reads: a memref that
  // reaches the gpu -> llvm-spv conversion outside a rewritten kernel is the
  // descriptor that SPIR-V rejects, and it would fail as an invalid module
  // rather than as this message.
  WalkResult left = gpuModule.walk([&](Operation *op) {
    if (isa<memref::GlobalOp>(op)) {
      error = "a `" + op->getName().getStringRef().str() +
              "` in the device module, which the SPIR-V argument flattening "
              "does not lower";
      return WalkResult::interrupt();
    }
    if (mentionsMemref(op)) {
      error = "a memref reaches `" + op->getName().getStringRef().str() +
              "` outside a flattened kernel";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return left.wasInterrupted() ? failure() : success();
}

} // namespace vx
} // namespace mlir
