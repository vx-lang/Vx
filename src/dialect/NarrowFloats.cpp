#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
// Conversions to and from the 8-bit and 4-bit float types, written as integer
// operations. LLVM has no type for these, so an `arith.extf` or `arith.truncf`
// on one cannot lower as it is. Upstream `arith-expand` covers f4E2M1FN, but
// it rounds a tie away from zero. Here every type rounds a tie to even, and a
// value too large for the type gives its largest finite value.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"

#include <limits>
#include <optional>

using namespace mlir;

namespace {

// The layout of a binary float format.
struct Format {
  unsigned width, expBits, mantBits;
  int64_t bias;
  // Encodings without the sign: the largest finite value, and the largest
  // value that is not NaN (infinity in a format that has it).
  uint64_t maxFinite, maxNotNan;
  // The NaN that a conversion gives, without the sign. A format with no NaN
  // gives its largest positive value for a NaN input, as upstream does.
  std::optional<uint64_t> nan;
};

std::optional<Format> narrowFormat(Type t) {
  if (isa<Float8E4M3FNType>(t))
    return Format{8, 4, 3, 7, 0x7E, 0x7E, 0x7F};
  if (isa<Float8E5M2Type>(t))
    return Format{8, 5, 2, 15, 0x7B, 0x7C, 0x7E};
  if (isa<Float4E2M1FNType>(t))
    return Format{4, 2, 1, 1, 0x7, 0x7, std::nullopt};
  return std::nullopt;
}

struct Expander {
  OpBuilder &b;
  Location loc;
  Type shapeOf; // a value of the shape (scalar or vector) being converted

  Type like(Type elem) {
    if (auto vt = dyn_cast<VectorType>(shapeOf))
      return VectorType::get(vt.getShape(), elem, vt.getScalableDims());
    return elem;
  }

  Value constant(Type ty, Attribute scalar) {
    if (auto vt = dyn_cast<VectorType>(ty))
      return b.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(vt, ArrayRef<Attribute>{scalar}));
    return b.create<arith::ConstantOp>(loc, cast<TypedAttr>(scalar));
  }

  Value intConst(unsigned width, uint64_t v) {
    Type et = b.getIntegerType(width);
    return constant(like(et), b.getIntegerAttr(et, APInt(width, v)));
  }

  Value f32Const(float v) {
    return constant(like(b.getF32Type()), b.getF32FloatAttr(v));
  }

  // `v >> s`, rounded to the nearest value, a tie to the even one. `s` > 0.
  Value shiftRoundingToEven(Value v, Value s, unsigned w) {
    Value one = intConst(w, 1);
    Value half = b.create<arith::ShRUIOp>(
        loc, b.create<arith::ShLIOp>(loc, one, s), one);
    Value lsb =
        b.create<arith::AndIOp>(loc, b.create<arith::ShRUIOp>(loc, v, s), one);
    Value bias = b.create<arith::AddIOp>(
        loc, b.create<arith::SubIOp>(loc, half, one), lsb);
    return b.create<arith::ShRUIOp>(loc, b.create<arith::AddIOp>(loc, v, bias),
                                    s);
  }

  // The f32 that a narrow value `x` holds. Exact: f32 holds every value of
  // every narrow format.
  Value decode(Value x, const Format &f) {
    Type i32 = like(b.getI32Type());
    Value bits =
        b.create<arith::BitcastOp>(loc, like(b.getIntegerType(f.width)), x);
    bits = b.create<arith::ExtUIOp>(loc, i32, bits);
    uint64_t signBit = uint64_t(1) << (f.width - 1);
    Value abs = b.create<arith::AndIOp>(loc, bits, intConst(32, signBit - 1));
    Value mant = b.create<arith::AndIOp>(
        loc, abs, intConst(32, (uint64_t(1) << f.mantBits) - 1));
    Value exp = b.create<arith::ShRUIOp>(loc, abs, intConst(32, f.mantBits));
    Value zero = intConst(32, 0);
    Value subnormal =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, exp, zero);
    Value sig = b.create<arith::SelectOp>(
        loc, subnormal, mant,
        b.create<arith::OrIOp>(loc, mant,
                               intConst(32, uint64_t(1) << f.mantBits)));
    Value e = b.create<arith::SelectOp>(loc, subnormal, intConst(32, 1), exp);
    // 2^(e - bias - mantBits), built from its f32 bits.
    Value scaleBits = b.create<arith::ShLIOp>(
        loc,
        b.create<arith::AddIOp>(loc, e,
                                intConst(32, 127 - f.bias - f.mantBits)),
        intConst(32, 23));
    Value value = b.create<arith::MulFOp>(
        loc, b.create<arith::UIToFPOp>(loc, like(b.getF32Type()), sig),
        b.create<arith::BitcastOp>(loc, like(b.getF32Type()), scaleBits));
    if (f.maxNotNan != f.maxFinite) {
      Value inf = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, abs,
                                          intConst(32, f.maxNotNan));
      value = b.create<arith::SelectOp>(
          loc, inf, f32Const(std::numeric_limits<float>::infinity()), value);
    }
    if (f.nan) {
      Value nan = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, abs,
                                          intConst(32, f.maxNotNan));
      value = b.create<arith::SelectOp>(
          loc, nan, f32Const(std::numeric_limits<float>::quiet_NaN()), value);
    }
    Value negative = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne,
        b.create<arith::AndIOp>(loc, bits, intConst(32, signBit)), zero);
    return b.create<arith::SelectOp>(
        loc, negative, b.create<arith::NegFOp>(loc, value), value);
  }

  // The narrow value nearest to `x`, an f32 or an f64.
  Value encode(Value x, const Format &f, Type narrow) {
    auto src = cast<FloatType>(getElementTypeOrSelf(x.getType()));
    unsigned w = src.getWidth();
    unsigned m = src.getFPMantissaWidth() - 1;
    int64_t bias = (int64_t(1) << (w - m - 2)) - 1;
    Type iw = like(b.getIntegerType(w));
    Value bits = b.create<arith::BitcastOp>(loc, iw, x);
    uint64_t signBit = uint64_t(1) << (w - 1);
    Value abs = b.create<arith::AndIOp>(loc, bits, intConst(w, signBit - 1));
    Value srcInf = intConst(w, ((uint64_t(1) << (w - m - 1)) - 1) << m);
    Value isNan =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, abs, srcInf);
    Value exp = b.create<arith::ShRUIOp>(loc, abs, intConst(w, m));
    int64_t rebias = bias - f.bias;

    // A result that is a normal number in the narrow format: subtract the
    // difference of the biases from the exponent, and round off the mantissa
    // bits the narrow format does not have. A carry out of the mantissa moves
    // into the exponent, which is the right answer.
    Value normal = shiftRoundingToEven(
        b.create<arith::SubIOp>(loc, abs, intConst(w, uint64_t(rebias) << m)),
        intConst(w, m - f.mantBits), w);

    // A result that is subnormal or 0: shift the whole significand. A shift
    // of more than m + 2 gives 0 just as m + 2 does.
    Value srcZeroExp = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               exp, intConst(w, 0));
    Value mant =
        b.create<arith::AndIOp>(loc, abs, intConst(w, (uint64_t(1) << m) - 1));
    Value sig = b.create<arith::SelectOp>(
        loc, srcZeroExp, mant,
        b.create<arith::OrIOp>(loc, mant, intConst(w, uint64_t(1) << m)));
    Value e = b.create<arith::SelectOp>(loc, srcZeroExp, intConst(w, 1), exp);
    Value shift = b.create<arith::SubIOp>(
        loc, intConst(w, uint64_t(bias + m + 1 - f.bias - f.mantBits)), e);
    shift = b.create<arith::MinUIOp>(loc, shift, intConst(w, m + 2));
    Value small = shiftRoundingToEven(sig, shift, w);

    Value isSmall = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule, exp,
                                            intConst(w, uint64_t(rebias)));
    Value r = b.create<arith::SelectOp>(loc, isSmall, small, normal);
    Value maxFinite = intConst(w, f.maxFinite);
    Value tooLarge =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, r, maxFinite);
    r = b.create<arith::SelectOp>(loc, tooLarge, maxFinite, r);
    if (f.nan)
      r = b.create<arith::SelectOp>(loc, isNan, intConst(w, *f.nan), r);
    Value sign = b.create<arith::ShRUIOp>(loc, bits, intConst(w, w - 1));
    r = b.create<arith::OrIOp>(
        loc, r, b.create<arith::ShLIOp>(loc, sign, intConst(w, f.width - 1)));
    if (!f.nan)
      r = b.create<arith::SelectOp>(loc, isNan, maxFinite, r);
    r = b.create<arith::TruncIOp>(loc, like(b.getIntegerType(f.width)), r);
    return b.create<arith::BitcastOp>(loc, like(narrow), r);
  }

  // `x`, any float, as an f32 or f64 that holds it exactly.
  Value widenForEncode(Value x) {
    Type et = getElementTypeOrSelf(x.getType());
    if (et.isF32() || et.isF64())
      return x;
    if (auto f = narrowFormat(et))
      return decode(x, *f);
    return b.create<arith::ExtFOp>(loc, like(b.getF32Type()), x);
  }

  // The f32 `v` as `dst`, a float type that holds it exactly or a narrow one.
  Value fromF32(Value v, Type dst) {
    Type et = getElementTypeOrSelf(dst);
    if (auto f = narrowFormat(et))
      return encode(v, *f, et);
    if (et.isF32())
      return v;
    if (cast<FloatType>(et).getWidth() > 32)
      return b.create<arith::ExtFOp>(loc, dst, v);
    return b.create<arith::TruncFOp>(loc, dst, v);
  }
};

bool isNarrow(Type t) {
  return narrowFormat(getElementTypeOrSelf(t)).has_value();
}

struct ExpandNarrowFloatsPass
    : public PassWrapper<ExpandNarrowFloatsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpandNarrowFloatsPass)

  llvm::StringRef getArgument() const override {
    return "vx-expand-narrow-floats";
  }

  llvm::StringRef getDescription() const override {
    return "Writes conversions to and from f8E4M3FN, f8E5M2 and f4E2M1FN as "
           "integer operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    SmallVector<Operation *> ops;
    getOperation().walk([&](Operation *op) {
      if (isa<arith::ExtFOp, arith::TruncFOp, arith::SIToFPOp, arith::UIToFPOp,
              arith::FPToSIOp, arith::FPToUIOp>(op) &&
          (isNarrow(op->getOperand(0).getType()) ||
           isNarrow(op->getResult(0).getType())))
        ops.push_back(op);
    });
    for (Operation *op : ops) {
      OpBuilder b(op);
      Value in = op->getOperand(0);
      Type outTy = op->getResult(0).getType();
      Expander ex{b, op->getLoc(), in.getType()};
      Value out;
      if (isa<arith::ExtFOp, arith::TruncFOp>(op)) {
        Value wide = ex.widenForEncode(in);
        Type outEt = getElementTypeOrSelf(outTy);
        if (auto f = narrowFormat(outEt))
          out = ex.encode(wide, *f, outEt);
        else if (getElementTypeOrSelf(wide.getType()) == outEt)
          out = wide;
        else
          out = ex.fromF32(wide, outTy);
      } else if (isa<arith::SIToFPOp, arith::UIToFPOp>(op)) {
        // Every narrow format's largest value is far below 2^24, so an integer
        // that f32 rounds is clamped anyway, and the result rounds once.
        Type f32 = ex.like(b.getF32Type());
        Value v =
            isa<arith::SIToFPOp>(op)
                ? b.create<arith::SIToFPOp>(op->getLoc(), f32, in).getResult()
                : b.create<arith::UIToFPOp>(op->getLoc(), f32, in).getResult();
        out = ex.fromF32(v, outTy);
      } else {
        Value v = ex.widenForEncode(in);
        out =
            isa<arith::FPToSIOp>(op)
                ? b.create<arith::FPToSIOp>(op->getLoc(), outTy, v).getResult()
                : b.create<arith::FPToUIOp>(op->getLoc(), outTy, v).getResult();
      }
      op->getResult(0).replaceAllUsesWith(out);
      op->erase();
    }
  }
};

} // namespace

namespace mlir {
namespace vx {
std::unique_ptr<Pass> createExpandNarrowFloatsPass() {
  return std::make_unique<ExpandNarrowFloatsPass>();
}
} // namespace vx
} // namespace mlir

#pragma clang diagnostic pop
