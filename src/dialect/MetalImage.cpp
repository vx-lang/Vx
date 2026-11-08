// Metal Shading Language for the kernels of a `gpu.module`, for the Apple GPU.
//
// A kernel arrives here as the NVPTX pipeline gets it: a CFG, with each local
// in a stack slot. Metal takes structured code. So this inlines the functions
// that a kernel calls, turns the stack slots of numbers into values, and lifts
// the CFG to `scf.while` and `scf.if`. Then it prints MSL. An op that the
// printer does not know is an error that names the op.
//
// The binding convention. runtime/metal_dispatch.mm reads the same one.
// - Each memref parameter is one buffer, `[[buffer(i)]]`, in parameter order.
// - After them, one buffer of 32-bit words holds everything else, in parameter
//   order. A memref takes 2 + 2 * rank words: the index of its aligned pointer
//   in its buffer, its offset, its sizes and its strides. A scalar of up to 32
//   bits takes one word. A 64-bit scalar takes two, the low word first.
// - A comment line `// vx.kernel <name> <param>...` before each kernel gives
//   its parameters: `m<rank>` for a memref, with `r` when the kernel only
//   reads it, and the MLIR type of a scalar (`i32`, `f32`, `index`, ...).
// - A kernel with asserts has a line `// vx.assert <name> <number> <message>`
//   for each, numbered from 1, with the message percent-encoded. It takes one
//   more buffer, a zeroed `uint`. A failed assert stores its number there.
// - `index` is a 32-bit `int`. The runtime refuses a buffer of 2^31 elements
//   or more, so no index can overflow.

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/IR/MemRefMemorySlot.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/CFGToSCF.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/Mem2Reg.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace mlir;

namespace {

// Lifts a kernel's CFG. A kernel returns nothing, so the end of a path that
// never returns is a `gpu.return`.
struct KernelToSCF : public ControlFlowToSCFTransformation {
  FailureOr<Operation *> createUnreachableTerminator(Location loc,
                                                     OpBuilder &builder,
                                                     Region &) override {
    return gpu::ReturnOp::create(builder, loc, ValueRange{}).getOperation();
  }
};

// The gpu dialect does not allow a function body in a `gpu.func`. So the
// inliner pass leaves every call in a kernel.
struct KernelInliner : public InlinerInterface {
  KernelInliner(MLIRContext *context) : InlinerInterface(context) {}
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &mapping) const override {
    return isa<gpu::GPUFuncOp>(dest->getParentOp()) ||
           InlinerInterface::isLegalToInline(dest, src, wouldBeCloned, mapping);
  }
};

// Where a memref's elements are: `ptr[origin + i0 * strides[0] + ...]`.
// `origin` is the index of the memref's first element in `ptr`.
struct View {
  std::string ptr;
  std::string origin;
  // The index of the memref's aligned pointer in `ptr`. A reinterpret_cast
  // sets a new offset from that pointer.
  std::string base;
  SmallVector<std::string> sizes;
  SmallVector<std::string> strides;
};

class Printer {
public:
  explicit Printer(gpu::GPUModuleOp module) : module(module), os(text) {}

  // Prints the whole module. Returns false with `error` set on failure.
  bool print(std::string &out, std::string &error);

private:
  gpu::GPUModuleOp module;
  std::string text;
  llvm::raw_string_ostream os;
  std::string failure;
  int depth = 0;
  unsigned next = 0;
  DenseMap<Value, std::string> names;
  DenseMap<Value, View> views;
  // The number of each assert in the kernel, from 1.
  DenseMap<Operation *, unsigned> asserts;

  bool fail(Operation *op, const Twine &why) {
    if (failure.empty()) {
      std::string where;
      llvm::raw_string_ostream w(where);
      if (op)
        w << "'" << op->getName() << "': ";
      failure = where + why.str();
    }
    return false;
  }

  llvm::raw_ostream &line() {
    os.indent(2 * depth);
    return os;
  }

  std::string fresh(StringRef stem = "v") {
    return (stem + Twine(next++)).str();
  }

  std::string name(Value v) {
    auto it = names.find(v);
    assert(it != names.end() && "a value is used before the printer named it");
    return it->second;
  }

  std::string define(Value v) {
    std::string n = fresh();
    names[v] = n;
    return n;
  }

  bool type(Type t, std::string &out, Operation *op);
  bool unsignedType(Type t, std::string &out, Operation *op);
  bool kernel(gpu::GPUFuncOp f);
  bool constantTable(memref::GlobalOp g);
  bool block(Block &b);
  bool op(Operation *op);
  bool arith(Operation *op);
  bool memrefOp(Operation *op);
  bool gpuOp(Operation *op);
  bool mathOp(Operation *op);
  bool ifOp(scf::IfOp op);
  bool whileOp(scf::WhileOp op);
  bool forOp(scf::ForOp op);
  bool switchOp(scf::IndexSwitchOp op);
  bool viewOf(Value memref, View &out, Operation *user);
  std::string element(const View &v, ValueRange indices);
  std::string foldResult(OpFoldResult r);
  bool layoutStrides(MemRefType t, const SmallVector<std::string> &sizes,
                     SmallVector<std::string> &strides, Operation *op);
  bool assignResults(ValueRange results, ValueRange values);
};

bool Printer::type(Type t, std::string &out, Operation *op) {
  if (t.isInteger(1))
    out = "bool";
  else if (t.isInteger(8))
    out = "char";
  else if (t.isInteger(16))
    out = "short";
  else if (t.isInteger(32) || t.isIndex())
    out = "int";
  else if (t.isInteger(64))
    out = "long";
  else if (t.isF16())
    out = "half";
  else if (t.isBF16())
    out = "bfloat";
  else if (t.isF32())
    out = "float";
  else {
    std::string s;
    llvm::raw_string_ostream ts(s);
    ts << t;
    return fail(op, "Metal has no type for " + s);
  }
  return true;
}

bool Printer::unsignedType(Type t, std::string &out, Operation *op) {
  if (t.isInteger(1))
    out = "bool";
  else if (t.isInteger(8))
    out = "uchar";
  else if (t.isInteger(16))
    out = "ushort";
  else if (t.isInteger(32) || t.isIndex())
    out = "uint";
  else if (t.isInteger(64))
    out = "ulong";
  else
    return type(t, out, op);
  return true;
}

std::string Printer::foldResult(OpFoldResult r) {
  if (auto v = dyn_cast<Value>(r))
    return name(v);
  return std::to_string(cast<IntegerAttr>(cast<Attribute>(r)).getInt());
}

// The strides of a memref with a static layout, as text.
bool Printer::layoutStrides(MemRefType t, const SmallVector<std::string> &sizes,
                            SmallVector<std::string> &strides, Operation *op) {
  SmallVector<int64_t> s;
  int64_t offset;
  if (failed(t.getStridesAndOffset(s, offset)))
    return fail(op, "a memref layout that is not strided");
  strides.clear();
  if (t.getLayout().isIdentity()) {
    strides.assign(t.getRank(), "1");
    for (int64_t k = t.getRank() - 2; k >= 0; --k)
      strides[k] = "(" + strides[k + 1] + " * " + sizes[k + 1] + ")";
    return true;
  }
  for (int64_t x : s) {
    if (ShapedType::isDynamic(x))
      return fail(op, "a memref with a dynamic stride that no parameter gives");
    strides.push_back(std::to_string(x));
  }
  return true;
}

bool Printer::viewOf(Value memref, View &out, Operation *user) {
  auto it = views.find(memref);
  if (it == views.end())
    return fail(user, "a memref that does not come from a parameter, a "
                      "stack slot, shared memory or a view of one of them");
  out = it->second;
  return true;
}

std::string Printer::element(const View &v, ValueRange indices) {
  std::string at = v.origin;
  for (auto [k, i] : llvm::enumerate(indices))
    at += " + " + name(i) + " * " + v.strides[k];
  return v.ptr + "[" + at + "]";
}

bool Printer::assignResults(ValueRange results, ValueRange values) {
  for (auto [r, v] : llvm::zip(results, values))
    line() << name(r) << " = " << name(v) << ";\n";
  return true;
}

bool Printer::print(std::string &out, std::string &error) {
  os << "// Written by vxc for the Apple GPU.\n";
  os << "#include <metal_stdlib>\n";
  os << "using namespace metal;\n";
  bool ok = true;
  for (Operation &o : module.getBody()->getOperations()) {
    if (o.hasTrait<OpTrait::IsTerminator>())
      continue;
    if (auto g = dyn_cast<memref::GlobalOp>(o)) {
      ok = ok && constantTable(g);
    } else if (auto f = dyn_cast<gpu::GPUFuncOp>(o)) {
      if (!f.isKernel()) {
        ok = fail(f, "a device function that is not a kernel");
        break;
      }
      ok = ok && kernel(f);
    } else if (auto f = dyn_cast<func::FuncOp>(o)) {
      ok = fail(f, "a function that a kernel calls and that could not be "
                   "inlined: " +
                       f.getSymName());
    } else {
      ok = fail(&o, "this op at the top of the device module");
    }
    if (!ok)
      break;
  }
  if (!ok || !failure.empty()) {
    error = failure;
    return false;
  }
  out = text;
  return true;
}

// A table of constants, at program scope. Shared memory (address space 3) is
// declared in the kernels that use it instead.
bool Printer::constantTable(memref::GlobalOp g) {
  auto t = g.getType();
  auto space = dyn_cast_or_null<IntegerAttr>(t.getMemorySpace());
  if ((space && space.getInt() == 3) ||
      isa_and_nonnull<gpu::AddressSpaceAttr>(t.getMemorySpace()))
    return true;
  auto init = g.getConstantInitValue();
  auto dense = dyn_cast_or_null<DenseElementsAttr>(init);
  if (!g.getConstant() || !dense || !t.hasStaticShape())
    return fail(g, "a global that is not a table of constants");
  std::string elem;
  if (!type(t.getElementType(), elem, g))
    return false;
  os << "\nconstant " << elem << " " << g.getSymName() << "["
     << std::max<int64_t>(t.getNumElements(), 1) << "] = {";
  bool first = true;
  if (dense.getElementType().isIntOrIndex()) {
    for (APInt v : dense.getValues<APInt>()) {
      os << (first ? "" : ", ") << v.getSExtValue();
      first = false;
    }
  } else {
    for (APFloat v : dense.getValues<APFloat>()) {
      os << (first ? "" : ", ") << "as_type<" << elem << ">(("
         << (elem == "float" ? "uint" : "ushort") << ")0x"
         << llvm::utohexstr(v.bitcastToAPInt().getZExtValue()) << "u)";
      first = false;
    }
  }
  os << "};\n";
  return true;
}

// `s` on one line, with `%`, `\` and control characters as `%XX`. A `\` at the
// end of a comment line would join the next line to the comment.
static std::string percentEncoded(StringRef s) {
  std::string out;
  for (unsigned char c : s) {
    if (c == '%' || c == '\\' || c < 0x20 || c == 0x7f)
      out += "%" + llvm::utohexstr(c, /*LowerCase=*/false, /*Width=*/2);
    else
      out += c;
  }
  return out;
}

bool Printer::kernel(gpu::GPUFuncOp f) {
  names.clear();
  views.clear();
  asserts.clear();
  next = 0;
  SmallVector<cf::AssertOp> checks;
  f.walk([&](cf::AssertOp a) { checks.push_back(a); });
  for (auto [k, a] : llvm::enumerate(checks))
    asserts[a] = k + 1;
  auto access = f->getAttrOfType<ArrayAttr>("vx.arg_access");

  auto readonly = [&](unsigned i) {
    return access && i < access.size() &&
           cast<StringAttr>(access[i]).getValue().contains("readonly");
  };

  // The comment line that the runtime checks the arguments against.
  os << "\n// vx.kernel " << f.getName();
  unsigned buffers = 0;
  for (auto [i, t] : llvm::enumerate(f.getArgumentTypes())) {
    if (auto m = dyn_cast<MemRefType>(t)) {
      os << " m" << m.getRank() << (readonly(i) ? "r" : "");
      ++buffers;
    } else if (t.isIndex()) {
      os << " index";
    } else if (t.isIntOrFloat()) {
      os << " " << t;
    } else {
      return fail(f,
                  "a kernel parameter that is neither a memref nor a number");
    }
  }
  os << "\n";
  for (auto [k, a] : llvm::enumerate(checks))
    os << "// vx.assert " << f.getName() << " " << k + 1 << " "
       << percentEncoded(a.getMsg()) << "\n";
  // Metal binds at most 31 buffers to a kernel. One holds the words, and one
  // holds the number of a failed assert.
  if (buffers > 30)
    return fail(f, "a kernel with more than 30 tensor parameters");
  if (buffers > 29 && !checks.empty())
    return fail(f,
                "a kernel with an assert and more than 29 tensor parameters");

  os << "kernel void " << f.getName() << "(\n";
  unsigned binding = 0;
  for (auto [i, t] : llvm::enumerate(f.getArgumentTypes())) {
    auto m = dyn_cast<MemRefType>(t);
    if (!m)
      continue;
    std::string elem;
    if (!type(m.getElementType(), elem, f))
      return false;
    os << "    device " << (readonly(i) ? "const " : "") << elem << " *p" << i
       << " [[buffer(" << binding++ << ")]],\n";
    names[f.getArgument(i)] = "p" + std::to_string(i);
  }
  os << "    constant uint *vx_w [[buffer(" << buffers << ")]],\n";
  if (!checks.empty())
    os << "    device atomic_uint *vx_fail [[buffer(" << buffers + 1
       << ")]],\n";
  os << "    uint3 vx_tgid [[threadgroup_position_in_grid]],\n";
  os << "    uint3 vx_tid [[thread_position_in_threadgroup]],\n";
  os << "    uint3 vx_ntg [[threadgroups_per_grid]],\n";
  os << "    uint3 vx_tptg [[threads_per_threadgroup]],\n";
  os << "    uint vx_lane [[thread_index_in_simdgroup]],\n";
  os << "    uint vx_simd [[threads_per_simdgroup]]) {\n";
  depth = 1;

  // Shared memory that this kernel uses.
  llvm::StringSet<> shared;
  f.walk([&](memref::GetGlobalOp get) {
    auto t = get.getType();
    auto space = dyn_cast_or_null<IntegerAttr>(t.getMemorySpace());
    if (!(space && space.getInt() == 3) &&
        !isa_and_nonnull<gpu::AddressSpaceAttr>(t.getMemorySpace()))
      return;
    if (!shared.insert(get.getName()).second)
      return;
    std::string elem;
    if (!type(t.getElementType(), elem, get))
      return;
    line() << "threadgroup " << elem << " " << get.getName() << "["
           << std::max<int64_t>(t.getNumElements(), 1) << "];\n";
  });
  if (!failure.empty())
    return false;

  // The parameters, from the words.
  unsigned word = 0;
  for (auto [i, arg] : llvm::enumerate(f.getArguments())) {
    std::string p = "p" + std::to_string(i);
    if (auto m = dyn_cast<MemRefType>(arg.getType())) {
      // A stride or an offset that the type fixes comes from the type, and
      // the others from the words, as the CPU code reads them.
      SmallVector<int64_t> fixedStrides;
      int64_t fixedOffset;
      if (failed(m.getStridesAndOffset(fixedStrides, fixedOffset)))
        return fail(f, "a tensor parameter whose layout is not strided");
      auto fromWord = [&](unsigned at) {
        return "as_type<int>(vx_w[" + std::to_string(at) + "])";
      };
      View v;
      v.ptr = p;
      v.base = p + "_b";
      line() << "const int " << p << "_b = " << fromWord(word) << ";\n";
      line() << "const int " << p << "_o = " << p << "_b + "
             << (ShapedType::isDynamic(fixedOffset)
                     ? fromWord(word + 1)
                     : std::to_string(fixedOffset))
             << ";\n";
      v.origin = p + "_o";
      for (int64_t k = 0; k < m.getRank(); ++k) {
        std::string n = p + "_n" + std::to_string(k);
        std::string s = p + "_s" + std::to_string(k);
        line() << "const int " << n << " = " << fromWord(word + 2 + k) << ";\n";
        line() << "const int " << s << " = "
               << (ShapedType::isDynamic(fixedStrides[k])
                       ? fromWord(word + 2 + m.getRank() + k)
                       : std::to_string(fixedStrides[k]))
               << ";\n";
        v.sizes.push_back(n);
        v.strides.push_back(s);
      }
      views[arg] = v;
      word += 2 + 2 * m.getRank();
      continue;
    }
    Type t = arg.getType();
    std::string ty;
    if (!type(t, ty, f))
      return false;
    names[arg] = p;
    if (t.isInteger(1))
      line() << "const bool " << p << " = vx_w[" << word << "] != 0;\n";
    else if (t.isInteger(64))
      line() << "const long " << p << " = as_type<long>(uint2(vx_w[" << word
             << "], vx_w[" << word + 1 << "]));\n";
    else if (t.isInteger(32) || t.isIndex() || t.isF32())
      line() << "const " << ty << " " << p << " = as_type<" << ty << ">(vx_w["
             << word << "]);\n";
    else if (t.isInteger(8) || t.isInteger(16))
      line() << "const " << ty << " " << p << " = (" << ty << ")vx_w[" << word
             << "];\n";
    else if (t.isF16() || t.isBF16())
      line() << "const " << ty << " " << p << " = as_type<" << ty
             << ">((ushort)vx_w[" << word << "]);\n";
    else
      return fail(f, "a scalar parameter that Metal cannot take");
    word += t.isInteger(64) ? 2 : 1;
  }

  if (!f.getBody().hasOneBlock())
    return fail(f, "a kernel whose branches could not be made structured");
  if (!block(f.getBody().front()))
    return false;
  depth = 0;
  os << "}\n";
  return true;
}

bool Printer::block(Block &b) {
  for (Operation &o : b.getOperations())
    if (!op(&o))
      return false;
  return true;
}

bool Printer::op(Operation *o) {
  if (isa<scf::YieldOp, scf::ConditionOp>(o))
    return true; // The parent op prints what these give.
  if (auto a = dyn_cast<cf::AssertOp>(o)) {
    // Metal cannot stop a kernel. The runtime reads the number when it waits
    // for the launch, and stops the program with the message.
    line() << "if (!" << name(a.getArg()) << ") {\n";
    line() << "  atomic_store_explicit(vx_fail, " << asserts.lookup(o)
           << "u, memory_order_relaxed);\n";
    line() << "  return;\n";
    line() << "}\n";
    return true;
  }
  if (auto i = dyn_cast<scf::IfOp>(o))
    return ifOp(i);
  if (auto w = dyn_cast<scf::WhileOp>(o))
    return whileOp(w);
  if (auto f = dyn_cast<scf::ForOp>(o))
    return forOp(f);
  if (auto s = dyn_cast<scf::IndexSwitchOp>(o))
    return switchOp(s);
  if (isa<gpu::ReturnOp>(o)) {
    line() << "return;\n";
    return true;
  }
  if (auto poison = dyn_cast<ub::PoisonOp>(o)) {
    std::string ty;
    if (!type(poison.getType(), ty, o))
      return false;
    line() << ty << " " << define(poison.getResult()) << " = " << ty
           << "(0);\n";
    return true;
  }
  if (auto call = dyn_cast<func::CallOp>(o))
    return fail(o, "a call to '" + call.getCallee() +
                       "', which could not be inlined");
  StringRef ns = o->getDialect() ? o->getDialect()->getNamespace() : "";
  if (ns == "arith")
    return arith(o);
  if (ns == "memref")
    return memrefOp(o);
  if (ns == "gpu")
    return gpuOp(o);
  if (ns == "math")
    return mathOp(o);
  return fail(o, "the Metal code generator cannot print this op");
}

bool Printer::arith(Operation *o) {
  if (auto c = dyn_cast<arith::ConstantOp>(o)) {
    Type t = c.getType();
    std::string ty;
    if (!type(t, ty, o))
      return false;
    std::string value;
    if (auto i = dyn_cast<IntegerAttr>(c.getValue())) {
      APInt v = i.getValue();
      if (t.isIndex()) {
        if (!v.isSignedIntN(32))
          return fail(o, "an index constant that does not fit in 32 bits");
        v = v.trunc(32);
      }
      if (t.isInteger(1))
        value = v.isZero() ? "false" : "true";
      else if (t.isInteger(64))
        value = "(long)" + std::to_string(v.getZExtValue()) + "ul";
      else
        value = "(" + ty + ")" + std::to_string(v.getZExtValue()) + "u";
    } else if (auto fl = dyn_cast<FloatAttr>(c.getValue())) {
      APFloat v = fl.getValue();
      // A decimal literal only when it reads back as the same bits.
      SmallString<32> s;
      bool exact = false;
      if (t.isF32() && v.isFinite()) {
        v.toString(s);
        if (StringRef(s).find_first_of(".eE") == StringRef::npos)
          s += ".0";
        APFloat back(APFloat::IEEEsingle());
        llvm::Expected<APFloat::opStatus> status =
            back.convertFromString(s, APFloat::rmNearestTiesToEven);
        if (status)
          exact = back.bitwiseIsEqual(v);
        else
          llvm::consumeError(status.takeError());
      }
      if (exact) {
        value = (s + "f").str();
      } else {
        unsigned bits = t.getIntOrFloatBitWidth();
        value = "as_type<" + ty + ">((" + (bits == 32 ? "uint" : "ushort") +
                ")0x" + llvm::utohexstr(v.bitcastToAPInt().getZExtValue()) +
                "u)";
      }
    } else {
      return fail(o, "a constant that is not a number");
    }
    line() << ty << " " << define(c.getResult()) << " = " << value << ";\n";
    return true;
  }

  if (o->getNumResults() != 1)
    return fail(o, "the Metal code generator cannot print this op");
  Value r = o->getResult(0);
  Type t = r.getType();
  std::string ty, uty;
  if (!type(t, ty, o) || !unsignedType(t, uty, o))
    return false;
  auto a = [&](unsigned k) { return name(o->getOperand(k)); };
  std::string expr;

  // Integer arithmetic wraps in MLIR. The printer uses unsigned types for it,
  // because signed overflow is undefined in MSL.
  static const llvm::StringMap<StringRef> wrapping = {
      {"arith.addi", "+"},  {"arith.subi", "-"},  {"arith.muli", "*"},
      {"arith.divui", "/"}, {"arith.remui", "%"}, {"arith.shli", "<<"},
      {"arith.shrui", ">>"}};
  static const llvm::StringMap<StringRef> plain = {
      {"arith.addf", "+"},  {"arith.subf", "-"},  {"arith.mulf", "*"},
      {"arith.divf", "/"},  {"arith.divsi", "/"}, {"arith.remsi", "%"},
      {"arith.andi", "&"},  {"arith.ori", "|"},   {"arith.xori", "^"},
      {"arith.shrsi", ">>"}};
  StringRef opName = o->getName().getStringRef();
  if (auto it = wrapping.find(opName); it != wrapping.end()) {
    if (t.isInteger(1))
      return fail(o, "this op on i1");
    expr = "(" + ty + ")((" + uty + ")" + a(0) + " " + it->second.str() + " (" +
           uty + ")" + a(1) + ")";
  } else if (auto it = plain.find(opName); it != plain.end()) {
    expr = a(0) + " " + it->second.str() + " " + a(1);
  } else if (isa<arith::RemFOp>(o)) {
    expr = "fmod(" + a(0) + ", " + a(1) + ")";
  } else if (isa<arith::NegFOp>(o)) {
    expr = "-" + a(0);
  } else if (isa<arith::MinSIOp>(o)) {
    expr = "min(" + a(0) + ", " + a(1) + ")";
  } else if (isa<arith::MaxSIOp>(o)) {
    expr = "max(" + a(0) + ", " + a(1) + ")";
  } else if (isa<arith::MinUIOp>(o)) {
    expr =
        "(" + ty + ")min((" + uty + ")" + a(0) + ", (" + uty + ")" + a(1) + ")";
  } else if (isa<arith::MaxUIOp>(o)) {
    expr =
        "(" + ty + ")max((" + uty + ")" + a(0) + ", (" + uty + ")" + a(1) + ")";
  } else if (isa<arith::MaxNumFOp>(o)) {
    expr = "fmax(" + a(0) + ", " + a(1) + ")";
  } else if (isa<arith::MinNumFOp>(o)) {
    expr = "fmin(" + a(0) + ", " + a(1) + ")";
  } else if (isa<arith::MaximumFOp, arith::MinimumFOp>(o)) {
    // A NaN gives NaN, and -0 is less than +0, as on the CPU.
    bool most = isa<arith::MaximumFOp>(o);
    std::string x = a(0), y = a(1);
    expr = "(isnan(" + x + ") || isnan(" + y + ")) ? (" + x + " + " + y +
           ") : (" + x + " == " + y + " ? (signbit(" + x + ") " +
           (most ? "? " + y + " : " + x : "? " + x + " : " + y) +
           ") : " + (most ? "max(" : "min(") + x + ", " + y + "))";
  } else if (auto cmp = dyn_cast<arith::CmpIOp>(o)) {
    Type ot = cmp.getLhs().getType();
    std::string ou;
    if (!unsignedType(ot, ou, o))
      return false;
    std::string x = a(0), y = a(1);
    std::string ux = "(" + ou + ")" + x, uy = "(" + ou + ")" + y;
    using P = arith::CmpIPredicate;
    bool isBool = ot.isInteger(1);
    switch (cmp.getPredicate()) {
    case P::eq:
      expr = x + " == " + y;
      break;
    case P::ne:
      expr = x + " != " + y;
      break;
    case P::ult:
      expr = ux + " < " + uy;
      break;
    case P::ule:
      expr = ux + " <= " + uy;
      break;
    case P::ugt:
      expr = ux + " > " + uy;
      break;
    case P::uge:
      expr = ux + " >= " + uy;
      break;
    default:
      if (isBool)
        return fail(o, "a signed comparison of i1 values");
      switch (cmp.getPredicate()) {
      case P::slt:
        expr = x + " < " + y;
        break;
      case P::sle:
        expr = x + " <= " + y;
        break;
      case P::sgt:
        expr = x + " > " + y;
        break;
      case P::sge:
        expr = x + " >= " + y;
        break;
      default:
        llvm_unreachable("every predicate is handled");
      }
    }
  } else if (auto cmp = dyn_cast<arith::CmpFOp>(o)) {
    std::string x = a(0), y = a(1);
    std::string nan = "(isnan(" + x + ") || isnan(" + y + "))";
    using P = arith::CmpFPredicate;
    switch (cmp.getPredicate()) {
    case P::AlwaysFalse:
      expr = "false";
      break;
    case P::AlwaysTrue:
      expr = "true";
      break;
    case P::OEQ:
      expr = x + " == " + y;
      break;
    case P::OGT:
      expr = x + " > " + y;
      break;
    case P::OGE:
      expr = x + " >= " + y;
      break;
    case P::OLT:
      expr = x + " < " + y;
      break;
    case P::OLE:
      expr = x + " <= " + y;
      break;
    case P::ONE:
      expr = "(" + x + " < " + y + " || " + x + " > " + y + ")";
      break;
    case P::ORD:
      expr = "!" + nan;
      break;
    case P::UEQ:
      expr = "!(" + x + " < " + y + " || " + x + " > " + y + ")";
      break;
    case P::UGT:
      expr = "!(" + x + " <= " + y + ")";
      break;
    case P::UGE:
      expr = "!(" + x + " < " + y + ")";
      break;
    case P::ULT:
      expr = "!(" + x + " >= " + y + ")";
      break;
    case P::ULE:
      expr = "!(" + x + " > " + y + ")";
      break;
    case P::UNE:
      expr = x + " != " + y;
      break;
    case P::UNO:
      expr = nan;
      break;
    }
  } else if (isa<arith::SelectOp>(o)) {
    if (isa<MemRefType>(t))
      return fail(o, "a choice between two memrefs");
    expr = a(0) + " ? " + a(1) + " : " + a(2);
  } else if (isa<arith::BitcastOp>(o)) {
    expr = "as_type<" + ty + ">(" + a(0) + ")";
  } else if (isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtSIOp,
                 arith::ExtUIOp, arith::TruncIOp, arith::SIToFPOp,
                 arith::UIToFPOp, arith::FPToSIOp, arith::FPToUIOp,
                 arith::ExtFOp, arith::TruncFOp>(o)) {
    Type from = o->getOperand(0).getType();
    std::string fromU;
    if (!unsignedType(from, fromU, o))
      return false;
    std::string x = a(0);
    bool zeroExtend =
        isa<arith::IndexCastUIOp, arith::ExtUIOp, arith::UIToFPOp>(o);
    if (from.isInteger(1) && !zeroExtend && !isa<arith::TruncIOp>(o))
      // A signed i1 is -1 or 0.
      expr = "(" + ty + ")(" + x + " ? -1 : 0)";
    else if (t.isInteger(1))
      expr = "((" + x + " & 1) != 0)";
    else if (zeroExtend)
      expr = "(" + ty + ")(" + fromU + ")" + x;
    else if (isa<arith::FPToUIOp>(o))
      expr = "(" + ty + ")(" + uty + ")" + x;
    else
      expr = "(" + ty + ")" + x;
  } else {
    return fail(o, "the Metal code generator cannot print this op");
  }
  line() << ty << " " << define(r) << " = " << expr << ";\n";
  return true;
}

bool Printer::mathOp(Operation *o) {
  if (o->getNumResults() != 1)
    return fail(o, "the Metal code generator cannot print this op");
  Value r = o->getResult(0);
  std::string ty;
  if (!type(r.getType(), ty, o))
    return false;
  auto a = [&](unsigned k) { return name(o->getOperand(k)); };
  std::string expr;
  // `precise::` gives the IEEE result where Metal has one, as the CPU does.
  static const llvm::StringMap<StringRef> unary = {
      {"math.sqrt", "precise::sqrt"},
      {"math.absf", "fabs"},
      {"math.floor", "floor"},
      {"math.ceil", "ceil"},
      {"math.trunc", "trunc"},
      {"math.round", "round"},
      {"math.roundeven", "rint"},
      {"math.exp", "precise::exp"},
      {"math.exp2", "precise::exp2"},
      {"math.log", "precise::log"},
      {"math.log2", "precise::log2"},
      {"math.sin", "precise::sin"},
      {"math.cos", "precise::cos"},
      {"math.tan", "precise::tan"},
      {"math.tanh", "precise::tanh"},
      {"math.atan", "precise::atan"},
      {"math.absi", "abs"}};
  static const llvm::StringMap<StringRef> binary = {
      {"math.powf", "precise::pow"},
      {"math.atan2", "precise::atan2"},
      {"math.copysign", "copysign"}};
  StringRef opName = o->getName().getStringRef();
  if (auto it = unary.find(opName); it != unary.end())
    expr = it->second.str() + "(" + a(0) + ")";
  else if (auto it = binary.find(opName); it != binary.end())
    expr = it->second.str() + "(" + a(0) + ", " + a(1) + ")";
  else if (isa<math::RsqrtOp>(o))
    expr = "(" + ty + ")1 / precise::sqrt(" + a(0) + ")";
  else if (isa<math::FmaOp>(o))
    expr = "fma(" + a(0) + ", " + a(1) + ", " + a(2) + ")";
  else
    return fail(o, "the Metal code generator cannot print this op");
  line() << ty << " " << define(r) << " = " << expr << ";\n";
  return true;
}

bool Printer::gpuOp(Operation *o) {
  auto builtin = [&](StringRef var, gpu::Dimension d, Value r) {
    static const char *axis[] = {"x", "y", "z"};
    line() << "int " << define(r) << " = (int)" << var << "."
           << axis[static_cast<int>(d)] << ";\n";
    return true;
  };
  if (auto i = dyn_cast<gpu::ThreadIdOp>(o))
    return builtin("vx_tid", i.getDimension(), i.getResult());
  if (auto i = dyn_cast<gpu::BlockIdOp>(o))
    return builtin("vx_tgid", i.getDimension(), i.getResult());
  if (auto i = dyn_cast<gpu::GridDimOp>(o))
    return builtin("vx_ntg", i.getDimension(), i.getResult());
  if (auto i = dyn_cast<gpu::BlockDimOp>(o))
    return builtin("vx_tptg", i.getDimension(), i.getResult());
  if (auto i = dyn_cast<gpu::LaneIdOp>(o)) {
    line() << "int " << define(i.getResult()) << " = (int)vx_lane;\n";
    return true;
  }
  if (auto i = dyn_cast<gpu::SubgroupSizeOp>(o)) {
    line() << "int " << define(i.getResult()) << " = (int)vx_simd;\n";
    return true;
  }
  if (isa<gpu::BarrierOp>(o)) {
    // As `bar.sync`: the threads of the block wait, and each sees the writes
    // to device and shared memory that the others made before it.
    line() << "threadgroup_barrier(mem_flags::mem_device | "
              "mem_flags::mem_threadgroup);\n";
    return true;
  }
  if (auto s = dyn_cast<gpu::ShuffleOp>(o)) {
    // A SIMD-group has 32 threads, so only a shuffle across 32 lanes is the
    // same operation.
    IntegerAttr width;
    if (!matchPattern(s.getWidth(), m_Constant(&width)) || width.getInt() != 32)
      return fail(o, "a shuffle that is not across 32 lanes");
    std::string ty;
    if (!type(s.getShuffleResult().getType(), ty, o))
      return false;
    const char *fn = nullptr;
    switch (s.getMode()) {
    case gpu::ShuffleMode::XOR:
      fn = "simd_shuffle_xor";
      break;
    case gpu::ShuffleMode::DOWN:
      fn = "simd_shuffle_down";
      break;
    case gpu::ShuffleMode::UP:
      fn = "simd_shuffle_up";
      break;
    case gpu::ShuffleMode::IDX:
      fn = "simd_shuffle";
      break;
    }
    line() << ty << " " << define(s.getShuffleResult()) << " = " << fn << "("
           << name(s.getValue()) << ", (ushort)" << name(s.getOffset())
           << ");\n";
    // A shuffle within 32 lanes always finds its lane, except that `down`
    // and `up` can go past the end. The result is then the thread's own
    // value, as on NVIDIA.
    std::string valid = "true";
    if (s.getMode() == gpu::ShuffleMode::DOWN)
      valid = "(int)vx_lane + " + name(s.getOffset()) + " < 32";
    if (s.getMode() == gpu::ShuffleMode::UP)
      valid = "(int)vx_lane >= " + name(s.getOffset());
    line() << "bool " << define(s.getValid()) << " = " << valid << ";\n";
    return true;
  }
  return fail(o, "the Metal code generator cannot print this op");
}

bool Printer::memrefOp(Operation *o) {
  if (auto load = dyn_cast<memref::LoadOp>(o)) {
    View v;
    if (!viewOf(load.getMemRef(), v, o))
      return false;
    std::string ty;
    if (!type(load.getType(), ty, o))
      return false;
    line() << ty << " " << define(load.getResult()) << " = "
           << element(v, load.getIndices()) << ";\n";
    return true;
  }
  if (auto store = dyn_cast<memref::StoreOp>(o)) {
    View v;
    if (!viewOf(store.getMemRef(), v, o))
      return false;
    line() << element(v, store.getIndices()) << " = "
           << name(store.getValueToStore()) << ";\n";
    return true;
  }
  if (auto dim = dyn_cast<memref::DimOp>(o)) {
    View v;
    if (!viewOf(dim.getSource(), v, o))
      return false;
    std::optional<int64_t> k = dim.getConstantIndex();
    if (!k || *k < 0 || *k >= static_cast<int64_t>(v.sizes.size()))
      return fail(o, "a dimension that is not a constant");
    line() << "int " << define(dim.getResult()) << " = " << v.sizes[*k]
           << ";\n";
    return true;
  }
  if (auto cast = dyn_cast<memref::CastOp>(o)) {
    View v;
    if (!viewOf(cast.getSource(), v, o))
      return false;
    views[cast.getResult()] = v;
    return true;
  }
  if (auto rc = dyn_cast<memref::ReinterpretCastOp>(o)) {
    View src;
    if (!viewOf(rc.getSource(), src, o))
      return false;
    View v;
    v.ptr = src.ptr;
    v.base = src.base;
    v.origin = fresh("o");
    line() << "const int " << v.origin << " = " << src.base << " + "
           << foldResult(rc.getMixedOffsets()[0]) << ";\n";
    for (OpFoldResult s : rc.getMixedSizes())
      v.sizes.push_back(foldResult(s));
    for (OpFoldResult s : rc.getMixedStrides())
      v.strides.push_back(foldResult(s));
    views[rc.getResult()] = v;
    return true;
  }
  if (auto sv = dyn_cast<memref::SubViewOp>(o)) {
    View src;
    if (!viewOf(sv.getSource(), src, o))
      return false;
    View v;
    v.ptr = src.ptr;
    v.base = src.base;
    v.origin = fresh("o");
    std::string at = src.origin;
    for (auto [k, off] : llvm::enumerate(sv.getMixedOffsets()))
      at += " + " + foldResult(off) + " * " + src.strides[k];
    line() << "const int " << v.origin << " = " << at << ";\n";
    llvm::SmallBitVector dropped = sv.getDroppedDims();
    for (auto [k, size] : llvm::enumerate(sv.getMixedSizes())) {
      if (dropped.test(k))
        continue;
      v.sizes.push_back(foldResult(size));
      v.strides.push_back("(" + src.strides[k] + " * " +
                          foldResult(sv.getMixedStrides()[k]) + ")");
    }
    views[sv.getResult()] = v;
    return true;
  }
  if (auto get = dyn_cast<memref::GetGlobalOp>(o)) {
    MemRefType t = get.getType();
    View v;
    v.ptr = get.getName().str();
    v.base = "0";
    v.origin = "0";
    for (int64_t s : t.getShape())
      v.sizes.push_back(std::to_string(s));
    if (!layoutStrides(t, v.sizes, v.strides, o))
      return false;
    views[get.getResult()] = v;
    return true;
  }
  if (auto alloca = dyn_cast<memref::AllocaOp>(o)) {
    MemRefType t = alloca.getType();
    if (!t.hasStaticShape())
      return fail(o, "a stack slot whose size is not known when compiling");
    std::string elem;
    if (!type(t.getElementType(), elem, o))
      return false;
    View v;
    v.ptr = fresh("a");
    v.base = "0";
    v.origin = "0";
    line() << "thread " << elem << " " << v.ptr << "["
           << std::max<int64_t>(t.getNumElements(), 1) << "];\n";
    for (int64_t s : t.getShape())
      v.sizes.push_back(std::to_string(s));
    if (!layoutStrides(t, v.sizes, v.strides, o))
      return false;
    views[alloca.getResult()] = v;
    return true;
  }
  return fail(o, "the Metal code generator cannot print this op");
}

bool Printer::ifOp(scf::IfOp i) {
  for (Value r : i.getResults()) {
    std::string ty;
    if (!type(r.getType(), ty, i))
      return false;
    line() << ty << " " << define(r) << ";\n";
  }
  line() << "if (" << name(i.getCondition()) << ") {\n";
  ++depth;
  if (!block(i.getThenRegion().front()))
    return false;
  assignResults(i.getResults(), i.thenYield().getOperands());
  --depth;
  if (!i.getElseRegion().empty()) {
    line() << "} else {\n";
    ++depth;
    if (!block(i.getElseRegion().front()))
      return false;
    assignResults(i.getResults(), i.elseYield().getOperands());
    --depth;
  }
  line() << "}\n";
  return true;
}

// `scf.while` as `while (true)`: the before region decides, and the after
// region makes the values of the next turn.
bool Printer::whileOp(scf::WhileOp w) {
  SmallVector<std::string> results, carried, passed;
  for (Value r : w.getResults()) {
    std::string ty;
    if (!type(r.getType(), ty, w))
      return false;
    line() << ty << " " << define(r) << ";\n";
  }
  line() << "{\n";
  ++depth;
  for (Value init : w.getInits()) {
    std::string ty;
    if (!type(init.getType(), ty, w))
      return false;
    carried.push_back(fresh("c"));
    line() << ty << " " << carried.back() << " = " << name(init) << ";\n";
  }
  line() << "while (true) {\n";
  ++depth;
  std::string go = fresh("go");
  line() << "bool " << go << ";\n";
  for (Value r : w.getResults()) {
    std::string ty;
    if (!type(r.getType(), ty, w))
      return false;
    passed.push_back(fresh("d"));
    line() << ty << " " << passed.back() << ";\n";
  }
  line() << "{\n";
  ++depth;
  Block &before = w.getBefore().front();
  for (auto [arg, c] : llvm::zip(before.getArguments(), carried)) {
    std::string ty;
    if (!type(arg.getType(), ty, w))
      return false;
    line() << ty << " " << define(arg) << " = " << c << ";\n";
  }
  if (!block(before))
    return false;
  scf::ConditionOp cond = w.getConditionOp();
  line() << go << " = " << name(cond.getCondition()) << ";\n";
  for (auto [d, v] : llvm::zip(passed, cond.getArgs()))
    line() << d << " = " << name(v) << ";\n";
  --depth;
  line() << "}\n";
  line() << "if (!" << go << ") {\n";
  ++depth;
  for (auto [r, d] : llvm::zip(w.getResults(), passed))
    line() << name(r) << " = " << d << ";\n";
  line() << "break;\n";
  --depth;
  line() << "}\n";
  line() << "{\n";
  ++depth;
  Block &after = w.getAfter().front();
  for (auto [arg, d] : llvm::zip(after.getArguments(), passed)) {
    std::string ty;
    if (!type(arg.getType(), ty, w))
      return false;
    line() << ty << " " << define(arg) << " = " << d << ";\n";
  }
  if (!block(after))
    return false;
  for (auto [c, v] : llvm::zip(carried, w.getYieldOp().getOperands()))
    line() << c << " = " << name(v) << ";\n";
  --depth;
  line() << "}\n";
  --depth;
  line() << "}\n";
  --depth;
  line() << "}\n";
  return true;
}

bool Printer::forOp(scf::ForOp f) {
  std::string ty;
  if (!type(f.getInductionVar().getType(), ty, f))
    return false;
  SmallVector<std::string> carried;
  for (Value r : f.getResults()) {
    std::string rt;
    if (!type(r.getType(), rt, f))
      return false;
    line() << rt << " " << define(r) << ";\n";
  }
  line() << "{\n";
  ++depth;
  for (Value init : f.getInitArgs()) {
    std::string it;
    if (!type(init.getType(), it, f))
      return false;
    carried.push_back(fresh("c"));
    line() << it << " " << carried.back() << " = " << name(init) << ";\n";
  }
  std::string iv = define(f.getInductionVar());
  line() << "for (" << ty << " " << iv << " = " << name(f.getLowerBound())
         << "; " << iv << " < " << name(f.getUpperBound()) << "; " << iv
         << " += " << name(f.getStep()) << ") {\n";
  ++depth;
  for (auto [arg, c] : llvm::zip(f.getRegionIterArgs(), carried)) {
    std::string at;
    if (!type(arg.getType(), at, f))
      return false;
    line() << at << " " << define(arg) << " = " << c << ";\n";
  }
  if (!block(*f.getBody()))
    return false;
  auto yield = cast<scf::YieldOp>(f.getBody()->getTerminator());
  for (auto [c, v] : llvm::zip(carried, yield.getOperands()))
    line() << c << " = " << name(v) << ";\n";
  --depth;
  line() << "}\n";
  for (auto [r, c] : llvm::zip(f.getResults(), carried))
    line() << name(r) << " = " << c << ";\n";
  --depth;
  line() << "}\n";
  return true;
}

bool Printer::switchOp(scf::IndexSwitchOp s) {
  for (Value r : s.getResults()) {
    std::string ty;
    if (!type(r.getType(), ty, s))
      return false;
    line() << ty << " " << define(r) << ";\n";
  }
  line() << "switch (" << name(s.getArg()) << ") {\n";
  auto body = [&](Region &region) {
    ++depth;
    line() << "{\n";
    ++depth;
    bool ok = block(region.front());
    if (ok)
      assignResults(
          s.getResults(),
          cast<scf::YieldOp>(region.front().getTerminator()).getOperands());
    line() << "break;\n";
    --depth;
    line() << "}\n";
    --depth;
    return ok;
  };
  for (auto [value, region] : llvm::zip(s.getCases(), s.getCaseRegions())) {
    line() << "case " << value << ":\n";
    if (!body(region))
      return false;
  }
  line() << "default:\n";
  if (!body(s.getDefaultRegion()))
    return false;
  line() << "}\n";
  return true;
}

// Makes each kernel of `m` structured: no calls, numbers in values instead of
// stack slots, and `scf` instead of a CFG.
LogicalResult structure(gpu::GPUModuleOp m, std::string &error) {
  MLIRContext *context = m.getContext();
  {
    PassManager pm(context, gpu::GPUModuleOp::getOperationName());
    pm.addPass(createLowerAffinePass());
    pm.addPass(createSCFToControlFlowPass());
    // This inlines a call from one function to another. The calls in a kernel
    // come next.
    pm.addPass(createInlinerPass());
    if (failed(pm.run(m))) {
      error = "inlining the functions that a kernel calls failed";
      return failure();
    }
  }
  // A call that stays here, for example a recursive one, is a printer error.
  SmallVector<func::CallOp> calls;
  for (gpu::GPUFuncOp f : m.getOps<gpu::GPUFuncOp>())
    f.walk([&](func::CallOp call) { calls.push_back(call); });
  SymbolTable symbols(m);
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
  // The lifetime of a stack slot means nothing to Metal.
  SmallVector<Operation *> markers;
  m.walk([&](Operation *op) {
    if (isa<LLVM::LifetimeStartOp, LLVM::LifetimeEndOp>(op))
      markers.push_back(op);
  });
  for (Operation *op : markers)
    op->erase();

  for (gpu::GPUFuncOp f : m.getOps<gpu::GPUFuncOp>()) {
    SmallVector<PromotableAllocationOpInterface> slots;
    f.walk([&](memref::AllocaOp a) {
      if (a.getType().getRank() == 0)
        if (auto p = dyn_cast<PromotableAllocationOpInterface>(*a))
          slots.push_back(p);
    });
    OpBuilder builder(context);
    DominanceInfo dominance(f);
    (void)tryToPromoteMemorySlots(slots, builder, DataLayout::closest(f),
                                  dominance);
  }
  {
    PassManager pm(context, gpu::GPUModuleOp::getOperationName());
    pm.addPass(createSymbolDCEPass());
    pm.addPass(createCanonicalizerPass());
    if (failed(pm.run(m))) {
      error = "simplifying a kernel failed";
      return failure();
    }
  }
  KernelToSCF lift;
  for (gpu::GPUFuncOp f : m.getOps<gpu::GPUFuncOp>()) {
    DominanceInfo dominance(f);
    if (failed(transformCFGToSCF(f.getBody(), lift, dominance))) {
      error = "the branches of kernel '" + f.getName().str() +
              "' could not be made structured";
      return failure();
    }
  }
  {
    PassManager pm(context, gpu::GPUModuleOp::getOperationName());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    if (failed(pm.run(m))) {
      error = "simplifying a structured kernel failed";
      return failure();
    }
  }
  return success();
}

} // namespace

/// The MSL of the kernels in `gpuModule`, or the empty string with `error`
/// set. Works on a copy, so `gpuModule` does not change.
std::string mslImageOf(gpu::GPUModuleOp gpuModule, std::string &error) {
  MLIRContext *context = gpuModule.getContext();
  DialectRegistry registry;
  func::registerInlinerExtension(registry);
  memref::registerMemorySlotExternalModels(registry);
  context->appendDialectRegistry(registry);
  context->getOrLoadDialect<ub::UBDialect>();
  context->getOrLoadDialect<scf::SCFDialect>();

  OwningOpRef<ModuleOp> holder = ModuleOp::create(gpuModule.getLoc());
  holder->getBody()->push_back(gpuModule->clone());
  auto copy = cast<gpu::GPUModuleOp>(holder->getBody()->front());
  if (failed(structure(copy, error)))
    return {};
  std::string image;
  Printer printer(copy);
  if (!printer.print(image, error))
    return {};
  return image;
}
