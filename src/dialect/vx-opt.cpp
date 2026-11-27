#include "VxDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <mutex>
#include <string>

namespace mlir {
namespace vx {
void registerVxPasses();
}
} // namespace mlir

extern "C" {

/// Entry point to run the MLIR optimizer with the Vx dialect registered.
/// `version` is the Vx version, which `--version` prints on a line before
/// LLVM's own lines. Returns 0 on success, non-zero on failure.
int run_vx_opt(int argc, char **argv, const char *version) {
  // The compiler calls this function many times in one process. Add the line
  // once.
  static std::once_flag versionPrinterAdded;
  std::call_once(versionPrinterAdded, [version]() {
    static const std::string vxVersion = version;
    llvm::cl::SetVersionPrinter([](llvm::raw_ostream &os) {
      os << "vx-opt " << vxVersion << "\n";
      llvm::cl::PrintVersionMessage();
    });
  });

  // Register all upstream MLIR passes
  mlir::registerAllPasses();

  // Register our custom Vx passes
  mlir::vx::registerVxPasses();

  // Register all upstream MLIR dialects
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  // The compiler registers these through the MLIR C API. Without them,
  // `convert-to-llvm` leaves the ops of a dialect such as `arith` unconverted.
  mlir::registerAllExtensions(registry);

  // Register our custom Vx dialect
  registry.insert<mlir::vx::VxDialect>();

  // Use MLIR's highly robust generic command line argument parser and pass
  // pipeline executor
  mlir::LogicalResult result =
      mlir::MlirOptMain(argc, argv, "Vx Optimizer Driver", registry);
  return result.succeeded() ? 0 : 1;
}

void registerVxPassesC() {
  // The compiler's own pass registry, not `vx-opt`'s. It held only the Vx
  // passes, so a pipeline naming an upstream pass --
  // `promote-buffers-to-stack`, say -- had nothing to resolve it to. Register
  // the upstream passes here too, exactly as `run_vx_opt` does, so the
  // pipelines in driver.rs and codegen/mod.rs can name any of them.
  mlir::registerAllPasses();
  mlir::vx::registerVxPasses();
}

} // extern "C"
