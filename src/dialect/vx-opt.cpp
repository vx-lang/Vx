#include "VxDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

namespace mlir {
namespace vx {
void registerVxPasses();
}
} // namespace mlir

extern "C" {

/// Entry point to run the MLIR optimizer with the Vx dialect registered.
/// Returns 0 on success, non-zero on failure.
int run_vx_opt(int argc, char **argv) {
  // Register all upstream MLIR passes
  mlir::registerAllPasses();

  // Register our custom Vx passes
  mlir::vx::registerVxPasses();

  // Register all upstream MLIR dialects
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  // Register our custom Vx dialect
  registry.insert<mlir::vx::VxDialect>();

  // Use MLIR's highly robust generic command line argument parser and pass
  // pipeline executor
  mlir::LogicalResult result =
      mlir::MlirOptMain(argc, argv, "Vx Optimizer Driver", registry);
  return result.succeeded() ? 0 : 1;
}

void registerVxPassesC() {
  mlir::vx::registerVxPasses();
}

} // extern "C"
