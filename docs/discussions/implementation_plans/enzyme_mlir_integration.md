# Integrating Enzyme MLIR Pass with Melior

This plan covers the design and execution of adopting "Melior all the way" for both object emission and Enzyme autodiff.

## Objective

Migrate the compiler's MLIR-to-LLVM backend from shelling out to CLI binaries (`mlir-opt`, `mlir-translate`, `llc`) to using native MLIR APIs within the same process.

## Design Decision: Enzyme's MLIR Interface via C++ Wrapper

We chose to use **Enzyme's MLIR interface** rather than running `opt` separately. This is the cleanest approach architecturally.

However, we hit a limitation with the `mlir-c` bindings (which `melior` relies on): **The MLIR C API does not currently expose a function to load dynamic pass plugins.**

Because of this, `melior` has no built-in way to load `Enzyme.dylib` into its `PassManager` dynamically at runtime.

### C++ Plugin Loader Wrapper

To strictly adhere to "Melior all the way" and ensure we still use **LLVM stuff as libraries rather than shelling out to binaries like `llc` or `opt`**, we built a minimal C++ wrapper that hooks into the MLIR C++ API's plugin loader, and exposed it to Rust.

1. **C++ Wrapper**: Created `src/plugin_loader.cpp`:
   ```cpp
   #include "mlir/Tools/Plugins/PassPlugin.h"
   #include <string>
   #include <iostream>

   extern "C" bool loadMlirPassPlugin(const char* pluginPath) {
       if (!pluginPath) return false;
       std::string path(pluginPath);
       auto plugin = mlir::PassPlugin::load(path);
       if (!plugin) return false;
       plugin.get().registerPassRegistryCallbacks();
       return true;
   }
   ```
1. **Build Script**: We updated `build.rs` to compile `src/plugin_loader.cpp` against LLVM libraries and archive it into `libplugin_loader.a`.
1. **FFI Binding in Rust**: We exposed this function in `src/melior_codegen.rs` to load the plugin from the `ENZYME_LIB` environment variable dynamically.
1. **Pipeline Injection**: If the MLIR pass plugin loads successfully, we insert `enzyme` into the native `melior` `PassManager` pass pipeline before lowering to LLVM dialect.

This solution avoids shelling out to `mlir-opt` for Enzyme completely, keeping the entire backend execution in-process.
