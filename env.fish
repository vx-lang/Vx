# env.fish - Local environment setup for Vx on macOS (Apple Silicon)

# 1. Point Rust crates to the LLVM 22 installation
set -gx LLVM_SYS_220_PREFIX /opt/homebrew/opt/llvm@22
set -gx MLIR_SYS_220_PREFIX /opt/homebrew/opt/llvm@22

# 2. Prepend LLVM 22's bin to PATH so 'llvm-config' and 'clang' resolve correctly
set -gx PATH /opt/homebrew/opt/llvm@22/bin $PATH

# 3. Linker paths: We MUST include both LLVM's lib and Homebrew's general lib (for zstd, xml2, etc.)
set -gx LIBRARY_PATH /opt/homebrew/opt/llvm@22/lib /opt/homebrew/lib $LIBRARY_PATH

# 4. pkg-config paths for C++ dependencies
set -gx PKG_CONFIG_PATH /opt/homebrew/opt/llvm@22/lib/pkgconfig /opt/homebrew/lib/pkgconfig $PKG_CONFIG_PATH

# 5. (Optional) Add the Vx toolchain install path if needed by MLIR
set -gx VX_TOOLCHAIN_PATH /opt/homebrew/opt/llvm@22

echo "Vx environment loaded (LLVM 22)"
