# Vx Ecosystem Packages

This directory contains the first-party ecosystem packages for the Vx programming language. 

While the core primitives (like `Tensor`, `Verified`, memory layout intrinsics) live in `stdlib/`, higher-level domain-specific code belongs here. These packages behave as standard third-party libraries, but are maintained natively in-house for the v3.0 release.

## Available Packages
- **`vx_nn`**: Core neural network layers (Conv2D, Linear, Transformers, Activations).
- **`vx_optim`**: Common optimizers (Adam, SGD, RMSProp) and learning rate schedulers.
- **`vx_linalg`**: Advanced matrix decompositions, solvers, and mathematical operations.
- **`vx_vision`**: Tools for image loading, pre-processing, and augmentation.
- **`vx_models`**: Fully implemented reference architectures (e.g. ResNet, LLaMA).

## Usage
Currently, these packages can be imported by providing their include paths to the `vxc` compiler via the `-I` flag. As module resolution evolves, they will be importable directly (e.g. `import vx_nn::layers`).

## Ecosystem Guidelines

1. **Self-Contained Testing**: All tests for a specific package must reside within that package's own directory (e.g., `packages/vx_nn/tests/`). Do not place ecosystem package tests in the root `tests/` directory.
2. **Minimal Sibling Dependencies**: Packages should strive to be as standalone as possible, relying primarily on the core `stdlib/`. If a package must depend on a sibling (e.g., `vx_models` depending on `vx_nn`), it should be done thoughtfully. 
3. **No Cyclic Dependencies**: Under no circumstances should two ecosystem packages depend on each other cyclically.
4. **Add dependency**: When adding a new library add a dependency graph to `docs/implementation_plans/libraries_design_deps.md`