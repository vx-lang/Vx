//===- mod.rs - Vx Compiler ------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file serves as the root module for the hardware plugin subsystem.
// It aggregates the plugin registry, hardware traits, and specific plugin
// implementations, providing a unified API for the compiler to query and utilize
// heterogeneous compute resources.
//
//===----------------------------------------------------------------------===//
pub mod apple_npe;
pub mod hardware_trait;
pub mod registry;
