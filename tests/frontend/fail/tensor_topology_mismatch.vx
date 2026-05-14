// RUN: ! akarc %s

fn requires_specific_topology(t: Tensor<f32, [128, 256], Topology::ANE>) {}

fn main() {
    let t: Tensor<f32, [128, 256], Topology::AMX> = 1.0;
    requires_specific_topology(t);
}
