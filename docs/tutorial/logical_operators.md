# Logical and Relational Operators in Vx

Vx provides comprehensive support for boolean logic and relation evaluations, both natively as scalars and across bounded Topology tensors.

## Relational Operators

You can compare tensors or scalars using standard operators. Vx evaluates these and inherently maps them to the `Tensor<Bool>` domain (lowering directly to hardware-efficient MLIR `i1` types).

- `==` : Equal
- `!=` : Not Equal
- `<` : Less Than
- `>` : Greater Than
- `<=` : Less Than or Equal
- `>=` : Greater Than or Equal

```vx
fn test_relations() -> Tensor<Bool> {
    let a: Tensor<i32> = 10;
    let b: Tensor<i32> = 20;

    let is_less = a < b;
    return is_less;
}
```

## Logical Operators

Complex boolean operations can be evaluated natively:

- `&&` : Logical AND
- `||` : Logical OR
- `!` : Logical NOT

```vx
fn validate_bounds(val: Tensor<i32>) -> Tensor<Bool> {
    let lower_bound: Tensor<i32> = 0;
    let upper_bound: Tensor<i32> = 100;

    // Check if within [0, 100]
    let is_valid = (val >= lower_bound) && (val <= upper_bound);
    return is_valid;
}
```

## Interaction with Generic Tensors

By abstracting boolean logic to `Tensor<Bool>`, Vx maintains full compatibility with its heterogeneous hardware abstraction. You can execute `&&` conditions seamlessly on a spawned NPU, and seamlessly pipe the resulting `Tensor<Bool>` back to the host for flow control!
