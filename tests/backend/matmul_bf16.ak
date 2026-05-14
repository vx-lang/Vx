// EXPECT:  [24.0,   24.0,   24.0,   24.0], 
// EXPECT:  [24.0,   24.0,   24.0,   24.0], 
// EXPECT:  [24.0,   24.0,   24.0,   24.0]]

fn custom_matmul(a: Ref<Tensor<bf16>, Memory::NPU_HBM>, b: Ref<Tensor<bf16>, Memory::NPU_HBM>) -> Verified<Tensor<bf16>> {
    spawn on(Topology::NPU[0]) {
        let mut result = Tensor<bf16>([4, 4]).with_memory(Memory::NPU_HBM);
        for i in 0..4 {
            for j in 0..4 {
                result[i][j] = 0.0;
                for k in 0..4 {
                    result[i][j] = result[i][j] + a[i][k] * b[k][j];
                }
            }
        }
        return Verified(result);
    }
}

fn main() -> Verified<Tensor<bf16>> {
    let mut a = Tensor<bf16>([4, 4]).with_memory(Memory::NPU_HBM);
    let mut b = Tensor<bf16>([4, 4]).with_memory(Memory::NPU_HBM);
    
    for i in 0..4 {
        for j in 0..4 {
            a[i][j] = 2.0;
            b[i][j] = 3.0;
        }
    }
    
    let result = custom_matmul(a, b);
    print(result);
    return result;
}
