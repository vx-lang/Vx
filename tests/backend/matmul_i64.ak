// EXPECT:  [24,   24,   24,   24], 
// EXPECT:  [24,   24,   24,   24], 
// EXPECT:  [24,   24,   24,   24]]

fn custom_matmul(a: Ref<Tensor<i64>, Memory::NPU_HBM>, b: Ref<Tensor<i64>, Memory::NPU_HBM>) -> Verified<Tensor<i64>> {
    spawn on(Topology::NPU[0]) {
        let mut result = Tensor<i64>([4, 4]).with_memory(Memory::NPU_HBM);
        for i in 0..4 {
            for j in 0..4 {
                result[i][j] = 0;
                for k in 0..4 {
                    result[i][j] = result[i][j] + a[i][k] * b[k][j];
                }
            }
        }
        return Verified(result);
    }
}

fn main() -> Verified<Tensor<i64>> {
    let mut a = Tensor<i64>([4, 4]).with_memory(Memory::NPU_HBM);
    let mut b = Tensor<i64>([4, 4]).with_memory(Memory::NPU_HBM);
    
    for i in 0..4 {
        for j in 0..4 {
            a[i][j] = 2;
            b[i][j] = 3;
        }
    }
    
    let result = custom_matmul(a, b);
    print(result);
    return result;
}
