fn custom_matmul(a: Ref<Tensor, Memory::NPU_HBM>, b: Ref<Tensor, Memory::NPU_HBM>) -> Verified<Tensor> {
    spawn on(Topology::NPU[0]) {
        let mut result: Tensor = Tensor([a.shape[0], b.shape[1]]).with_memory(Memory::NPU_HBM);
        for i in 0..a.shape[0] {
            for j in 0..b.shape[1] {
                result[i][j] = 0;
                for k in 0..a.shape[1] {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return Verified(result);
    }
}

fn distributed_matmul(a: Ref<Tensor, Memory::Host_DRAM>, b: Ref<Tensor, Memory::Host_DRAM>) -> Verified<Tensor> {
    spawn on(Topology::NPU[0]) {
        let local_a = transfer(a, Memory::NPU_HBM);
        let local_b = transfer(b, Memory::NPU_HBM);
        let result = custom_matmul(local_a, local_b);
        return result;
    }
}
