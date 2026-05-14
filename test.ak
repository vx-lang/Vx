fn distributed_matmul(a: Ref<Tensor, Memory::Host_DRAM>) -> Verified<Tensor> {
    spawn on(Topology::NPU[0]) {
        let local_data = transfer(a, Memory::NPU_HBM);
        let result = custom_matmul(local_data);
        return transfer(result, Memory::Host_DRAM);
    }
}
