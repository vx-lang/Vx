// Scores fleet/node-2gpu-a100.vx's declared peer edge, which that file calls "the one figure the
// demo actually rests on". It is declared at the PESSIMISTIC end (PCIe Gen4, 31.5 GB/s) because a
// rented pod does not say what it gave you, and a bound the hardware can only beat is the right
// direction for a refusal to be trustworthy. `nvidia-smi topo -m` on this pod reports PHB, so
// there is no NVLink and the pessimistic branch is the one that applies.
#include <cstdio>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("ERR %s @%d\n",cudaGetErrorString(e),__LINE__);return 1;} } while(0)
int main(){
  int n=0; CK(cudaGetDeviceCount(&n));
  printf("devices,%d\n", n);
  if(n<2){ printf("need 2 GPUs\n"); return 1; }
  int can01=0, can10=0;
  CK(cudaDeviceCanAccessPeer(&can01,0,1));
  CK(cudaDeviceCanAccessPeer(&can10,1,0));
  printf("p2p_capable_0to1,%d\np2p_capable_1to0,%d\n", can01, can10);
  CK(cudaSetDevice(0)); if(can01) cudaDeviceEnablePeerAccess(1,0); cudaGetLastError();
  CK(cudaSetDevice(1)); if(can10) cudaDeviceEnablePeerAccess(0,0); cudaGetLastError();
  const size_t SIZES[]={4096ul,65536ul,1048576ul,16777216ul,268435456ul};
  printf("seam,bytes,unit,median,q1,q3,derived_rate_GBps,reps,note\n");
  for(size_t bytes : SIZES){
    void *a=nullptr,*b=nullptr;
    CK(cudaSetDevice(0)); CK(cudaMalloc(&a,bytes)); CK(cudaMemset(a,1,bytes));
    CK(cudaSetDevice(1)); CK(cudaMalloc(&b,bytes));
    cudaEvent_t ev0, ev1; CK(cudaSetDevice(0)); CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));
    std::vector<double> v;
    for(int r=0;r<12;r++){
      CK(cudaEventRecord(ev0));
      CK(cudaMemcpyPeer(b,1,a,0,bytes));
      CK(cudaEventRecord(ev1)); CK(cudaEventSynchronize(ev1));
      float ms=0; CK(cudaEventElapsedTime(&ms,ev0,ev1));
      if(r) v.push_back((double)ms*1e9); // ps
    }
    std::sort(v.begin(),v.end());
    double med=v[v.size()/2], q1=v[v.size()/4], q3=v[3*v.size()/4];
    printf("HBM->PEER_HBM,%zu,ps,%.1f,%.1f,%.1f,%.2f,11,cudaMemcpyPeer over PHB\n",
           bytes, med, q1, q3, bytes/med*1e12/1e9);
    cudaFree(a); CK(cudaSetDevice(1)); cudaFree(b);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
  }
  return 0;
}
