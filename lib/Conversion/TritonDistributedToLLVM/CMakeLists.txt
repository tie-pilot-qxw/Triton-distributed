add_triton_library(TritonDistributedToLLVM
    NVIDIA/ConvertTritonDistributedToLLVM.cpp
    NVIDIA/DistributedOpToLLVM.cpp
    SIMTOpToLLVM.cpp
    AMD/LibDeviceToLLVM.cpp
    AMD/DistributedOpToLLVM.cpp
    AMD/ConvertAMDDistributedToLLVM.cpp

    DEPENDS
    TritonDistributedToLLVMConversionPassIncGen
    NVGPUAttrDefsIncGen

    LINK_LIBS PUBLIC
    TritonGPUToLLVM
    DistributedIR
    SIMTIR
    TritonAMDGPUIR
    TritonAMDGPUToLLVM
    TritonAMDGPUDialectToLLVM
    TritonAMDGPUTransforms
    TritonAMDUtils
    TritonAMDAnalysis
)