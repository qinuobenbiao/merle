cmake_minimum_required(VERSION 3.24)
set(WAH_CUDA_ARCH "86" CACHE STRING "CUDA architecture for WAH")
project(wahAttempt VERSION 0.0.1 LANGUAGES C CUDA)

add_executable(wahConv host/wahConv.c host/wahCpu.c host/wahConvMain.c)
add_executable(wahProfileGPU
  host/wahConv.c host/wahCpu.c
  wahGpuCmprs.cu wahGpuDirect.cu Andrzejewski.cu
  wahDeviceStruct.cu profileDat.cu profileMain.cu runTestCases.cu
  host/roaring.c
  # teb/thirdparty/roaring/roaring.c
)
add_executable(wahSynthGPU
  host/nbitRandGen.c host/wahRandGen.c
  wahGpuCmprs.cu wahGpuDirect.cu Andrzejewski.cu
  synthMain.cu
)

target_link_libraries(wahProfileGPU PUBLIC cuda)
target_include_directories(wahProfileGPU PUBLIC /opt/cuda/include)
set_target_properties(wahProfileGPU PROPERTIES
  CUDA_ARCHITECTURES ${WAH_CUDA_ARCH}
)
target_link_libraries(wahSynthGPU PUBLIC cuda)
set_target_properties(wahSynthGPU PROPERTIES
  CUDA_ARCHITECTURES ${WAH_CUDA_ARCH}
)

set(C_FLAGS -Wall -Wextra)
set(CUDA_FLAGS
  -Wall -Wextra
  --extended-lambda 
  --expt-relaxed-constexpr
  --use_fast_math
  $<$<CXX_COMPILER_ID:GNU>:-O3> # Host optimize-level
  # --verbose
  # --debug # Host debug
  # --device-debug # Device debug
)

target_compile_definitions(wahProfileGPU PUBLIC __WAH_POPCNT)
set(C_FLAGS -Wall -Wextra -march=x86-64-v3)
target_compile_options(wahConv PUBLIC
  $<$<COMPILE_LANGUAGE:C>:${C_FLAGS}>
  $<$<COMPILE_LANGUAGE:CXX>:${C_FLAGS}>
)
target_compile_options(wahProfileGPU PUBLIC
  $<$<COMPILE_LANGUAGE:C>:${C_FLAGS}>
  $<$<COMPILE_LANGUAGE:CXX>:${C_FLAGS}>
  $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>
)
#set_property(TARGET wahProfileGPU PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_options(wahSynthGPU PUBLIC
  $<$<COMPILE_LANGUAGE:C>:${C_FLAGS}>
  $<$<COMPILE_LANGUAGE:CXX>:${C_FLAGS}>
  $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>
)
#set_property(TARGET wahSynthGPU PROPERTY POSITION_INDEPENDENT_CODE ON)
