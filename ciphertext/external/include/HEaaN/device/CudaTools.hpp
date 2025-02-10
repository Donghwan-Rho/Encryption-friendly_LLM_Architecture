////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2024 Crypto Lab Inc.                                    //
//                                                                            //
// - This file is part of HEaaN homomorphic encryption library.               //
// - HEaaN cannot be copied and/or distributed without the express permission //
//  of Crypto Lab Inc.                                                        //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "HEaaN/HEaaNExport.hpp"
#include "HEaaN/Integers.hpp"
#include <utility>

namespace HEaaN::CudaTools {

HEAAN_API bool isAvailable();
HEAAN_API void cudaDeviceSynchronize();
HEAAN_API int cudaGetDevice();
HEAAN_API int cudaGetDeviceCount();
HEAAN_API void cudaSetDevice(int device_id);
HEAAN_API void nvtxPush(const char *msg);
HEAAN_API void nvtxPop(void);
// Return [free, total] memory bytes of the current CUDA device.
HEAAN_API std::pair<u64, u64> getCudaMemoryInfo();

} // namespace HEaaN::CudaTools
