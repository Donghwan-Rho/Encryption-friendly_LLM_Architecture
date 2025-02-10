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

#include <array>
#include <ostream>

namespace HEaaN {

constexpr const u64 SEED_SIZE_IN_BYTES{32};
using SeedType = std::array<u64, SEED_SIZE_IN_BYTES / 8>; // 256-bit seed

///@brief Get the random seed of the current HEaaN instance
///@returns The internal random seed
HEAAN_API SeedType getSeed();

///@brief Set the random seed
///@param[in] new_global_seed The internal random seed is to be set with this
/// parameter.
HEAAN_API void setSeed(const SeedType &new_global_seed);

} // namespace HEaaN
