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

namespace HEaaN {

///
///@brief Enum class for security levels
///@details This enum class represents the security level of the homomorphic
/// encryption parameters. The prefix `Classical` in the enum value names
/// denotes that the security level in question is measured assuming classical
/// computers, not quantum ones.
///
enum class HEAAN_API SecurityLevel : int {
    None = 0, ///< signifies that no security level assumptions are imposed,
              ///< suitable for parameters for experimental purposes.
    Classical128 = 128, ///< at least 128-bit security level classically.
    Classical192 = 192, ///< at least 192-bit security level classically.
    Classical256 = 256, ///< at least 256-bit security level classically.
};
} // namespace HEaaN
