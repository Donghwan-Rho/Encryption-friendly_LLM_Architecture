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

#include "HEaaN/EncryptionType.hpp"
#include "HEaaN/HEaaNExport.hpp"
#include <cstdint>

namespace HEaaN {
///
///@brief Class of Parameter presets.
///@details The first alphabet denotes whether a parameter is full or somewhat
/// homomorphic encryption. Somewhat homomorphic encryption parameters can use
/// only a finite number of multiplications while full homomorphic encryption
/// parameters can use infinitely many multiplications via bootstrapping. The
/// second alphabet denotes the size of log2(N), where N denotes the degree
/// of the base ring. V(Venti), G(Grande), T(Tall), S(Short), D(Demi) represent
/// log2(N) = 17, 16, 15, 14, 13, respectively. For somewhat parameters, a
/// number that comes after these alphabets indicates total available
/// multiplication number.
///
enum class HEAAN_API ParameterPreset : uint32_t {
    FVa, // Depth optimal FV parameter
    FVb, // High precision FV parameter
    FGa, // Precision optimal FG parameter
    FGb, // Depth optimal FG parameter
    FTa, // Depth optimal FT parameter
    FTb, // Precision optimal FT parameter
    ST19,
    ST14,
    ST11,
    ST8,
    ST7,
    SS7,
    SD3,
    CUSTOM, // Parameter preset used to create custom parameters
    FVc,    // Precision optimal FV parameter
    FX,     // Small bootstrappable parameter for test
    /* Reserved parameters;
    those are for development and should not be used */
    FGd,  // FG parameter for experimental sparse secret encapsulation support
    SGd0, // A zero-depth parameter which uses compatible prime with FGd
    SD3D6R2,   // Deg 6 rank 2 parameter that uses same primes to SD3
    FGbL0,     // Deg 16 level 0 parameter that uses same primes to FGb
    FGbD12L0,  // Deg 12 level 0 parameter that uses same primes to FGb
    FGbD8R4L0, // Deg 8 rank 4 level 0 parameter that uses same primes to FGb
    FGbD6R6L0  // Deg 6 rank 6 level 0 parameter that uses same primes to FGb
};

/// @brief Returns the available encryption type of the given parameter preset
/// @param preset
/// @details For a ParameterPreset, only one among the two encryption types,
/// EncryptionType::RLWE and Encryption::MLWE is available. Ciphertexts and
/// SecretKeys are only be able to constructed with the available encryption
/// type by giving it as a template parameter, i.e. Ciphertext<enc_type> and
/// SecretKey<enc_type>.
HEAAN_API EncryptionType getEncryptionType(ParameterPreset preset);

///@brief Returns the parameter preset which is required to perform
/// sparse secret encapsulation on bootstrapping for certain parameters.
///@details The context of the sparse parameter should be constructed
/// and provided to construct modules for the parameters.
/// Sparse Secret Encapsulation is a technique to ease bootstrapping complexity
/// while maintaing homomorphic capacity. For more details, please refer to the
/// paper : <a href="https://eprint.iacr.org/2022/024">Bootstrapping for
/// Approximate Homomorphic Encryption with Negligible Failure-Probability by
/// Using Sparse-Secret Encapsulation</a>
HEAAN_API ParameterPreset getSparseParameterPresetFor(ParameterPreset preset);

} // namespace HEaaN
