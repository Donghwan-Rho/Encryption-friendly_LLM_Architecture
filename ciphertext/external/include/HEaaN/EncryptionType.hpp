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
#include <cstdint>

namespace HEaaN {

/// @brief Metadata to describe how the message of a ciphertext is encoded
/// @details Encoding is map which embeds an element of the message space to a
/// plaintext.
/// SLOT : The message of a ciphertext is a complex vector, of length N/2, where
/// N is the degree of ring R[X] / (X^N + 1). The message is encoded to an
/// element of the ring through (the inverse of) discrete Fourier transform so
/// the element-wise multiplication on the plaintext side could be led to the
/// same operation on the message side.
/// COEFF : The message of a ciphertext is a real vector, of length N, where N
/// is defined same to the SLOT case. The message directly corresponds to each
/// (scaled-down) coefficient of the plaintext. Unlike SLOT case, the
/// element-wise multiplication is not homomorphic, but ring packing method
/// could be performed to obtain ciphertext(s) encrypting (de)composed message
/// by (de)composing the ciphertext.
enum class HEAAN_API EncodingType : uint32_t { SLOT, COEFF };

/// @brief Metadata to describe how the message of a ciphertext is encrypted
/// @details Encryption maps a plaintext, which is an element of the ring R[X] /
/// (X^N + 1), to a pair of elements (RLWE) or a vector (length >= 2) of
/// elements (MLWE).
/// A RLWE ciphertext is encrypted into a form of (b, a) and is decrypted as b +
/// a * s, where a is an uniformly sampled random element of R[X] / (X^N + 1)
/// and s is a secret key. For the exceptional case of a multiplied RLWE
/// ciphertext but yet to be relinearized, the ciphertext may hold more than two
/// elements.
/// For a MLWE ciphertext, a and s are vectors of elements, and the ciphertext
/// (b, a) is decrypted as b + <a, s>. The length of a and s is equal to the
/// rank of Context which constructed the ciphertext with.
enum class HEAAN_API EncryptionType : uint32_t { MLWE, RLWE };

} // namespace HEaaN
