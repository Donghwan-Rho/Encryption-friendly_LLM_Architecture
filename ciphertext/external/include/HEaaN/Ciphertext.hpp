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

#include "HEaaN/Context.hpp"
#include "HEaaN/EncryptionType.hpp"
#include "HEaaN/HEaaNExport.hpp"
#include "HEaaN/Pointer.hpp"
#include "HEaaN/Real.hpp"
#include "HEaaN/device/Device.hpp"

namespace HEaaN {
template <EncryptionType> class CiphertextImpl;
class Polynomial;

///
///@brief A class of RLWE/MLWE ciphertexts each of which contains a vector of
/// polynomials of length > 1.
///
template <EncryptionType enc_type> class HEAAN_API CiphertextBase {
    friend class BootstrapperImpl;
    friend class RingPacker;

public:
    explicit CiphertextBase(const Context &context, bool is_extended = false);

    ParameterPreset getParameterPreset() const;

    ///@brief Get the number of polynomials in a ciphertext
    u64 getNumPoly() const;

    ///@brief Get the number of polynomials in a ciphertext.
    ///@deprecated In favor of `getNumPoly()`.
    [[deprecated("We will use a more specific name for this function; use "
                 "getNumPoly() instead.")]] u64
    getSize() const {
        return getNumPoly();
    }

    ///@brief Set the number of polynomials of a ciphertext
    ///@throws if the number is less than 2.
    void setNumPoly(u64 size);

    ///@brief Set the number of polynomials of a ciphertext
    ///@throws if the number is less than 2.
    ///@deprecated In favor of `setNumPoly()`
    [[deprecated("We will use a more specific name for this function; use "
                 "setNumPoly() instead.")]] void
    setSize(u64 size) {
        setNumPoly(size);
    }

    ///@brief Set log(number of slots) of a ciphertext
    ///@param[in] log_slots
    void setLogSlots(u64 log_slots);

    ///@brief Get log(number of slots) of a ciphertext
    ///@returns log(number of slots)
    u64 getLogSlots() const;
    u64 getNumberOfSlots() const;

    ///@brief Get prime of current level
    u64 getCurrentPrime() const;

    ///@brief Get scale factor of current level
    Real getCurrentScaleFactor() const;

    ///@brief Get the i-th part of ciphertext.
    Polynomial &getPoly(u64 i);
    ///@brief Get the i-th part of ciphertext.
    const Polynomial &getPoly(u64 i) const;
    ///@brief Get the @p level -th data of the i-th part.
    ///@details the @p level -th data is a modulo q_{level} information of the
    /// polynomial
    u64 *getPolyData(u64 i, u64 level) const;

    ///@brief True if it is a mod-up ciphertext, False otherwise
    bool isModUp() const;

    ///@brief Save a ciphertext to a file
    void save(const std::string &path) const;

    ///@brief Save a ciphertext
    void save(std::ostream &stream) const;

    ///@brief Load a ciphertext from a file
    void load(const std::string &path);

    ///@brief Load a ciphertext
    void load(std::istream &stream);

    ///@brief Rescaling flag
    ///@returns The amount of extra deltas multiplied.
    int getRescaleCounter() const;

    ///@brief set rescale counter
    void setRescaleCounter(int r_counter);

    ///@brief Get level of a cipherext.
    ///@returns The current level of the ciphertext.
    u64 getLevel() const;

    ///@brief Set level of a ciphertext.
    ///@param[in] level
    void setLevel(u64 level);

    ///@brief Get device which a ciphertext reside in.
    ///@returns The device in which the ciphertext resides
    const Device &getDevice() const;

    ///@brief Send a ciphertext to given device.
    ///@param[in] device
    void to(const Device &device);

    ///@brief Allocate memory for a ciphertext at given device.
    ///@param[in] device
    void allocate(const Device &device);

    /// @brief Get whether the ciphertext is encoding its message on slots or
    /// coefficients
    EncodingType getEncodingType() const;

    /// @brief Set whether the ciphertext is encoding its message on slots or
    /// coefficients
    void setEncodingType(EncodingType encode_type);

    ///@brief Get Context context
    const Context &getContext() const;

private:
    Pointer<CiphertextImpl<enc_type>> impl_;

    ///@brief Construct a CiphertextBase containing polynomial of same value
    /// only valid when the input ciphertext constructed for context
    /// sharing primes
    explicit CiphertextBase(const Context &context, const CiphertextBase &ctxt);
};

using Ciphertext = CiphertextBase<EncryptionType::RLWE>;
using MLWECiphertext = CiphertextBase<EncryptionType::MLWE>;

} // namespace HEaaN
