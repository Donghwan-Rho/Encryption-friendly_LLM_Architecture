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

#include "EncryptionType.hpp"
#include "HEaaN/Context.hpp"
#include "HEaaN/HEaaNExport.hpp"
#include "HEaaN/Pointer.hpp"
#include "HEaaN/device/Device.hpp"

namespace HEaaN {
template <EncryptionType> class SecretKeyImpl;
class Polynomial;

///
///@brief RLWE/MLWE Secret key class
///

template <EncryptionType enc_type> class HEAAN_API SecretKeyBase {
public:
    using Coefficients = int *;
    ///@brief Generate random secret key
    explicit SecretKeyBase(const Context &context);
    ///@brief Load secret key from stream
    ///@details The key can be loaded regardless of whether the stream is saving
    /// the full key or its seed only.
    explicit SecretKeyBase(const Context &context, std::istream &stream);
    ///@brief Load secret key from file
    ///@details The key can be loaded regardless of whether the stream is saving
    /// the full key or its seed only.
    explicit SecretKeyBase(const Context &context,
                           const std::string &key_dir_path);

    ///@brief Generate a secret key whose coefficients are
    /// copied from @p coefficients and fits @p context.
    ///@details The key cannot be saved or loaded by its seed.
    /// Instead of using a uniform random bit generator, it fills the secret
    /// key's coefficients with integers from coefficients[0] to
    /// coefficients[context->getDegree() * @p num_poly - 1].
    explicit SecretKeyBase(const Context &context,
                           const Coefficients &coefficients);

    ///@brief Save a secret key to file
    void save(const std::string &path) const;
    ///@brief Save a secret key to stream
    void save(std::ostream &stream) const;

    ///@brief Save the seed of a secret key to file
    ///@details The seed can reproduce the key under different parameter
    ///(Context) too.
    void saveSeedOnly(const std::string &path) const;
    ///@brief Save the seed of a secret key to stream
    ///@details The seed can reproduce the key under different parameter
    ///(Context) too.
    void saveSeedOnly(std::ostream &stream) const;

    ///@brief Get Context context
    const Context &getContext() const;

    ///@brief Get i-th sx part of secret key.
    Polynomial &getSx(u64 i = 0);
    ///@brief Get const i-th sx part of secret key.
    const Polynomial &getSx(u64 i = 0) const;

    ///@brief Get integer representation of coefficients.
    Coefficients getCoefficients() const;

    ///@brief Get device a secret key reside in.
    ///@returns The device in which the secret key resides
    const Device &getDevice() const;

    ///@brief Send a secret key to given device.
    ///@param[in] device
    void to(const Device &device);

private:
    Pointer<SecretKeyImpl<enc_type>> impl_;
};

using SecretKey = SecretKeyBase<EncryptionType::RLWE>;
using MLWESecretKey = SecretKeyBase<EncryptionType::MLWE>;

} // namespace HEaaN
