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

#include "HEaaN/Ciphertext.hpp"
#include "HEaaN/Context.hpp"
#include "HEaaN/HEaaNExport.hpp"
#include "HEaaN/Message.hpp"
#include "HEaaN/SecretKey.hpp"

namespace HEaaN {

class KeyPack;
class Plaintext;

///
///@brief Abstract entity for encrypting messages into ciphertexts
///
class HEAAN_API Encryptor {
public:
    explicit Encryptor(const Context &context);

    ///@brief Encrypt a message using a secret key, to the maximal
    /// supported level, to rescale counter zero.
    ///@tparam enc_type
    ///@param[in] msg
    ///@param[in] sk
    ///@param[out] ctxt
    ///@throws RuntimeException if msg and key are at different devices.
    template <EncryptionType enc_type>
    void encrypt(const Message &msg, const SecretKeyBase<enc_type> &sk,
                 CiphertextBase<enc_type> &ctxt) const;

    ///@brief Encrypt a message using a secret key, to a certain level, to a
    /// certain rescale counter.
    ///@tparam enc_type
    ///@param[in] msg
    ///@param[in] sk
    ///@param[in] level
    ///@param[in] r_counter
    ///@param[out] ctxt
    ///@throws RuntimeException if msg and key are at different devices.
    template <EncryptionType enc_type>
    void encrypt(const Message &msg, const SecretKeyBase<enc_type> &sk,
                 CiphertextBase<enc_type> &ctxt, u64 level,
                 int r_counter = 0) const;

    ///@brief Encrypt a message using a keypack (Public key encryption),
    /// to the maximal supported level, to rescale counter zero.
    ///@param[in] msg
    ///@param[in] keypack
    ///@param[out] ctxt
    ///@throws RuntimeException if msg and key are at different devices.
    void encrypt(const Message &msg, const KeyPack &keypack,
                 Ciphertext &ctxt) const;

    ///@brief Encrypt a message using a keypack (Public key encryption),
    /// to a certain level, to a certain rescale counter.
    ///@param[in] msg
    ///@param[in] keypack
    ///@param[in] level
    ///@param[in] r_counter
    ///@param[out] ctxt
    ///@throws RuntimeException if msg and key are at different devices.
    void encrypt(const Message &msg, const KeyPack &keypack, Ciphertext &ctxt,
                 u64 level, int r_counter = 0) const;

    ///@brief Encrypt a plaintext using a secret key
    ///@tparam enc_type
    ///@param[in] ptxt
    ///@param[in] sk
    ///@param[out] ctxt
    ///@details compute (a, -as + e + m)
    ///@throws RuntimeException if ptxt and key are at different devices.
    template <EncryptionType enc_type>
    void encrypt(const Plaintext &ptxt, const SecretKeyBase<enc_type> &sk,
                 CiphertextBase<enc_type> &ctxt) const;

    ///@brief Encrypt a plaintext using an encryption key
    ///@param[in] ptxt
    ///@param[in] keypack
    ///@param[out] ctxt
    ///@details compute (va + e_1, vb + e_2 + m) where (a, b) = (a, -as + e_0)
    /// is an encryption key
    ///@throws RuntimeException if ptxt and key are at different devices.
    void encrypt(const Plaintext &ptxt, const KeyPack &keypack,
                 Ciphertext &ctxt) const;

    ///@brief Encrypt a coeff message using a secret key, to the maximal
    /// supported level, to rescale counter zero.
    ///@tparam enc_type
    ///@param[in] msg
    ///@param[in] sk
    ///@param[out] ctxt
    template <EncryptionType enc_type>
    void encrypt(const CoeffMessage &msg, const SecretKeyBase<enc_type> &sk,
                 CiphertextBase<enc_type> &ctxt) const;

    ///@brief Encrypt a coeff message using a secret key, to a certain level, to
    /// a certain rescale counter.
    ///@tparam enc_type
    ///@param[in] msg
    ///@param[in] sk
    ///@param[in] level
    ///@param[in] r_counter
    ///@param[out] ctxt
    template <EncryptionType enc_type>
    void encrypt(const CoeffMessage &msg, const SecretKeyBase<enc_type> &sk,
                 CiphertextBase<enc_type> &ctxt, u64 level,
                 int r_counter = 0) const;

private:
    ///@brief A context with which Encryptor is associated
    const Context context_;
};
} // namespace HEaaN
