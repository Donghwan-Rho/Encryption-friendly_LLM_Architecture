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

#include <memory>

#include "HEaaN/Context.hpp"
#include "HEaaN/HEaaNExport.hpp"
#include "HEaaN/device/Device.hpp"

namespace HEaaN {

class EncryptionKey;
class EvaluationKey;
class SparseSecretEncapsulationKey;
class KeyPackImpl;

///
///@brief Class managing public keys
///
class HEAAN_API KeyPack {
    friend class KeyGenerator;

public:
    ///@brief Create a KeyPack object
    explicit KeyPack(const Context &context);

    ///@brief Create a KeyPack object which is capable of holding key for sparse
    /// secret encapsulation
    explicit KeyPack(const Context &context, const Context &context_sparse);

    ///@brief Create a KeyPack object with a designated directory for saved keys
    ///@throws RuntimeException if either keyDirPath or keyDirPath + "/PK" (the
    /// designated location for public keys) is not a valid directory
    explicit KeyPack(const Context &context, const std::string &key_dir_path);

    ///@brief Create a KeyPack object with a designated directory for saved
    /// keys,
    /// for parameter supporting sparse secret encapsulation
    ///@throws RuntimeException if either keyDirPath or keyDirPath + "/PK" (the
    /// designated location for public keys) is not a valid directory
    ///@throws RuntimeException if context_sparse is not a context constructed
    /// with the corresponding sparse parameter of which constructed context.
    /// Please refer to SparseParameterPresetFor() on ParameterPreset.hpp
    /// for the sparse parameters.
    explicit KeyPack(const Context &context, const Context &context_sparse,
                     const std::string &key_dir_path);

    ///////////////////////////////
    // Functions for key loading //
    ///////////////////////////////

    ///@brief Load encryption key from a file to the memory
    ///@details If the key is already loaded in the memory or it cannot find the
    /// key file, this function does nothing.
    /// If a stream is specified, load from the stream.
    void loadEncKey(void);
    void loadEncKey(std::istream &stream);

    ///@brief Load multiplication key from a file to the memory
    ///@details If the key is already loaded in the memory or it cannot find the
    /// key file, this function does nothing.
    /// If stream is specified, load from the stream.
    void loadMultKey(void);
    void loadMultKey(std::istream &stream);

    ///@brief Load conjugation key from file to memory
    ///@details If the key is already loaded in the memory or it cannot find the
    /// key file, this function does nothing.
    /// If stream is specified, load from the stream.
    void loadConjKey(void);
    void loadConjKey(std::istream &stream);

    ///@brief Load left rotation key (by rot) from file to memory
    ///@param[in] rot Rotation index (left rotation)
    ///@details If the key is already loaded in the memory or it cannot find the
    /// key file, this function does nothing.
    /// If stream is specified, load from the stream.
    void loadLeftRotKey(const u64 rot);
    void loadLeftRotKey(const u64 rot, std::istream &stream);

    ///@brief Load right rotation key (by rot) from file to memory
    ///@param[in] rot Rotation index (right rotation)
    ///@details If the key is already loaded in the memory or it cannot find the
    /// key file, this function does nothing.
    /// If stream is specified, load from the stream.
    void loadRightRotKey(const u64 rot);
    void loadRightRotKey(const u64 rot, std::istream &stream);

    ///@brief Load key for sparse secret encapsulation from file to memory
    ///@details If the key is already loaded in the memory or it cannot find the
    /// key file, this function does nothing.
    /// If stream is specified, load from the stream.
    void loadSparseSecretEncapsulationKey(void);
    void loadSparseSecretEncapsulationKey(std::istream &stream);

    ////////////////////////////////////////////////////
    // Functions for checking keys in the main memory //
    ////////////////////////////////////////////////////

    ///@brief Check whether the Encryption Key is loaded in memory
    bool isEncKeyLoaded(void) const;

    ///@brief Check whether the Multiplication Key is loaded in memory
    bool isMultKeyLoaded(void) const;

    ///@brief Check whether the Left Rotation Key is loaded in memory
    ///@param[in] rot Rotation Key index.
    bool isLeftRotKeyLoaded(const u64 rot) const;

    ///@brief Check whether the Right Rotation Key is loaded in memory
    ///@param[in] rot Rotation Key index.
    bool isRightRotKeyLoaded(const u64 rot) const;

    ///@brief Check whether the Conjugation Key is loaded in memory
    bool isConjKeyLoaded(void) const;

    ///@brief Check whether the Left Rotation Key file is accessible
    bool isLeftRotKeyFileAvailable(const u64 rot) const;

    ///@brief Check whether the sparse secret encapsulation key is loaded in
    /// memory
    bool isSparseSecretEncapsulationKeyLoaded() const;

    /////////////////
    // Key Getters //
    /////////////////

    ///@brief Get a pointer to the encryption key
    ///@details This function first checks if the Encryption Key is loaded in
    /// the memory (isEncKeyLoaded()).  If it fails, then it tries to load the
    /// key from the file in the path, and if it fails again, then this function
    /// finally returns nullptr.
    std::shared_ptr<EncryptionKey> getEncKey(void) const;

    ///@brief Get a pointer to the multiplication key
    ///@details This function first checks if the Multiplication Key is loaded
    /// in the memory (isMultKeyLoaded()).  If it fails, then it tries to load
    /// the key from the file in the path, and if it fails again, then this
    /// function finally returns nullptr.
    std::shared_ptr<EvaluationKey> getMultKey(void) const;

    ///@brief Get a pointer to the left rotation key
    ///@param[in] rot The rotation index
    ///@details This function first checks if the Rotation Key is loaded in the
    /// memory (isLeftRotKeyLoaded()).  If it fails, then it tries to load the
    /// key from the file in the path, and if it fails again, then this function
    /// finally returns nullptr.
    std::shared_ptr<EvaluationKey> getLeftRotKey(const u64 rot) const;

    ///@brief Get a pointer to the right roation key
    ///@param[in] rot Thhe rotation index
    ///@details This function first checks if the Rotation Key is loaded in the
    /// memory (isRightRotKeyLoaded()).  If it fails, then it tries to load the
    /// key from the file in the path, and if it fails again, then this function
    /// finally returns nullptr.
    std::shared_ptr<EvaluationKey> getRightRotKey(const u64 rot) const;

    ///@brief Get a pointer to the conjugation key
    ///@details This function first checks if the Conjugation Key is loaded in
    /// the memory (isConjKeyLoaded()).  If it fails, then it tries to load the
    /// key from the file in the path, and if it fails again, then this function
    /// finally returns nullptr.
    std::shared_ptr<EvaluationKey> getConjKey(void) const;

    ///@brief Get a pointer to the key for sparse secret encapsulation
    ///@details This function first checks if sparse secret encryption keys are
    /// loaded in the memory (isSparseSecretEncapsulationKeyLoaded()).  If it
    /// fails, then it tries to load the keys from the file in the path, and if
    /// it fails again, then this function finally returns nullptr.
    std::shared_ptr<SparseSecretEncapsulationKey>
    getSparseSecretEncapsulationKey(void) const;

    /////////////////////////////////////
    // Miscellaneous/Utility functions //
    /////////////////////////////////////

    ///@brief Set the key directory for the keys in the current object
    void setKeyDirPath(const std::string &key_dir_path);

    ///@brief Make all the keys only usable at `device`. This can be useful
    /// in memory-constrained environment since it frees the spaces allocated
    /// for keys in other devices.
    void to(const Device &device);

    ///@brief Save multiplication, conjugation, encryption, sparse secret
    /// encapsulation, and all rotation keys to @p stream .
    void save(std::ostream &stream) const;

    ///@brief Load multiplication, conjugation, encryption, sparse secret
    /// encapsulation, and all rotation keys from @p stream . The context used
    /// to construct this `KeyPack` should match that of the keys in the stream;
    /// otherwise, it is undefined behavior.
    void load(std::istream &stream);

    ///@brief Save multiplication, conjugation, encryption, sparse secret
    /// encapsulation, and all rotation keys under a directory.
    void save(const std::string &dir_path) const;

    ///@brief Make all the keys usable at `device`.
    void copyTo(const Device &device);

private:
    ///@brief Pointer to the implementation class
    std::shared_ptr<KeyPackImpl> impl_;

    ///@brief Maximal number of slots for the current context object.
    u64 num_max_slots_;
};

///@brief Save @p key to a given @p stream.
HEAAN_API void save(const EvaluationKey &key, std::ostream &stream);
HEAAN_API void save(const EncryptionKey &key, std::ostream &stream);
HEAAN_API void save(const SparseSecretEncapsulationKey &key,
                    std::ostream &stream);

///@brief Load @p key from a given @p stream.
HEAAN_API void load(EvaluationKey &key, std::istream &stream);
HEAAN_API void load(EncryptionKey &key, std::istream &stream);
HEAAN_API void load(SparseSecretEncapsulationKey &key, std::istream &stream);

} // namespace HEaaN
