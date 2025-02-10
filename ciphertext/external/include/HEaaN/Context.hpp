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
#include <string>
#include <vector>

#include "HEaaN/Integers.hpp"
#include "HEaaN/ParameterPreset.hpp"
#include "HEaaN/SecurityLevel.hpp"
#include "HEaaN/device/Device.hpp"
#include "HEaaNExport.hpp"
#include "Real.hpp"

namespace HEaaN {

class ContextContent;
using Context = std::shared_ptr<ContextContent>;

///@brief Make a context object based on a given parameter preset
///@param[in] preset Parameter preset
///@param[in] cuda_device_ids Optional CUDA device IDs to be used. You can use
/// only the specified cuda devices with this Context.
///@returns The context object generated from the predetermined parameters in
/// the given ParameterPreset.
///@throws RuntimeException if one calls this function with
/// preset == ParameterPreset::CUSTOM.  In order to make sense, one must use
/// the other overloaded `makeContext()` function to specify custom parameters
/// explicitly.
HEAAN_API Context makeContext(const ParameterPreset &preset,
                              const CudaDeviceIds &cuda_device_ids = {});

///@brief Make a context based on custom parameters
///@param[in] log_degree Logarithmic (with base 2) degree of the
/// ciphertexts and the keys.  This means that in this homomorphic encryption
/// context, the polynomials constituting the ciphertexts and the public/secret
/// keys must be elements of the ring R[X] / (X^N + 1), where N is called the
/// degree of the polynomial (or of the ciphertexts).  The value must be >=
/// 10 and <= 20.
///@param[in] chain_length This is the number of primes in the RNS decomposition
/// of each polynomial constituting the ciphertexts or the keys in the current
/// homomorphic encryption context.  There are the base prime (the prime at
/// level 0) and the quantization primes at the higher levels, so chain_length
/// is equal to the sum of the number of base primes (usually this number is 1)
/// and the number of quantization primes.  The value must be <= 50.
///@param[in] bpsize The size of the base prime in bits.  The value must be
/// greater than or equal to qpsize, less than or equal to 61.
///@param[in] qpsize The size of the quantization primes in bits.  The value
/// must be greater than or equal to 36, less than or equal to bpsize.
///@param[in] tpsize The size of the temporary primes in bits. The value must
/// be greater than qpsize + (bpsize - qpsize) / numTP, less than or equal to
/// 61. Note that numTP = chain_length / gadget_rank.
///@param[in] gadget_rank This is the number of decomposed polynomials when one
/// does the "modup" process in the middle of keyswitching process.  More
/// precisely, in the modup process, the maximal ciphertext modulus Q which is
/// also a product of word-sized primes, is divided into gadget_rank number of
/// pieces and the modup process is to be applied to each of these pieces.  This
/// also means that the size of the evaluation keys is roughly gadget_rank times
/// of the fresh ciphertexts.  Using smaller or bigger number of gadget_rank has
/// its advantages and disadvantages: using bigger number means that the
/// polynomial is divided more finely, and this causes bigger total size for the
/// evaluation keys, meanwhile smaller gadget_rank makes the size of the
/// evaluation keys smaller.  However, since the security level of the parameter
/// is determined by the size of each modup polynomial, so in order to retain
/// the same security level, using smaller gadget_rank causes smaller ciphertext
/// modulus size, i.e., less multiplication depth is allowed.  Microsoft SEAL
/// uses the maximum gadget_rank, i.e. equal to chain_length, while HEaaN allows
/// smaller gadget_rank in order to accelerate homomorphic operations and reduce
/// switching key size.
///@param[in] cuda_device_ids Optional CUDA device IDs to be used. You can use
/// only the specified cuda devices with this Context???
///@param[in] rank The length of a=(a_0,a_1,...) where each a_j
/// belongs to the ring R[X] / (X^N + 1). The ciphertext (b, a)
/// decrypts as b+ <a, s>, given that s is another vector of elements of the
/// ring.
///@returns The context object generated from the input parameters.
///@details One must use this function to create a custom "somewhat" parameters,
/// meaning ones with a fixed multiplication depth and without bootstrapping.
HEAAN_API Context makeContext(const u64 log_degree, const u64 chain_length,
                              const u64 bpsize, const u64 qpsize,
                              const u64 tpsize, const u64 gadget_rank,
                              const u64 rank,
                              const CudaDeviceIds &cuda_device_ids = {});

///@brief Make a context based on custom parameters
///@param[in] log_degree Logarithmic (with base 2) degree of the
/// ciphertexts and the keys.  This means that in this homomorphic encryption
/// context, the polynomials constituting the ciphertexts and the public/secret
/// keys must be elements of the ring R[X] / (X^N + 1), where N is called the
/// degree of the polynomial (or of the ciphertexts).  The value must be >=
/// 10 and <= 20.
///@param[in] chain_length This is the number of primes in the RNS decomposition
/// of each polynomial constituting the ciphertexts or the keys in the current
/// homomorphic encryption context.  There are the base prime (the prime at
/// level 0) and the quantization primes at the higher levels, so chain_length
/// is equal to the sum of the number of base primes (usually this number is 1)
/// and the number of quantization primes.  The value must be <= 50.
///@param[in] bpsize The size of the base prime in bits.  The value must be
/// greater than or equal to qpsize, less than or equal to 61.
///@param[in] qpsize The size of the quantization primes in bits.  The value
/// must be greater than or equal to 36, less than or equal to bpsize.
///@param[in] tpsize The size of the temporary primes in bits. The value must
/// be greater than qpsize + (bpsize - qpsize) / numTP, less than or equal to
/// 61. Note that numTP = chain_length / gadget_rank.
///@param[in] gadget_rank This is the number of decomposed polynomials when one
/// does the "modup" process in the middle of keyswitching process.  More
/// precisely, in the modup process, the maximal ciphertext modulus Q which is
/// also a product of word-sized primes, is divided into gadget_rank number of
/// pieces and the modup process is to be applied to each of these pieces.  This
/// also means that the size of the evaluation keys is roughly gadget_rank times
/// of the fresh ciphertexts.  Using smaller or bigger number of gadget_rank has
/// its advantages and disadvantages: using bigger number means that the
/// polynomial is divided more finely, and this causes bigger total size for the
/// evaluation keys, meanwhile smaller gadget_rank makes the size of the
/// evaluation keys smaller.  However, since the security level of the parameter
/// is determined by the size of each modup polynomial, so in order to retain
/// the same security level, using smaller gadget_rank causes smaller ciphertext
/// modulus size, i.e., less multiplication depth is allowed.  Microsoft SEAL
/// uses the maximum gadget_rank, i.e. equal to chain_length, while HEaaN allows
/// smaller gadget_rank in order to accelerate homomorphic operations and reduce
/// switching key size.
///@param[in] cuda_device_ids Optional CUDA device IDs to be used. You can use
/// only the specified cuda devices with this Context.
///@returns The context object generated from the input parameters.
///@details One must use this function to create a custom "somewhat" parameters,
/// meaning ones with a fixed multiplication depth and without bootstrapping.
///@deprecated In the future releases, it is necessary to specify rank parameter
/// in order for making a context object.
[[deprecated("Use makeContext() with additional rank argument.")]] HEAAN_API
    Context
    makeContext(const u64 log_degree, const u64 chain_length, const u64 bpsize,
                const u64 qpsize, const u64 tpsize, const u64 gadget_rank,
                const CudaDeviceIds &cuda_device_ids = {});

///@brief Make a context object from a "context file"
///@param[in] filename It designates the path of the file to be read inside
/// this function.
///@param[in] cuda_device_ids Optional CUDA device IDs to be used. You can use
/// only the specified cuda devices with this Context.
///@returns The generated context object.
///@details A context file is one created by `saveContextToFile`.
///@throws RuntimeException if it fails to open `filename` in read mode.
HEAAN_API Context makeContextFromFile(
    const std::string &filename, const CudaDeviceIds &cuda_device_ids = {});

///@brief Save a context object into a file.
///@param[in] context A context object to be saved.
///@param[in] filename File path to be written.
///@throws RuntimeException if it fails to open `filename` in write mode.
HEAAN_API void saveContextToFile(const Context &context,
                                 const std::string &filename);

///@brief Get the maximal logarithmic (base of 2) number of slots for the given
/// context object.
///@param[in] context
///@details The maximal number of slots is equal to N / 2 if N is the degree
/// of the context.  The basic intention is to use this function with the
/// constructor of the `Message` class, which takes the logarithmic number of
/// slots in order to allocate sufficient memory.
HEAAN_API u64 getLogFullSlots(const Context &context);

///@brief Get the degree of the given context.
///@param[in] context
///@details The degree of context is the degree of modulo polynomial in the
/// ring R[X] / (X^N + 1). This value equals to the two times of maximal number
/// of slots.
HEAAN_API u64 getDegree(const Context &context);

///@brief Get the number of polynomials used to encrypt a(X) part of a
/// ciphertext
///@param[in] context
///@details The MLWE of rank 1 is the same as RLWE. In the case of MLWE, the
/// rank is eqaul to the number of polynomials of a(X) part of a ciphertext.
/// However, there can be more than one polynomial for a(X) part of RLWE
/// ciphertexts which are not relinearized after multiplication.
HEAAN_API u64 getRank(const Context &context);

///@brief Get the level of a fresh ciphertext, which is the maximum level that
/// users can encrypt a ciphertext to.
///@param[in] context
///@details For somewhat homomorphic encryption parameters, it is equal to the
/// maximum level of Q part. For some full homomorphic encryption parameters,
/// however, some levels are reserved for bootstrapping and encrypting a
/// ciphertext over this level is not allowed.
HEAAN_API u64 getEncryptionLevel(const Context &context);

///@brief Get the default list of scale factors
///@param[in] context
///@details The i-th element corresponds to level i. HEaaN uses fixed scale
/// factor system, which fixes the scale factor with respect to each level.
/// It helps managing scale factor properly, and saves some level.
HEAAN_API std::vector<Real> getDefaultScaleFactorList(const Context &context);

///@brief Get the list of primes
///@param[in] context
///@details The i-th element corresponds to level i
HEAAN_API std::vector<u64> getPrimeList(const Context &context);

///@brief Get whether the bootstrapping is supported.
///@param[in] context
HEAAN_API bool isBootstrappableParameter(const Context &context);

///@brief Get whether the given context supports sparse secret encapsulation or
/// not.
///@param[in] context
///@details Returns true for the context using a parameter with dense hamming
/// weight on secret key, which can be key-switched to a corresponding parameter
/// with a sparse secret key during bootstrapping.
HEAAN_API bool isSparseSecretEncapsulationSupported(const Context &context);

///@brief Get whether the bootstrapping for extended range supported.
///@param[in] context
HEAAN_API bool isExtendedBootstrapSupported(const Context &context);

///@brief Get the list of rotation key index for the bootstrapping process
///@param[in] context
///@param[in] log_slots
HEAAN_API std::set<i64> getRotIndicesForBootstrap(const Context &context,
                                                  u64 log_slots);

///@brief Get the security level of the given context
///@param[in] context Context object
///@details The security level is
/// chosen according to the [homomorphic encryption standard
/// documentation](http://homomorphicencryption.org/wp-content/uploads/2018/11/HomomorphicEncryptionStandardv1.1.pdf),
/// Table 1, distribution (-1,1) (ternary uniform with elements -1, 0 and 1) and
/// CryptoLab's own [experimental
/// results](https://deciduous-cause-137.notion.site/Security-Level-of-Parameters-3ecb6810c57843e4b55e788f34b36108).
///@returns The security level as described in HEaaN::SecurityLevel
HEAAN_API SecurityLevel getSecurityLevel(const Context &context);

} // namespace HEaaN
