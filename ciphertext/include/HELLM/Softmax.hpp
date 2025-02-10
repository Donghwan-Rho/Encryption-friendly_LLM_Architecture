////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2024 Crypto Lab Inc. All rights reserved.               //
//                                                                            //
// This source code is protected under international copyright law.           //
// All rights reserved and protected by the copyright holders.                //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "HEaaN/HEaaN.hpp"

#include "HELLM/HETensor.hpp"

namespace HELLM::Softmax {

// HETAL//

HEaaN::Real degreeThreePolySolver(const std::vector<HEaaN::Real> &coeffs);

void computeDegreeSevenOddPolyWithSol(const HEaaN::HomEvaluator &eval,
                                      const std::vector<HEaaN::Real> &coeff,
                                      const HEaaN::Real sol,
                                      const HEaaN::Ciphertext &ctx,
                                      HEaaN::Ciphertext &res,
                                      const HEaaN::Real scale = 1.0);

HEaaN::Ciphertext approxDomainExtension(
    const HEaaN::HomEvaluator &eval, const HEaaN::Bootstrapper &btp,
    const HEaaN::Ciphertext &ctxt, const HEaaN::Real base_range,
    const HEaaN::Real extended_range, const HEaaN::Real domain_extension_rate);

HEaaN::Ciphertext
approxDomainExtensionInverse(const HEaaN::HomEvaluator &eval,
                             const HEaaN::Ciphertext &ctxt,
                             const HEaaN::u64 domain_extension_order,
                             const HEaaN::Real domain_extension_rate);

HEaaN::Message genColumnMask(const HEaaN::u64 log_slots,
                             const HEaaN::u64 num_class,
                             const HEaaN::Real scale);

void approxSign(const HEaaN::HomEvaluator &eval, const HEaaN::Bootstrapper &btp,
                const HEaaN::Ciphertext &ctxt, HEaaN::Ciphertext &ctxt_out,
                const HEaaN::u64 num_iter_g, const HEaaN::u64 num_iter_f,
                const HEaaN::Real scale);

void approxMax(const HEaaN::HomEvaluator &eval, const HEaaN::Bootstrapper &btp,
               const HEaaN::Ciphertext &ctxt, HEaaN::Ciphertext &ctxt_out,
               const int num_data_par, const int num_group);

void approxMax_Parallel(const HEaaN::HomEvaluator &eval,
                        const HEaaN::Bootstrapper &btp,
                        const std::vector<HEaaN::Ciphertext> &ctxt,
                        HEaaN::Ciphertext &ctxt_out, const int num_data_par,
                        const int num_group);

void approxExpWide(const HEaaN::HomEvaluator &eval,
                   const HEaaN::Bootstrapper &btp,
                   const HEaaN::Ciphertext &ctxt, HEaaN::Ciphertext &ctxt_out);

void approxInv(const HEaaN::HomEvaluator &eval, const HEaaN::Bootstrapper &btp,
               const HEaaN::Ciphertext &ctxt, HEaaN::Ciphertext &ctxt_out,
               const HEaaN::Real initial, const HEaaN::u64 num_iter,
               const HEaaN::u64 num_log_slots);

void approxSoftmaxWide_Parallel(
    const HEaaN::HomEvaluator &eval, const HEaaN::Bootstrapper &btp,
    const HEaaN::Decryptor &dec, const HEaaN::SecretKey &sk,
    std::vector<CtxtTensor> &ctxt, std::vector<CtxtTensor> &ctxt_out,
    const HEaaN::u64 num_data, const HEaaN::u64 M, const int layer_n);

//////////////////
////// BERT //////
//////////////////

HEaaN::Ciphertext linearTransform_exp(const HEaaN::HomEvaluator &eval,
                                      const HEaaN::Ciphertext &ctxt,
                                      const int k, const int N);

HEaaN::Message generateMask_V2(const HEaaN::u64 log_slots,
                               const HEaaN::u64 num_data);

HEaaN::Message generateMask_garbage(const HEaaN::u64 log_slots,
                                    const HEaaN::u64 num_data);

void bootstrap_02(const HEaaN::HomEvaluator &eval,
                  const HEaaN::Bootstrapper &btp, HEaaN::Ciphertext &op);

void bootstrap2_02(const HEaaN::HomEvaluator &eval,
                   const HEaaN::Bootstrapper &btp, HEaaN::Ciphertext &op1,
                   HEaaN::Ciphertext &op2);

void rotateSum_masking(const HEaaN::HomEvaluator &eval,
                       const HEaaN::Ciphertext &op, HEaaN::Ciphertext &res,
                       const HEaaN::Message &mask, const HEaaN::u64 num_data);

void rotateSum_masking_first(const HEaaN::HomEvaluator &eval,
                             const HEaaN::Ciphertext &op,
                             HEaaN::Ciphertext &res, const HEaaN::Message &mask,
                             const HEaaN::u64 num_data);

void rotateSum_masking_after(const HEaaN::HomEvaluator &eval,
                             const HEaaN::Ciphertext &op,
                             HEaaN::Ciphertext &res, const HEaaN::Message &mask,
                             const HEaaN::u64 num_data);

void Softmax_UNI_approxExp_BSGS(const HEaaN::HomEvaluator &eval,
                                const HEaaN::Bootstrapper &btp,
                                const HEaaN::Ciphertext &op,
                                HEaaN::Ciphertext &res, const int k,
                                const int N);

void Softmax_UNI_approxInverseSqrt_63_BSGS(const HEaaN::HomEvaluator &eval,
                                           const HEaaN::Bootstrapper &btp,
                                           const HEaaN::Ciphertext &op,
                                           HEaaN::Ciphertext &res);

void Softmax_UNI128_Scaled_approxInverseSqrt_15_BSGS_GH(
    const HEaaN::HomEvaluator &eval, const HEaaN::Bootstrapper &btp,
    double scale, const HEaaN::Ciphertext &op, HEaaN::Ciphertext &res);

void Softmax_UNI128_LAST_approxInverseSqrt_31_BSGS(
    const HEaaN::HomEvaluator &eval, const HEaaN::Bootstrapper &btp,
    const HEaaN::Ciphertext &op, HEaaN::Ciphertext &res);

// N = 128, range= [-256,256]
void Softmax_128_512(const HEaaN::HomEvaluator &eval,
                     const HEaaN::Bootstrapper &btp,
                     const HEaaN::Ciphertext &op, HEaaN::Ciphertext &res);

void Softmax_128_512_Parallel(const HEaaN::HomEvaluator &eval,
                              const HEaaN::Bootstrapper &btp,
                              const std::vector<HELLM::CtxtTensor> &op,
                              std::vector<HELLM::CtxtTensor> &res);

} // namespace HELLM::Softmax
