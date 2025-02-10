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

namespace HELLM::Exp {

void exp_iter(const HEaaN::HomEvaluator &eval, const HEaaN::Bootstrapper &btp,
              const HEaaN::Ciphertext &op, HEaaN::Ciphertext &res,
              const int num_iter);

void exp_iter_Parallel(const HEaaN::HomEvaluator &eval,
                       const HEaaN::Bootstrapper &btp,
                       std::vector<HELLM::CtxtTensor> &op, const int num_iter);

HEaaN::Ciphertext approxDomainExtension(
    const HEaaN::HomEvaluator &eval, const HEaaN::Bootstrapper &btp,
    const HEaaN::Ciphertext &ctxt, const HEaaN::Real base_range,
    const HEaaN::Real extended_range, const HEaaN::Real domain_extension_rate);

HEaaN::Ciphertext
approxDomainExtensionInverse(const HEaaN::HomEvaluator &eval,
                             const HEaaN::Ciphertext &ctxt,
                             const HEaaN::u64 domain_extension_order,
                             const HEaaN::Real domain_extension_rate);

void approxExpWide(const HEaaN::HomEvaluator &eval,
                   const HEaaN::Bootstrapper &btp,
                   const HEaaN::Ciphertext &ctxt, HEaaN::Ciphertext &ctxt_out);

} // namespace HELLM::Exp
