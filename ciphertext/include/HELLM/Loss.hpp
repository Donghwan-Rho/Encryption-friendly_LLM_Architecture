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

namespace HELLM::Loss {

void approxExp15(const HEaaN::HomEvaluator &eval,
                 const HEaaN::Bootstrapper &btp, const HEaaN::Ciphertext &op,
                 HEaaN::Ciphertext &res);

void approxExp15_SST2(const HEaaN::HomEvaluator &eval,
                      const HEaaN::Bootstrapper &btp,
                      const HEaaN::Ciphertext &op, HEaaN::Ciphertext &res);

void approxInv63(const HEaaN::HomEvaluator &eval,
                 const HEaaN::Bootstrapper &btp, const HEaaN::Ciphertext &op,
                 HEaaN::Ciphertext &res);

void approxInv15_SST2(const HEaaN::HomEvaluator &eval,
                      const HEaaN::Bootstrapper &btp,
                      const HEaaN::Ciphertext &op, HEaaN::Ciphertext &res);

void approxMax(const HEaaN::HomEvaluator &eval, const HEaaN::Bootstrapper &btp,
               const HEaaN::Ciphertext &op, HEaaN::Ciphertext &res);

} // namespace HELLM::Loss
