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

namespace HELLM::ReLU {

void multWithoutRelin(const HEaaN::HomEvaluator &eval,
                      const HEaaN::Ciphertext &ctxt1,
                      const HEaaN::Ciphertext &ctxt2, HEaaN::Ciphertext &res);

void oddSetUp(const HEaaN::HomEvaluator &eval, HEaaN::Ciphertext &ctxt,
              std::vector<HEaaN::Ciphertext> &oddBS_basis,
              std::vector<HEaaN::Ciphertext> &evenBS_basis,
              std::vector<HEaaN::Ciphertext> &GS_basis, const int k,
              const int l);

void oddBabyStep(const HEaaN::HomEvaluator &eval,
                 const std::vector<HEaaN::Ciphertext> &oddBS_basis,
                 const std::vector<double> &polynomial,
                 HEaaN::Ciphertext &ctxt_result, const int k);

std::vector<double> vectorSlice(const std::vector<double> &input, int a, int b);

void oddGiantStep(const HEaaN::HomEvaluator &eval,
                  const std::vector<HEaaN::Ciphertext> &oddBS_basis,
                  const std::vector<HEaaN::Ciphertext> &GS_basis,
                  const std::vector<double> &polynomial,
                  HEaaN::Ciphertext &ctxt_result, int k, int l);

void evalOddPolynomial(const HEaaN::HomEvaluator &eval, HEaaN::Ciphertext &ctxt,
                       HEaaN::Ciphertext &ctxt_poly,
                       const std::vector<double> &polynomial, int k, int l);

void ApproxReLU(const HEaaN::HomEvaluator &eval, const HEaaN::Bootstrapper &btp,
                HEaaN::Ciphertext &ctxt, HEaaN::Ciphertext &ctxt_relu,
                HEaaN::Ciphertext &ctxt_train);

} // namespace HELLM::ReLU
