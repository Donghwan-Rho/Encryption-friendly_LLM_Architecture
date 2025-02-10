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

namespace HELLM::Tanh {

void approxTanh(const HEaaN::HomEvaluator &eval, const HEaaN::Bootstrapper &btp,
                const HEaaN::Ciphertext &op, HEaaN::Ciphertext &res,
                const int layer_n);

void approxTanh_wide_12(const HEaaN::HomEvaluator &eval,
                        const HEaaN::Bootstrapper &btp,
                        const HEaaN::Ciphertext &op, HEaaN::Ciphertext &res,
                        const int layer_n);

void approxTanh_wide_16(const HEaaN::HomEvaluator &eval,
                        const HEaaN::Bootstrapper &btp,
                        const HEaaN::Ciphertext &op, HEaaN::Ciphertext &res,
                        const int layer_n);

void approxTanh_wide(const HEaaN::HomEvaluator &eval,
                     const HEaaN::Bootstrapper &btp,
                     const HEaaN::Ciphertext &op, HEaaN::Ciphertext &res,
                     const int layer_n);

} // namespace HELLM::Tanh
