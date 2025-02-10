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

namespace HELLM::LayerNorm {

void approxInverseSqrtLN(const HEaaN::HomEvaluator &eval,
                         const HEaaN::Bootstrapper &btp,
                         const HEaaN::Ciphertext &op, HEaaN::Ciphertext &res,
                         const int layer_n, const HEaaN::u64 num_iter);

} // namespace HELLM::LayerNorm
