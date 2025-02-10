////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2024 Crypto Lab Inc. All rights reserved.               //
//                                                                            //
// This source code is protected under international copyright law.           //
// All rights reserved and protected by the copyright holders.                //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "HELLM/TorchTransformerBlock.hpp"

#include <ATen/core/TensorBody.h>
#include <ATen/ops/split.h>
#include <c10/core/ScalarType.h>
#include <cmath>
#include <memory>
#include <string>
#include <torch/types.h>
#include <utility>
#include <vector>

#include "HELLM/DevUtils.hpp"
#include "HELLM/HEMMer.hpp"
#include "HELLM/HETensor.hpp"
#include "HELLM/ModelArgs.hpp"
#include "HELLM/utils/check_macros.hpp"

namespace HELLM {

void TorchTransformerBlock::generateInitialLoraWeight(
    const std::string &lora_type) const {
    lora_module_->generateInitialLoraWeight(lora_type);
}

void TorchTransformerBlock::zeroGrad(const std::string &lora_type) const {
    lora_module_->zeroGrad(lora_type);
}

} // namespace HELLM
