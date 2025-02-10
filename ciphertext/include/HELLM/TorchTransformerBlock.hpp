////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2024 Crypto Lab Inc. All rights reserved.               //
//                                                                            //
// This source code is protected under international copyright law.           //
// All rights reserved and protected by the copyright holders.                //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "HELLM/ModelArgs.hpp"
#include "HEMMer.hpp"
#include "LoRA.hpp"
#include "torch/torch.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace HELLM {

class TorchTransformerBlock {
public:
    TorchTransformerBlock(HEMMer *hemmer, const torch::jit::Module &container,
                          int layer_n)
        : hemmer_{hemmer}, container_{container}, layer_n_{layer_n} {
        lora_module_ = std::make_shared<LoRA::LoraModule>(hemmer_, layer_n_);
    }

    void generateInitialLoraWeight(const std::string &lora_type) const;
    void zeroGrad(const std::string &lora_type) const;

private:
    HEMMer *hemmer_;
    const torch::jit::Module &container_;
    const int layer_n_;
    std::shared_ptr<LoRA::LoraModule> lora_module_;
};
} // namespace HELLM
