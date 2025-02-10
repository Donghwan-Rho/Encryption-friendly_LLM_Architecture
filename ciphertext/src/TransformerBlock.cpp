////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2024 Crypto Lab Inc. All rights reserved.               //
//                                                                            //
// This source code is protected under international copyright law.           //
// All rights reserved and protected by the copyright holders.                //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "HELLM/TransformerBlock.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "HELLM/DevUtils.hpp"
#include "HELLM/HEMMer.hpp"
#include "HELLM/HETensor.hpp"
#include "HELLM/ModelArgs.hpp"
#include "HELLM/utils/check_macros.hpp"
#include "HEaaN/Message.hpp"

#include "HEaaN/HEaaN.hpp"

#ifdef HELLM_MULTIGPU
#include <mpi.h>
#endif

#define HELLM_DEBUG
#if defined(HELLM_DEBUG)
#ifndef HDEBUG
#define HDEBUG(...)                                                            \
    do {                                                                       \
        __VA_ARGS__;                                                           \
    } while (0)
#endif
#else
#ifndef HDEBUG
#define HDEBUG(...)
#endif
#endif

namespace {

int getOutlierIndex(int layer_n, int index) {
    switch (layer_n) {
    case 1:
        // 5843 = 22 * 256 + 211
        // 7890 = 30 * 256 + 210
        // 40
        if (index == 0)
            return 22;
        if (index == 1)
            return 30;
        if (index == 2)
            return 40;
        return -1;
    case 29:
        // 5162 = 20 * 256 + 42
        return index == 0 ? -1 : 20;
    case 30:
        // 3721 = 14 * 256 + 137
        // 7006 = 27 * 256 + 94
        return index == 0 ? 14 : 27;
    case 31:
        // 8733 = 34 * 256 + 29
        return index == 0 ? -1 : 34;
    default:
        return -1;
    }
}

int getFFNMatrixDevice(int layer_n, HEaaN::u64 index, int max_rank) {
    int idx = static_cast<int>(index);
    return (idx + 4) / 6 / (8 / max_rank);

    if (getOutlierIndex(layer_n, 1) == -1)
        return (idx + 4) / 6 / (8 / max_rank);
    if (idx == getOutlierIndex(layer_n, 0) ||
        idx == getOutlierIndex(layer_n, 1))
        return 7 / (8 / max_rank);
    if (idx < getOutlierIndex(layer_n, 0))
        return (idx + 2) / 6 / (8 / max_rank);
    if (idx < getOutlierIndex(layer_n, 1))
        return idx / 6 / (8 / max_rank);
    return (idx - 2) / 6 / (8 / max_rank);
}

} // namespace

namespace HELLM {

static std::chrono::high_resolution_clock::time_point start;

void TransformerBlock::printElapsedTime(const char *str) const {
    const auto rank = static_cast<unsigned int>(hemmer_->getRank());
    if (benchmark_) {
        CudaTools::cudaDeviceSynchronize();
        std::chrono::duration<double> elapsed =
            std::chrono::high_resolution_clock::now() - start;
        std::cout << rank << ", " << str << ": " << elapsed.count()
                  << std::endl;
        start = std::chrono::high_resolution_clock::now();
    }
}

std::vector<CtxtTensor>
TransformerBlock::forward_bert(std::vector<CtxtTensor> &input,
                               const Message &exp_mask) {
    /* if (input.size() == 100 ){
        //garbage value.
        std::cout << exp_mask[0].real() << std::endl;
    } */
    attention_bert(input, exp_mask);
    feedForward_bert(input);
    return input;
}

std::vector<CtxtTensor>
TransformerBlock::forward_bert_final(std::vector<CtxtTensor> &input) {
    pooling_bert(input);
    return input;
}

std::vector<CtxtTensor>
TransformerBlock::forward_bert_multi(std::vector<CtxtTensor> &input,
                                     const Message &exp_message) {
    attention_bert_multi(input, exp_message);
    feedForward_bert_multi(input);
    return input;
}

// forward2: forward pass for a fine-tuning.
std::vector<CtxtTensor>
TransformerBlock::forward2_bert(std::vector<CtxtTensor> &input,
                                const Message &exp_mask,
                                const std::string &lora_type) {

    attention2_bert(input, exp_mask, lora_type);
    feedForward2_bert(input);

    return input;
}

std::vector<CtxtTensor>
TransformerBlock::forward2_bert_test(std::vector<CtxtTensor> &input,
                                     const Message &exp_mask,
                                     const std::string &lora_type) {

    attention2_bert_test(input, exp_mask, lora_type);
    feedForward2_bert_test(input);

    return input;
}

std::vector<CtxtTensor>
TransformerBlock::forward2_bert_eval(std::vector<CtxtTensor> &input,
                                     const Message &exp_mask,
                                     const std::string &lora_type) {

    attention2_bert_loraOpti_eval(input, exp_mask, lora_type);
    feedForward2_bert_test(input);

    return input;
}

std::vector<CtxtTensor>
TransformerBlock::forward2_bert_SM(std::vector<CtxtTensor> &input,
                                   const std::string &lora_type) {

    attention_bert_SM(input, lora_type);
    feedForward2_bert_time(input);

    return input;
}

std::vector<CtxtTensor>
TransformerBlock::forward2_bert_time(std::vector<CtxtTensor> &input,
                                     const Message &exp_mask,
                                     const std::string &lora_type) {

    attention2_bert_time(input, exp_mask, lora_type);
    feedForward2_bert_time(input);

    return input;
}

std::vector<CtxtTensor>
TransformerBlock::forward2_bert_loraOpti(std::vector<CtxtTensor> &input,
                                         const Message &exp_mask,
                                         const std::string &lora_type) {

    attention2_bert_loraOpti(input, exp_mask, lora_type);
    feedForward2_bert(input);

    return input;
}

std::vector<CtxtTensor>
TransformerBlock::forward2_bert_loraOpti_time(std::vector<CtxtTensor> &input,
                                              const Message &exp_mask,
                                              const std::string &lora_type) {

    attention2_bert_loraOpti_time(input, exp_mask, lora_type);
    // feedForward2_bert_time(input);

    return input;
}

void TransformerBlock::forward2_pooling_bert(std::vector<CtxtTensor> &input,
                                             std::vector<CtxtTensor> &output,
                                             const u64 label) {
    // std::cout << "pooling input " << input[0].get().getLevel() << std::endl;
    pooling2_bert(input, output, label);
    // std::cout << "pooling output " << output[0].get().getLevel() <<
    // std::endl;
}

void TransformerBlock::forward3_pooling_bert(std::vector<CtxtTensor> &input,
                                             CtxtTensor &output,
                                             const u64 label) {
    // std::cout << "pooling input " << input[0].get().getLevel() << std::endl;
    pooling3_bert(input, output, label);
    // std::cout << "pooling output " << output[0].get().getLevel() <<
    // std::endl;
}

void TransformerBlock::forward3_pooling_bert_stsb(
    std::vector<CtxtTensor> &input, CtxtTensor &output, const u64 label) {
    // std::cout << "pooling input " << input[0].get().getLevel() << std::endl;
    pooling3_bert_stsb(input, output, label);
    // std::cout << "pooling output " << output[0].get().getLevel() <<
    // std::endl;
}

void TransformerBlock::forward3_pooling_bert_sst2(
    std::vector<CtxtTensor> &input, CtxtTensor &output, const u64 label) {
    // std::cout << "pooling input " << input[0].get().getLevel() << std::endl;
    pooling3_bert_sst2(input, output, label);
    // std::cout << "pooling output " << output[0].get().getLevel() <<
    // std::endl;
}

void TransformerBlock::forward3_pooling_bert_time(
    std::vector<CtxtTensor> &input, CtxtTensor &output, const u64 label) {
    pooling3_bert_time(input, output, label);
}

void TransformerBlock::forward3_pooling_bert_test(
    std::vector<CtxtTensor> &input) {
    pooling3_bert_test(input);
}

// TODO: combine with calculating loss function.
std::vector<CtxtTensor>
TransformerBlock::pooling_res_repack(CtxtTensor &tensor) {

    std::vector<CtxtTensor> output;
    output.reserve(4);

    hemmer_->maskFirsteleInplace(tensor);

    for (i64 rot = 1; rot < 128; rot <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().rightRotate(tensor.get(),
                                       static_cast<u64>(rot) * 256, tmp);
        hemmer_->getEval().add(tensor.get(), tmp, tensor.get());
    }

    Ciphertext tmp_rot(hemmer_->getContext());
    hemmer_->getEval().rightRotate(tensor.get(), 128, tmp_rot);
    hemmer_->getEval().add(tensor.get(), tmp_rot, tensor.get());

    for (u64 i = 0; i < 4; ++i) {
        output.push_back(tensor);
    }

    return output;
}

std::vector<CtxtTensor> TransformerBlock::pooling_loss_grad(CtxtTensor &tensor,
                                                            const u64 label) {

    std::vector<CtxtTensor> output;
    output.reserve(4);

    hemmer_->maskFirsteleInplace(tensor); // 10 -> 9
    hemmer_->lossExpInplace(tensor);      // 9 -> 4
    auto exp_output = tensor;             // 4

    Ciphertext tmp(hemmer_->getContext());
    hemmer_->getEval().leftRotate(tensor.get(), 1, tmp);
    hemmer_->getEval().add(tensor.get(), tmp, tensor.get());

    // 0.1 value masking! >> should recover after inv eval.
    hemmer_->maskFirsteleOnlyInplace(tensor);

    torch::Tensor inv_mask_tensor = torch::ones({128, 128}) * 0.5;
    torch::Tensor garbage = torch::ones({128, 128}) * 0.5;
    inv_mask_tensor[0][0] = 0.0;
    auto inv_mask = hemmer_->message2(inv_mask_tensor, garbage);

    hemmer_->getEval().add(tensor.get(), inv_mask, tensor.get());
    hemmer_->lossInvInplace(tensor);
    hemmer_->maskFirsteleOnlyInplace(tensor);

    hemmer_->getEval().rightRotate(tensor.get(), 1, tmp);
    hemmer_->getEval().add(tensor.get(), tmp, tensor.get());

    hemmer_->hadamardMultInplace(tensor, exp_output);

    torch::Tensor label_mask_tensor = torch::zeros({128, 128});
    torch::Tensor zeros = torch::zeros({128, 128});
    label_mask_tensor[0][static_cast<int>(label)] = 1.0;
    auto label_mask = hemmer_->message2(label_mask_tensor, zeros);

    hemmer_->getEval().sub(tensor.get(), label_mask, tensor.get());

    for (i64 rot = 1; rot < 128; rot <<= 1) {
        // Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().rightRotate(tensor.get(),
                                       static_cast<u64>(rot) * 256, tmp);
        hemmer_->getEval().add(tensor.get(), tmp, tensor.get());
    }

    Ciphertext tmp_rot(hemmer_->getContext());
    hemmer_->getEval().rightRotate(tensor.get(), 128, tmp_rot);
    hemmer_->getEval().add(tensor.get(), tmp_rot, tensor.get());

    hemmer_->getBtp().bootstrap(tensor.get(), tensor.get());

    /* //test for real value.
    torch::Tensor row = torch::tensor({0.4935394824,-0.4935393929});
    torch::Tensor repeated_tensor = row.repeat({128, 1});
    torch::Tensor zero_tensor = torch::zeros({128, 128});
    zero_tensor.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)})
    = repeated_tensor;

    auto test = hemmer_->encrypt2(zero_tensor, zero_tensor); */

    for (u64 i = 0; i < 4; ++i) {
        output.push_back(tensor);
    }

    return output;
}

CtxtTensor TransformerBlock::pooling3_loss_grad(CtxtTensor &tensor,
                                                const u64 label) {

    CtxtTensor output{tensor};

    // remaining (0,0), (1,0)
    hemmer_->maskFirstele2Inplace(tensor); // 11 -> 10
    if (hemmer_->getRank() == 0) {
        std::cout << "exp input in loss: " << std::endl;
        printing_masking(tensor);
    }
    hemmer_->lossExpInplace(tensor); // 9 -> 4
    if (hemmer_->getRank() == 0) {
        std::cout << "exp output in loss: " << std::endl;
        printing_masking(tensor);
    }
    auto exp_output = tensor; // 4

    Ciphertext tmp(hemmer_->getContext());
    hemmer_->getEval().leftRotate(tensor.get(), 1 * 256, tmp);
    hemmer_->getEval().add(tensor.get(), tmp, tensor.get());

    // 0.1 value masking! (for inv evaluation)  >> should recover after inv
    // eval.
    hemmer_->maskFirsteleOnlyInplace(tensor);

    torch::Tensor inv_mask_tensor = torch::ones({128, 128}) * 0.5;
    torch::Tensor garbage = torch::ones({128, 128}) * 0.5;
    inv_mask_tensor[0][0] = 0.0;
    auto inv_mask = hemmer_->message2(inv_mask_tensor, garbage);

    hemmer_->getEval().add(tensor.get(), inv_mask, tensor.get());
    if (hemmer_->getRank() == 0) {
        std::cout << "inptut inv in loss: " << std::endl;
        printing_masking(tensor);
    }
    hemmer_->lossInvInplace(tensor);
    if (hemmer_->getRank() == 0) {
        std::cout << "output inv in loss: " << std::endl;
        printing_masking(tensor);
    }
    hemmer_->maskFirsteleOnlyInplace(tensor);

    hemmer_->getEval().rightRotate(tensor.get(), 1 * 256, tmp);
    hemmer_->getEval().add(tensor.get(), tmp, tensor.get());

    hemmer_->hadamardMultInplace(tensor, exp_output);

    torch::Tensor label_mask_tensor = torch::zeros({128, 128});
    torch::Tensor zeros = torch::zeros({128, 128});
    label_mask_tensor[static_cast<int>(label)][0] = 1.0;
    auto label_mask = hemmer_->message2(label_mask_tensor, zeros);

    hemmer_->getEval().sub(tensor.get(), label_mask, tensor.get());

    for (u64 i = 1; i < 4; i <<= 1) {
        if (i == 1) {
            for (i64 rot = 1; rot < 128; rot <<= 1) {
                hemmer_->getEval().rightRotate(tensor.get(),
                                               static_cast<u64>(rot), tmp);
                hemmer_->getEval().add(tensor.get(), tmp, tensor.get());
            }
        }
        hemmer_->getEval().rightRotate(tensor.get(),
                                       i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
        hemmer_->getEval().add(tensor.get(), tmp, tensor.get());
    }

    hemmer_->getEval().rightRotate(tensor.get(), 128, tmp);
    hemmer_->getEval().add(tensor.get(), tmp, output.get());

    hemmer_->getBtp().bootstrap(output.get(), output.get());

    return output;
}

CtxtTensor TransformerBlock::pooling3_loss_grad_sst2(CtxtTensor &tensor,
                                                     const u64 label) {

    CtxtTensor output{tensor};
    CtxtTensor max{tensor};

    hemmer_->maskFirstele2Inplace(tensor);
    /* if (hemmer_->getRank() == 0) {
        std::cout << "pooling3_loss input in loss: " << std::endl;
        printing_masking(tensor);
    } */

    // Getting max
    hemmer_->maskFirstele2InplaceSST2(max); // 11 -> 10
    /* if (hemmer_->getRank() == 0) {
        std::cout << "max input in loss: " << std::endl;
        printing_masking(max);
    } */
    hemmer_->lossMax(max);                     // 9
    hemmer_->maskFirsteleOnlyInplaceSST2(max); // 8
    /* if (hemmer_->getRank() == 0) {
        std::cout << "max output in loss: " << std::endl;
        printing_masking(max);
    } */
    hemmer_->getEval().rightRotate(max.get(), 256, output.get());
    hemmer_->getEval().add(max.get(), output.get(), max.get());

    hemmer_->getEval().sub(tensor.get(), max.get(), tensor.get()); // 8

    // remaining (0,0), (1,0)
    /* if( hemmer_->getRank() == 0) {
        std::cout << "exp input in loss: " << std::endl;
        printing_masking(tensor);
    } */
    hemmer_->lossExpInplaceSST2(tensor);                     // 8 -> 3
    hemmer_->getBtp().bootstrap(tensor.get(), tensor.get()); // 12
    /* if( hemmer_->getRank() == 0) {
        std::cout << "exp output in loss: " << std::endl;
        printing_masking(tensor);
    } */
    auto exp_output = tensor; // 12

    Ciphertext tmp(hemmer_->getContext());
    hemmer_->getEval().leftRotate(tensor.get(), 1 * 256, tmp);
    hemmer_->getEval().add(tensor.get(), tmp, tensor.get());

    hemmer_->maskFirstele1OnlyInplace(tensor);

    torch::Tensor inv_mask_tensor = torch::ones({128, 128});
    torch::Tensor garbage = torch::ones({128, 128});
    inv_mask_tensor[0][0] = 0.0;
    auto inv_mask = hemmer_->message2(inv_mask_tensor, garbage);

    hemmer_->getEval().add(tensor.get(), inv_mask, tensor.get());
    /* if( hemmer_->getRank() == 0) {
        std::cout << "inptut inv in loss: " << std::endl;
        printing_masking(tensor);
    } */
    hemmer_->lossInvInplaceSST2(tensor); // 7
    /* if( hemmer_->getRank() == 0) {
        std::cout << "output inv in loss: " << std::endl;
        printing_masking(tensor);
    } */
    hemmer_->maskFirstele1OnlyInplace(tensor); // 6

    hemmer_->getEval().rightRotate(tensor.get(), 1 * 256, tmp);
    hemmer_->getEval().add(tensor.get(), tmp, tensor.get());

    hemmer_->hadamardMultInplace(tensor, exp_output); // 5

    torch::Tensor label_mask_tensor = torch::zeros({128, 128});
    torch::Tensor zeros = torch::zeros({128, 128});
    label_mask_tensor[static_cast<int>(label)][0] = 1.0;
    auto label_mask = hemmer_->message2(label_mask_tensor, zeros);

    hemmer_->getEval().sub(tensor.get(), label_mask, tensor.get());

    for (u64 i = 1; i < 4; i <<= 1) {
        if (i == 1) {
            for (i64 rot = 1; rot < 128; rot <<= 1) {
                hemmer_->getEval().rightRotate(tensor.get(),
                                               static_cast<u64>(rot), tmp);
                hemmer_->getEval().add(tensor.get(), tmp, tensor.get());
            }
        }
        hemmer_->getEval().rightRotate(tensor.get(),
                                       i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
        hemmer_->getEval().add(tensor.get(), tmp, tensor.get());
    }

    hemmer_->getEval().rightRotate(tensor.get(), 128, tmp);
    hemmer_->getEval().add(tensor.get(), tmp, output.get());

    hemmer_->getBtp().bootstrap(output.get(), output.get());

    return output;
}

CtxtTensor TransformerBlock::pooling3_loss_grad_mse(CtxtTensor &tensor,
                                                    const u64 label) {

    CtxtTensor output{tensor};

    // (0,0) = 1 , otherwise = 0
    hemmer_->maskFirstele1OnlyInplace(tensor); // 11 -> 10

    torch::Tensor label_mask_tensor = torch::zeros({128, 128});
    torch::Tensor zeros = torch::zeros({128, 128});
    label_mask_tensor[0][0] = static_cast<double>(label);
    /* if (hemmer_->getRank() == 0) {
        std::cout << "label: " << label << std::endl;
    } */
    auto label_mask = hemmer_->message2(label_mask_tensor, zeros);

    hemmer_->getEval().sub(tensor.get(), label_mask, tensor.get());

    // hemmer_->getEval().square(tensor.get(), tensor.get()); // 10 -> 9
    hemmer_->getEval().mult(tensor.get(), 2, tensor.get());

    /* if (hemmer_->getRank() == 0) {
        std::cout << "mse loss output" << std::endl;
        printing_masking(tensor);
    } */

    Ciphertext tmp{tensor.get()};
    // repeated packing
    for (u64 i = 1; i < 4; i <<= 1) {
        if (i == 1) {
            for (i64 rot = 1; rot < 128; rot <<= 1) {
                hemmer_->getEval().rightRotate(tensor.get(),
                                               static_cast<u64>(rot), tmp);
                hemmer_->getEval().add(tensor.get(), tmp, tensor.get());
            }
        }
        hemmer_->getEval().rightRotate(tensor.get(),
                                       i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
        hemmer_->getEval().add(tensor.get(), tmp, tensor.get());
    }

    hemmer_->getEval().rightRotate(tensor.get(), 128, tmp);
    hemmer_->getEval().add(tensor.get(), tmp, output.get());

    hemmer_->getBtp().bootstrap(output.get(), output.get());

    return output;
}

// TODO: fix a return structure.
std::vector<CtxtTensor>
TransformerBlock::backward2_bert(std::vector<CtxtTensor> &grad_y,
                                 const std::string &lora_type) {

    /* const auto rank = hemmer_->getRank();
    if (rank == 0) {
        lora_module_->zeroAggGrad(lora_type);
    } */

    // std::cout << "backward ffn input" << std::endl;
    backwardfeedForward2_bert(grad_y);
    /* std::cout << "backward after ffn " << std::endl;
    printing(grad_y); */

    backwardattention2_bert(grad_y, lora_type);
    /* std::cout <<  "backward after attn" << std::endl;
    printing(grad_y); */

    hemmer_->bootstrap2(grad_y[0], grad_y[1]);
    hemmer_->bootstrap(grad_y[2]);

    return grad_y;
}

// TODO: fix a return structure.
std::vector<CtxtTensor>
TransformerBlock::backward2_bert_time(std::vector<CtxtTensor> &grad_y,
                                      const std::string &lora_type) {

    backwardfeedForward2_bert_time(grad_y);
    std::cout << "backffn done" << std::endl;
    backwardattention2_bert_time(grad_y, lora_type);
    std::cout << "backattn done" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    hemmer_->bootstrap2(grad_y[0], grad_y[1]);
    hemmer_->bootstrap(grad_y[2]);
    printElapsedTime("BTS");

    return grad_y;
}

// TODO: fix a return structure.
std::vector<CtxtTensor>
TransformerBlock::backward2_bert_loraOpti(std::vector<CtxtTensor> &grad_y,
                                          const std::string &lora_type) {

    backwardfeedForward2_bert(grad_y);
    backwardattention2_bert_loraOpti(grad_y, lora_type);

    hemmer_->bootstrap2(grad_y[0], grad_y[1]);
    hemmer_->bootstrap(grad_y[2]);

    return grad_y;
}

// TODO: fix a return structure.
std::vector<CtxtTensor>
TransformerBlock::backward2_bert_loraOpti_time(std::vector<CtxtTensor> &grad_y,
                                               const std::string &lora_type) {

    backwardfeedForward2_bert_time(grad_y);
    backwardattention2_bert_loraOpti_time(grad_y, lora_type);

    hemmer_->bootstrap2(grad_y[0], grad_y[1]);
    hemmer_->bootstrap(grad_y[2]);

    printElapsedTime("BTS");

    return grad_y;
}

// TODO: fix a return structure.
std::vector<CtxtTensor>
TransformerBlock::backward2_bert_SM(std::vector<CtxtTensor> &grad_y,
                                    const std::string &lora_type) {

    backwardfeedForward2_bert(grad_y);
    std::cout << "backffn done" << std::endl;
    backwardattention2_bert_SM(grad_y, lora_type);
    std::cout << "backattn done" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    hemmer_->bootstrap2(grad_y[0], grad_y[1]);
    hemmer_->bootstrap(grad_y[2]);
    printElapsedTime("BTS");

    return grad_y;
}

std::vector<CtxtTensor>
TransformerBlock::backward2_pooling_bert(std::vector<CtxtTensor> &grad_y) {
    return backwardpooling2_bert(grad_y);
}

std::vector<CtxtTensor>
TransformerBlock::backward3_pooling_bert(CtxtTensor &grad_y) {
    return backwardpooling3_bert(grad_y);
}

std::vector<CtxtTensor>
TransformerBlock::backward3_pooling_bert_stsb(CtxtTensor &grad_y) {
    return backwardpooling3_bert_stsb(grad_y);
}

std::vector<CtxtTensor>
TransformerBlock::backward3_pooling_bert_time(CtxtTensor &grad_y) {
    return backwardpooling3_bert_time(grad_y);
}

void TransformerBlock::generateInitialLoraWeight(
    const std::string &lora_type) const {
    lora_module_->generateInitialLoraWeight(lora_type);
}

void TransformerBlock::compareLoraWeight(const std::string &lora_type) const {
    lora_module_->compareLoraWeight(lora_type);
}

void TransformerBlock::zeroGrad(const std::string &lora_type) const {
    lora_module_->zeroGrad(lora_type);
}

void TransformerBlock::printing(
    const std::vector<CtxtTensor> &tensor_vec) const {

    for (u64 m = 0; m < 1; ++m) {
        auto dec_tensor = hemmer_->decrypt2(tensor_vec[m]);

        for (HELLM::i64 k = 0; k < 1; ++k) {
            for (HELLM::i64 i = 0; i < 1; ++i) {
                for (HELLM::i64 j = 0; j < 4; ++j) {
                    std::cout.precision(10);
                    std::cout << dec_tensor[k].index({i, j}).item<double>()
                              << ", ";
                }
                std::cout << std::endl;
            }
        }
    }
}

void TransformerBlock::printing_whole(
    const std::vector<CtxtTensor> &tensor_vec) const {

    for (u64 m = 0; m < 1; ++m) {
        auto dec_tensor = hemmer_->decrypt2(tensor_vec[m]);

        std::cout << "index = " << m << std::endl;

        for (HELLM::i64 k = 0; k < 1; ++k) {
            for (HELLM::i64 i = 0; i < 1; ++i) {
                for (HELLM::i64 j = 0; j < 64; ++j) {
                    std::cout.precision(10);
                    std::cout << dec_tensor[k].index({i, j}).item<double>()
                              << ", ";
                }
                std::cout << std::endl;
            }
        }
    }
}

void TransformerBlock::printing_exp(
    const std::vector<CtxtTensor> &tensor_vec) const {

    for (u64 m = 0; m < 1; ++m) {
        auto dec_tensor = hemmer_->decrypt2(tensor_vec[m]);

        for (HELLM::i64 k = 0; k < 1; ++k) {
            for (HELLM::i64 i = 0; i < 4; ++i) {
                for (HELLM::i64 j = 40; j < 40 + 4; ++j) {
                    std::cout.precision(10);
                    std::cout << dec_tensor[k].index({i, j}).item<double>()
                              << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

void TransformerBlock::printing_masking(const CtxtTensor &tensor_vec) const {

    auto dec_tensor = hemmer_->decrypt2(tensor_vec);

    for (HELLM::i64 k = 0; k < 1; ++k) {
        for (HELLM::i64 i = 0; i < 1; ++i) {
            for (HELLM::i64 j = 0; j < 0 + 4; ++j) {
                std::cout.precision(10);
                std::cout << dec_tensor[k].index({i, j}).item<double>() << ", ";
            }
            std::cout << std::endl;
        }
    }
}

void TransformerBlock::printingfirstCol(const CtxtTensor &tensor_vec) const {

    auto dec_tensor = hemmer_->decrypt2(tensor_vec);
    auto rank = hemmer_->getRank();

    std::cout << rank << ", ";
    for (HELLM::i64 i = 0; i < 2; ++i) {
        std::cout << dec_tensor[0].index({i, 0}).item<double>() << ", ";
    }
    std::cout << std::endl;
}

void TransformerBlock::tensor_save(const std::vector<CtxtTensor> &tensor_vec,
                                   const std::string &name,
                                   const int layer_n) const {
    for (u64 i = 0; i < tensor_vec.size(); ++i) {
        auto tensor = hemmer_->decrypt2(tensor_vec[i]);
        torch::save(tensor, "./tensor/" + name + "_" +
                                std::to_string(hemmer_->getRank()) + "_" +
                                std::to_string(layer_n) + "_" +
                                std::to_string(i) + ".pth");
    }
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//////////////////////////////// PRIVATE //////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// single GPU version.
// Assume input leval is 12.
void TransformerBlock::attention_bert(std::vector<CtxtTensor> &input,
                                      const Message &exp_message) {
    const u64 rank = static_cast<u64>(hemmer_->getRank());
    const u64 max_rank = static_cast<u64>(hemmer_->getMaxRank());

    // Caution: N_HEAD = 12 -> 6 (based on the number of block, which is 128 x
    // 128 tensor)
    // const u64 n_iter = ModelArgs::N_HEAD / max_rank;
    const u64 n_iter = 6 / max_rank;
    // const u64 n_iter_half = ModelArgs::N_HEAD / 2 / max_rank;
    const u64 n_iter_half = 3 / max_rank;
    // const u64 n_iter_quarter = ModelArgs::N_HEAD / 4 / max_rank;
    const u64 n_iter_quarter = 2 / max_rank;
    /* Step 1. Prepare Query, Key, Value tensors. */
    // Important: N_HEAD counting!

    start = std::chrono::high_resolution_clock::now();
    std::vector<CtxtTensor> cur =
        hemmer_->LayerNorm(input, "atn", layer_n_); // 12 -> 6
    for (u64 i = 0; i < n_iter_half; ++i) {
        auto idx = n_iter_half * rank + i;
        // TODO: replace name with fixed one.
        const auto &weight = getWeightMsg("norm1_w", idx);
        const auto &bias = getWeightMsg("norm1_b", idx);
        hemmer_->hadamardMultInplace(cur[i], weight);             // 6 -> 5
        hemmer_->getEval().add(cur[i].get(), bias, cur[i].get()); // 5 -> 5
    }
    printElapsedTime("atn LN");
    // printing(cur);

    hemmer_->bootstrap2(cur[0], cur[1]);
    hemmer_->bootstrap(cur[2]);
    printElapsedTime("BTS");

    std::vector<CtxtTensor> temp_q, temp_k, temp_v;
    temp_q.reserve(6);
    temp_k.reserve(6);
    temp_v.reserve(6);
    std::vector<PtxtTensor> weights_q, weights_k, weights_v;
    weights_q.reserve(6);
    weights_k.reserve(6);
    weights_v.reserve(6);
    for (u64 i = 0; i < n_iter_quarter; ++i) {
        // if loop >> not i but rank
        std::vector<Ciphertext> tmp;
        // check
        if (i == 0) {
            hemmer_->matMulPre(
                hemmer_->complexPacking(cur[i * 2], cur[i * 2 + 1]), tmp);
        } else {
            hemmer_->matMulPre(cur[2], tmp);
        }
        // pre-compute weights for q, k, v for each head
        weights_q.clear();
        weights_k.clear();
        weights_v.clear();
        for (u64 j = 0; j < 6; ++j) {
            // n_iter_quarter << should fix as the same value.
            // define n_iter_quarter = 2
            // auto idx = n_iter_quarter * rank + i;
            auto idx = n_iter_quarter * rank + i;
            weights_q.push_back(getWeight("wq", idx, j));
            weights_k.push_back(getWeight("wk", idx, j));
            weights_v.push_back(getWeight("wv", idx, j));
        }

        // perform matmul and addition
        for (u64 j = 0; j < 6; ++j) {
            auto result_q = hemmer_->matMulReUse(tmp, weights_q[j]);
            auto result_k = hemmer_->matMulReUse(tmp, weights_k[j]);
            auto result_v = hemmer_->matMulReUse(tmp, weights_v[j]);

            if (i == 0) {
                temp_q.push_back(result_q);
                temp_k.push_back(result_k);
                temp_v.push_back(result_v);
            } else {
                hemmer_->addInplace(temp_q[j], result_q);
                hemmer_->addInplace(temp_k[j], result_k);
                hemmer_->addInplace(temp_v[j], result_v);
            }
        }
    }
    printElapsedTime("q,k,v mm");

    std::vector<CtxtTensor> query, key, value;
    query.reserve(n_iter_half);
    key.reserve(n_iter_half);
    value.reserve(n_iter_half);
    // repacking temp_q, temp_k, temp_v into query, key, value
    for (u64 i = 0; i < n_iter_half; ++i) {
        auto idx = (rank * n_iter_half + i) * 2;

        query.push_back(hemmer_->repack(temp_q[idx], temp_q[idx + 1])); // 11
        key.push_back(hemmer_->repack(temp_k[idx], temp_k[idx + 1]));   // 11
        value.push_back(hemmer_->repack(temp_v[idx], temp_v[idx + 1])); // 11

        hemmer_->getEval().levelDown(query[i].get(), 10, query[i].get());
        hemmer_->getEval().levelDown(value[i].get(), 9, value[i].get());
        hemmer_->getEval().levelDown(key[i].get(), 11, key[i].get());
    }
    printElapsedTime("q,k,v repack");

    /* Step 2. Perform the attention.. */
    // initialization
    std::vector<CtxtTensor> masked_query, masked_key;
    masked_query.reserve(n_iter);
    masked_key.reserve(n_iter);
    for (u64 i = 0; i < n_iter_half; i++) {
        for (int j = 0; j < 2; j++) {
            masked_query.push_back(query[i]);
            masked_key.push_back(key[i]);
        }
    }
    std::vector<CtxtTensor> one_matmul_query, one_matmul_key;
    one_matmul_query = masked_query;
    one_matmul_key = masked_key;

    // Q. query element -> divide q1 q2 q3 q4 >> q1 0 q3 0 , 0 q2 0 q4
    for (u64 i = 0; i < n_iter_half; ++i) {
        // masked matmul
        hemmer_->maskRightLeft(query[i], masked_query[2 * i],
                               masked_query[2 * i + 1]); // 10 -> 9
        hemmer_->maskRightLeft(key[i], masked_key[2 * i],
                               masked_key[2 * i + 1]); // 11 -> 10

        for (u64 j = 0; j < 2; ++j) {
            hemmer_->getEval().square(
                masked_query[2 * i + j].get(),
                one_matmul_query[2 * i + j].get()); // 9 -> 8
            hemmer_->getEval().square(
                masked_key[2 * i + j].get(),
                one_matmul_key[2 * i + j].get()); // 10 -> 9
        }
    }
    printElapsedTime("GK masking and squaring");

    for (u64 i = 0; i < n_iter_half; ++i) {

        hemmer_->oneMatRotSumInplace(one_matmul_query[2 * i],
                                     one_matmul_query[2 * i + 1]); // 6 -> 5
        hemmer_->oneMatRotSumInplace(one_matmul_key[2 * i],
                                     one_matmul_key[2 * i + 1]); // 7 -> 6

        // target level: ?
        for (u64 j = 0; j < 2; ++j) {
            hemmer_->transposeInplace(one_matmul_key[2 * i + j],
                                      4); // 6 -> 4 (LD)
        }
    }
    printElapsedTime("GK norm");

    // init
    std::vector<CtxtTensor> cur_qk;
    cur_qk.reserve(n_iter);
    for (u64 i = 0; i < n_iter_half; ++i) {
        for (u64 j = 0; j < 2; ++j) {
            cur_qk.push_back(query[i]);
        }
    }

    for (u64 i = 0; i < n_iter; ++i) {
        hemmer_->complexPackingInplace(masked_query[i]);        // 8 -> 7
        hemmer_->transposeComplexPackingInplace(masked_key[i]); //  9 -> 7
        cur_qk[i] =
            hemmer_->packedMatMul(masked_query[i], masked_key[i]); // 7 -> 4
        cur_qk[i] = hemmer_->repackCC(cur_qk[i]);                  // 4 -> 4

        hemmer_->getEval().mult(cur_qk[i].get(), -2, cur_qk[i].get()); // 4 -> 4

        hemmer_->getEval().add(cur_qk[i].get(), one_matmul_query[i].get(),
                               cur_qk[i].get()); // 4
        hemmer_->getEval().add(cur_qk[i].get(), one_matmul_key[i].get(),
                               cur_qk[i].get()); // 4
    }
    printElapsedTime("qk CC");

    for (u64 i = 0; i < n_iter_half; ++i) {
        hemmer_->bootstrap2(cur_qk[i * 2], cur_qk[i * 2 + 1]); // 4 -> 12
    }
    printElapsedTime("BTS");

    // 12 -> 11
    for (u64 i = 0; i < n_iter; ++i) {
        hemmer_->getEval().mult(cur_qk[i].get(), -1.0 / (2 * std::sqrt(64)),
                                cur_qk[i].get()); // 12 -> 11
        hemmer_->getEval().add(cur_qk[i].get(), exp_message, cur_qk[i].get());
    }
    printElapsedTime("dividing and summation of exp_mask");

    // exp evaluation.
    // Caution: should consider output level of exp by comparing with the output
    // preicison. 11 -> 7 ( 8 for complexpacking)
    hemmer_->expParallelInplace(cur_qk, layer_n_, false);
    printElapsedTime("exp eval");

    for (u64 i = 0; i < 3; ++i) {
        hemmer_->bootstrap2_exponly(cur_qk[2 * i], cur_qk[2 * i + 1]);
        for (u64 j = 0; j < 2; ++j) {
            hemmer_->getEval().levelDown(cur_qk[2 * i + j].get(), 8,
                                         cur_qk[2 * i + j].get());
        }
    }
    printElapsedTime("BTS");

    // TODO: clean up loop index: n_iter_half*2 etc.
    std::vector<CtxtTensor> masked_value;
    masked_value.reserve(n_iter);
    for (u64 i = 0; i < n_iter_half; ++i) {
        for (u64 j = 0; j < 2; ++j) {
            masked_value.push_back(value[i]); // 8
        }
    }

    // masked value
    // v11 v12 v13 v14 >> v11 0 v13 0 | 0 v12 0 v14 (even index)
    // v12 v22 v23 v24 >> v12 0 v23 0 | 0 v22 0 v24 (odd index)
    for (u64 i = 0; i < n_iter_half; ++i)
        hemmer_->maskRightLeft(value[i], masked_value[2 * i],
                               masked_value[2 * i + 1]); // 9 -> 8
    printElapsedTime("masking");

    for (u64 i = 0; i < n_iter; ++i) {
        // cur_qk[i] = hemmer_->singleCCMatMul(cur_qk[i], masked_value[i], 4);
        // // 7 -> 4
        hemmer_->complexPackingInplace(cur_qk[i]);          // 8 -> 7
        hemmer_->complexPackingRowInplace(masked_value[i]); // 8 -> 7

        cur_qk[i] = hemmer_->packedMatMul(
            cur_qk[i], masked_value[i]); // automatic level down...
    }
    printElapsedTime("value MM");

    for (u64 i = 0; i < n_iter; ++i) {            // 7 -> 4
        cur_qk[i] = hemmer_->repackCC(cur_qk[i]); // level consumpotion x
    }
    printElapsedTime("value MM repack");

    for (u64 i = 0; i < n_iter_half; ++i) {
        hemmer_->bootstrap2(cur_qk[i * 2], cur_qk[i * 2 + 1]); // 12
    }
    printElapsedTime("BTS");

    // collect into one ciphertext
    std::vector<CtxtTensor> cur_reuse;
    cur_reuse.reserve(n_iter_half);
    for (u64 i = 0; i < n_iter_half; i++) {
        hemmer_->addInplace(cur_qk[i * 2], cur_qk[i * 2 + 1]); // 12
        cur_reuse.push_back(cur_qk[i * 2]);                    // 12
    }
    cur_qk.clear();

    std::vector<CtxtTensor> output;
    output.reserve(6);
    for (u64 i = 0; i < n_iter_quarter; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(
                hemmer_->complexPacking(cur_reuse[i * 2], cur_reuse[i * 2 + 1]),
                tmp);
        } else {
            hemmer_->matMulPre(cur_reuse[2], tmp);
        }

        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("wd", 2 * rank + i, j);
            auto result = hemmer_->matMulReUse(tmp, weight);

            if (i == 0) {
                output.push_back(result);
            } else {
                hemmer_->addInplace(output[j], result);
            }
        }
    }
    printElapsedTime("weight MM");

    if (max_rank > 1) {
        reduceWrapper(output);

        for (u64 i = 0; i < n_iter; ++i) {
            hemmer_->getEval().modReduct(output[rank * n_iter + i].get());
        }
    }

    // printElapsedTime("modReduct");

    // std::cout << "after reducewrapper" << std::endl;
    // printing(output);

    /* Step 3. Post-process the result.. */
    for (u64 i = 0; i < n_iter_half; ++i) {
        auto idx = (rank * n_iter_half + i) * 2;
        auto repacked_output = hemmer_->repack(output[idx], output[idx + 1]);

        // hemmer_->dropoutInplace(repacked_output, i); //11

        hemmer_->addInplace(input[i], repacked_output); // 11
    }
    printElapsedTime("repacking");
    // std::cout << "attn output" << std::endl;
    // printing(input);
}

// Fine-tuning version
// TODO: Implemenation of AdamW optimizer.
void TransformerBlock::attention2_bert(std::vector<CtxtTensor> &input,
                                       const Message &exp_message,
                                       const std::string &lora_type) {
    const u64 rank = static_cast<u64>(hemmer_->getRank());
    // const u64 max_rank = static_cast<u64>(hemmer_->getMaxRank());

    start = std::chrono::high_resolution_clock::now();

    std::vector<CtxtTensor> cur =
        hemmer_->LayerNorm(input, "atn", layer_n_, true); // 12 -> 6
    for (u64 i = 0; i < 3; ++i) {
        // TODO: replace name with fixed one.
        const auto &weight = getWeightMsg("norm1_w", i);
        const auto &bias = getWeightMsg("norm1_b", i);
        hemmer_->hadamardMultInplace(cur[i], weight);             // 6 -> 5
        hemmer_->getEval().add(cur[i].get(), bias, cur[i].get()); // 5 -> 5
    }
    // printElapsedTime("atn LN");

    if (rank == 0) {
        std::cout << "atn LN" << std::endl;
        printing(cur);
    }

    hemmer_->bootstrap2(cur[0], cur[1]);
    hemmer_->bootstrap(cur[2]);
    // printElapsedTime("BTS");

    std::vector<CtxtTensor> temp_q, temp_k, temp_v;
    temp_q.reserve(6);
    temp_k.reserve(6);
    temp_v.reserve(6);
    std::vector<PtxtTensor> weights_q, weights_k, weights_v;
    weights_q.reserve(6);
    weights_k.reserve(6);
    weights_v.reserve(6);
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(
                hemmer_->complexPacking(cur[i * 2], cur[i * 2 + 1]), tmp);
        } else {
            hemmer_->matMulPre(cur[2], tmp);
        }
        weights_q.clear();
        weights_k.clear();
        weights_v.clear();
        for (u64 j = 0; j < 6; ++j) {
            weights_q.push_back(getWeight("wq", i, j));
            weights_k.push_back(getWeight("wk", i, j));
            weights_v.push_back(getWeight("wv", i, j));
        }
        for (u64 j = 0; j < 6; ++j) {
            auto result_q = hemmer_->matMulReUse(tmp, weights_q[j]);
            auto result_k = hemmer_->matMulReUse(tmp, weights_k[j]);
            auto result_v = hemmer_->matMulReUse(tmp, weights_v[j]);

            if (i == 0) {
                temp_q.push_back(result_q);
                temp_k.push_back(result_k);
                temp_v.push_back(result_v);
            } else {
                hemmer_->addInplace(temp_q[j], result_q);
                hemmer_->addInplace(temp_k[j], result_k);
                hemmer_->addInplace(temp_v[j], result_v);
            }
        }
    }
    // printElapsedTime("q,k,v mm");

    std::vector<CtxtTensor> query, key, value;
    query.reserve(3);
    key.reserve(3);
    value.reserve(3);
    for (u64 i = 0; i < 3; ++i) {
        query.push_back(
            hemmer_->repack(temp_q[i * 2], temp_q[i * 2 + 1]));           // 11
        key.push_back(hemmer_->repack(temp_k[i * 2], temp_k[i * 2 + 1])); // 11
        value.push_back(
            hemmer_->repack(temp_v[i * 2], temp_v[i * 2 + 1])); // 11

        hemmer_->getEval().levelDown(query[i].get(), 10, query[i].get());
        hemmer_->getEval().levelDown(value[i].get(), 9, value[i].get());
        hemmer_->getEval().levelDown(key[i].get(), 11, key[i].get());
    }
    // printElapsedTime("q,k,v repack");

    /* if (rank == 0) {
        std::cout << "q,k,v mm done" << std::endl;
        printing(query);
    } */

    /* LoRA */
    // forward
    // std::unordered_map<char, CtxtTensor> lora_a;
    std::unordered_map<char, std::vector<CtxtTensor>> lora_a;

    start = std::chrono::high_resolution_clock::now();
    for (const char t : lora_type) {
        const std::string lora_t = std::string(1, t);
        auto lora_wa = lora_module_->getCtxtTensor_lora("lora_wa_" + lora_t, 0,
                                                        0, 0); // 9 or 8

        hemmer_->getEval().levelDown(lora_wa.get(), 9, lora_wa.get());

        // origin version
        /* auto lora_a_tmp = hemmer_->matMulHighLow(
            cur[0], hemmer_->getLowColBlock(lora_wa, 0), 0, 5);
        lora_a.emplace(t, lora_a_tmp);
        for (u64 i = 1 ; i < 3 ; ++i) {
            auto it = lora_a.find(t);
            assert(it != lora_a.end());
            hemmer_->addInplace(
                it->second,
                hemmer_->matMulHighLow(
                    cur[i], hemmer_->getLowColBlock(lora_wa, i), i, 5));
        } */

        // hard coding for rank 2
        std::vector<std::vector<CtxtTensor>> lora_a_weight;
        lora_a_weight.reserve(3);
        std::vector<CtxtTensor> lora_a_output;
        lora_a_output.reserve(2);

        for (u64 j = 0; j < 2; ++j) {
            lora_a_output.emplace_back(lora_wa);
        }

        auto tmp = lora_wa;

        // split weights
        hemmer_->maskFirstRowInplace(lora_a_output[0]);
        hemmer_->getEval().leftRotate(lora_a_output[1].get(), 1 * 256,
                                      lora_a_output[1].get());
        hemmer_->maskFirstRowInplace(lora_a_output[1]);
        lora_a_weight.emplace_back(lora_a_output);
        for (u64 i = 1; i < 3; ++i) {
            lora_a_output.clear();
            hemmer_->getEval().leftRotate(lora_wa.get(), i * 2 * 256,
                                          tmp.get());
            for (u64 j = 0; j < 2; ++j) {
                lora_a_output.emplace_back(tmp);
            }
            hemmer_->maskFirstRowInplace(lora_a_output[0]);
            hemmer_->getEval().leftRotate(lora_a_output[1].get(), 1 * 256,
                                          lora_a_output[1].get());
            hemmer_->maskFirstRowInplace(lora_a_output[1]);
            lora_a_weight.emplace_back(lora_a_output);
        }

        // repeated packing
        for (u64 i = 0; i < 3; ++i) {
            for (u64 j = 0; j < 2; ++j) {
                for (u64 rot = 1; rot < 128; rot <<= 1) {
                    hemmer_->getEval().rightRotate(lora_a_weight[i][j].get(),
                                                   rot * 256, tmp.get());
                    hemmer_->addInplace(lora_a_weight[i][j], tmp);
                }
            }
        }

        // hadamult
        hemmer_->getEval().mult(cur[0].get(), lora_a_weight[0][0].get(),
                                lora_a_output[0].get());
        hemmer_->getEval().mult(cur[0].get(), lora_a_weight[0][1].get(),
                                lora_a_output[1].get());
        for (u64 i = 1; i < 3; ++i) {
            hemmer_->getEval().mult(cur[i].get(), lora_a_weight[i][0].get(),
                                    tmp.get());
            hemmer_->addInplace(lora_a_output[0], tmp);
            hemmer_->getEval().mult(cur[i].get(), lora_a_weight[i][1].get(),
                                    tmp.get());
            hemmer_->addInplace(lora_a_output[1], tmp);
        }

        // addition
        for (u64 i = 0; i < 2; ++i) {
            for (u64 rot = 1; rot < 256; rot <<= 1) {
                hemmer_->getEval().leftRotate(lora_a_output[i].get(), rot,
                                              tmp.get());
                hemmer_->addInplace(lora_a_output[i], tmp);
            }
            hemmer_->maskFirstColOnlyInplace(lora_a_output[i]);
        }
        lora_a.emplace(t, lora_a_output);
    }

    // printElapsedTime("lora_a mm");

    // original version
    /* for (const char t : lora_type) {
        auto it = lora_a.find(t);
        it->second = hemmer_->repackToOneCol(it->second, 0);
        hemmer_->bootstrap(it->second);
    } */
    // printElapsedTime("lora_a repack");

    std::unordered_map<char, std::vector<CtxtTensor>> lora_output;
    start = std::chrono::high_resolution_clock::now();
    for (const char t : lora_type) {
        const std::string lora_t = std::string(1, t);

        auto lora_wb =
            lora_module_->getCtxtTensor_lora("lora_wb_" + lora_t, 0, 0, 0);
        auto lora_a_output = lora_a[t];
        // const std::string key = "lora_wb_" + lora_t + "_" +
        // std::to_string(layer_n_); auto lora_wb = weights[key]

        /* auto it = lora_a.find(t);
        std::vector<CtxtTensor> tmp;
        for (u64 i = 0 ; i < 3 ; ++i) {
            tmp.emplace_back(hemmer_->matMulLowLow(
                it->second, hemmer_->getLowRowBlock(lora_wb, i), 0 , i ));
        } */

        // split weights
        std::vector<std::vector<CtxtTensor>> lora_b_weight;
        lora_b_weight.reserve(3);
        std::vector<CtxtTensor> tmp_vector;
        tmp_vector.reserve(2);

        auto tmp = lora_wb;

        for (u64 i = 0; i < 2; ++i) {
            tmp_vector.emplace_back(lora_wb);
        }
        hemmer_->maskFirstRowInplace(tmp_vector[0]);
        hemmer_->getEval().leftRotate(tmp_vector[1].get(), 1 * 256,
                                      tmp_vector[1].get());
        hemmer_->maskFirstRowInplace(tmp_vector[1]);
        lora_b_weight.emplace_back(tmp_vector);

        for (u64 i = 1; i < 3; ++i) {
            tmp_vector.clear();
            hemmer_->getEval().leftRotate(lora_wb.get(), i * 2 * 256,
                                          tmp.get());
            for (u64 j = 0; j < 2; ++j) {
                tmp_vector.emplace_back(tmp);
            }
            hemmer_->maskFirstRowInplace(tmp_vector[0]);
            hemmer_->getEval().leftRotate(tmp_vector[1].get(), 1 * 256,
                                          tmp.get());
            hemmer_->maskFirstRowInplace(tmp_vector[1]);

            lora_b_weight.emplace_back(tmp_vector);
        }
        tmp_vector.clear();

        // repeated packing
        for (u64 i = 0; i < 2; ++i) {
            for (u64 rot = 1; rot < 256; rot <<= 1) {
                hemmer_->getEval().rightRotate(lora_a_output[i].get(), rot,
                                               tmp.get());
                hemmer_->addInplace(lora_a_output[i], tmp);
            }
        }

        for (u64 i = 0; i < 3; ++i) {
            for (u64 j = 0; j < 2; ++j) {
                for (u64 rot = 1; rot < 128; rot <<= 1) {
                    hemmer_->getEval().rightRotate(lora_b_weight[i][j].get(),
                                                   rot * 256, tmp.get());
                    hemmer_->addInplace(lora_b_weight[i][j], tmp);
                }
            }
        }

        std::vector<CtxtTensor> lora_b_output;
        lora_b_output.reserve(3);
        for (u64 i = 0; i < 3; ++i) {
            lora_b_output.emplace_back(lora_wb);
        }

        // hadamult & addition
        for (u64 i = 0; i < 3; ++i) {
            hemmer_->getEval().mult(lora_a_output[0].get(),
                                    lora_b_weight[i][0].get(),
                                    lora_b_output[i].get());
            hemmer_->getEval().mult(lora_a_output[1].get(),
                                    lora_b_weight[i][1].get(), tmp.get());
            hemmer_->addInplace(lora_b_output[i], tmp);
        }

        lora_output.emplace(t, lora_b_output);
    }
    // printElapsedTime("lora_b");

    // origianl
    /* for (const char t : lora_type) {
        hemmer_->bootstrap2(lora_output[t][0], lora_output[t][1]);
        hemmer_->bootstrap(lora_output[t][2]);

        if (rank == 0) {
            std::cout << "lora output " << t << std::endl;
            printing(lora_output[t]);
        }
    } */

    // hard coding
    hemmer_->bootstrap2(lora_output['q'][0], lora_output['q'][1]);
    hemmer_->bootstrap2(lora_output['q'][2], lora_output['k'][0]);
    hemmer_->bootstrap2(lora_output['k'][1], lora_output['k'][2]);
    hemmer_->bootstrap2(lora_output['v'][0], lora_output['v'][1]);
    hemmer_->bootstrap(lora_output['v'][2]);
    // printElapsedTime("lora BTS");

    // origianl code
    /* for (u64 i = 0 ; i < 3 ; ++i) {
        auto tmp = cur[i];
        hemmer_->transposeInplace(tmp);
        lora_module_->saveCtxtTensor(tmp, "tr_lora_in_a", 0,0,i);
    } */

    // hard coding
    for (u64 i = 0; i < 3; ++i) {
        auto tmp = cur[i];
        lora_module_->saveCtxtTensor(tmp, "tr_lora_in_a", 0, 0, i);
    }
    // printElapsedTime("lora_a tr and save");

    // origianl code
    /* for (const char t : lora_type) {
        const std::string lora_t = std::string(1,t);
        auto it = lora_a.find(t);
        hemmer_->transposeInplace(it->second);
        lora_module_->saveCtxtTensor(it->second, "tr_lora_in_b_" + lora_t, 0, 0,
    0);
    } */

    // hard coding
    // index = col position
    for (const char t : lora_type) {
        const std::string lora_t = std::string(1, t);
        auto lora_a_output = lora_a[t];
        for (u64 i = 0; i < 2; ++i) {
            lora_module_->saveCtxtTensor(lora_a_output[i],
                                         "tr_lora_in_b_" + lora_t, 0, 0, i);
        }
    }
    // printElapsedTime("lora_b tr and save");

    for (u64 i = 0; i < 3; ++i) {
        if (lora_output.count('q') == 1) {
            hemmer_->addInplace(query[i], lora_output['q'][i]);
        }
        if (lora_output.count('k') == 1) {
            hemmer_->addInplace(key[i], lora_output['k'][i]);
        }
        if (lora_output.count('v') == 1) {
            hemmer_->addInplace(value[i], lora_output['v'][i]);
        }
    }
    // printElapsedTime("lora addition");

    /* attention.. */
    // initialization
    std::vector<CtxtTensor> masked_query, masked_key;
    masked_query.reserve(6);
    masked_key.reserve(6);
    for (u64 i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            masked_query.push_back(query[i]);
            masked_key.push_back(key[i]);
        }
    }
    std::vector<CtxtTensor> one_matmul_query, one_matmul_key;
    one_matmul_query = masked_query;
    one_matmul_key = masked_key;

    // Q. query element -> divide q1 q2 q3 q4 >> q1 0 q3 0 , 0 q2 0 q4
    for (u64 i = 0; i < 3; ++i) {
        // masked matmul
        hemmer_->maskRightLeft(query[i], masked_query[2 * i],
                               masked_query[2 * i + 1]); // 10 -> 9
        hemmer_->maskRightLeft(key[i], masked_key[2 * i],
                               masked_key[2 * i + 1]); // 11 -> 10

        for (u64 j = 0; j < 2; ++j) {
            // lora forward
            lora_module_->saveCtxtTensor(masked_query[2 * i + j],
                                         "forward_res_q", 0, 0, 2 * i + j); // 9
            lora_module_->saveCtxtTensor(masked_key[2 * i + j], "forward_res_k",
                                         0, 0, 2 * i + j); // 10

            hemmer_->getEval().square(
                masked_query[2 * i + j].get(),
                one_matmul_query[2 * i + j].get()); // 9 -> 8
            hemmer_->getEval().square(
                masked_key[2 * i + j].get(),
                one_matmul_key[2 * i + j].get()); // 10 -> 9
        }
    }
    // printElapsedTime("GK masking and squaring");

    for (u64 i = 0; i < 3; ++i) {

        hemmer_->oneMatRotSumInplace(one_matmul_query[2 * i],
                                     one_matmul_query[2 * i + 1]); // 6 -> 5
        hemmer_->oneMatRotSumInplace(one_matmul_key[2 * i],
                                     one_matmul_key[2 * i + 1]); // 7 -> 6

        // target level: ?
        for (u64 j = 0; j < 2; ++j) {
            hemmer_->transposeInplace(one_matmul_key[2 * i + j],
                                      4); // 6 -> 4 (LD)
        }
    }
    // printElapsedTime("GK norm");

    // init
    std::vector<CtxtTensor> cur_qk;
    cur_qk.reserve(6);
    for (u64 i = 0; i < 3; ++i) {
        for (u64 j = 0; j < 2; ++j) {
            cur_qk.push_back(query[i]);
        }
    }

    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingInplace(masked_query[i]); // 8 -> 7
        hemmer_->transposeInplace(masked_key[i]);
        hemmer_->complexPackingRowInplace(masked_key[i]);
        // hemmer_->transposeComplexPackingInplace(masked_key[i]); //  9 -> 7
        cur_qk[i] =
            hemmer_->packedMatMul(masked_query[i], masked_key[i]); // 7 -> 4
        cur_qk[i] = hemmer_->repackCC(cur_qk[i]);                  // 4 -> 4

        hemmer_->getEval().mult(cur_qk[i].get(), -2, cur_qk[i].get()); // 4 -> 4

        hemmer_->getEval().add(cur_qk[i].get(), one_matmul_query[i].get(),
                               cur_qk[i].get()); // 4
        hemmer_->getEval().add(cur_qk[i].get(), one_matmul_key[i].get(),
                               cur_qk[i].get()); // 4
    }
    // printElapsedTime("qk CC");

    for (u64 i = 0; i < 3; ++i) {
        hemmer_->bootstrap2(cur_qk[i * 2], cur_qk[i * 2 + 1]); // 4 -> 12
    }
    // printElapsedTime("BTS");

    // 12 -> 11
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->getEval().mult(cur_qk[i].get(), -1.0 / (2 * std::sqrt(64)),
                                cur_qk[i].get()); // 12 -> 11
        hemmer_->getEval().add(cur_qk[i].get(), exp_message, cur_qk[i].get());

        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().add(cur_qk[i].get(), exp_message, tmp);
    }
    // printElapsedTime("dividing and summation of exp_mask");

    // exp evaluation.
    // Caution: should consider output level of exp by comparing with the output
    // preicison. 11 -> 5 ( 8 for complexpacking)
    hemmer_->expParallelInplace(cur_qk, layer_n_, true);
    // printElapsedTime("exp eval");

    /* if (rank == 0) {
        std::cout << "forward exp output" << std::endl;
        printing(cur_qk);
    } */

    for (u64 i = 0; i < 3; ++i) {
        hemmer_->bootstrap2_exponly(cur_qk[2 * i], cur_qk[2 * i + 1]);
        for (u64 j = 0; j < 2; ++j) {
            // for lora backward
            lora_module_->saveCtxtTensor(cur_qk[2 * i + j], "forward_res_exp",
                                         0, 0, 2 * i + j);
            hemmer_->getEval().levelDown(cur_qk[2 * i + j].get(), 9,
                                         cur_qk[2 * i + j].get());
        }
    }
    // std::cout << "exp outuput " << std::endl;
    // printing(cur_qk);

    // 9 -> 8
    for (u64 i = 0; i < cur_qk.size(); ++i) {
        hemmer_->dropoutExpInplace(cur_qk[i], "atn_exp", layer_n_, i);
    }

    // TODO: clean up loop index: n_iter_half*2 etc.
    std::vector<CtxtTensor> masked_value;
    masked_value.reserve(6);
    for (u64 i = 0; i < 3; ++i) {
        for (u64 j = 0; j < 2; ++j) {
            masked_value.push_back(value[i]); // 8
        }
    }

    // masked value
    // v11 v12 v13 v14 >> v11 0 v13 0 | 0 v12 0 v14 (even index)
    // v12 v22 v23 v24 >> v12 0 v23 0 | 0 v22 0 v24 (odd index)
    for (u64 i = 0; i < 3; ++i)
        hemmer_->maskRightLeft(value[i], masked_value[2 * i],
                               masked_value[2 * i + 1]); // 9 -> 8
    // printElapsedTime("masking");

    for (u64 i = 0; i < 6; ++i) {
        lora_module_->saveCtxtTensor(masked_value[i], "forward_res_v", 0, 0,
                                     i); // 8
    }

    for (u64 i = 0; i < 6; ++i) {
        // cur_qk[i] = hemmer_->singleCCMatMul(cur_qk[i], masked_value[i], 4);
        // // 7 -> 4
        hemmer_->complexPackingInplace(cur_qk[i]);          // 8 -> 7
        hemmer_->complexPackingRowInplace(masked_value[i]); // 8 -> 7

        cur_qk[i] = hemmer_->packedMatMul(
            cur_qk[i], masked_value[i]); // automatic level down...
    }
    // printElapsedTime("value MM");

    for (u64 i = 0; i < 6; ++i) {                 // 7 -> 4
        cur_qk[i] = hemmer_->repackCC(cur_qk[i]); // level consumpotion x
    }
    // printElapsedTime("value MM repack");

    for (u64 i = 0; i < 3; ++i) {
        hemmer_->bootstrap2(cur_qk[i * 2], cur_qk[i * 2 + 1]); // 12
    }
    // printElapsedTime("BTS");

    // collect into one ciphertext
    std::vector<CtxtTensor> cur_reuse;
    cur_reuse.reserve(3);
    for (u64 i = 0; i < 3; i++) {
        hemmer_->addInplace(cur_qk[i * 2], cur_qk[i * 2 + 1]); // 12
        cur_reuse.push_back(cur_qk[i * 2]);                    // 12
    }
    cur_qk.clear();

    std::vector<CtxtTensor> output;
    output.reserve(6);
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(
                hemmer_->complexPacking(cur_reuse[i * 2], cur_reuse[i * 2 + 1]),
                tmp);
        } else {
            hemmer_->matMulPre(cur_reuse[2], tmp);
        }

        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("wd", i, j);
            auto result = hemmer_->matMulReUse(tmp, weight);

            if (i == 0) {
                output.push_back(result);
            } else {
                hemmer_->addInplace(output[j], result);
            }
        }
    }
    // printElapsedTime("weight MM");

    /* Step 3. Post-process the result.. */
    for (u64 i = 0; i < 3; ++i) {
        auto repacked_output =
            hemmer_->repack(output[i * 2], output[i * 2 + 1]); // 12
        hemmer_->dropoutInplace(repacked_output, "atn_output", layer_n_,
                                i);                     // 10
        hemmer_->addInplace(input[i], repacked_output); // 10
    }
    // printElapsedTime("repacking");
    // std::cout << "attn output" << std::endl;
    // printing(input);
}

// Fine-tuning version
// TODO: Implemenation of AdamW optimizer.
void TransformerBlock::attention2_bert_time(std::vector<CtxtTensor> &input,
                                            const Message &exp_message,
                                            const std::string &lora_type) {
    // const u64 rank = static_cast<u64>(hemmer_->getRank());

    start = std::chrono::high_resolution_clock::now();

    std::vector<CtxtTensor> cur =
        hemmer_->LayerNorm(input, "atn", layer_n_, true); // 12 -> 6
    for (u64 i = 0; i < 3; ++i) {
        // TODO: replace name with fixed one.
        const auto &weight = getWeightMsg("norm1_w", i);
        const auto &bias = getWeightMsg("norm1_b", i);
        hemmer_->hadamardMultInplace(cur[i], weight);             // 6 -> 5
        hemmer_->getEval().add(cur[i].get(), bias, cur[i].get()); // 5 -> 5
    }
    printElapsedTime("atn LN");

    hemmer_->bootstrap2(cur[0], cur[1]);
    hemmer_->bootstrap(cur[2]);
    printElapsedTime("BTS");

    std::vector<CtxtTensor> temp_q, temp_k, temp_v;
    temp_q.reserve(6);
    temp_k.reserve(6);
    temp_v.reserve(6);
    std::vector<PtxtTensor> weights_q, weights_k, weights_v;
    weights_q.reserve(6);
    weights_k.reserve(6);
    weights_v.reserve(6);
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            // merged BTS >> traget_level = 5
            hemmer_->matMulPre(
                hemmer_->complexPacking(cur[i * 2], cur[i * 2 + 1]), tmp);
            // std::cout << "pre output " << tmp[0].getLevel() << std::endl;
        } else {
            hemmer_->matMulPre(cur[2], tmp);
        }
        weights_q.clear();
        weights_k.clear();
        weights_v.clear();
        for (u64 j = 0; j < 6; ++j) {
            weights_q.push_back(getWeight("wq", i, j));
            weights_k.push_back(getWeight("wk", i, j));
            weights_v.push_back(getWeight("wv", i, j));
        }
        for (u64 j = 0; j < 6; ++j) {
            auto result_q = hemmer_->matMulReUse(tmp, weights_q[j]);
            auto result_k = hemmer_->matMulReUse(tmp, weights_k[j]);
            auto result_v = hemmer_->matMulReUse(tmp, weights_v[j]);

            if (i == 0) {
                temp_q.push_back(result_q);
                temp_k.push_back(result_k);
                temp_v.push_back(result_v);
            } else {
                hemmer_->addInplace(temp_q[j], result_q);
                hemmer_->addInplace(temp_k[j], result_k);
                hemmer_->addInplace(temp_v[j], result_v);
            }
        }
    }
    printElapsedTime("q,k,v mm");

    // std::cout << "q,k,v mm ouptut " << temp_q[0].get().getLevel() <<
    // std::endl;

    std::vector<CtxtTensor> query, key, value;
    query.reserve(3);
    key.reserve(3);
    value.reserve(3);
    for (u64 i = 0; i < 3; ++i) {
        /* if (i == 0) {
            std::cout << "repack in: " << temp_q[0].get().getLevel() <<
        std::endl;
        } */
        query.push_back(
            hemmer_->repack(temp_q[i * 2], temp_q[i * 2 + 1])); // 11
        /* if (i == 0) {
            std::cout << "repack out: " << query[0].get().getLevel() <<
        std::endl; printing_masking(query[0]);
        } */
        key.push_back(hemmer_->repack(temp_k[i * 2], temp_k[i * 2 + 1])); // 11
        value.push_back(
            hemmer_->repack(temp_v[i * 2], temp_v[i * 2 + 1])); // 11

        hemmer_->getEval().levelDown(query[i].get(), 10, query[i].get());
        hemmer_->getEval().levelDown(value[i].get(), 9, value[i].get());
        hemmer_->getEval().levelDown(key[i].get(), 11, key[i].get());
    }
    printElapsedTime("q,k,v repack");

    /* std::cout << "cur[0] " << std::endl;
    printing_masking(cur[0]); */

    /* LoRA */
    // forward
    std::unordered_map<char, CtxtTensor> lora_a;

    start = std::chrono::high_resolution_clock::now();
    for (const char t : lora_type) {
        const std::string lora_t = std::string(1, t);
        auto lora_wa = lora_module_->getCtxtTensor_lora("lora_wa_" + lora_t, 0,
                                                        0, 0); // 9 or 8

        auto lora_a_tmp = hemmer_->matMulHighLow(
            cur[0], hemmer_->getLowColBlock(lora_wa, 0), 0, 5);
        lora_a.emplace(t, lora_a_tmp);
        for (u64 i = 1; i < 3; ++i) {
            auto it = lora_a.find(t);
            assert(it != lora_a.end());
            hemmer_->addInplace(
                it->second,
                hemmer_->matMulHighLow(
                    cur[i], hemmer_->getLowColBlock(lora_wa, i), i, 5));
        }
    }
    printElapsedTime("lora_a mm");

    // Caution: We have to consider LOW_RANK.
    for (const char t : lora_type) {
        auto it = lora_a.find(t);
        it->second = hemmer_->repackToOneCol(it->second, 0);
        hemmer_->bootstrap(it->second);

        /* if (t == 'q') {
            std::cout << "lora_a output" << std::endl;
            printing_masking(it->second);
        } */
    }
    printElapsedTime("lora_a repack");

    std::unordered_map<char, std::vector<CtxtTensor>> lora_output;
    start = std::chrono::high_resolution_clock::now();
    for (const char t : lora_type) {
        const std::string lora_t = std::string(1, t);
        auto lora_wb =
            lora_module_->getCtxtTensor_lora("lora_wb_" + lora_t, 0, 0, 0);

        auto it = lora_a.find(t);
        std::vector<CtxtTensor> tmp;
        for (u64 i = 0; i < 3; ++i) {
            tmp.emplace_back(hemmer_->matMulLowLow(
                it->second, hemmer_->getLowRowBlock(lora_wb, i), 0, i));
        }
        lora_output.emplace(t, tmp);
    }
    printElapsedTime("lora_b");

    /* std::cout << "lora_output[q]" << std::endl;
    printing(lora_output['q']); */

    // hard coding for qkv
    hemmer_->bootstrap2(lora_output['q'][0], lora_output['q'][1]);
    hemmer_->bootstrap2(lora_output['q'][2], lora_output['k'][0]);
    hemmer_->bootstrap2(lora_output['k'][1], lora_output['k'][2]);
    hemmer_->bootstrap2(lora_output['v'][0], lora_output['v'][1]);
    hemmer_->bootstrap(lora_output['v'][2]);
    printElapsedTime("lora BTS");

    for (u64 i = 0; i < 3; ++i) {
        auto tmp = cur[i];
        hemmer_->transposeInplace(tmp);
        lora_module_->saveCtxtTensor(tmp, "tr_lora_in_a", 0, 0, i);
    }
    printElapsedTime("lora_a tr and save");

    for (const char t : lora_type) {
        const std::string lora_t = std::string(1, t);
        auto it = lora_a.find(t);
        hemmer_->transposeInplace(it->second);
        lora_module_->saveCtxtTensor(it->second, "tr_lora_in_b_" + lora_t, 0, 0,
                                     0);
    }
    printElapsedTime("lora_b tr and save");

    for (u64 i = 0; i < 3; ++i) {
        if (lora_output.count('q') == 1) {
            hemmer_->addInplace(query[i], lora_output['q'][i]);
        }
        if (lora_output.count('k') == 1) {
            hemmer_->addInplace(key[i], lora_output['k'][i]);
        }
        if (lora_output.count('v') == 1) {
            hemmer_->addInplace(value[i], lora_output['v'][i]);
        }
    }
    printElapsedTime("lora addition");
    /* std::cout << "after addition, query[0]" << std::endl;
    printing_masking(query[0]); */

    /* hemmer_->bootstrap2(query[0], query[1]);
    hemmer_->bootstrap2(query[2], key[0]);
    hemmer_->bootstrap2(key[1], key[2]);
    hemmer_->bootstrap2(value[0], value[1]);
    hemmer_->bootstrap(value[2]);
    printElapsedTime("after addition BTS");

    for (u64 i = 0 ; i < 3 ; ++i) {
        hemmer_->getEval().levelDown(query[i].get(), 10, query[i].get());
        hemmer_->getEval().levelDown(value[i].get(), 9, value[i].get());
        hemmer_->getEval().levelDown(key[i].get(), 11, key[i].get());
    } */

    /* attention.. */
    // initialization
    std::vector<CtxtTensor> masked_query, masked_key;
    masked_query.reserve(6);
    masked_key.reserve(6);
    for (u64 i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            masked_query.push_back(query[i]);
            masked_key.push_back(key[i]);
        }
    }
    std::vector<CtxtTensor> one_matmul_query, one_matmul_key;
    one_matmul_query = masked_query;
    one_matmul_key = masked_key;

    // Q. query element -> divide q1 q2 q3 q4 >> q1 0 q3 0 , 0 q2 0 q4
    start = std::chrono::high_resolution_clock::now();
    for (u64 i = 0; i < 3; ++i) {
        // masked matmul
        hemmer_->maskRightLeft(query[i], masked_query[2 * i],
                               masked_query[2 * i + 1]); // 10 -> 9
        hemmer_->maskRightLeft(key[i], masked_key[2 * i],
                               masked_key[2 * i + 1]); // 11 -> 10

        for (u64 j = 0; j < 2; ++j) {
            // lora forward
            lora_module_->saveCtxtTensor(masked_query[2 * i + j],
                                         "forward_res_q", 0, 0, 2 * i + j); // 9
            lora_module_->saveCtxtTensor(masked_key[2 * i + j], "forward_res_k",
                                         0, 0, 2 * i + j); // 10

            hemmer_->getEval().square(
                masked_query[2 * i + j].get(),
                one_matmul_query[2 * i + j].get()); // 9 -> 8
            hemmer_->getEval().square(
                masked_key[2 * i + j].get(),
                one_matmul_key[2 * i + j].get()); // 10 -> 9
        }
    }
    printElapsedTime("GK masking and squaring / saving");

    start = std::chrono::high_resolution_clock::now();
    for (u64 i = 0; i < 3; ++i) {

        hemmer_->oneMatRotSumInplace(one_matmul_query[2 * i],
                                     one_matmul_query[2 * i + 1]); // 6 -> 5
        hemmer_->oneMatRotSumInplace(one_matmul_key[2 * i],
                                     one_matmul_key[2 * i + 1]); // 7 -> 6
        // target level: ?
        for (u64 j = 0; j < 2; ++j) {
            hemmer_->transposeInplace(one_matmul_key[2 * i + j],
                                      4); // 6 -> 4 (LD)
        }
    }
    printElapsedTime("GK norm");

    // init
    std::vector<CtxtTensor> cur_qk;
    cur_qk.reserve(6);
    for (u64 i = 0; i < 3; ++i) {
        for (u64 j = 0; j < 2; ++j) {
            cur_qk.push_back(query[i]);
        }
    }

    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingInplace(masked_query[i]); // 8 -> 7
        hemmer_->transposeInplace(masked_key[i]);
        hemmer_->complexPackingRowInplace(masked_key[i]);
        cur_qk[i] =
            hemmer_->packedMatMul(masked_query[i], masked_key[i]); // 7 -> 4
        cur_qk[i] = hemmer_->repackCC(cur_qk[i]);                  // 4 -> 4

        hemmer_->getEval().mult(cur_qk[i].get(), -2, cur_qk[i].get()); // 4 -> 4

        hemmer_->getEval().add(cur_qk[i].get(), one_matmul_query[i].get(),
                               cur_qk[i].get()); // 4
        hemmer_->getEval().add(cur_qk[i].get(), one_matmul_key[i].get(),
                               cur_qk[i].get()); // 4
    }
    printElapsedTime("qk CC");

    for (u64 i = 0; i < 3; ++i) {
        hemmer_->bootstrap2(cur_qk[i * 2], cur_qk[i * 2 + 1]); // 4 -> 12
    }
    printElapsedTime("BTS");

    // 12 -> 11
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->getEval().mult(cur_qk[i].get(), -1.0 / (2 * std::sqrt(64)),
                                cur_qk[i].get()); // 12 -> 11
        hemmer_->getEval().add(cur_qk[i].get(), exp_message, cur_qk[i].get());

        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().add(cur_qk[i].get(), exp_message, tmp);
    }
    printElapsedTime("dividing and summation of exp_mask");

    // exp evaluation.
    // Caution: should consider output level of exp by comparing with the output
    // preicison. 11 -> 5 ( 8 for complexpacking)
    hemmer_->expParallelInplace(cur_qk, layer_n_, true);
    printElapsedTime("exp eval");

    for (u64 i = 0; i < 3; ++i) {
        // hemmer_->bootstrap2_exponly(cur_qk[2*i], cur_qk[2*i+1]);
        for (u64 j = 0; j < 2; ++j) {
            // for lora backward
            lora_module_->saveCtxtTensor(cur_qk[2 * i + j], "forward_res_exp",
                                         0, 0, 2 * i + j);
            hemmer_->getEval().levelDown(cur_qk[2 * i + j].get(), 9,
                                         cur_qk[2 * i + j].get());
        }
    }
    printElapsedTime("saving");

    // 9 -> 8
    for (u64 i = 0; i < cur_qk.size(); ++i) {
        hemmer_->dropoutExpInplace(cur_qk[i], "atn_exp", layer_n_, i);
    }
    printElapsedTime("dropout");

    // TODO: clean up loop index: n_iter_half*2 etc.
    std::vector<CtxtTensor> masked_value;
    masked_value.reserve(6);
    for (u64 i = 0; i < 3; ++i) {
        for (u64 j = 0; j < 2; ++j) {
            masked_value.push_back(value[i]); // 8
        }
    }

    // masked value
    // v11 v12 v13 v14 >> v11 0 v13 0 | 0 v12 0 v14 (even index)
    // v12 v22 v23 v24 >> v12 0 v23 0 | 0 v22 0 v24 (odd index)
    for (u64 i = 0; i < 3; ++i)
        hemmer_->maskRightLeft(value[i], masked_value[2 * i],
                               masked_value[2 * i + 1]); // 9 -> 8
    printElapsedTime("masking");

    for (u64 i = 0; i < 6; ++i) {
        lora_module_->saveCtxtTensor(masked_value[i], "forward_res_v", 0, 0,
                                     i); // 8
    }
    printElapsedTime("saving");

    for (u64 i = 0; i < 6; ++i) {
        // cur_qk[i] = hemmer_->singleCCMatMul(cur_qk[i], masked_value[i], 4);
        // // 7 -> 4
        hemmer_->complexPackingInplace(cur_qk[i]);          // 8 -> 7
        hemmer_->complexPackingRowInplace(masked_value[i]); // 8 -> 7
    }
    printElapsedTime("value packing");

    for (u64 i = 0; i < 6; ++i) {

        cur_qk[i] = hemmer_->packedMatMul(
            cur_qk[i], masked_value[i]); // automatic level down...
    }
    printElapsedTime("value MM");

    for (u64 i = 0; i < 6; ++i) {                 // 7 -> 4
        cur_qk[i] = hemmer_->repackCC(cur_qk[i]); // level consumpotion x
    }
    printElapsedTime("value MM repack");

    for (u64 i = 0; i < 3; ++i) {
        hemmer_->bootstrap2(cur_qk[i * 2], cur_qk[i * 2 + 1]); // 12
    }
    printElapsedTime("BTS");

    // collect into one ciphertext
    std::vector<CtxtTensor> cur_reuse;
    cur_reuse.reserve(3);
    for (u64 i = 0; i < 3; i++) {
        hemmer_->addInplace(cur_qk[i * 2], cur_qk[i * 2 + 1]); // 12
        cur_reuse.push_back(cur_qk[i * 2]);                    // 12
    }
    cur_qk.clear();

    std::vector<CtxtTensor> output;
    output.reserve(6);
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(
                hemmer_->complexPacking(cur_reuse[i * 2], cur_reuse[i * 2 + 1]),
                tmp);
        } else {
            hemmer_->matMulPre(cur_reuse[2], tmp);
        }

        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("wd", i, j);
            auto result = hemmer_->matMulReUse(tmp, weight);

            if (i == 0) {
                output.push_back(result);
            } else {
                hemmer_->addInplace(output[j], result);
            }
        }
    }
    printElapsedTime("weight MM");

    /* Step 3. Post-process the result.. */
    for (u64 i = 0; i < 3; ++i) {
        auto repacked_output =
            hemmer_->repack(output[i * 2], output[i * 2 + 1]); // 12
        hemmer_->dropoutInplace(repacked_output, "atn_output", layer_n_,
                                i);                     // 10
        hemmer_->addInplace(input[i], repacked_output); // 10
    }
    printElapsedTime("repack and dropout");
    // std::cout << "attn output" << std::endl;
    // printing(input);
}

// Fine-tuning version
// TODO: Implemenation of AdamW optimizer.
void TransformerBlock::attention2_bert_loraOpti(std::vector<CtxtTensor> &input,
                                                const Message &exp_message,
                                                const std::string &lora_type) {
    const u64 rank = static_cast<u64>(hemmer_->getRank());

    start = std::chrono::high_resolution_clock::now();

    std::vector<CtxtTensor> cur =
        hemmer_->LayerNorm(input, "atn", layer_n_, true); // 12 -> 6
    for (u64 i = 0; i < 3; ++i) {
        // TODO: replace name with fixed one.
        const auto &weight = getWeightMsg("norm1_w", i);
        const auto &bias = getWeightMsg("norm1_b", i);
        hemmer_->hadamardMultInplace(cur[i], weight);             // 6 -> 5
        hemmer_->getEval().add(cur[i].get(), bias, cur[i].get()); // 5 -> 5
    }
    // printElapsedTime("atn LN");

    if (rank == 0) {
        std::cout << "atn LN" << std::endl;
        printing(cur);
    }

    hemmer_->bootstrap2(cur[0], cur[1]);
    hemmer_->bootstrap(cur[2]);
    // printElapsedTime("BTS");

    std::vector<CtxtTensor> temp_q, temp_k, temp_v;
    temp_q.reserve(6);
    temp_k.reserve(6);
    temp_v.reserve(6);
    std::vector<PtxtTensor> weights_q, weights_k, weights_v;
    weights_q.reserve(6);
    weights_k.reserve(6);
    weights_v.reserve(6);
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(
                hemmer_->complexPacking(cur[i * 2], cur[i * 2 + 1]), tmp);
        } else {
            hemmer_->matMulPre(cur[2], tmp);
        }
        weights_q.clear();
        weights_k.clear();
        weights_v.clear();
        for (u64 j = 0; j < 6; ++j) {
            weights_q.push_back(getWeight("wq", i, j));
            weights_k.push_back(getWeight("wk", i, j));
            weights_v.push_back(getWeight("wv", i, j));
        }
        for (u64 j = 0; j < 6; ++j) {
            auto result_q = hemmer_->matMulReUse(tmp, weights_q[j]);
            auto result_k = hemmer_->matMulReUse(tmp, weights_k[j]);
            auto result_v = hemmer_->matMulReUse(tmp, weights_v[j]);

            if (i == 0) {
                temp_q.push_back(result_q);
                temp_k.push_back(result_k);
                temp_v.push_back(result_v);
            } else {
                hemmer_->addInplace(temp_q[j], result_q);
                hemmer_->addInplace(temp_k[j], result_k);
                hemmer_->addInplace(temp_v[j], result_v);
            }
        }
    }
    // printElapsedTime("q,k,v mm");

    std::vector<CtxtTensor> query, key, value;
    query.reserve(3);
    key.reserve(3);
    value.reserve(3);
    for (u64 i = 0; i < 3; ++i) {
        query.push_back(
            hemmer_->repack(temp_q[i * 2], temp_q[i * 2 + 1]));           // 11
        key.push_back(hemmer_->repack(temp_k[i * 2], temp_k[i * 2 + 1])); // 11
        value.push_back(
            hemmer_->repack(temp_v[i * 2], temp_v[i * 2 + 1])); // 11

        hemmer_->getEval().levelDown(query[i].get(), 10, query[i].get());
        hemmer_->getEval().levelDown(value[i].get(), 9, value[i].get());
        hemmer_->getEval().levelDown(key[i].get(), 11, key[i].get());
    }
    // printElapsedTime("q,k,v repack");

    /* LoRA */
    // forward
    std::unordered_map<char, std::vector<CtxtTensor>> lora_a;
    start = std::chrono::high_resolution_clock::now();
    for (const char t : lora_type) {
        const std::string lora_t = std::string(1, t);
        auto lora_wa = lora_module_->getCtxtTensor_lora("lora_wa_" + lora_t, 0,
                                                        0, 0); // 9 or 8

        // hard coding for rank 2
        start = std::chrono::high_resolution_clock::now();
        hemmer_->getEval().levelDown(lora_wa.get(), 9, lora_wa.get());

        start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<CtxtTensor>> lora_a_weight;
        lora_a_weight.reserve(3);
        std::vector<CtxtTensor> lora_a_output;
        lora_a_output.reserve(2);

        for (u64 j = 0; j < 2; ++j) {
            lora_a_output.emplace_back(lora_wa);
        }

        auto tmp = lora_wa;

        // split weights
        hemmer_->maskFirstRowInplace(lora_a_output[0]);
        hemmer_->getEval().leftRotate(lora_a_output[1].get(), 1 * 256,
                                      lora_a_output[1].get());
        hemmer_->maskFirstRowInplace(lora_a_output[1]);
        lora_a_weight.emplace_back(lora_a_output);
        for (u64 i = 1; i < 3; ++i) {
            lora_a_output.clear();
            hemmer_->getEval().leftRotate(lora_wa.get(), i * 2 * 256,
                                          tmp.get());
            for (u64 j = 0; j < 2; ++j) {
                lora_a_output.emplace_back(tmp);
            }
            hemmer_->maskFirstRowInplace(lora_a_output[0]);
            hemmer_->getEval().leftRotate(lora_a_output[1].get(), 1 * 256,
                                          lora_a_output[1].get());
            hemmer_->maskFirstRowInplace(lora_a_output[1]);
            lora_a_weight.emplace_back(lora_a_output);
        }

        // repeated packing
        for (u64 i = 0; i < 3; ++i) {
            for (u64 j = 0; j < 2; ++j) {
                for (u64 rot = 1; rot < 128; rot <<= 1) {
                    hemmer_->getEval().rightRotate(lora_a_weight[i][j].get(),
                                                   rot * 256, tmp.get());
                    hemmer_->addInplace(lora_a_weight[i][j], tmp);
                }
            }
        }

        // hadamult
        hemmer_->getEval().mult(cur[0].get(), lora_a_weight[0][0].get(),
                                lora_a_output[0].get());
        hemmer_->getEval().mult(cur[0].get(), lora_a_weight[0][1].get(),
                                lora_a_output[1].get());
        for (u64 i = 1; i < 3; ++i) {
            hemmer_->getEval().mult(cur[i].get(), lora_a_weight[i][0].get(),
                                    tmp.get());
            hemmer_->addInplace(lora_a_output[0], tmp);
            hemmer_->getEval().mult(cur[i].get(), lora_a_weight[i][1].get(),
                                    tmp.get());
            hemmer_->addInplace(lora_a_output[1], tmp);
        }

        // addition
        for (u64 i = 0; i < 2; ++i) {
            for (u64 rot = 1; rot < 256; rot <<= 1) {
                hemmer_->getEval().leftRotate(lora_a_output[i].get(), rot,
                                              tmp.get());
                hemmer_->addInplace(lora_a_output[i], tmp);
            }
            hemmer_->maskFirstColOnlyInplace(lora_a_output[i]);
        }
        // printElapsedTime("lora_a hard coding ");
        lora_a.emplace(t, lora_a_output);

        // std::cout << "lora_a output level: " <<
        // lora_a_output[0].get().getLevel() << std::endl;
    }
    // printElapsedTime("lora_a mm");

    std::unordered_map<char, std::vector<CtxtTensor>> lora_output;
    start = std::chrono::high_resolution_clock::now();
    for (const char t : lora_type) {
        const std::string lora_t = std::string(1, t);
        auto lora_wb =
            lora_module_->getCtxtTensor_lora("lora_wb_" + lora_t, 0, 0, 0);

        // hard coding for rank 2
        hemmer_->getEval().levelDown(lora_wb.get(), 6, lora_wb.get());
        auto lora_a_output = lora_a[t];
        // split weights
        std::vector<std::vector<CtxtTensor>> lora_b_weight;
        lora_b_weight.reserve(3);
        std::vector<CtxtTensor> tmp_vector;
        tmp_vector.reserve(2);

        auto tmp = lora_wb;

        for (u64 i = 0; i < 2; ++i) {
            tmp_vector.emplace_back(lora_wb);
        }
        hemmer_->maskFirstRowInplace(tmp_vector[0]);
        hemmer_->getEval().leftRotate(tmp_vector[1].get(), 1 * 256,
                                      tmp_vector[1].get());
        hemmer_->maskFirstRowInplace(tmp_vector[1]);
        lora_b_weight.emplace_back(tmp_vector);

        for (u64 i = 1; i < 3; ++i) {
            tmp_vector.clear();
            hemmer_->getEval().leftRotate(lora_wb.get(), i * 2 * 256,
                                          tmp.get());
            for (u64 j = 0; j < 2; ++j) {
                tmp_vector.emplace_back(tmp);
            }
            hemmer_->maskFirstRowInplace(tmp_vector[0]);
            hemmer_->getEval().leftRotate(tmp_vector[1].get(), 1 * 256,
                                          tmp_vector[1].get());
            hemmer_->maskFirstRowInplace(tmp_vector[1]);

            lora_b_weight.emplace_back(tmp_vector);
        }
        tmp_vector.clear();

        // repeated packing
        for (u64 i = 0; i < 2; ++i) {
            for (u64 rot = 1; rot < 256; rot <<= 1) {
                hemmer_->getEval().rightRotate(lora_a_output[i].get(), rot,
                                               tmp.get());
                hemmer_->addInplace(lora_a_output[i], tmp);
            }
        }

        for (u64 i = 0; i < 3; ++i) {
            for (u64 j = 0; j < 2; ++j) {
                for (u64 rot = 1; rot < 128; rot <<= 1) {
                    hemmer_->getEval().rightRotate(lora_b_weight[i][j].get(),
                                                   rot * 256, tmp.get());
                    hemmer_->addInplace(lora_b_weight[i][j], tmp);
                }
            }
        }

        std::vector<CtxtTensor> lora_b_output;
        lora_b_output.reserve(3);
        for (u64 i = 0; i < 3; ++i) {
            lora_b_output.emplace_back(lora_wb);
        }

        // hadamult & addition
        for (u64 i = 0; i < 3; ++i) {
            hemmer_->getEval().mult(lora_a_output[0].get(),
                                    lora_b_weight[i][0].get(),
                                    lora_b_output[i].get());
            hemmer_->getEval().mult(lora_a_output[1].get(),
                                    lora_b_weight[i][1].get(), tmp.get());
            hemmer_->addInplace(lora_b_output[i], tmp);
        }

        lora_output.emplace(t, lora_b_output);
    }
    // printElapsedTime("lora_b mm");

    // hard coding for qkv
    hemmer_->bootstrap2(lora_output['q'][0], lora_output['q'][1]);
    hemmer_->bootstrap2(lora_output['q'][2], lora_output['k'][0]);
    hemmer_->bootstrap2(lora_output['k'][1], lora_output['k'][2]);
    hemmer_->bootstrap2(lora_output['v'][0], lora_output['v'][1]);
    hemmer_->bootstrap(lora_output['v'][2]);
    // printElapsedTime("lora BTS");

    // hard coding
    for (u64 i = 0; i < 3; ++i) {
        auto tmp = cur[i];
        hemmer_->getEval().levelDown(tmp.get(), 6, tmp.get());
        lora_module_->saveCtxtTensor(tmp, "tr_lora_in_a", 0, 0, i);
    }
    // printElapsedTime("save");

    // hard coding
    // index = col position
    for (const char t : lora_type) {
        const std::string lora_t = std::string(1, t);
        auto lora_a_output = lora_a[t];
        for (u64 i = 0; i < 2; ++i) {
            lora_module_->saveCtxtTensor(lora_a_output[i],
                                         "tr_lora_in_b_" + lora_t, 0, 0, i);
        }
    }
    // printElapsedTime("lora_b save");

    for (u64 i = 0; i < 3; ++i) {
        if (lora_output.count('q') == 1) {
            hemmer_->addInplace(query[i], lora_output['q'][i]);
        }
        if (lora_output.count('k') == 1) {
            hemmer_->addInplace(key[i], lora_output['k'][i]);
        }
        if (lora_output.count('v') == 1) {
            hemmer_->addInplace(value[i], lora_output['v'][i]);
        }
    }
    // printElapsedTime("lora addition");

    /* attention.. */
    // initialization
    std::vector<CtxtTensor> masked_query, masked_key;
    masked_query.reserve(6);
    masked_key.reserve(6);
    for (u64 i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            masked_query.push_back(query[i]);
            masked_key.push_back(key[i]);
        }
    }
    std::vector<CtxtTensor> one_matmul_query, one_matmul_key;
    one_matmul_query = masked_query;
    one_matmul_key = masked_key;

    // Q. query element -> divide q1 q2 q3 q4 >> q1 0 q3 0 , 0 q2 0 q4
    start = std::chrono::high_resolution_clock::now();
    for (u64 i = 0; i < 3; ++i) {
        // masked matmul
        hemmer_->maskRightLeft(query[i], masked_query[2 * i],
                               masked_query[2 * i + 1]); // 10 -> 9
        hemmer_->maskRightLeft(key[i], masked_key[2 * i],
                               masked_key[2 * i + 1]); // 11 -> 10

        for (u64 j = 0; j < 2; ++j) {
            // lora forward
            lora_module_->saveCtxtTensor(masked_query[2 * i + j],
                                         "forward_res_q", 0, 0, 2 * i + j); // 9
            lora_module_->saveCtxtTensor(masked_key[2 * i + j], "forward_res_k",
                                         0, 0, 2 * i + j); // 10

            hemmer_->getEval().square(
                masked_query[2 * i + j].get(),
                one_matmul_query[2 * i + j].get()); // 9 -> 8
            hemmer_->getEval().square(
                masked_key[2 * i + j].get(),
                one_matmul_key[2 * i + j].get()); // 10 -> 9
        }
    }
    // printElapsedTime("GK masking and squaring / saving");

    start = std::chrono::high_resolution_clock::now();
    for (u64 i = 0; i < 3; ++i) {

        hemmer_->oneMatRotSumInplace(one_matmul_query[2 * i],
                                     one_matmul_query[2 * i + 1]); // 6 -> 5
        hemmer_->oneMatRotSumInplace(one_matmul_key[2 * i],
                                     one_matmul_key[2 * i + 1]); // 7 -> 6
        // target level: ?
        for (u64 j = 0; j < 2; ++j) {
            hemmer_->transposeInplace(one_matmul_key[2 * i + j],
                                      4); // 6 -> 4 (LD)
        }
    }
    // printElapsedTime("GK norm");

    // init
    std::vector<CtxtTensor> cur_qk;
    cur_qk.reserve(6);
    for (u64 i = 0; i < 3; ++i) {
        for (u64 j = 0; j < 2; ++j) {
            cur_qk.push_back(query[i]);
        }
    }

    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingInplace(masked_query[i]); // 8 -> 7
        hemmer_->transposeInplace(masked_key[i]);
        hemmer_->complexPackingRowInplace(masked_key[i]);
        // hemmer_->transposeComplexPackingInplace(masked_key[i]); //  9 -> 7
        cur_qk[i] =
            hemmer_->packedMatMul(masked_query[i], masked_key[i]); // 7 -> 4
        cur_qk[i] = hemmer_->repackCC(cur_qk[i]);                  // 4 -> 4

        hemmer_->getEval().mult(cur_qk[i].get(), -2, cur_qk[i].get()); // 4 -> 4

        hemmer_->getEval().add(cur_qk[i].get(), one_matmul_query[i].get(),
                               cur_qk[i].get()); // 4
        hemmer_->getEval().add(cur_qk[i].get(), one_matmul_key[i].get(),
                               cur_qk[i].get()); // 4
    }
    // printElapsedTime("qk CC");

    for (u64 i = 0; i < 3; ++i) {
        hemmer_->bootstrap2(cur_qk[i * 2], cur_qk[i * 2 + 1]); // 4 -> 12
    }
    // printElapsedTime("BTS");

    // 12 -> 11
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->getEval().mult(cur_qk[i].get(), -1.0 / (2 * std::sqrt(64)),
                                cur_qk[i].get()); // 12 -> 11
        hemmer_->getEval().add(cur_qk[i].get(), exp_message, cur_qk[i].get());

        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().add(cur_qk[i].get(), exp_message, tmp);
    }
    // printElapsedTime("dividing and summation of exp_mask");

    // exp evaluation.
    // Caution: should consider output level of exp by comparing with the output
    // preicison. 11 -> 5 ( 8 for complexpacking)
    hemmer_->expParallelInplace(cur_qk, layer_n_, true);
    // printElapsedTime("exp eval");

    if (rank == 0) {
        std::cout << "exp output: " << std::endl;
        printing(cur_qk);
    }

    for (u64 i = 0; i < 3; ++i) {
        for (u64 j = 0; j < 2; ++j) {
            // for lora backward
            lora_module_->saveCtxtTensor(cur_qk[2 * i + j], "forward_res_exp",
                                         0, 0, 2 * i + j);

            // dropout version
            // hemmer_->getEval().levelDown(cur_qk[2*i+j].get(), 9,
            // cur_qk[2*i+j].get());

            // No dropout version
            hemmer_->getEval().levelDown(cur_qk[2 * i + j].get(), 8,
                                         cur_qk[2 * i + j].get());
        }
    }
    // printElapsedTime("bts and saving");

    // 9 -> 8
    /* for (u64 i = 0 ; i < cur_qk.size(); ++i) {
        hemmer_->dropoutExpInplace(cur_qk[i], "atn_exp", layer_n_, i);
    } */
    // printElapsedTime("dropout");

    // TODO: clean up loop index: n_iter_half*2 etc.
    std::vector<CtxtTensor> masked_value;
    masked_value.reserve(6);
    for (u64 i = 0; i < 3; ++i) {
        for (u64 j = 0; j < 2; ++j) {
            masked_value.push_back(value[i]); // 8
        }
    }

    // masked value
    // v11 v12 v13 v14 >> v11 0 v13 0 | 0 v12 0 v14 (even index)
    // v12 v22 v23 v24 >> v12 0 v23 0 | 0 v22 0 v24 (odd index)
    for (u64 i = 0; i < 3; ++i)
        hemmer_->maskRightLeft(value[i], masked_value[2 * i],
                               masked_value[2 * i + 1]); // 9 -> 8
    // printElapsedTime("masking");

    for (u64 i = 0; i < 6; ++i) {
        lora_module_->saveCtxtTensor(masked_value[i], "forward_res_v", 0, 0,
                                     i); // 8
    }
    // printElapsedTime("saving");

    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingInplace(cur_qk[i]);          // 8 -> 7
        hemmer_->complexPackingRowInplace(masked_value[i]); // 8 -> 7
    }
    // printElapsedTime("value packing");

    for (u64 i = 0; i < 6; ++i) {
        cur_qk[i] = hemmer_->packedMatMul(
            cur_qk[i], masked_value[i]); // automatic level down.
    }
    // printElapsedTime("value MM");

    for (u64 i = 0; i < 6; ++i) {                 // 7 -> 4
        cur_qk[i] = hemmer_->repackCC(cur_qk[i]); // level consumpotion x
    }
    // printElapsedTime("value MM repack");

    for (u64 i = 0; i < 3; ++i) {
        hemmer_->bootstrap2(cur_qk[i * 2], cur_qk[i * 2 + 1]); // 12
    }
    // printElapsedTime("BTS");

    // collect into one ciphertext
    std::vector<CtxtTensor> cur_reuse;
    cur_reuse.reserve(3);
    for (u64 i = 0; i < 3; i++) {
        hemmer_->addInplace(cur_qk[i * 2], cur_qk[i * 2 + 1]); // 12
        cur_reuse.push_back(cur_qk[i * 2]);                    // 12
    }
    cur_qk.clear();

    std::vector<CtxtTensor> output;
    output.reserve(6);
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(
                hemmer_->complexPacking(cur_reuse[i * 2], cur_reuse[i * 2 + 1]),
                tmp);
        } else {
            hemmer_->matMulPre(cur_reuse[2], tmp);
        }

        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("wd", i, j);
            auto result = hemmer_->matMulReUse(tmp, weight);

            if (i == 0) {
                output.push_back(result);
            } else {
                hemmer_->addInplace(output[j], result);
            }
        }
    }
    // printElapsedTime("weight MM");

    /* Step 3. Post-process the result.. */
    for (u64 i = 0; i < 3; ++i) {
        auto repacked_output =
            hemmer_->repack(output[i * 2], output[i * 2 + 1]); // 12
        // hemmer_->dropoutInplace(repacked_output, "atn_output", layer_n_, i);
        // //10
        hemmer_->addInplace(input[i], repacked_output); // 10
    }
    // printElapsedTime("repack and dropout");
}

// Fine-tuning version
// TODO: Implemenation of AdamW optimizer.
void TransformerBlock::attention2_bert_loraOpti_eval(
    std::vector<CtxtTensor> &input, const Message &exp_message,
    const std::string &lora_type) {
    const u64 rank = static_cast<u64>(hemmer_->getRank());

    start = std::chrono::high_resolution_clock::now();

    std::vector<CtxtTensor> cur =
        hemmer_->LayerNorm(input, "atn", layer_n_, false); // 12 -> 6
    for (u64 i = 0; i < 3; ++i) {
        // TODO: replace name with fixed one.
        const auto &weight = getWeightMsg("norm1_w", i);
        const auto &bias = getWeightMsg("norm1_b", i);
        hemmer_->hadamardMultInplace(cur[i], weight);             // 6 -> 5
        hemmer_->getEval().add(cur[i].get(), bias, cur[i].get()); // 5 -> 5
    }
    // printElapsedTime("atn LN");

    /* if (rank == 0) {
        std::cout << "after atn LN" << std::endl;
        printing(cur);
    } */

    hemmer_->bootstrap2(cur[0], cur[1]);
    hemmer_->bootstrap(cur[2]);
    // printElapsedTime("BTS");

    /* if (rank == 0) {
        std::cout << "after BTS" << std::endl;
        printing(cur);
    } */

    std::vector<CtxtTensor> temp_q, temp_k, temp_v;
    temp_q.reserve(6);
    temp_k.reserve(6);
    temp_v.reserve(6);
    std::vector<PtxtTensor> weights_q, weights_k, weights_v;
    weights_q.reserve(6);
    weights_k.reserve(6);
    weights_v.reserve(6);
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(
                hemmer_->complexPacking(cur[i * 2], cur[i * 2 + 1]), tmp);
        } else {
            hemmer_->matMulPre(cur[2], tmp);
        }
        weights_q.clear();
        weights_k.clear();
        weights_v.clear();
        for (u64 j = 0; j < 6; ++j) {
            weights_q.push_back(getWeight("wq", i, j));
            weights_k.push_back(getWeight("wk", i, j));
            weights_v.push_back(getWeight("wv", i, j));
        }
        for (u64 j = 0; j < 6; ++j) {
            auto result_q = hemmer_->matMulReUse(tmp, weights_q[j]);
            auto result_k = hemmer_->matMulReUse(tmp, weights_k[j]);
            auto result_v = hemmer_->matMulReUse(tmp, weights_v[j]);

            if (i == 0) {
                temp_q.push_back(result_q);
                temp_k.push_back(result_k);
                temp_v.push_back(result_v);
            } else {
                hemmer_->addInplace(temp_q[j], result_q);
                hemmer_->addInplace(temp_k[j], result_k);
                hemmer_->addInplace(temp_v[j], result_v);
            }
        }
    }
    // printElapsedTime("q,k,v mm");

    std::vector<CtxtTensor> query, key, value;
    query.reserve(3);
    key.reserve(3);
    value.reserve(3);
    for (u64 i = 0; i < 3; ++i) {
        query.push_back(
            hemmer_->repack(temp_q[i * 2], temp_q[i * 2 + 1]));           // 11
        key.push_back(hemmer_->repack(temp_k[i * 2], temp_k[i * 2 + 1])); // 11
        value.push_back(
            hemmer_->repack(temp_v[i * 2], temp_v[i * 2 + 1])); // 11

        hemmer_->getEval().levelDown(query[i].get(), 10, query[i].get());
        hemmer_->getEval().levelDown(value[i].get(), 9, value[i].get());
        hemmer_->getEval().levelDown(key[i].get(), 11, key[i].get());
    }
    // printElapsedTime("q,k,v repack");

    /* if (rank == 0) {
        std::cout << "after MM" << std::endl;
        printing(query);
    } */

    /* LoRA */
    // forward
    std::unordered_map<char, std::vector<CtxtTensor>> lora_a;
    // std::unordered_map<char, CtxtTensor> lora_a;
    start = std::chrono::high_resolution_clock::now();
    for (const char t : lora_type) {
        const std::string lora_t = std::string(1, t);
        auto lora_wa = lora_module_->getCtxtTensor_lora_test(
            "lora_wa_" + lora_t, 0, 0, 0); // 9 or 8

        /* if (t == 'q' && rank == 0) {
            std::cout << "weight: " << std::endl;
            printing_masking(lora_wa);
        } */

        if (rank == 0) {
            auto pth_weight = hemmer_->decrypt2(lora_wa);
            lora_module_->saveTorchTensor(pth_weight, "lora_wa_" + lora_t, 0);
        }

        // hard coding for rank 2
        start = std::chrono::high_resolution_clock::now();
        hemmer_->getEval().levelDown(lora_wa.get(), 9, lora_wa.get());

        start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<CtxtTensor>> lora_a_weight;
        lora_a_weight.reserve(3);
        std::vector<CtxtTensor> lora_a_output;
        lora_a_output.reserve(2);

        for (u64 j = 0; j < 2; ++j) {
            lora_a_output.emplace_back(lora_wa);
        }

        auto tmp = lora_wa;

        // split weights
        hemmer_->maskFirstRowInplace(lora_a_output[0]);
        hemmer_->getEval().leftRotate(lora_a_output[1].get(), 1 * 256,
                                      lora_a_output[1].get());
        hemmer_->maskFirstRowInplace(lora_a_output[1]);
        lora_a_weight.emplace_back(lora_a_output);
        for (u64 i = 1; i < 3; ++i) {
            lora_a_output.clear();
            hemmer_->getEval().leftRotate(lora_wa.get(), i * 2 * 256,
                                          tmp.get());
            for (u64 j = 0; j < 2; ++j) {
                lora_a_output.emplace_back(tmp);
            }
            hemmer_->maskFirstRowInplace(lora_a_output[0]);
            hemmer_->getEval().leftRotate(lora_a_output[1].get(), 1 * 256,
                                          lora_a_output[1].get());
            hemmer_->maskFirstRowInplace(lora_a_output[1]);
            lora_a_weight.emplace_back(lora_a_output);
        }

        // repeated packing
        for (u64 i = 0; i < 3; ++i) {
            for (u64 j = 0; j < 2; ++j) {
                for (u64 rot = 1; rot < 128; rot <<= 1) {
                    hemmer_->getEval().rightRotate(lora_a_weight[i][j].get(),
                                                   rot * 256, tmp.get());
                    hemmer_->addInplace(lora_a_weight[i][j], tmp);
                }
            }
        }

        // hadamult
        hemmer_->getEval().mult(cur[0].get(), lora_a_weight[0][0].get(),
                                lora_a_output[0].get());
        hemmer_->getEval().mult(cur[0].get(), lora_a_weight[0][1].get(),
                                lora_a_output[1].get());
        for (u64 i = 1; i < 3; ++i) {
            hemmer_->getEval().mult(cur[i].get(), lora_a_weight[i][0].get(),
                                    tmp.get());
            hemmer_->addInplace(lora_a_output[0], tmp);
            hemmer_->getEval().mult(cur[i].get(), lora_a_weight[i][1].get(),
                                    tmp.get());
            hemmer_->addInplace(lora_a_output[1], tmp);
        }

        // addition
        for (u64 i = 0; i < 2; ++i) {
            for (u64 rot = 1; rot < 256; rot <<= 1) {
                hemmer_->getEval().leftRotate(lora_a_output[i].get(), rot,
                                              tmp.get());
                hemmer_->addInplace(lora_a_output[i], tmp);
            }
            hemmer_->maskFirstColOnlyInplace(lora_a_output[i]);
        }
        // printElapsedTime("lora_a hard coding ");
        lora_a.emplace(t, lora_a_output);
    }
    // printElapsedTime("lora_a mm");

    std::unordered_map<char, std::vector<CtxtTensor>> lora_output;
    start = std::chrono::high_resolution_clock::now();
    for (const char t : lora_type) {
        const std::string lora_t = std::string(1, t);
        auto lora_wb =
            lora_module_->getCtxtTensor_lora_test("lora_wb_" + lora_t, 0, 0, 0);

        if (rank == 0) {
            auto pth_weight = hemmer_->decrypt2(lora_wb);
            lora_module_->saveTorchTensor(pth_weight, "lora_wb_" + lora_t, 0);
        }

        // hard coding for rank 2
        hemmer_->getEval().levelDown(lora_wb.get(), 6, lora_wb.get());
        auto lora_a_output = lora_a[t];
        // split weights
        std::vector<std::vector<CtxtTensor>> lora_b_weight;
        lora_b_weight.reserve(3);
        std::vector<CtxtTensor> tmp_vector;
        tmp_vector.reserve(2);

        auto tmp = lora_wb;

        for (u64 i = 0; i < 2; ++i) {
            tmp_vector.emplace_back(lora_wb);
        }
        hemmer_->maskFirstRowInplace(tmp_vector[0]);
        hemmer_->getEval().leftRotate(tmp_vector[1].get(), 1 * 256,
                                      tmp_vector[1].get());
        hemmer_->maskFirstRowInplace(tmp_vector[1]);
        lora_b_weight.emplace_back(tmp_vector);

        for (u64 i = 1; i < 3; ++i) {
            tmp_vector.clear();
            hemmer_->getEval().leftRotate(lora_wb.get(), i * 2 * 256,
                                          tmp.get());
            for (u64 j = 0; j < 2; ++j) {
                tmp_vector.emplace_back(tmp);
            }
            hemmer_->maskFirstRowInplace(tmp_vector[0]);
            hemmer_->getEval().leftRotate(tmp_vector[1].get(), 1 * 256,
                                          tmp_vector[1].get());
            hemmer_->maskFirstRowInplace(tmp_vector[1]);

            lora_b_weight.emplace_back(tmp_vector);
        }
        tmp_vector.clear();

        // repeated packing
        for (u64 i = 0; i < 2; ++i) {
            for (u64 rot = 1; rot < 256; rot <<= 1) {
                hemmer_->getEval().rightRotate(lora_a_output[i].get(), rot,
                                               tmp.get());
                hemmer_->addInplace(lora_a_output[i], tmp);
            }
        }

        for (u64 i = 0; i < 3; ++i) {
            for (u64 j = 0; j < 2; ++j) {
                for (u64 rot = 1; rot < 128; rot <<= 1) {
                    hemmer_->getEval().rightRotate(lora_b_weight[i][j].get(),
                                                   rot * 256, tmp.get());
                    hemmer_->addInplace(lora_b_weight[i][j], tmp);
                }
            }
        }

        std::vector<CtxtTensor> lora_b_output;
        lora_b_output.reserve(3);
        for (u64 i = 0; i < 3; ++i) {
            lora_b_output.emplace_back(lora_wb);
        }

        // hadamult & addition
        for (u64 i = 0; i < 3; ++i) {
            hemmer_->getEval().mult(lora_a_output[0].get(),
                                    lora_b_weight[i][0].get(),
                                    lora_b_output[i].get());
            hemmer_->getEval().mult(lora_a_output[1].get(),
                                    lora_b_weight[i][1].get(), tmp.get());
            hemmer_->addInplace(lora_b_output[i], tmp);
        }
        lora_output.emplace(t, lora_b_output);
    }
    // printElapsedTime("lora_b mm");

    /* if (rank == 0) {
        std::cout << "after lora output" << std::endl;
        printing(lora_output['q']);
    } */

    // hard coding for qkv
    if (std::find(lora_type.begin(), lora_type.end(), 'q') != lora_type.end()) {
        hemmer_->bootstrap2(lora_output['q'][0], lora_output['q'][1]);
        hemmer_->bootstrap2(lora_output['q'][2], lora_output['k'][0]);
        hemmer_->bootstrap2(lora_output['k'][1], lora_output['k'][2]);
        hemmer_->bootstrap2(lora_output['v'][0], lora_output['v'][1]);
        hemmer_->bootstrap(lora_output['v'][2]);
    }

    /* if (rank == 0) {
        std::cout << "after lora output" << std::endl;
        printing(lora_output['q']);
    } */

    /* for (const char t : lora_type) {

        hemmer_->bootstrap2(lora_output[t][0], lora_output[t][1]);
        hemmer_->bootstrap(lora_output[t][2]);

        if (rank == 0 && t == 'q') {
            std::cout << "lora_output " << std::string(1,t) << std::endl;
            printing(lora_output[t]);
        }
    } */
    // printElapsedTime("lora BTS");

    for (u64 i = 0; i < 3; ++i) {
        if (lora_output.count('q') == 1) {
            hemmer_->addInplace(query[i], lora_output['q'][i]);
        }
        if (lora_output.count('k') == 1) {
            hemmer_->addInplace(key[i], lora_output['k'][i]);
        }
        if (lora_output.count('v') == 1) {
            hemmer_->addInplace(value[i], lora_output['v'][i]);
        }
    }
    // printElapsedTime("lora addition");

    /* if (rank == 0) {
        std::cout << "after lora addition, q " << std::endl;
        printing(query);

        std::cout << "after lora addition, k " << std::endl;
        printing(key);

        std::cout << "after lora addition, v " << std::endl;
        printing(value);
    } */

    /* attention.. */
    // initialization
    std::vector<CtxtTensor> masked_query, masked_key;
    masked_query.reserve(6);
    masked_key.reserve(6);
    for (u64 i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            masked_query.push_back(query[i]);
            masked_key.push_back(key[i]);
        }
    }
    std::vector<CtxtTensor> one_matmul_query, one_matmul_key;
    one_matmul_query = masked_query;
    one_matmul_key = masked_key;

    // Q. query element -> divide q1 q2 q3 q4 >> q1 0 q3 0 , 0 q2 0 q4
    for (u64 i = 0; i < 3; ++i) {
        // masked matmul
        hemmer_->maskRightLeft(query[i], masked_query[2 * i],
                               masked_query[2 * i + 1]); // 10 -> 9
        hemmer_->maskRightLeft(key[i], masked_key[2 * i],
                               masked_key[2 * i + 1]); // 11 -> 10
        for (u64 j = 0; j < 2; ++j) {
            hemmer_->getEval().square(
                masked_query[2 * i + j].get(),
                one_matmul_query[2 * i + j].get()); // 9 -> 8
            hemmer_->getEval().square(
                masked_key[2 * i + j].get(),
                one_matmul_key[2 * i + j].get()); // 10 -> 9
        }
    }
    // printElapsedTime("GK masking and squaring");

    for (u64 i = 0; i < 3; ++i) {

        hemmer_->oneMatRotSumInplace(one_matmul_query[2 * i],
                                     one_matmul_query[2 * i + 1]); // 6 -> 5
        hemmer_->oneMatRotSumInplace(one_matmul_key[2 * i],
                                     one_matmul_key[2 * i + 1]); // 7 -> 6

        // target level: ?
        for (u64 j = 0; j < 2; ++j) {
            hemmer_->transposeInplace(one_matmul_key[2 * i + j],
                                      4); // 6 -> 4 (LD)
        }
    }
    // printElapsedTime("GK norm");

    // init
    std::vector<CtxtTensor> cur_qk;
    cur_qk.reserve(6);
    for (u64 i = 0; i < 3; ++i) {
        for (u64 j = 0; j < 2; ++j) {
            cur_qk.push_back(query[i]);
        }
    }

    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingInplace(masked_query[i]); // 8 -> 7
        hemmer_->transposeInplace(masked_key[i]);
        hemmer_->complexPackingRowInplace(masked_key[i]);
        // hemmer_->transposeComplexPackingInplace(masked_key[i]); //  9 -> 7
        cur_qk[i] =
            hemmer_->packedMatMul(masked_query[i], masked_key[i]); // 7 -> 4
        cur_qk[i] = hemmer_->repackCC(cur_qk[i]);                  // 4 -> 4

        hemmer_->getEval().mult(cur_qk[i].get(), -2, cur_qk[i].get()); // 4 -> 4

        hemmer_->getEval().add(cur_qk[i].get(), one_matmul_query[i].get(),
                               cur_qk[i].get()); // 4
        hemmer_->getEval().add(cur_qk[i].get(), one_matmul_key[i].get(),
                               cur_qk[i].get()); // 4
    }
    // printElapsedTime("qk CC");

    for (u64 i = 0; i < 3; ++i) {
        hemmer_->bootstrap2(cur_qk[i * 2], cur_qk[i * 2 + 1]); // 4 -> 12
    }
    // printElapsedTime("BTS");

    // 12 -> 11
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->getEval().mult(cur_qk[i].get(), -1.0 / (2 * std::sqrt(64)),
                                cur_qk[i].get()); // 12 -> 11
        hemmer_->getEval().add(cur_qk[i].get(), exp_message, cur_qk[i].get());

        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().add(cur_qk[i].get(), exp_message, tmp);
    }
    // printElapsedTime("dividing and summation of exp_mask");

    // exp evaluation.
    // Caution: should consider output level of exp by comparing with the output
    // preicison. 11 -> 5 ( 8 for complexpacking)
    /* if (rank == 0) {
        std::cout << "exp input" << std::endl;
        printing(cur_qk);
    } */

    hemmer_->expParallelInplace(cur_qk, layer_n_, false);
    // printElapsedTime("exp eval");

    /* if (rank == 0) {
        std::cout << "after exp eval" << std::endl;
        printing(cur_qk);
    } */

    for (u64 i = 0; i < 3; ++i) {
        // hemmer_->bootstrap2_exponly(cur_qk[2*i], cur_qk[2*i+1]);
        for (u64 j = 0; j < 2; ++j) {
            hemmer_->getEval().levelDown(cur_qk[2 * i + j].get(), 8,
                                         cur_qk[2 * i + j].get());
        }
    }
    // printElapsedTime("BTS");
    // std::cout << "exp outuput " << std::endl;
    // printing(cur_qk);

    // 9 -> 8
    /* for (u64 i = 0 ; i < cur_qk.size(); ++i) {
        hemmer_->dropoutExpInplace(cur_qk[i], "atn_exp", layer_n_, i);
    } */

    // TODO: clean up loop index: n_iter_half*2 etc.
    std::vector<CtxtTensor> masked_value;
    masked_value.reserve(6);
    for (u64 i = 0; i < 3; ++i) {
        for (u64 j = 0; j < 2; ++j) {
            masked_value.push_back(value[i]); // 8
        }
    }

    // masked value
    // v11 v12 v13 v14 >> v11 0 v13 0 | 0 v12 0 v14 (even index)
    // v12 v22 v23 v24 >> v12 0 v23 0 | 0 v22 0 v24 (odd index)
    for (u64 i = 0; i < 3; ++i)
        hemmer_->maskRightLeft(value[i], masked_value[2 * i],
                               masked_value[2 * i + 1]); // 9 -> 8
    // printElapsedTime("masking");

    for (u64 i = 0; i < 6; ++i) {
        // cur_qk[i] = hemmer_->singleCCMatMul(cur_qk[i], masked_value[i], 4);
        // // 7 -> 4
        hemmer_->complexPackingInplace(cur_qk[i]);          // 8 -> 7
        hemmer_->complexPackingRowInplace(masked_value[i]); // 8 -> 7

        cur_qk[i] = hemmer_->packedMatMul(
            cur_qk[i], masked_value[i]); // automatic level down...
    }
    // printElapsedTime("value MM");

    for (u64 i = 0; i < 6; ++i) {                 // 7 -> 4
        cur_qk[i] = hemmer_->repackCC(cur_qk[i]); // level consumption x
    }
    // printElapsedTime("value MM repack");

    for (u64 i = 0; i < 3; ++i) {
        hemmer_->bootstrap2(cur_qk[i * 2], cur_qk[i * 2 + 1]); // 12
    }
    // printElapsedTime("BTS");

    // collect into one ciphertext
    std::vector<CtxtTensor> cur_reuse;
    cur_reuse.reserve(3);
    for (u64 i = 0; i < 3; i++) {
        hemmer_->addInplace(cur_qk[i * 2], cur_qk[i * 2 + 1]); // 12
        cur_reuse.push_back(cur_qk[i * 2]);                    // 12
    }
    cur_qk.clear();

    /* if (rank == 0) {
        std::cout << "score @ v mm " << std::endl;
        printing(cur_reuse);
    } */

    std::vector<CtxtTensor> output;
    output.reserve(6);
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(
                hemmer_->complexPacking(cur_reuse[i * 2], cur_reuse[i * 2 + 1]),
                tmp);
        } else {
            hemmer_->matMulPre(cur_reuse[2], tmp);
        }

        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("wd", i, j);
            auto result = hemmer_->matMulReUse(tmp, weight);

            if (i == 0) {
                output.push_back(result);
            } else {
                hemmer_->addInplace(output[j], result);
            }
        }
    }
    // printElapsedTime("weight MM");
    /* if (rank == 0 ){
        std::cout << "after dense mm" << std::endl;
        printing(output);
    } */

    /* Step 3. Post-process the result.. */
    for (u64 i = 0; i < 3; ++i) {
        auto repacked_output =
            hemmer_->repack(output[i * 2], output[i * 2 + 1]); // 12

        // hemmer_->dropoutInplace(repacked_output, "atn_output", layer_n_, i);
        // //10

        hemmer_->addInplace(input[i], repacked_output); // 10
    }
    // printElapsedTime("repacking");

    /* std::cout << "attn output" << std::endl;
    printing(input); */
}

// Fine-tuning version
// TODO: Implemenation of AdamW optimizer.
void TransformerBlock::attention2_bert_loraOpti_time(
    std::vector<CtxtTensor> &input, const Message &exp_message,
    const std::string &lora_type) {
    // const u64 rank = static_cast<u64>(hemmer_->getRank());

    start = std::chrono::high_resolution_clock::now();

    std::vector<CtxtTensor> cur =
        hemmer_->LayerNorm(input, "atn", layer_n_, true); // 12 -> 6
    for (u64 i = 0; i < 3; ++i) {
        // TODO: replace name with fixed one.
        const auto &weight = getWeightMsg("norm1_w", i);
        const auto &bias = getWeightMsg("norm1_b", i);
        hemmer_->hadamardMultInplace(cur[i], weight);             // 6 -> 5
        hemmer_->getEval().add(cur[i].get(), bias, cur[i].get()); // 5 -> 5
    }
    printElapsedTime("atn LN");

    /* if (rank == 0) {
        std::cout << "atn LN" << std::endl;
        printing(cur);
    } */

    hemmer_->bootstrap2(cur[0], cur[1]);
    hemmer_->bootstrap(cur[2]);
    printElapsedTime("BTS");

    std::vector<CtxtTensor> temp_q, temp_k, temp_v;
    temp_q.reserve(6);
    temp_k.reserve(6);
    temp_v.reserve(6);
    std::vector<PtxtTensor> weights_q, weights_k, weights_v;
    weights_q.reserve(6);
    weights_k.reserve(6);
    weights_v.reserve(6);
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(
                hemmer_->complexPacking(cur[i * 2], cur[i * 2 + 1]), tmp);
        } else {
            hemmer_->matMulPre(cur[2], tmp);
        }
        weights_q.clear();
        weights_k.clear();
        weights_v.clear();
        for (u64 j = 0; j < 6; ++j) {
            weights_q.push_back(getWeight("wq", i, j));
            weights_k.push_back(getWeight("wk", i, j));
            weights_v.push_back(getWeight("wv", i, j));
        }
        for (u64 j = 0; j < 6; ++j) {
            auto result_q = hemmer_->matMulReUse(tmp, weights_q[j]);
            auto result_k = hemmer_->matMulReUse(tmp, weights_k[j]);
            auto result_v = hemmer_->matMulReUse(tmp, weights_v[j]);

            if (i == 0) {
                temp_q.push_back(result_q);
                temp_k.push_back(result_k);
                temp_v.push_back(result_v);
            } else {
                hemmer_->addInplace(temp_q[j], result_q);
                hemmer_->addInplace(temp_k[j], result_k);
                hemmer_->addInplace(temp_v[j], result_v);
            }
        }
    }
    printElapsedTime("q,k,v mm");

    std::vector<CtxtTensor> query, key, value;
    query.reserve(3);
    key.reserve(3);
    value.reserve(3);
    for (u64 i = 0; i < 3; ++i) {
        query.push_back(
            hemmer_->repack(temp_q[i * 2], temp_q[i * 2 + 1]));           // 11
        key.push_back(hemmer_->repack(temp_k[i * 2], temp_k[i * 2 + 1])); // 11
        value.push_back(
            hemmer_->repack(temp_v[i * 2], temp_v[i * 2 + 1])); // 11

        hemmer_->getEval().levelDown(query[i].get(), 10, query[i].get());
        hemmer_->getEval().levelDown(value[i].get(), 9, value[i].get());
        hemmer_->getEval().levelDown(key[i].get(), 11, key[i].get());
    }
    printElapsedTime("q,k,v repack");

    return;

    /* LoRA */
    // forward
    std::unordered_map<char, std::vector<CtxtTensor>> lora_a;
    start = std::chrono::high_resolution_clock::now();
    for (const char t : lora_type) {
        const std::string lora_t = std::string(1, t);
        auto lora_wa = lora_module_->getCtxtTensor_lora("lora_wa_" + lora_t, 0,
                                                        0, 0); // 9 or 8

        // hard coding for rank 2
        start = std::chrono::high_resolution_clock::now();
        hemmer_->getEval().levelDown(lora_wa.get(), 9, lora_wa.get());

        start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<CtxtTensor>> lora_a_weight;
        lora_a_weight.reserve(3);
        std::vector<CtxtTensor> lora_a_output;
        lora_a_output.reserve(2);

        for (u64 j = 0; j < 2; ++j) {
            lora_a_output.emplace_back(lora_wa);
        }

        auto tmp = lora_wa;

        // split weights
        hemmer_->maskFirstRowInplace(lora_a_output[0]);
        hemmer_->getEval().leftRotate(lora_a_output[1].get(), 1 * 256,
                                      lora_a_output[1].get());
        hemmer_->maskFirstRowInplace(lora_a_output[1]);
        lora_a_weight.emplace_back(lora_a_output);
        for (u64 i = 1; i < 3; ++i) {
            lora_a_output.clear();
            hemmer_->getEval().leftRotate(lora_wa.get(), i * 2 * 256,
                                          tmp.get());
            for (u64 j = 0; j < 2; ++j) {
                lora_a_output.emplace_back(tmp);
            }
            hemmer_->maskFirstRowInplace(lora_a_output[0]);
            hemmer_->getEval().leftRotate(lora_a_output[1].get(), 1 * 256,
                                          lora_a_output[1].get());
            hemmer_->maskFirstRowInplace(lora_a_output[1]);
            lora_a_weight.emplace_back(lora_a_output);
        }

        // repeated packing
        for (u64 i = 0; i < 3; ++i) {
            for (u64 j = 0; j < 2; ++j) {
                for (u64 rot = 1; rot < 128; rot <<= 1) {
                    hemmer_->getEval().rightRotate(lora_a_weight[i][j].get(),
                                                   rot * 256, tmp.get());
                    hemmer_->addInplace(lora_a_weight[i][j], tmp);
                }
            }
        }

        // hadamult
        hemmer_->getEval().mult(cur[0].get(), lora_a_weight[0][0].get(),
                                lora_a_output[0].get());
        hemmer_->getEval().mult(cur[0].get(), lora_a_weight[0][1].get(),
                                lora_a_output[1].get());
        for (u64 i = 1; i < 3; ++i) {
            hemmer_->getEval().mult(cur[i].get(), lora_a_weight[i][0].get(),
                                    tmp.get());
            hemmer_->addInplace(lora_a_output[0], tmp);
            hemmer_->getEval().mult(cur[i].get(), lora_a_weight[i][1].get(),
                                    tmp.get());
            hemmer_->addInplace(lora_a_output[1], tmp);
        }

        // addition
        for (u64 i = 0; i < 2; ++i) {
            for (u64 rot = 1; rot < 256; rot <<= 1) {
                hemmer_->getEval().leftRotate(lora_a_output[i].get(), rot,
                                              tmp.get());
                hemmer_->addInplace(lora_a_output[i], tmp);
            }
            hemmer_->maskFirstColOnlyInplace(lora_a_output[i]);
        }
        // printElapsedTime("lora_a hard coding ");
        lora_a.emplace(t, lora_a_output);

        // std::cout << "lora_a output level: " <<
        // lora_a_output[0].get().getLevel() << std::endl;
    }
    printElapsedTime("lora_a mm");

    std::unordered_map<char, std::vector<CtxtTensor>> lora_output;
    start = std::chrono::high_resolution_clock::now();
    for (const char t : lora_type) {
        const std::string lora_t = std::string(1, t);
        auto lora_wb =
            lora_module_->getCtxtTensor_lora("lora_wb_" + lora_t, 0, 0, 0);

        // hard coding for rank 2
        hemmer_->getEval().levelDown(lora_wb.get(), 6, lora_wb.get());
        auto lora_a_output = lora_a[t];
        // split weights
        std::vector<std::vector<CtxtTensor>> lora_b_weight;
        lora_b_weight.reserve(3);
        std::vector<CtxtTensor> tmp_vector;
        tmp_vector.reserve(2);

        auto tmp = lora_wb;

        for (u64 i = 0; i < 2; ++i) {
            tmp_vector.emplace_back(lora_wb);
        }
        hemmer_->maskFirstRowInplace(tmp_vector[0]);
        hemmer_->getEval().leftRotate(tmp_vector[1].get(), 1 * 256,
                                      tmp_vector[1].get());
        hemmer_->maskFirstRowInplace(tmp_vector[1]);
        lora_b_weight.emplace_back(tmp_vector);

        for (u64 i = 1; i < 3; ++i) {
            tmp_vector.clear();
            hemmer_->getEval().leftRotate(lora_wb.get(), i * 2 * 256,
                                          tmp.get());
            for (u64 j = 0; j < 2; ++j) {
                tmp_vector.emplace_back(tmp);
            }
            hemmer_->maskFirstRowInplace(tmp_vector[0]);
            hemmer_->getEval().leftRotate(tmp_vector[1].get(), 1 * 256,
                                          tmp_vector[1].get());
            hemmer_->maskFirstRowInplace(tmp_vector[1]);

            lora_b_weight.emplace_back(tmp_vector);
        }
        tmp_vector.clear();

        // repeated packing
        for (u64 i = 0; i < 2; ++i) {
            for (u64 rot = 1; rot < 256; rot <<= 1) {
                hemmer_->getEval().rightRotate(lora_a_output[i].get(), rot,
                                               tmp.get());
                hemmer_->addInplace(lora_a_output[i], tmp);
            }
        }

        for (u64 i = 0; i < 3; ++i) {
            for (u64 j = 0; j < 2; ++j) {
                for (u64 rot = 1; rot < 128; rot <<= 1) {
                    hemmer_->getEval().rightRotate(lora_b_weight[i][j].get(),
                                                   rot * 256, tmp.get());
                    hemmer_->addInplace(lora_b_weight[i][j], tmp);
                }
            }
        }

        std::vector<CtxtTensor> lora_b_output;
        lora_b_output.reserve(3);
        for (u64 i = 0; i < 3; ++i) {
            lora_b_output.emplace_back(lora_wb);
        }

        // hadamult & addition
        for (u64 i = 0; i < 3; ++i) {
            hemmer_->getEval().mult(lora_a_output[0].get(),
                                    lora_b_weight[i][0].get(),
                                    lora_b_output[i].get());
            hemmer_->getEval().mult(lora_a_output[1].get(),
                                    lora_b_weight[i][1].get(), tmp.get());
            hemmer_->addInplace(lora_b_output[i], tmp);
        }

        lora_output.emplace(t, lora_b_output);
    }
    printElapsedTime("lora_b mm");

    // hard coding for qkv
    hemmer_->bootstrap2(lora_output['q'][0], lora_output['q'][1]);
    hemmer_->bootstrap2(lora_output['q'][2], lora_output['k'][0]);
    hemmer_->bootstrap2(lora_output['k'][1], lora_output['k'][2]);
    hemmer_->bootstrap2(lora_output['v'][0], lora_output['v'][1]);
    hemmer_->bootstrap(lora_output['v'][2]);
    printElapsedTime("lora BTS");

    // hard coding
    for (u64 i = 0; i < 3; ++i) {
        auto tmp = cur[i];
        hemmer_->getEval().levelDown(tmp.get(), 6, tmp.get());
        lora_module_->saveCtxtTensor(tmp, "tr_lora_in_a", 0, 0, i);
    }
    printElapsedTime("save");

    // hard coding
    // index = col position
    for (const char t : lora_type) {
        const std::string lora_t = std::string(1, t);
        auto lora_a_output = lora_a[t];
        for (u64 i = 0; i < 2; ++i) {
            lora_module_->saveCtxtTensor(lora_a_output[i],
                                         "tr_lora_in_b_" + lora_t, 0, 0, i);
        }
    }
    printElapsedTime("lora_b save");

    for (u64 i = 0; i < 3; ++i) {
        if (lora_output.count('q') == 1) {
            hemmer_->addInplace(query[i], lora_output['q'][i]);
        }
        if (lora_output.count('k') == 1) {
            hemmer_->addInplace(key[i], lora_output['k'][i]);
        }
        if (lora_output.count('v') == 1) {
            hemmer_->addInplace(value[i], lora_output['v'][i]);
        }
    }
    printElapsedTime("lora addition");

    /* attention.. */
    // initialization
    std::vector<CtxtTensor> masked_query, masked_key;
    masked_query.reserve(6);
    masked_key.reserve(6);
    for (u64 i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            masked_query.push_back(query[i]);
            masked_key.push_back(key[i]);
        }
    }
    std::vector<CtxtTensor> one_matmul_query, one_matmul_key;
    one_matmul_query = masked_query;
    one_matmul_key = masked_key;

    // Q. query element -> divide q1 q2 q3 q4 >> q1 0 q3 0 , 0 q2 0 q4
    start = std::chrono::high_resolution_clock::now();
    for (u64 i = 0; i < 3; ++i) {
        // masked matmul
        hemmer_->maskRightLeft(query[i], masked_query[2 * i],
                               masked_query[2 * i + 1]); // 10 -> 9
        hemmer_->maskRightLeft(key[i], masked_key[2 * i],
                               masked_key[2 * i + 1]); // 11 -> 10

        for (u64 j = 0; j < 2; ++j) {
            // lora forward
            lora_module_->saveCtxtTensor(masked_query[2 * i + j],
                                         "forward_res_q", 0, 0, 2 * i + j); // 9
            lora_module_->saveCtxtTensor(masked_key[2 * i + j], "forward_res_k",
                                         0, 0, 2 * i + j); // 10

            hemmer_->getEval().square(
                masked_query[2 * i + j].get(),
                one_matmul_query[2 * i + j].get()); // 9 -> 8
            hemmer_->getEval().square(
                masked_key[2 * i + j].get(),
                one_matmul_key[2 * i + j].get()); // 10 -> 9
        }
    }
    printElapsedTime("GK masking and squaring / saving");

    start = std::chrono::high_resolution_clock::now();
    for (u64 i = 0; i < 3; ++i) {

        hemmer_->oneMatRotSumInplace(one_matmul_query[2 * i],
                                     one_matmul_query[2 * i + 1]); // 6 -> 5
        hemmer_->oneMatRotSumInplace(one_matmul_key[2 * i],
                                     one_matmul_key[2 * i + 1]); // 7 -> 6
        // target level: ?
        for (u64 j = 0; j < 2; ++j) {
            hemmer_->transposeInplace(one_matmul_key[2 * i + j],
                                      4); // 6 -> 4 (LD)
        }
    }
    printElapsedTime("GK norm");

    // init
    std::vector<CtxtTensor> cur_qk;
    cur_qk.reserve(6);
    for (u64 i = 0; i < 3; ++i) {
        for (u64 j = 0; j < 2; ++j) {
            cur_qk.push_back(query[i]);
        }
    }

    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingInplace(masked_query[i]); // 8 -> 7
        hemmer_->transposeInplace(masked_key[i]);
        hemmer_->complexPackingRowInplace(masked_key[i]);
        // hemmer_->transposeComplexPackingInplace(masked_key[i]); //  9 -> 7
        cur_qk[i] =
            hemmer_->packedMatMul(masked_query[i], masked_key[i]); // 7 -> 4
        cur_qk[i] = hemmer_->repackCC(cur_qk[i]);                  // 4 -> 4

        hemmer_->getEval().mult(cur_qk[i].get(), -2, cur_qk[i].get()); // 4 -> 4

        hemmer_->getEval().add(cur_qk[i].get(), one_matmul_query[i].get(),
                               cur_qk[i].get()); // 4
        hemmer_->getEval().add(cur_qk[i].get(), one_matmul_key[i].get(),
                               cur_qk[i].get()); // 4
    }
    printElapsedTime("qk CC");

    for (u64 i = 0; i < 3; ++i) {
        hemmer_->bootstrap2(cur_qk[i * 2], cur_qk[i * 2 + 1]); // 4 -> 12
    }
    printElapsedTime("BTS");

    // 12 -> 11
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->getEval().mult(cur_qk[i].get(), -1.0 / (2 * std::sqrt(64)),
                                cur_qk[i].get()); // 12 -> 11
        hemmer_->getEval().add(cur_qk[i].get(), exp_message, cur_qk[i].get());

        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().add(cur_qk[i].get(), exp_message, tmp);
    }
    printElapsedTime("dividing and summation of exp_mask");

    // exp evaluation.
    // Caution: should consider output level of exp by comparing with the output
    // preicison. 11 -> 5 ( 8 for complexpacking)
    hemmer_->expParallelInplace(cur_qk, layer_n_, true);
    printElapsedTime("exp eval");

    for (u64 i = 0; i < 3; ++i) {
        for (u64 j = 0; j < 2; ++j) {
            // for lora backward
            lora_module_->saveCtxtTensor(cur_qk[2 * i + j], "forward_res_exp",
                                         0, 0, 2 * i + j);
            hemmer_->getEval().levelDown(cur_qk[2 * i + j].get(), 9,
                                         cur_qk[2 * i + j].get());
        }
    }
    printElapsedTime("bts and saving");

    // 9 -> 8
    /* for (u64 i = 0 ; i < cur_qk.size(); ++i) {
        hemmer_->dropoutExpInplace(cur_qk[i], "atn_exp", layer_n_, i);
    }
    printElapsedTime("dropout"); */

    // TODO: clean up loop index: n_iter_half*2 etc.
    std::vector<CtxtTensor> masked_value;
    masked_value.reserve(6);
    for (u64 i = 0; i < 3; ++i) {
        for (u64 j = 0; j < 2; ++j) {
            masked_value.push_back(value[i]); // 8
        }
    }

    // masked value
    // v11 v12 v13 v14 >> v11 0 v13 0 | 0 v12 0 v14 (even index)
    // v12 v22 v23 v24 >> v12 0 v23 0 | 0 v22 0 v24 (odd index)
    for (u64 i = 0; i < 3; ++i)
        hemmer_->maskRightLeft(value[i], masked_value[2 * i],
                               masked_value[2 * i + 1]); // 9 -> 8
    printElapsedTime("masking");

    for (u64 i = 0; i < 6; ++i) {
        lora_module_->saveCtxtTensor(masked_value[i], "forward_res_v", 0, 0,
                                     i); // 8
    }
    printElapsedTime("saving");

    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingInplace(cur_qk[i]);          // 8 -> 7
        hemmer_->complexPackingRowInplace(masked_value[i]); // 8 -> 7
    }
    printElapsedTime("value packing");

    for (u64 i = 0; i < 6; ++i) {
        cur_qk[i] = hemmer_->packedMatMul(
            cur_qk[i], masked_value[i]); // automatic level down.
    }
    printElapsedTime("value MM");

    for (u64 i = 0; i < 6; ++i) {                 // 7 -> 4
        cur_qk[i] = hemmer_->repackCC(cur_qk[i]); // level consumpotion x
    }
    printElapsedTime("value MM repack");

    for (u64 i = 0; i < 3; ++i) {
        hemmer_->bootstrap2(cur_qk[i * 2], cur_qk[i * 2 + 1]); // 12
    }
    printElapsedTime("BTS");

    // collect into one ciphertext
    std::vector<CtxtTensor> cur_reuse;
    cur_reuse.reserve(3);
    for (u64 i = 0; i < 3; i++) {
        hemmer_->addInplace(cur_qk[i * 2], cur_qk[i * 2 + 1]); // 12
        cur_reuse.push_back(cur_qk[i * 2]);                    // 12
    }
    cur_qk.clear();

    std::vector<CtxtTensor> output;
    output.reserve(6);
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(
                hemmer_->complexPacking(cur_reuse[i * 2], cur_reuse[i * 2 + 1]),
                tmp);
        } else {
            hemmer_->matMulPre(cur_reuse[2], tmp);
        }

        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("wd", i, j);
            auto result = hemmer_->matMulReUse(tmp, weight);

            if (i == 0) {
                output.push_back(result);
            } else {
                hemmer_->addInplace(output[j], result);
            }
        }
    }
    printElapsedTime("weight MM");

    /* Step 3. Post-process the result.. */
    for (u64 i = 0; i < 3; ++i) {
        auto repacked_output =
            hemmer_->repack(output[i * 2], output[i * 2 + 1]); // 12
        // hemmer_->dropoutInplace(repacked_output, "atn_output", layer_n_, i);
        // //10
        hemmer_->addInplace(input[i], repacked_output); // 10
    }
    printElapsedTime("repack and dropout");
}

// Fine-tuning version
// TODO: Implemenation of AdamW optimizer.
void TransformerBlock::attention2_bert_test(std::vector<CtxtTensor> &input,
                                            const Message &exp_message,
                                            const std::string &lora_type) {
    // const u64 rank = static_cast<u64>(hemmer_->getRank());

    start = std::chrono::high_resolution_clock::now();

    // TODO: weight, bias backward.
    std::vector<CtxtTensor> cur =
        hemmer_->LayerNorm(input, "atn", layer_n_, false); // 12 -> 6
    for (u64 i = 0; i < 3; ++i) {
        // TODO: replace name with fixed one.
        const auto &weight = getWeightMsg("norm1_w", i);
        const auto &bias = getWeightMsg("norm1_b", i);
        hemmer_->hadamardMultInplace(cur[i], weight);             // 6 -> 5
        hemmer_->getEval().add(cur[i].get(), bias, cur[i].get()); // 5 -> 5
    }
    // printElapsedTime("LN");

    /* if (rank == 0) {
        std::cout << "after attn LN" << std::endl;
        printing(cur);
    } */

    hemmer_->bootstrap2(cur[0], cur[1]);
    hemmer_->bootstrap(cur[2]);
    // printElapsedTime("BTS");

    std::vector<CtxtTensor> temp_q, temp_k, temp_v;
    temp_q.reserve(6);
    temp_k.reserve(6);
    temp_v.reserve(6);
    std::vector<PtxtTensor> weights_q, weights_k, weights_v;
    weights_q.reserve(6);
    weights_k.reserve(6);
    weights_v.reserve(6);
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(
                hemmer_->complexPacking(cur[i * 2], cur[i * 2 + 1]), tmp);
        } else {
            hemmer_->matMulPre(cur[2], tmp);
        }
        weights_q.clear();
        weights_k.clear();
        weights_v.clear();
        for (u64 j = 0; j < 6; ++j) {
            weights_q.push_back(getWeight("wq", i, j));
            weights_k.push_back(getWeight("wk", i, j));
            weights_v.push_back(getWeight("wv", i, j));
        }
        for (u64 j = 0; j < 6; ++j) {
            auto result_q = hemmer_->matMulReUse(tmp, weights_q[j]);
            auto result_k = hemmer_->matMulReUse(tmp, weights_k[j]);
            auto result_v = hemmer_->matMulReUse(tmp, weights_v[j]);

            if (i == 0) {
                temp_q.push_back(result_q);
                temp_k.push_back(result_k);
                temp_v.push_back(result_v);
            } else {
                hemmer_->addInplace(temp_q[j], result_q);
                hemmer_->addInplace(temp_k[j], result_k);
                hemmer_->addInplace(temp_v[j], result_v);
            }
        }
    }
    // printElapsedTime("q,k,v mm");

    std::vector<CtxtTensor> query, key, value;
    query.reserve(3);
    key.reserve(3);
    value.reserve(3);

    for (u64 i = 0; i < 3; ++i) {
        query.push_back(
            hemmer_->repack(temp_q[i * 2], temp_q[i * 2 + 1]));           // 11
        key.push_back(hemmer_->repack(temp_k[i * 2], temp_k[i * 2 + 1])); // 11
        value.push_back(
            hemmer_->repack(temp_v[i * 2], temp_v[i * 2 + 1])); // 11

        hemmer_->getEval().levelDown(query[i].get(), 10, query[i].get());
        hemmer_->getEval().levelDown(value[i].get(), 9, value[i].get());
        hemmer_->getEval().levelDown(key[i].get(), 11, key[i].get());
    }
    // printElapsedTime("q,k,v repack");

    /* LoRA */
    // forward
    std::unordered_map<char, CtxtTensor> lora_a;
    // std::unordered_map<char, std::vector<CtxtTensor>> lora_a;
    for (const char t : lora_type) {
        const std::string lora_t = std::string(1, t);
        auto lora_wa = lora_module_->getCtxtTensor_lora_test(
            "lora_wa_" + lora_t, 0, 0, 0); // 9 or 8
        hemmer_->transposeInplace(lora_wa);

        /* if (rank == 0) {
            auto pth_weight = hemmer_->decrypt2(lora_wa);
            lora_module_->saveTorchTensor(pth_weight, "lora_wa_" + lora_t, 0);
        } */

        auto lora_a_tmp = hemmer_->matMulHighLow(
            cur[0], hemmer_->getLowColBlock(lora_wa, 0), 0, 5);
        lora_a.emplace(t, lora_a_tmp);
        for (u64 i = 1; i < 3; ++i) {
            auto it = lora_a.find(t);
            assert(it != lora_a.end());
            hemmer_->addInplace(
                it->second,
                hemmer_->matMulHighLow(
                    cur[i], hemmer_->getLowColBlock(lora_wa, i), i, 5));
        }
    }
    // printElapsedTime("lora_a mm");

    // repackToOneCol: we might reduce 3 rotations in the first for loop.
    // Caution: We have to consider LOW_RANK.
    for (const char t : lora_type) {
        auto it = lora_a.find(t);
        it->second = hemmer_->repackToOneCol(it->second, 0);
        hemmer_->bootstrap(it->second);

        /* if ( t == 'q') {
            std::cout << "lora_a[q] output " << std::endl;
            printing_masking(it->second);
        } */
    }
    // printElapsedTime("lora_a repack");

    std::unordered_map<char, std::vector<CtxtTensor>> lora_output;
    start = std::chrono::high_resolution_clock::now();
    for (const char t : lora_type) {
        const std::string lora_t = std::string(1, t);
        auto lora_wb =
            lora_module_->getCtxtTensor_lora_test("lora_wb_" + lora_t, 0, 0, 0);

        auto it = lora_a.find(t);
        std::vector<CtxtTensor> tmp;
        for (u64 i = 0; i < 3; ++i) {
            tmp.emplace_back(hemmer_->matMulLowLow(
                it->second, hemmer_->getLowRowBlock(lora_wb, i), 0, i));
        }
        lora_output.emplace(t, tmp);
    }
    // printElapsedTime("lora_b");

    for (const char t : lora_type) {

        hemmer_->bootstrap2(lora_output[t][0], lora_output[t][1]);
        hemmer_->bootstrap(lora_output[t][2]);

        /* if (rank == 0 && t == 'q') {
            std::cout << "lora_output " << std::string(1,t) << std::endl;
            printing(lora_output[t]);
        } */
    }
    // printElapsedTime("lora BTS");

    for (u64 i = 0; i < 3; ++i) {
        if (lora_output.count('q') == 1) {
            hemmer_->addInplace(query[i], lora_output['q'][i]);
        }
        if (lora_output.count('k') == 1) {
            hemmer_->addInplace(key[i], lora_output['k'][i]);
        }
        if (lora_output.count('v') == 1) {
            hemmer_->addInplace(value[i], lora_output['v'][i]);
        }
    }
    // printElapsedTime("lora addition");

    /* if (rank == 0) {
        std::cout << "after lora addition, q " << std::endl;
        printing(query);

        std::cout << "after lora addition, k " << std::endl;
        printing(key);

        std::cout << "after lora addition, v " << std::endl;
        printing(value);
    } */

    /* attention.. */
    // initialization
    std::vector<CtxtTensor> masked_query, masked_key;
    masked_query.reserve(6);
    masked_key.reserve(6);
    for (u64 i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            masked_query.push_back(query[i]);
            masked_key.push_back(key[i]);
        }
    }
    std::vector<CtxtTensor> one_matmul_query, one_matmul_key;
    one_matmul_query = masked_query;
    one_matmul_key = masked_key;

    // Q. query element -> divide q1 q2 q3 q4 >> q1 0 q3 0 , 0 q2 0 q4
    for (u64 i = 0; i < 3; ++i) {
        // masked matmul
        hemmer_->maskRightLeft(query[i], masked_query[2 * i],
                               masked_query[2 * i + 1]); // 10 -> 9
        hemmer_->maskRightLeft(key[i], masked_key[2 * i],
                               masked_key[2 * i + 1]); // 11 -> 10
        for (u64 j = 0; j < 2; ++j) {
            hemmer_->getEval().square(
                masked_query[2 * i + j].get(),
                one_matmul_query[2 * i + j].get()); // 9 -> 8
            hemmer_->getEval().square(
                masked_key[2 * i + j].get(),
                one_matmul_key[2 * i + j].get()); // 10 -> 9
        }
    }
    // printElapsedTime("GK masking and squaring");

    for (u64 i = 0; i < 3; ++i) {

        hemmer_->oneMatRotSumInplace(one_matmul_query[2 * i],
                                     one_matmul_query[2 * i + 1]); // 6 -> 5
        hemmer_->oneMatRotSumInplace(one_matmul_key[2 * i],
                                     one_matmul_key[2 * i + 1]); // 7 -> 6

        // target level: ?
        for (u64 j = 0; j < 2; ++j) {
            hemmer_->transposeInplace(one_matmul_key[2 * i + j],
                                      4); // 6 -> 4 (LD)
        }
    }
    // printElapsedTime("GK norm");

    // init
    std::vector<CtxtTensor> cur_qk;
    cur_qk.reserve(6);
    for (u64 i = 0; i < 3; ++i) {
        for (u64 j = 0; j < 2; ++j) {
            cur_qk.push_back(query[i]);
        }
    }

    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingInplace(masked_query[i]); // 8 -> 7
        hemmer_->transposeInplace(masked_key[i]);
        hemmer_->complexPackingRowInplace(masked_key[i]);
        // hemmer_->transposeComplexPackingInplace(masked_key[i]); //  9 -> 7
        cur_qk[i] =
            hemmer_->packedMatMul(masked_query[i], masked_key[i]); // 7 -> 4
        cur_qk[i] = hemmer_->repackCC(cur_qk[i]);                  // 4 -> 4

        hemmer_->getEval().mult(cur_qk[i].get(), -2, cur_qk[i].get()); // 4 -> 4

        hemmer_->getEval().add(cur_qk[i].get(), one_matmul_query[i].get(),
                               cur_qk[i].get()); // 4
        hemmer_->getEval().add(cur_qk[i].get(), one_matmul_key[i].get(),
                               cur_qk[i].get()); // 4
    }
    // printElapsedTime("qk CC");

    for (u64 i = 0; i < 3; ++i) {
        hemmer_->bootstrap2(cur_qk[i * 2], cur_qk[i * 2 + 1]); // 4 -> 12
    }
    // printElapsedTime("BTS");

    // 12 -> 11
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->getEval().mult(cur_qk[i].get(), -1.0 / (2 * std::sqrt(64)),
                                cur_qk[i].get()); // 12 -> 11
        hemmer_->getEval().add(cur_qk[i].get(), exp_message, cur_qk[i].get());

        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().add(cur_qk[i].get(), exp_message, tmp);
    }
    // printElapsedTime("dividing and summation of exp_mask");

    // exp evaluation.
    // Caution: should consider output level of exp by comparing with the output
    // preicison. 11 -> 5 ( 8 for complexpacking)
    /* if (rank == 0) {
        std::cout << "exp input" << std::endl;
        printing(cur_qk);
    } */

    hemmer_->expParallelInplace(cur_qk, layer_n_, false);
    // printElapsedTime("exp eval");

    /* if (rank == 0) {
        std::cout << "after exp eval" << std::endl;
        printing(cur_qk);
    } */

    for (u64 i = 0; i < 3; ++i) {
        // hemmer_->bootstrap2_exponly(cur_qk[2*i], cur_qk[2*i+1]);
        for (u64 j = 0; j < 2; ++j) {
            hemmer_->getEval().levelDown(cur_qk[2 * i + j].get(), 8,
                                         cur_qk[2 * i + j].get());
        }
    }
    // printElapsedTime("BTS");
    // std::cout << "exp outuput " << std::endl;
    // printing(cur_qk);

    // 9 -> 8
    /* for (u64 i = 0 ; i < cur_qk.size(); ++i) {
        hemmer_->dropoutExpInplace(cur_qk[i], "atn_exp", layer_n_, i);
    } */

    // TODO: clean up loop index: n_iter_half*2 etc.
    std::vector<CtxtTensor> masked_value;
    masked_value.reserve(6);
    for (u64 i = 0; i < 3; ++i) {
        for (u64 j = 0; j < 2; ++j) {
            masked_value.push_back(value[i]); // 8
        }
    }

    // masked value
    // v11 v12 v13 v14 >> v11 0 v13 0 | 0 v12 0 v14 (even index)
    // v12 v22 v23 v24 >> v12 0 v23 0 | 0 v22 0 v24 (odd index)
    for (u64 i = 0; i < 3; ++i)
        hemmer_->maskRightLeft(value[i], masked_value[2 * i],
                               masked_value[2 * i + 1]); // 9 -> 8
    // printElapsedTime("masking");

    for (u64 i = 0; i < 6; ++i) {
        // cur_qk[i] = hemmer_->singleCCMatMul(cur_qk[i], masked_value[i], 4);
        // // 7 -> 4
        hemmer_->complexPackingInplace(cur_qk[i]);          // 8 -> 7
        hemmer_->complexPackingRowInplace(masked_value[i]); // 8 -> 7

        cur_qk[i] = hemmer_->packedMatMul(
            cur_qk[i], masked_value[i]); // automatic level down...
    }
    // printElapsedTime("value MM");

    for (u64 i = 0; i < 6; ++i) {                 // 7 -> 4
        cur_qk[i] = hemmer_->repackCC(cur_qk[i]); // level consumption x
    }
    // printElapsedTime("value MM repack");

    for (u64 i = 0; i < 3; ++i) {
        hemmer_->bootstrap2(cur_qk[i * 2], cur_qk[i * 2 + 1]); // 12
    }
    // printElapsedTime("BTS");

    // collect into one ciphertext
    std::vector<CtxtTensor> cur_reuse;
    cur_reuse.reserve(3);
    for (u64 i = 0; i < 3; i++) {
        hemmer_->addInplace(cur_qk[i * 2], cur_qk[i * 2 + 1]); // 12
        cur_reuse.push_back(cur_qk[i * 2]);                    // 12
    }
    cur_qk.clear();

    /* if (rank == 0) {
        std::cout << "score @ v mm " << std::endl;
        printing(cur_reuse);
    } */

    std::vector<CtxtTensor> output;
    output.reserve(6);
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(
                hemmer_->complexPacking(cur_reuse[i * 2], cur_reuse[i * 2 + 1]),
                tmp);
        } else {
            hemmer_->matMulPre(cur_reuse[2], tmp);
        }

        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("wd", i, j);
            auto result = hemmer_->matMulReUse(tmp, weight);

            if (i == 0) {
                output.push_back(result);
            } else {
                hemmer_->addInplace(output[j], result);
            }
        }
    }
    // printElapsedTime("weight MM");
    /* if (rank == 0 ){
        std::cout << "after dense mm" << std::endl;
        printing(output);
    } */

    /* Step 3. Post-process the result.. */
    for (u64 i = 0; i < 3; ++i) {
        auto repacked_output =
            hemmer_->repack(output[i * 2], output[i * 2 + 1]); // 12

        // hemmer_->dropoutInplace(repacked_output, "atn_output", layer_n_, i);
        // //10

        hemmer_->addInplace(input[i], repacked_output); // 10
    }
    // printElapsedTime("repacking");

    /* std::cout << "attn output" << std::endl;
    printing(input); */
}

// Assume input leval is 12.
void TransformerBlock::attention_bert_SM(std::vector<CtxtTensor> &input,
                                         const std::string &lora_type) {

    const u64 rank = static_cast<u64>(hemmer_->getRank());

    start = std::chrono::high_resolution_clock::now();

    std::vector<CtxtTensor> cur =
        hemmer_->LayerNorm(input, "atn", layer_n_, true); // 12 -> 6
    for (u64 i = 0; i < 3; ++i) {
        const auto &weight = getWeightMsg("norm1_w", i);
        const auto &bias = getWeightMsg("norm1_b", i);
        hemmer_->hadamardMultInplace(cur[i], weight);             // 6 -> 5
        hemmer_->getEval().add(cur[i].get(), bias, cur[i].get()); // 5 -> 5
    }
    printElapsedTime("atn LN");

    hemmer_->bootstrap2(cur[0], cur[1]);
    hemmer_->bootstrap(cur[2]);
    printElapsedTime("BTS");

    std::vector<CtxtTensor> temp_q, temp_k, temp_v;
    temp_q.reserve(6);
    temp_k.reserve(6);
    temp_v.reserve(6);
    std::vector<PtxtTensor> weights_q, weights_k, weights_v;
    weights_q.reserve(6);
    weights_k.reserve(6);
    weights_v.reserve(6);
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(
                hemmer_->complexPacking(cur[i * 2], cur[i * 2 + 1]), tmp);
        } else {
            hemmer_->matMulPre(cur[2], tmp);
        }
        weights_q.clear();
        weights_k.clear();
        weights_v.clear();
        for (u64 j = 0; j < 6; ++j) {
            weights_q.push_back(getWeight("wq", i, j));
            weights_k.push_back(getWeight("wk", i, j));
            weights_v.push_back(getWeight("wv", i, j));
        }
        for (u64 j = 0; j < 6; ++j) {
            auto result_q = hemmer_->matMulReUse(tmp, weights_q[j]);
            auto result_k = hemmer_->matMulReUse(tmp, weights_k[j]);
            auto result_v = hemmer_->matMulReUse(tmp, weights_v[j]);

            if (i == 0) {
                temp_q.push_back(result_q);
                temp_k.push_back(result_k);
                temp_v.push_back(result_v);
            } else {
                hemmer_->addInplace(temp_q[j], result_q);
                hemmer_->addInplace(temp_k[j], result_k);
                hemmer_->addInplace(temp_v[j], result_v);
            }
        }
    }
    printElapsedTime("q,k,v mm");

    std::vector<CtxtTensor> query, key, value;
    query.reserve(3);
    key.reserve(3);
    value.reserve(3);
    for (u64 i = 0; i < 3; ++i) {
        query.push_back(
            hemmer_->repack(temp_q[i * 2], temp_q[i * 2 + 1]));           // 11
        key.push_back(hemmer_->repack(temp_k[i * 2], temp_k[i * 2 + 1])); // 11
        value.push_back(
            hemmer_->repack(temp_v[i * 2], temp_v[i * 2 + 1])); // 11

        hemmer_->getEval().levelDown(query[i].get(), 10, query[i].get());
        hemmer_->getEval().levelDown(value[i].get(), 9, value[i].get());
        hemmer_->getEval().levelDown(key[i].get(), 11, key[i].get());
    }
    printElapsedTime("q,k,v repack");

    /* LoRA */
    // forward
    std::unordered_map<char, CtxtTensor> lora_a;

    start = std::chrono::high_resolution_clock::now();
    for (const char t : lora_type) {
        const std::string lora_t = std::string(1, t);
        auto lora_wa = lora_module_->getCtxtTensor_lora("lora_wa_" + lora_t, 0,
                                                        0, 0); // 9 or 8

        auto lora_a_tmp = hemmer_->matMulHighLow(
            cur[0], hemmer_->getLowColBlock(lora_wa, 0), 0, 5);
        lora_a.emplace(t, lora_a_tmp);
        for (u64 i = 1; i < 3; ++i) {
            auto it = lora_a.find(t);
            assert(it != lora_a.end());
            hemmer_->addInplace(
                it->second,
                hemmer_->matMulHighLow(
                    cur[i], hemmer_->getLowColBlock(lora_wa, i), i, 5));
        }
    }
    printElapsedTime("lora_a mm");

    // Caution: We have to consider LOW_RANK.
    for (const char t : lora_type) {
        auto it = lora_a.find(t);
        it->second = hemmer_->repackToOneCol(it->second, 0);
        hemmer_->bootstrap(it->second);
    }
    printElapsedTime("lora_a repack");

    std::unordered_map<char, std::vector<CtxtTensor>> lora_output;
    start = std::chrono::high_resolution_clock::now();
    for (const char t : lora_type) {
        const std::string lora_t = std::string(1, t);
        auto lora_wb =
            lora_module_->getCtxtTensor_lora("lora_wb_" + lora_t, 0, 0, 0);

        auto it = lora_a.find(t);
        std::vector<CtxtTensor> tmp;
        for (u64 i = 0; i < 3; ++i) {
            tmp.emplace_back(hemmer_->matMulLowLow(
                it->second, hemmer_->getLowRowBlock(lora_wb, i), 0, i));
        }
        lora_output.emplace(t, tmp);
    }
    printElapsedTime("lora_b");

    for (const char t : lora_type) {
        hemmer_->bootstrap2(lora_output[t][0], lora_output[t][1]);
        hemmer_->bootstrap(lora_output[t][2]);
    }
    printElapsedTime("lora BTS");

    for (u64 i = 0; i < 3; ++i) {
        auto tmp = cur[i];
        hemmer_->transposeInplace(tmp);
        lora_module_->saveCtxtTensor(tmp, "tr_lora_in_a", 0, 0, i);
    }
    printElapsedTime("tr and save");

    for (const char t : lora_type) {
        const std::string lora_t = std::string(1, t);
        auto it = lora_a.find(t);
        hemmer_->transposeInplace(it->second);
        lora_module_->saveCtxtTensor(it->second, "tr_lora_in_b_" + lora_t, 0, 0,
                                     0);
    }
    printElapsedTime("lora_b tr and save");

    for (u64 i = 0; i < 3; ++i) {
        if (lora_output.count('q') == 1) {
            hemmer_->addInplace(query[i], lora_output['q'][i]);
        }
        if (lora_output.count('k') == 1) {
            hemmer_->addInplace(key[i], lora_output['k'][i]);
        }
        if (lora_output.count('v') == 1) {
            hemmer_->addInplace(value[i], lora_output['v'][i]);
        }
    }
    printElapsedTime("lora addition");

    std::vector<CtxtTensor> masked_query, masked_key;
    masked_query.reserve(6);
    masked_key.reserve(6);
    for (u64 i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            masked_query.push_back(query[i]);
            masked_key.push_back(key[i]);
        }
    }

    // Q. query element -> divide q1 q2 q3 q4 >> q1 0 q3 0 , 0 q2 0 q4
    for (u64 i = 0; i < 3; ++i) {
        hemmer_->maskRightLeft(query[i], masked_query[2 * i],
                               masked_query[2 * i + 1]); // 10 -> 9
        hemmer_->maskRightLeft(key[i], masked_key[2 * i],
                               masked_key[2 * i + 1]); // 11 -> 10

        for (u64 j = 0; j < 2; ++j) {
            // lora forward
            lora_module_->saveCtxtTensor(masked_query[2 * i + j],
                                         "forward_res_q", 0, 0, 2 * i + j); // 9
            lora_module_->saveCtxtTensor(masked_key[2 * i + j], "forward_res_k",
                                         0, 0, 2 * i + j); // 10
        }
    }
    printElapsedTime("masking");

    std::vector<CtxtTensor> cur_qk;
    cur_qk.reserve(6);
    for (u64 i = 0; i < 3; ++i) {
        for (u64 j = 0; j < 2; ++j) {
            cur_qk.push_back(query[i]);
        }
    }

    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingInplace(masked_query[i]);
        hemmer_->transposeComplexPackingInplace(masked_key[i], 7);

        cur_qk[i] =
            hemmer_->packedMatMul(masked_query[i], masked_key[i]); // 9 -> 6
        cur_qk[i] = hemmer_->repackCC(cur_qk[i]);
    }
    printElapsedTime("qk CC");

    /* Step 2. Perform the attention.. */
    for (u64 i = 0; i < 3; ++i) {
        hemmer_->bootstrap2(cur_qk[2 * i], cur_qk[2 * i + 1]);
    }
    printElapsedTime("BTS");

    auto qk_dec = hemmer_->decrypt2(cur_qk[0]);
    auto max = qk_dec.max().item<double>();
    auto min = qk_dec.min().item<double>();
    std::cout << "qk_mm min: " << min << ", max: " << max << std::endl;

    // 12 -> 4
    hemmer_->softmaxVectorInplaceHETAL(cur_qk, layer_n_, 0, true,
                                       hemmer_->getDec(), hemmer_->getsk());
    // hemmer_->softmaxVectorInplaceCCS(cur_qk, layer_n_, true);
    printElapsedTime("Softmax");

    for (u64 i = 0; i < 3; ++i) {
        for (u64 j = 0; j < 2; ++j) {
            lora_module_->saveCtxtTensor(cur_qk[2 * i + j], "forward_res_exp",
                                         0, 0, 2 * i + j);
            hemmer_->getEval().levelDown(cur_qk[2 * i + j].get(), 9,
                                         cur_qk[2 * i + j].get());
        }
    }
    printElapsedTime("save and leveldown");

    // 9 -> 8
    for (u64 i = 0; i < cur_qk.size(); ++i) {
        hemmer_->dropoutExpInplace(cur_qk[i], "atn_exp", layer_n_, i);
    }
    printElapsedTime("dropout");

    // TODO: clean up loop index: n_iter_half*2 etc.
    std::vector<CtxtTensor> masked_value;
    masked_value.reserve(6);
    for (u64 i = 0; i < 3; ++i) {
        for (u64 j = 0; j < 2; ++j) {
            masked_value.push_back(value[i]); // 8
        }
    }

    // masked value
    // v11 v12 v13 v14 >> v11 0 v13 0 | 0 v12 0 v14 (even index)
    // v12 v22 v23 v24 >> v12 0 v23 0 | 0 v22 0 v24 (odd index)
    for (u64 i = 0; i < 3; ++i) {
        hemmer_->maskRightLeft(value[i], masked_value[2 * i],
                               masked_value[2 * i + 1]); // 11 -> 10
    }
    printElapsedTime("masking");

    for (u64 i = 0; i < 6; ++i) {
        lora_module_->saveCtxtTensor(masked_value[i], "forward_res_v", 0, 0,
                                     i); // 8
    }
    printElapsedTime("saving");

    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingInplace(cur_qk[i]);          // 8 -> 7
        hemmer_->complexPackingRowInplace(masked_value[i]); // 8 -> 7

        cur_qk[i] = hemmer_->packedMatMul(
            cur_qk[i], masked_value[i]); // automatic level down...
    }
    printElapsedTime("value MM");

    for (u64 i = 0; i < 6; ++i) {                 // 7 -> 4
        cur_qk[i] = hemmer_->repackCC(cur_qk[i]); // level consumpotion x
    }
    printElapsedTime("value MM repack");

    for (u64 i = 0; i < 3; ++i) {
        hemmer_->bootstrap2(cur_qk[i * 2], cur_qk[i * 2 + 1]); // 12
    }
    printElapsedTime("BTS");

    // collect into one ciphertext
    std::vector<CtxtTensor> cur_reuse;
    cur_reuse.reserve(3);
    for (u64 i = 0; i < 3; i++) {
        hemmer_->addInplace(cur_qk[i * 2], cur_qk[i * 2 + 1]); // 12
        cur_reuse.push_back(cur_qk[i * 2]);                    // 12
    }
    cur_qk.clear();

    start = std::chrono::high_resolution_clock::now();
    std::vector<CtxtTensor> output;
    output.reserve(6);
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(
                hemmer_->complexPacking(cur_reuse[i * 2], cur_reuse[i * 2 + 1]),
                tmp);
        } else {
            hemmer_->matMulPre(cur_reuse[2], tmp);
        }

        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("wd", 2 * rank + i, j);
            auto result = hemmer_->matMulReUse(tmp, weight);

            if (i == 0) {
                output.push_back(result);
            } else {
                hemmer_->addInplace(output[j], result);
            }
        }
    }
    printElapsedTime("weight MM");

    /* Step 3. Post-process the result.. */
    for (u64 i = 0; i < 3; ++i) {
        auto repacked_output =
            hemmer_->repack(output[i * 2], output[i * 2 + 1]);
        hemmer_->dropoutInplace(repacked_output, "atn_output", layer_n_,
                                i);                     // 10
        hemmer_->addInplace(input[i], repacked_output); // 11
    }
    printElapsedTime("repacking");
    // std::cout << "attn output" << std::endl;
    // printing(input);
}

// multi-GPU version·
void TransformerBlock::attention_bert_multi(std::vector<CtxtTensor> &input,
                                            const Message &exp_mask) {
    // const int seqlen = static_cast<int>(input[0].getHeight());
    const u64 rank = static_cast<u64>(hemmer_->getRank());
    const u64 max_rank = static_cast<u64>(hemmer_->getMaxRank());

    // Assume using 2 GPUs.
    std::cout << "rank = " << rank << ", size = " << input.size() << std::endl;

    start = std::chrono::high_resolution_clock::now();
    // Instead of conveying rank number, we can use getRank() in the function.
    std::vector<CtxtTensor> cur =
        hemmer_->LayerNorm_multi(input, "atn", layer_n_); // 12 -> 6
    if (rank == 0) {
        for (u64 i = 0; i < 2; ++i) {
            const auto &weight = getWeightMsg("norm1_w", i);
            const auto &bias = getWeightMsg("norm1_b", i);
            hemmer_->hadamardMultInplace(cur[i], weight);             // 6 -> 5
            hemmer_->getEval().add(cur[i].get(), bias, cur[i].get()); // 5 -> 5
        }
    } else {
        const auto &weight = getWeightMsg("norm1_w", 2);
        const auto &bias = getWeightMsg("norm1_b", 2);
        hemmer_->hadamardMultInplace(cur[0], weight);
        hemmer_->getEval().add(cur[0].get(), bias, cur[0].get());
    }
    if (rank == 0)
        printElapsedTime("atn LN");

    if (rank == 0) {
        hemmer_->bootstrap2(cur[0], cur[1]);
    } else {
        hemmer_->bootstrap(cur[0]);
    }

    if (rank == 0)
        printElapsedTime("BTS");

    std::vector<CtxtTensor> temp_q, temp_k, temp_v;
    // ModelArgs::N_HEAD = 6
    temp_q.reserve(6);
    temp_k.reserve(6);
    temp_v.reserve(6);
    std::vector<PtxtTensor> weights_q, weights_k, weights_v;
    weights_q.reserve(6);
    weights_k.reserve(6);
    weights_v.reserve(6);
    for (u64 i = 0; i < 1; ++i) {
        // if loop >> not i but rank
        std::vector<Ciphertext> tmp;
        // check
        if (rank == 0) {
            hemmer_->matMulPre(hemmer_->complexPacking(cur[0], cur[1]), tmp);
        } else {
            hemmer_->matMulPre(cur[0], tmp);
        }
        // pre-compute weights for q, k, v for each head
        weights_q.clear();
        weights_k.clear();
        weights_v.clear();
        for (u64 j = 0; j < 6; ++j) {
            // n_iter_quarter << should fix as the same value.
            // define n_iter_quarter = 2
            // auto idx = n_iter_quarter * rank + i;
            weights_q.push_back(getWeight("wq", rank, j));
            weights_k.push_back(getWeight("wk", rank, j));
            weights_v.push_back(getWeight("wv", rank, j));
        }

        // perform matmul and addition
        for (u64 j = 0; j < 6; ++j) {
            auto result_q = hemmer_->matMulReUse(tmp, weights_q[j]);
            auto result_k = hemmer_->matMulReUse(tmp, weights_k[j]);
            auto result_v = hemmer_->matMulReUse(tmp, weights_v[j]);

            if (i == 0) {
                temp_q.push_back(result_q);
                temp_k.push_back(result_k);
                temp_v.push_back(result_v);
            } else {
                hemmer_->addInplace(temp_q[j], result_q);
                hemmer_->addInplace(temp_k[j], result_k);
                hemmer_->addInplace(temp_v[j], result_v);
            }
        }
    }

    if (max_rank > 1) {
        for (u64 i = 0; i < 6; ++i) {
            allReduceWrapper(temp_q[i]);
            allReduceWrapper(temp_k[i]);
            allReduceWrapper(temp_v[i]);
        }

        if (rank == 0) {
            for (u64 i = 0; i < 4; ++i) {
                auto idx = i;
                hemmer_->getEval().modReduct(temp_q[idx].get());
                hemmer_->getEval().modReduct(temp_k[idx].get());
                hemmer_->getEval().modReduct(temp_v[idx].get());
            }

        } else {
            for (u64 i = 0; i < 2; ++i) {
                // idx = 4*rank + i
                auto idx = 4 + i;
                hemmer_->getEval().modReduct(temp_q[idx].get());
                hemmer_->getEval().modReduct(temp_k[idx].get());
                hemmer_->getEval().modReduct(temp_v[idx].get());
            }
        }
    }
    if (rank == 0)
        printElapsedTime("qkv generation");

    std::vector<CtxtTensor> query, key, value;
    query.clear();
    key.clear();
    value.clear();
    // repacking temp_q, temp_k, temp_v into query, key, value
    if (rank == 0) {
        query.reserve(2);
        key.reserve(2);
        value.reserve(2);
        for (u64 i = 0; i < 2; ++i) {
            auto idx = i * 2;
            query.push_back(
                hemmer_->repack(temp_q[idx], temp_q[idx + 1]));           // 11
            key.push_back(hemmer_->repack(temp_k[idx], temp_k[idx + 1])); // 11
            value.push_back(
                hemmer_->repack(temp_v[idx], temp_v[idx + 1])); // 11

            hemmer_->getEval().levelDown(query[i].get(), 10, query[i].get());
            hemmer_->getEval().levelDown(value[i].get(), 8, value[i].get());
            hemmer_->getEval().levelDown(key[i].get(), 11, key[i].get());
        }
    } else {
        query.reserve(1);
        key.reserve(1);
        value.reserve(1);

        query.push_back(hemmer_->repack(temp_q[4], temp_q[5]));
        key.push_back(hemmer_->repack(temp_k[4], temp_k[5]));
        value.push_back(hemmer_->repack(temp_v[4], temp_v[5]));

        hemmer_->getEval().levelDown(query[0].get(), 10, query[0].get());
        hemmer_->getEval().levelDown(value[0].get(), 8, value[0].get());
        hemmer_->getEval().levelDown(key[0].get(), 11, key[0].get());
    }

    if (rank == 0)
        printElapsedTime("qkv attention");

    /* Step 2. Perform the attention.. */
    // initialization
    std::vector<CtxtTensor> masked_query, masked_key;
    std::vector<CtxtTensor> one_matmul_query, one_matmul_key;
    if (rank == 0) {
        masked_query.reserve(4);
        masked_key.reserve(4);
        for (u64 i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                masked_query.push_back(query[i]);
                masked_key.push_back(key[i]);
            }
        }
        one_matmul_query = masked_query;
        one_matmul_key = masked_key;
    } else {
        masked_query.reserve(2);
        masked_key.reserve(2);
        for (u64 i = 0; i < 1; i++) {
            for (int j = 0; j < 2; j++) {
                masked_query.push_back(query[i]);
                masked_key.push_back(key[i]);
            }
        }
        one_matmul_query = masked_query;
        one_matmul_key = masked_key;
    }

    // query element -> divide q1 q2 q3 q4 >> q1 0 q3 0 , 0 q2 0 q4
    if (rank == 0) {
        for (u64 i = 0; i < 2; ++i) {
            // masked matmul
            // measure level consumption.
            hemmer_->maskRightLeft(query[i], masked_query[2 * i],
                                   masked_query[2 * i + 1]); // 10 -> 9
            hemmer_->maskRightLeft(key[i], masked_key[2 * i],
                                   masked_key[2 * i + 1]); // 11 -> 10

            // TODO: reducing 2 mult times to one.
            for (u64 j = 0; j < 2; ++j) {
                hemmer_->getEval().square(
                    masked_query[2 * i + j].get(),
                    one_matmul_query[2 * i + j].get()); // 9 -> 8
                hemmer_->getEval().square(
                    masked_key[2 * i + j].get(),
                    one_matmul_key[2 * i + j].get()); // 10 -> 9
            }
        }
    } else {
        hemmer_->maskRightLeft(query[0], masked_query[0],
                               masked_query[1]); // 10 -> 9
        hemmer_->maskRightLeft(key[0], masked_key[0],
                               masked_key[1]); // 11 -> 10

        // TODO: reducing 2 mult times to one.
        for (u64 j = 0; j < 2; ++j) {
            hemmer_->getEval().square(masked_query[j].get(),
                                      one_matmul_query[j].get()); // 9 -> 8
            hemmer_->getEval().square(masked_key[j].get(),
                                      one_matmul_key[j].get()); // 10 -> 9
        }
    }
    if (rank == 0)
        printElapsedTime("GK masking and squaring");

    if (rank == 0) {
        for (u64 i = 0; i < 2; ++i) {
            hemmer_->oneMatRotSumInplace(one_matmul_query[2 * i],
                                         one_matmul_query[2 * i + 1]); // 6 -> 5
            hemmer_->oneMatRotSumInplace(one_matmul_key[2 * i],
                                         one_matmul_key[2 * i + 1]); // 7 -> 6

            // target level: ?
            for (u64 j = 0; j < 2; ++j) {
                hemmer_->transposeInplace(one_matmul_key[2 * i + j],
                                          4); // 6 -> 4 (LD)
            }
        }
    } else {
        hemmer_->oneMatRotSumInplace(one_matmul_query[0],
                                     one_matmul_query[1]); // 6 -> 5
        hemmer_->oneMatRotSumInplace(one_matmul_key[0],
                                     one_matmul_key[1]); // 7 -> 6

        // target level: ?
        for (u64 j = 0; j < 2; ++j) {
            hemmer_->transposeInplace(one_matmul_key[j], 4); // 6 -> 4 (LD)
        }
    }

    if (rank == 0)
        printElapsedTime("GK 1CMM");
    /* std::cout << "1CMM" << std::endl;
    printing(one_matmul_query); */

    // init
    std::vector<CtxtTensor> cur_qk;
    if (rank == 0) {
        cur_qk.reserve(4);
        for (u64 i = 0; i < 2; ++i) {
            for (u64 j = 0; j < 2; ++j) {
                cur_qk.push_back(query[i]);
            }
        }
        for (u64 i = 0; i < 4; ++i) {
            hemmer_->complexPackingInplace(masked_query[i]);        // 8 -> 7
            hemmer_->transposeComplexPackingInplace(masked_key[i]); //  9 -> 7
            cur_qk[i] =
                hemmer_->packedMatMul(masked_query[i], masked_key[i]); // 7 -> 4
            cur_qk[i] = hemmer_->repackCC(cur_qk[i]);                  // 4 -> 4

            hemmer_->getEval().mult(cur_qk[i].get(), -2.0, cur_qk[i].get());

            hemmer_->getEval().add(cur_qk[i].get(), one_matmul_query[i].get(),
                                   cur_qk[i].get());
            hemmer_->getEval().add(cur_qk[i].get(), one_matmul_key[i].get(),
                                   cur_qk[i].get());
        }
    } else {
        cur_qk.reserve(2);
        for (u64 i = 0; i < 1; ++i) {
            for (u64 j = 0; j < 2; ++j) {
                cur_qk.push_back(query[i]);
            }
        }
        for (u64 i = 0; i < 2; ++i) {
            hemmer_->complexPackingInplace(masked_query[i]);        // 8 -> 7
            hemmer_->transposeComplexPackingInplace(masked_key[i]); //  9 -> 7
            cur_qk[i] =
                hemmer_->packedMatMul(masked_query[i], masked_key[i]); // 7 -> 4
            cur_qk[i] = hemmer_->repackCC(cur_qk[i]);                  // 4 -> 4

            hemmer_->getEval().mult(cur_qk[i].get(), -2, cur_qk[i].get());

            hemmer_->getEval().add(cur_qk[i].get(), one_matmul_query[i].get(),
                                   cur_qk[i].get());
            hemmer_->getEval().add(cur_qk[i].get(), one_matmul_key[i].get(),
                                   cur_qk[i].get());
        }
    }

    if (rank == 0)
        printElapsedTime("GK CC");

    if (rank == 0) {
        for (u64 i = 0; i < 2; ++i) {
            hemmer_->bootstrap2(cur_qk[i * 2], cur_qk[i * 2 + 1]); // 4 -> 12
            for (u64 j = 0; j < 2; ++j) {
                hemmer_->getEval().mult(cur_qk[i * 2 + j].get(),
                                        -1.0 / (2 * std::sqrt(64)),
                                        cur_qk[i * 2 + j].get());
                hemmer_->getEval().add(cur_qk[i * 2 + j].get(), exp_mask,
                                       cur_qk[i * 2 + j].get());
            }
        }
    } else {
        hemmer_->bootstrap2(cur_qk[0], cur_qk[1]);
        for (u64 j = 0; j < 2; ++j) {
            hemmer_->getEval().mult(cur_qk[j].get(), -1.0 / (2 * std::sqrt(64)),
                                    cur_qk[j].get());
            hemmer_->getEval().add(cur_qk[j].get(), exp_mask, cur_qk[j].get());
        }
    }
    if (rank == 0)
        printElapsedTime("BTS and addition");
    // 12 -> 11

    hemmer_->expParallelInplace(cur_qk, layer_n_, false);
    printElapsedTime("exp eval");
    if (rank == 0) {
        printElapsedTime("exp eval");
    }

    // TODO: clean up loop index: n_iter_half*2 etc.
    std::vector<CtxtTensor> masked_value;
    // masked value
    // v11 v12 v13 v14 >> v11 0 v13 0 | 0 v12 0 v14 (even index)
    // v12 v22 v23 v24 >> v12 0 v23 0 | 0 v22 0 v24 (odd index)
    if (rank == 0) {
        masked_value.clear();
        masked_value.reserve(4);
        for (u64 i = 0; i < 2; ++i) {
            for (u64 j = 0; j < 2; ++j) {
                masked_value.push_back(value[i]); // 8
            }
        }
        for (u64 i = 0; i < 2; ++i)
            hemmer_->maskRightLeft(value[i], masked_value[2 * i],
                                   masked_value[2 * i + 1]); // 8 -> 7

        for (u64 i = 0; i < 4; ++i) {
            cur_qk[i] = hemmer_->singleCCMatMul(cur_qk[i], masked_value[i],
                                                4); // 7 -> 4
        }
    } else {
        masked_value.clear();
        masked_value.reserve(2);
        for (u64 i = 0; i < 1; ++i) {
            for (u64 j = 0; j < 2; ++j) {
                masked_value.push_back(value[i]); // 8
            }
        }
        hemmer_->maskRightLeft(value[0], masked_value[0], masked_value[1]);

        for (u64 i = 0; i < 2; ++i) {
            cur_qk[i] = hemmer_->singleCCMatMul(cur_qk[i], masked_value[i],
                                                4); // 7 -> 4
        }
    }
    if (rank == 0)
        printElapsedTime("value MM");

    if (rank == 0) {
        for (u64 i = 0; i < 2; ++i) {
            hemmer_->bootstrap2(cur_qk[i * 2], cur_qk[i * 2 + 1]); // 12
        }
    } else {
        hemmer_->bootstrap2(cur_qk[0], cur_qk[1]);
    }
    if (rank == 0)
        printElapsedTime("BTS");

    // collect into one ciphertext
    std::vector<CtxtTensor> cur_reuse;
    if (rank == 0) {
        cur_reuse.reserve(2);
        for (u64 i = 0; i < 2; i++) {
            hemmer_->addInplace(cur_qk[i * 2], cur_qk[i * 2 + 1]); // 12
            cur_reuse.push_back(cur_qk[i * 2]);                    // 12
        }

    } else {
        cur_reuse.reserve(1);
        hemmer_->addInplace(cur_qk[0], cur_qk[1]);
        cur_reuse.push_back(cur_qk[0]);
    }
    cur_qk.clear();

    std::vector<CtxtTensor> output;
    output.reserve(6);
    std::vector<PtxtTensor> weight;
    weight.reserve(6);
    for (u64 i = 0; i < 1; ++i) {
        std::vector<Ciphertext> tmp;
        if (rank == 0) {
            hemmer_->matMulPre(
                hemmer_->complexPacking(cur_reuse[0], cur_reuse[1]), tmp);
        } else {
            hemmer_->matMulPre(cur_reuse[0], tmp);
        }

        for (u64 j = 0; j < 6; ++j) {
            // auto weight = getWeight("wd", rank, j);
            // auto result = hemmer_->matMulReUse(tmp, weight);
            weight.push_back(getWeight("wd", rank, j));
        }

        for (u64 j = 0; j < 6; ++j) {
            auto result = hemmer_->matMulReUse(tmp, weight[j]);

            if (i == 0) {
                output.push_back(result);
            } else {
                hemmer_->addInplace(output[j], result);
            }
        }

        /* if (i == 0) {
            output.push_back(result);
        } else {
            hemmer_->addInplace(output[j], result);
        } */
    }
    if (rank == 0)
        printElapsedTime("qkv dense");

    if (max_rank > 1) {
        for (u64 i = 0; i < 6; ++i) {
            allReduceWrapper(output[i]);
        }

        if (rank == 0) {
            for (u64 i = 0; i < 4; ++i) {
                hemmer_->getEval().modReduct(output[i].get());
            }
        } else {
            for (u64 i = 4; i < 6; ++i) {
                hemmer_->getEval().modReduct(output[i].get());
            }
        }
    }
    /* std::cout << "reduce wrapper done" << std::endl;
    std::cout << "rank " << rank << std::endl;
    if( rank == 0) {
        printing(output);
    } */

    /* Step 3. Post-process the result.. */
    if (rank == 0) {
        for (u64 i = 0; i < 2; ++i) {
            auto idx = i * 2;
            auto repacked_output =
                hemmer_->repack(output[idx], output[idx + 1]);
            // hemmer_->dropoutInplace(repacked_output, i); //11

            hemmer_->addInplace(input[i], repacked_output); // 11
        }
    } else {
        auto repacked_output = hemmer_->repack(output[4], output[5]);
        // hemmer_->dropoutInplace(repacked_output, i); //11

        hemmer_->addInplace(input[0], repacked_output); // 11
    }
    // printElapsedTime("repacking");
    // std::cout << "atn done" << std::endl;
    if (rank == 0)
        printElapsedTime("repacking");
}

void TransformerBlock::feedForward_bert(std::vector<CtxtTensor> &input) {
    const u64 rank = static_cast<u64>(hemmer_->getRank());
    const u64 max_rank = static_cast<u64>(hemmer_->getMaxRank());
    // Caution: N_HEAD: not 12 but 6 (since, 768 = 128 x 6) <<< Check!!!
    // const u64 n_iter = ModelArgs::N_HEAD / max_rank;
    // const u64 n_iter = 6 / max_rank;
    // const u64 n_iter_half = ModelArgs::N_HEAD / 2 / max_rank;
    const u64 n_iter_half = 3 / max_rank;
    // const u64 n_iter_quarter = ModelArgs::N_HEAD / 4 / max_rank;
    const u64 n_iter_quarter = 2 / max_rank;

    std::vector<CtxtTensor> cur =
        hemmer_->LayerNorm(input, "ffn", layer_n_); // 12 -> 6
    for (u64 i = 0; i < n_iter_half; ++i) {
        auto idx = n_iter_half * rank + i;
        const auto &weight = getWeightMsg("norm2_w", idx);
        const auto &bias = getWeightMsg("norm2_b", idx);
        hemmer_->hadamardMultInplace(cur[i], weight);             // 6 -> 5
        hemmer_->getEval().add(cur[i].get(), bias, cur[i].get()); // 5 -> 5
    }
    printElapsedTime("FFN LN");

    // for (u64 i = 0 ; i < n_iter_quarter; ++i)
    //     hemmer_->bootstrap2(cur[i*2], cur[i*2+1]); // 5 -> 12
    hemmer_->bootstrap2(cur[0], cur[1]);
    hemmer_->bootstrap(cur[2]);
    printElapsedTime("BTS");

    std::vector<CtxtTensor> temp_gate;
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(
                hemmer_->complexPacking(cur[i * 2], cur[i * 2 + 1]), tmp);
        } else {
            hemmer_->matMulPre(cur[2], tmp);
        }
        auto idx = n_iter_quarter * rank + i;
        // Cuation: should fix N_HIDDEN_SUBMATRIX to a right value.
        //  3072 = 768 x 4 = 128 x 6 x 4 >> N_HIDDEN_SUB = 24.
        for (u64 j = 0; j < 24; ++j) {
            // TODO: weight naming.
            auto weight = getWeight("wdin", idx, j);
            auto gate_res = hemmer_->matMulReUse(tmp, weight);
            if (i == 0) {
                temp_gate.push_back(gate_res);
            } else {
                hemmer_->addInplace(temp_gate[j], gate_res);
            }
        }
    }
    printElapsedTime("wdin MM");

    std::vector<CtxtTensor> gate_proj;
    std::vector<u64> sub_mat_idx_div_2;

    gate_proj = std::move(temp_gate);
    hemmer_->repackVector(gate_proj); // 12
    sub_mat_idx_div_2.resize(gate_proj.size());
    // filling indices sequentially
    std::iota(sub_mat_idx_div_2.begin(), sub_mat_idx_div_2.end(), 0);

    printElapsedTime("wdin repack");

    std::vector<CtxtTensor> gate_proj_half;
    // gate_proj.size() = 12;
    gate_proj_half.reserve(gate_proj.size() / 2);
    for (u64 i = gate_proj.size() / 2; i < gate_proj.size(); ++i)
        gate_proj_half.push_back(gate_proj[i]);

    hemmer_->reluVectorInplace(gate_proj_half, layer_n_); // 11 ->
    printElapsedTime("relu");

    // TODO: position of BTS. consider a level consumption.
    for (u64 i = 0; i < gate_proj_half.size(); ++i) {
        hemmer_->hadamardMultInplace(gate_proj_half[i], gate_proj[i]);
    }
    printElapsedTime("hadamardMult");

    std::vector<CtxtTensor> output;
    output.reserve(ModelArgs::N_HEAD);

    for (u64 i = 0; i < (gate_proj_half.size() + 1) / 2; ++i) {
        std::vector<Ciphertext> tmp;
        auto idx = i * 2;
        // use complexPacking unless it is the last odd element
        auto packed_input =
            ((i == gate_proj_half.size() / 2) &&
             (gate_proj_half.size() % 2 != 0))
                ? gate_proj_half[idx]
                : hemmer_->complexPacking(gate_proj_half[idx],
                                          gate_proj_half[idx + 1]);

        hemmer_->matMulPre(packed_input, tmp);

        u64 sub_mat_idx_div_4 = sub_mat_idx_div_2[i * 2] / 2;
        // caution: should consider the right value of N_HEAD.
        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("wdout", sub_mat_idx_div_4, j);
            auto result = hemmer_->matMulReUse(tmp, weight);

            if (i == 0) {
                output.push_back(result);
            } else {
                hemmer_->addInplace(output[j], result);
            }
        }
    }
    printElapsedTime("wdout MM");

    // printElapsedTime("wdout MM");
    for (u64 i = 0; i < n_iter_half; ++i) {
        auto idx = (n_iter_half * rank + i) * 2;
        auto repacked_output = hemmer_->repack(output[idx], output[idx + 1]);

        hemmer_->addInplace(input[i], repacked_output);
    }
    printElapsedTime("wdout repack and addition");
    // std::cout << "ffn output" << std::endl;
    // printing(input);
}

// Fine-tuning version.
void TransformerBlock::feedForward2_bert(std::vector<CtxtTensor> &input) {
    const u64 rank = static_cast<u64>(hemmer_->getRank());

    std::vector<CtxtTensor> cur =
        hemmer_->LayerNorm(input, "ffn", layer_n_, true); // 12 -> 6
    for (u64 i = 0; i < 3; ++i) {
        auto idx = i;
        const auto &weight = getWeightMsg("norm2_w", idx);
        const auto &bias = getWeightMsg("norm2_b", idx);
        hemmer_->hadamardMultInplace(cur[i], weight);             // 6 -> 5
        hemmer_->getEval().add(cur[i].get(), bias, cur[i].get()); // 5 -> 5
    }
    // printElapsedTime("FFN LN");
    if (rank == 0) {
        std::cout << "ffn LN" << std::endl;
        printing(cur);
    }

    hemmer_->bootstrap2(cur[0], cur[1]);
    hemmer_->bootstrap(cur[2]);
    // printElapsedTime("BTS");

    std::vector<CtxtTensor> temp_gate;
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(
                hemmer_->complexPacking(cur[i * 2], cur[i * 2 + 1]), tmp);
        } else {
            hemmer_->matMulPre(cur[2], tmp);
        }
        auto idx = i;
        // Cuation: should fix N_HIDDEN_SUBMATRIX to a right value.
        //  3072 = 768 x 4 = 128 x 6 x 4 >> N_HIDDEN_SUB = 24.
        for (u64 j = 0; j < 24; ++j) {
            // TODO: weight naming.
            auto weight = getWeight("wdin", idx, j);
            auto gate_res = hemmer_->matMulReUse(tmp, weight);
            if (i == 0) {
                temp_gate.push_back(gate_res);
            } else {
                hemmer_->addInplace(temp_gate[j], gate_res);
            }
        }
    }
    // printElapsedTime("wdin MM");

    std::vector<CtxtTensor> gate_proj;
    std::vector<u64> sub_mat_idx_div_2;

    gate_proj = std::move(temp_gate);
    hemmer_->repackVector(gate_proj); // 12
    sub_mat_idx_div_2.resize(gate_proj.size());
    // filling indices sequentially
    std::iota(sub_mat_idx_div_2.begin(), sub_mat_idx_div_2.end(), 0);

    // printElapsedTime("wdin repack");

    std::vector<CtxtTensor> gate_proj_half;
    // gate_proj.size() = 12;
    gate_proj_half.reserve(gate_proj.size() / 2);
    for (u64 i = gate_proj.size() / 2; i < gate_proj.size(); ++i)
        gate_proj_half.push_back(gate_proj[i]);

    hemmer_->reluVectorInplace(gate_proj_half, layer_n_, true); // 11 ->
    // printElapsedTime("relu");
    if (rank == 0) {
        std::cout << "relu output" << std::endl;
        printing(gate_proj_half);
    }

    // TODO: position of BTS. consider a level consumption.
    for (u64 i = 0; i < gate_proj_half.size(); ++i) {
        // For fine-tuning
        // relu_forward: evaluating relu function part.
        // id_forward: not evaluting relu funciton part.
        gate_proj_half[i].get().save(
            hemmer_->getHEPath() + "/relu_forward_" + std::to_string(rank) +
            "_" + std::to_string(layer_n_) + "_" + std::to_string(i) + ".bin");
        gate_proj[i].get().save(
            hemmer_->getHEPath() + "/id_forward_" + std::to_string(rank) + "_" +
            std::to_string(layer_n_) + "_" + std::to_string(i) + ".bin");

        hemmer_->hadamardMultInplace(gate_proj_half[i], gate_proj[i]);
    }
    // printElapsedTime("hadamardMult");

    std::vector<CtxtTensor> output;
    output.reserve(ModelArgs::N_HEAD);
    for (u64 i = 0; i < (gate_proj_half.size() + 1) / 2; ++i) {
        std::vector<Ciphertext> tmp;
        auto idx = i * 2;
        // use complexPacking unless it is the last odd element
        auto packed_input =
            ((i == gate_proj_half.size() / 2) &&
             (gate_proj_half.size() % 2 != 0))
                ? gate_proj_half[idx]
                : hemmer_->complexPacking(gate_proj_half[idx],
                                          gate_proj_half[idx + 1]);

        hemmer_->matMulPre(packed_input, tmp);

        u64 sub_mat_idx_div_4 = sub_mat_idx_div_2[i * 2] / 2;
        // caution: should consider the right value of N_HEAD.
        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("wdout", sub_mat_idx_div_4, j);
            auto result = hemmer_->matMulReUse(tmp, weight);

            if (i == 0) {
                output.push_back(result);
            } else {
                hemmer_->addInplace(output[j], result);
            }
        }
    }
    // printElapsedTime("wdout MM");

    for (u64 i = 0; i < 3; ++i) {
        auto repacked_output =
            hemmer_->repack(output[i * 2], output[i * 2 + 1]);

        // hemmer_->dropoutInplace(input[i], "ffn_res", layer_n_, i);

        hemmer_->addInplace(input[i], repacked_output);
    }
    // printElapsedTime("wdout repack and addition");
    // std::cout << "ffn output" << std::endl;
    // printing(input);
}

// Fine-tuning version.
void TransformerBlock::feedForward2_bert_time(std::vector<CtxtTensor> &input) {
    const u64 rank = static_cast<u64>(hemmer_->getRank());

    start = std::chrono::high_resolution_clock::now();

    std::vector<CtxtTensor> cur =
        hemmer_->LayerNorm(input, "ffn", layer_n_, true); // 12 -> 6
    for (u64 i = 0; i < 3; ++i) {
        auto idx = i;
        const auto &weight = getWeightMsg("norm2_w", idx);
        const auto &bias = getWeightMsg("norm2_b", idx);
        hemmer_->hadamardMultInplace(cur[i], weight);             // 6 -> 5
        hemmer_->getEval().add(cur[i].get(), bias, cur[i].get()); // 5 -> 5
    }
    printElapsedTime("FFN LN");

    hemmer_->bootstrap2(cur[0], cur[1]);
    hemmer_->bootstrap(cur[2]);
    printElapsedTime("BTS");

    std::vector<CtxtTensor> temp_gate;
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(
                hemmer_->complexPacking(cur[i * 2], cur[i * 2 + 1]), tmp);
        } else {
            hemmer_->matMulPre(cur[2], tmp);
        }
        auto idx = i;
        // Cuation: should fix N_HIDDEN_SUBMATRIX to a right value.
        //  3072 = 768 x 4 = 128 x 6 x 4 >> N_HIDDEN_SUB = 24.
        for (u64 j = 0; j < 24; ++j) {
            // TODO: weight naming.
            auto weight = getWeight("wdin", idx, j);
            auto gate_res = hemmer_->matMulReUse(tmp, weight);
            if (i == 0) {
                temp_gate.push_back(gate_res);
            } else {
                hemmer_->addInplace(temp_gate[j], gate_res);
            }
        }
    }
    printElapsedTime("wdin MM");

    std::vector<CtxtTensor> gate_proj;
    std::vector<u64> sub_mat_idx_div_2;

    gate_proj = std::move(temp_gate);
    hemmer_->repackVector(gate_proj); // 12
    printElapsedTime("wdin repack");
    sub_mat_idx_div_2.resize(gate_proj.size());
    // filling indices sequentially
    std::iota(sub_mat_idx_div_2.begin(), sub_mat_idx_div_2.end(), 0);

    std::vector<CtxtTensor> gate_proj_half;
    // gate_proj.size() = 12;
    gate_proj_half.reserve(gate_proj.size() / 2);
    for (u64 i = gate_proj.size() / 2; i < gate_proj.size(); ++i)
        gate_proj_half.push_back(gate_proj[i]);

    hemmer_->reluVectorInplace(gate_proj_half, layer_n_, true); // 11 ->
    printElapsedTime("relu");

    // TODO: position of BTS. consider a level consumption.
    for (u64 i = 0; i < gate_proj_half.size(); ++i) {
        // For fine-tuning
        // relu_forward: evaluating relu function part.
        // id_forward: not evaluting relu funciton part.
        start = std::chrono::high_resolution_clock::now();
        gate_proj_half[i].get().save(
            hemmer_->getHEPath() + "/relu_forward_" + std::to_string(rank) +
            "_" + std::to_string(layer_n_) + "_" + std::to_string(i) + ".bin");
        gate_proj[i].get().save(
            hemmer_->getHEPath() + "/id_forward_" + std::to_string(rank) + "_" +
            std::to_string(layer_n_) + "_" + std::to_string(i) + ".bin");
        printElapsedTime("saving!");

        hemmer_->hadamardMultInplace(gate_proj_half[i], gate_proj[i]);
    }
    printElapsedTime("hadamardMult and saving");

    std::vector<CtxtTensor> output;
    output.reserve(ModelArgs::N_HEAD);
    for (u64 i = 0; i < (gate_proj_half.size() + 1) / 2; ++i) {
        std::vector<Ciphertext> tmp;
        auto idx = i * 2;
        // use complexPacking unless it is the last odd element
        auto packed_input =
            ((i == gate_proj_half.size() / 2) &&
             (gate_proj_half.size() % 2 != 0))
                ? gate_proj_half[idx]
                : hemmer_->complexPacking(gate_proj_half[idx],
                                          gate_proj_half[idx + 1]);

        hemmer_->matMulPre(packed_input, tmp);

        u64 sub_mat_idx_div_4 = sub_mat_idx_div_2[i * 2] / 2;
        // caution: should consider the right value of N_HEAD.
        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("wdout", sub_mat_idx_div_4, j);
            auto result = hemmer_->matMulReUse(tmp, weight);

            if (i == 0) {
                output.push_back(result);
            } else {
                hemmer_->addInplace(output[j], result);
            }
        }
    }
    printElapsedTime("wdout MM");

    for (u64 i = 0; i < 3; ++i) {
        auto repacked_output =
            hemmer_->repack(output[i * 2], output[i * 2 + 1]);
        hemmer_->dropoutInplace(input[i], "ffn_res", layer_n_, i);
        hemmer_->addInplace(input[i], repacked_output);
    }
    printElapsedTime("wdout repack and addition");
    // std::cout << "ffn output" << std::endl;
    // printing(input);
}

// Fine-tuning version.
void TransformerBlock::feedForward2_bert_test(std::vector<CtxtTensor> &input) {
    // const u64 rank = static_cast<u64>(hemmer_->getRank());
    // const u64 max_rank = static_cast<u64>(hemmer_->getMaxRank());

    // std::cout << "rank = " << rank << " ,max_rank = " << max_rank <<
    // std::endl;

    std::vector<CtxtTensor> cur =
        hemmer_->LayerNorm(input, "ffn", layer_n_, false); // 12 -> 6
    for (u64 i = 0; i < 3; ++i) {
        auto idx = i;
        const auto &weight = getWeightMsg("norm2_w", idx);
        const auto &bias = getWeightMsg("norm2_b", idx);
        hemmer_->hadamardMultInplace(cur[i], weight);             // 6 -> 5
        hemmer_->getEval().add(cur[i].get(), bias, cur[i].get()); // 5 -> 5
    }
    // printElapsedTime("LN");
    /* if (rank == 0) {
        std::cout << "after ffn LN" << std::endl;
        printing(cur);
    } */

    hemmer_->bootstrap2(cur[0], cur[1]);
    hemmer_->bootstrap(cur[2]);
    // printElapsedTime("BTS");

    std::vector<CtxtTensor> temp_gate;
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(
                hemmer_->complexPacking(cur[i * 2], cur[i * 2 + 1]), tmp);
        } else {
            hemmer_->matMulPre(cur[2], tmp);
        }
        auto idx = i;
        // Cuation: should fix N_HIDDEN_SUBMATRIX to a right value.
        //  3072 = 768 x 4 = 128 x 6 x 4 >> N_HIDDEN_SUB = 24.
        for (u64 j = 0; j < 24; ++j) {
            // TODO: weight naming.
            auto weight = getWeight("wdin", idx, j);
            auto gate_res = hemmer_->matMulReUse(tmp, weight);
            if (i == 0) {
                temp_gate.push_back(gate_res);
            } else {
                hemmer_->addInplace(temp_gate[j], gate_res);
            }
        }
    }
    // printElapsedTime("wdin MM");

    std::vector<CtxtTensor> gate_proj;
    std::vector<u64> sub_mat_idx_div_2;

    gate_proj = std::move(temp_gate);
    hemmer_->repackVector(gate_proj); // 12
    sub_mat_idx_div_2.resize(gate_proj.size());
    // filling indices sequentially
    std::iota(sub_mat_idx_div_2.begin(), sub_mat_idx_div_2.end(), 0);
    // printElapsedTime("wdin repack");

    /* if (rank == 0) {
        std::cout << "after ffn dense in mm" << std::endl;
        printing(gate_proj);
    } */

    std::vector<CtxtTensor> gate_proj_half;
    // gate_proj.size() = 12;
    gate_proj_half.reserve(gate_proj.size() / 2);
    for (u64 i = gate_proj.size() / 2; i < gate_proj.size(); ++i)
        gate_proj_half.push_back(gate_proj[i]);

    hemmer_->reluVectorInplace(gate_proj_half, layer_n_, false); // 11 ->
    // printElapsedTime("relu");

    /* if (rank == 0) {
        std::cout << "relu output" << std::endl;
        printing(gate_proj_half);
    } */

    // TODO: position of BTS. consider a level consumption.
    for (u64 i = 0; i < gate_proj_half.size(); ++i) {
        hemmer_->hadamardMultInplace(gate_proj_half[i], gate_proj[i]);
    }
    // printElapsedTime("hadamardMult");

    std::vector<CtxtTensor> output;
    output.reserve(ModelArgs::N_HEAD);
    for (u64 i = 0; i < (gate_proj_half.size() + 1) / 2; ++i) {
        std::vector<Ciphertext> tmp;
        auto idx = i * 2;
        // use complexPacking unless it is the last odd element
        auto packed_input =
            ((i == gate_proj_half.size() / 2) &&
             (gate_proj_half.size() % 2 != 0))
                ? gate_proj_half[idx]
                : hemmer_->complexPacking(gate_proj_half[idx],
                                          gate_proj_half[idx + 1]);

        hemmer_->matMulPre(packed_input, tmp);

        u64 sub_mat_idx_div_4 = sub_mat_idx_div_2[i * 2] / 2;
        // caution: should consider the right value of N_HEAD.
        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("wdout", sub_mat_idx_div_4, j);
            auto result = hemmer_->matMulReUse(tmp, weight);

            if (i == 0) {
                output.push_back(result);
            } else {
                hemmer_->addInplace(output[j], result);
            }
        }
    }
    // printElapsedTime("wdout MM");
    /* if(rank == 0) {
        std::cout << "after ffn dense out mm" << std::endl;
        printing(output);
    } */

    for (u64 i = 0; i < 3; ++i) {
        auto repacked_output =
            hemmer_->repack(output[i * 2], output[i * 2 + 1]);

        // hemmer_->dropoutInplace(input[i], "ffn_res", layer_n_, i);

        hemmer_->addInplace(input[i], repacked_output);
    }
    // printElapsedTime("wdout repack and addition");
    // std::cout << "ffn output" << std::endl;
    // printing(input);
}

void TransformerBlock::feedForward_bert_multi(std::vector<CtxtTensor> &input) {
    const u64 rank = static_cast<u64>(hemmer_->getRank());
    const u64 max_rank = static_cast<u64>(hemmer_->getMaxRank());

    std::vector<CtxtTensor> cur =
        hemmer_->LayerNorm_multi(input, "ffn", layer_n_); // 12 -> 6
    if (rank == 0) {
        for (u64 i = 0; i < 2; ++i) {
            const auto &weight = getWeightMsg("norm2_w", i);
            const auto &bias = getWeightMsg("norm2_b", i);
            hemmer_->hadamardMultInplace(cur[i], weight);             // 6 -> 5
            hemmer_->getEval().add(cur[i].get(), bias, cur[i].get()); // 5 -> 5
        }
    } else {
        const auto &weight = getWeightMsg("norm2_w", 2);
        const auto &bias = getWeightMsg("norm2_b", 2);
        hemmer_->hadamardMultInplace(cur[0], weight);
        hemmer_->getEval().add(cur[0].get(), bias, cur[0].get());
    }
    if (rank == 0) {
        printElapsedTime("FFN LN");
    }

    if (rank == 0) {
        hemmer_->bootstrap2(cur[0], cur[1]);
    } else {
        hemmer_->bootstrap(cur[0]);
    }
    if (rank == 0)
        printElapsedTime("BTS");

    std::vector<CtxtTensor> temp_gate;
    for (u64 i = 0; i < 1; ++i) {
        std::vector<Ciphertext> tmp;
        if (rank == 0) {
            hemmer_->matMulPre(hemmer_->complexPacking(cur[0], cur[1]), tmp);
        } else {
            hemmer_->matMulPre(cur[0], tmp);
        }
        // auto idx = n_iter_quarter * rank + i;
        // Cuation: should fix N_HIDDEN_SUBMATRIX to a right value.
        //  3072 = 768 x 4 = 128 x 6 x 4 >> N_HIDDEN_SUB = 24.
        for (u64 j = 0; j < 24; ++j) {
            // TODO: weight naming.
            auto weight = getWeight("wdin", rank, j);
            auto gate_res = hemmer_->matMulReUse(tmp, weight);
            if (i == 0) {
                temp_gate.push_back(gate_res);
            } else {
                hemmer_->addInplace(temp_gate[j], gate_res);
            }
        }
    }
    if (rank == 0)
        printElapsedTime("wdin MM");

    std::vector<CtxtTensor> gate_proj;
    std::vector<u64> sub_mat_idx_div_2;
    if (max_rank > 1) {
        // reduceWrapperHidden(temp_gate);
        for (u64 i = 0; i < temp_gate.size(); ++i)
            allReduceWrapper(temp_gate[i]);

        for (u64 i = 0; i < 12; ++i) {

            sub_mat_idx_div_2.push_back(i);

            auto idx = i * 2;
            hemmer_->getEval().modReduct(temp_gate[idx].get());
            hemmer_->getEval().modReduct(temp_gate[idx + 1].get());
            gate_proj.push_back(
                hemmer_->repack(temp_gate[idx], temp_gate[idx + 1])); // 11
        }

        /* if (rank == 0){
            for (u64 i = 0; i < 6 ; ++i) {

                sub_mat_idx_div_2.push_back(i);

                auto idx = i * 2;
                hemmer_->getEval().modReduct(temp_gate[idx].get());
                hemmer_->getEval().modReduct(temp_gate[idx + 1].get());
                gate_proj.push_back(
                    hemmer_->repack(temp_gate[idx], temp_gate[idx + 1])); //11
            }
        } else {
            for (u64 i = 0; i < 6 ; ++i) {
                sub_mat_idx_div_2.push_back(i);

                auto idx = 12 + i * 2;
                hemmer_->getEval().modReduct(temp_gate[idx].get());
                hemmer_->getEval().modReduct(temp_gate[idx + 1].get());
                gate_proj.push_back(
                    hemmer_->repack(temp_gate[idx], temp_gate[idx + 1])); //11
            }
        } */
    } else {
        gate_proj = std::move(temp_gate);
        hemmer_->repackVector(gate_proj); // 12
        sub_mat_idx_div_2.resize(gate_proj.size());
        // filling indices sequentially
        std::iota(sub_mat_idx_div_2.begin(), sub_mat_idx_div_2.end(), 0);
    }

    if (rank == 0)
        printElapsedTime("repack");
    // std::cout << "wdin output" << std::endl;
    // printing(gate_proj);

    std::vector<CtxtTensor> gate_proj_half;
    gate_proj_half.reserve(6);
    /* for (u64 i = 6 ; i < 12 ; ++i) {
        gate_proj_half.push_back(gate_proj[i]);
    } */
    for (u64 i = 6; i < 12; ++i) {
        gate_proj_half.push_back(gate_proj[i]);
    }
    // evaluate ReLU onto 12 ~ 23 indices.

    hemmer_->reluVectorInplace(gate_proj_half, layer_n_);
    if (rank == 0)
        printElapsedTime("relu");

    // element의 copy가 필요!

    // TODO: position of BTS. consider a level consumption.
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->getEval().mult(gate_proj[i].get(), gate_proj_half[i].get(),
                                gate_proj_half[i].get());
    }

    /* if(rank == 0){
        std::cout << "after hmult" << std::endl;
        printing(gate_proj_half);
    }*/

    std::vector<CtxtTensor> output;
    output.reserve(ModelArgs::N_HEAD);
    if (rank == 0) {
        for (u64 i = 0; i < 2; ++i) {
            std::vector<Ciphertext> tmp;

            auto packed_input = hemmer_->complexPacking(
                gate_proj_half[2 * i], gate_proj_half[2 * i + 1]);

            hemmer_->matMulPre(packed_input, tmp);

            for (u64 j = 0; j < 6; ++j) {
                auto weight = getWeight("wdout", i, j);
                auto result = hemmer_->matMulReUse(tmp, weight);

                if (i == 0) {
                    output.push_back(result);
                } else {
                    hemmer_->addInplace(output[j], result);
                }
            }
        }
    } else {
        std::vector<Ciphertext> tmp;
        auto packed_input =
            hemmer_->complexPacking(gate_proj_half[4], gate_proj_half[5]);
        hemmer_->matMulPre(packed_input, tmp);

        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("wdout", 2, j);
            auto result = hemmer_->matMulReUse(tmp, weight);

            output.push_back(result);
        }
    }
    if (rank == 0)
        printElapsedTime("FFN dense out");

    /* for (u64 i = 0; i < (gate_proj_half.size() + 1) / 2; ++i) {
        std::vector<Ciphertext> tmp;
        auto idx = i * 2;
        // use complexPacking unless it is the last odd element
        auto packed_input =
            ((i == gate_proj_half.size() / 2) && (gate_proj_half.size() % 2 !=
    0)) ? gate_proj_half[idx] : hemmer_->complexPacking(gate_proj_half[idx],
    gate_proj_half[idx + 1]);

        hemmer_->matMulPre(packed_input, tmp);

        u64 sub_mat_idx_div_4 = sub_mat_idx_div_2[i * 2] / 2;
        // caution: should consider the right value of N_HEAD.
        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("wdout", sub_mat_idx_div_4, j);
            auto result = hemmer_->matMulReUse(tmp, weight);

            if (i == 0) {
                output.push_back(result);
            } else {
                hemmer_->addInplace(output[j], result);
            }
        }
    } */

    if (max_rank > 1) {
        reduceWrapper(output);

        if (rank == 0) {
            for (u64 i = 0; i < 4; ++i) {
                hemmer_->getEval().modReduct(output[i].get());
            }
        } else {
            for (u64 i = 4; i < 6; ++i) {
                hemmer_->getEval().modReduct(output[i].get());
            }
        }
    }
    if (rank == 0)
        printElapsedTime("wdout MM");

    if (rank == 0) {
        for (u64 i = 0; i < 2; ++i) {
            auto idx = i * 2;
            auto repacked_output =
                hemmer_->repack(output[idx], output[idx + 1]);
            // TODO: put drop_out function before addition.
            hemmer_->addInplace(input[i], repacked_output);
        }
    } else {
        auto repacked_output = hemmer_->repack(output[4], output[5]);
        // TODO: put drop_out function before addition.
        hemmer_->addInplace(input[0], repacked_output);
    }
    if (rank == 0)
        printElapsedTime("repack");
    // std::cout << "ffn output" << std::endl;
    // printing(input);
}

void TransformerBlock::pooling_bert(std::vector<CtxtTensor> &input) {
    const u64 rank = static_cast<u64>(hemmer_->getRank());
    const u64 max_rank = static_cast<u64>(hemmer_->getMaxRank());
    // const u64 n_iter = 6 / max_rank;
    const u64 n_iter_half = 3 / max_rank;
    const u64 n_iter_quarter = 2 / max_rank;

    std::vector<CtxtTensor> cur =
        hemmer_->LayerNorm(input, "final", layer_n_); // 12 -> 6
    for (u64 i = 0; i < n_iter_half; ++i) {
        auto idx = n_iter_half * rank + i;
        const auto &weight = getWeightMsg("final_norm_w", idx);
        const auto &bias = getWeightMsg("final_norm_b", idx);
        hemmer_->hadamardMultInplace(cur[i], weight);             // 6 -> 5
        hemmer_->getEval().add(cur[i].get(), bias, cur[i].get()); // 5 -> 5
    }
    // printElapsedTime("pooler LN");
    // printing(cur);

    hemmer_->bootstrap2(cur[0], cur[1]);
    hemmer_->bootstrap(cur[2]);

    std::vector<CtxtTensor> temp_gate;
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(
                hemmer_->complexPacking(cur[i * 2], cur[i * 2 + 1]), tmp);
        } else {
            hemmer_->matMulPre(cur[2], tmp);
        }
        auto idx = n_iter_quarter * rank + i;
        // Cuation: should fix N_HIDDEN_SUBMATRIX to a right value.
        //  3072 = 768 x 4 = 128 x 6 x 4 >> N_HIDDEN_SUB = 24.
        for (u64 j = 0; j < 8; ++j) {
            // TODO: weight naming.
            auto weight = getWeight("wfinal1", idx, j);
            auto gate_res = hemmer_->matMulReUse(tmp, weight);
            if (i == 0) {
                temp_gate.push_back(gate_res);
            } else {
                hemmer_->addInplace(temp_gate[j], gate_res);
            }
        }
    }
    // printElapsedTime("wfinal1 MM");

    std::vector<CtxtTensor> gate_proj;
    std::vector<u64> sub_mat_idx_div_2;

    if (max_rank > 1) {
        reduceWrapperHidden(temp_gate, layer_n_);

        for (u64 i = 0; i < 12; ++i) {
            if (getFFNMatrixDevice(layer_n_, i, hemmer_->getMaxRank()) !=
                static_cast<int>(rank))
                continue;

            sub_mat_idx_div_2.push_back(i);

            auto idx = i * 2;
            hemmer_->getEval().modReduct(temp_gate[idx].get());
            hemmer_->getEval().modReduct(temp_gate[idx + 1].get());
            gate_proj.push_back(
                hemmer_->repack(temp_gate[idx], temp_gate[idx + 1])); // 11
        }
    } else {
        gate_proj = std::move(temp_gate);
        hemmer_->repackVector(gate_proj); // 12
        sub_mat_idx_div_2.resize(gate_proj.size());
        // filling indices sequentially
        std::iota(sub_mat_idx_div_2.begin(), sub_mat_idx_div_2.end(), 0);
    }

    // printElapsedTime("repack");
    // printing(gate_proj);
    // std::cout << "wdin output" << std::endl;
    // printing(gate_proj);

    for (u64 i = 0; i < 4; ++i) {
        auto idx = 4 * rank + i;
        const auto &bias = getWeightMsg("wfinal1_b", idx);
        hemmer_->getEval().add(gate_proj[i].get(), bias, gate_proj[i].get());
    }

    // tanh approximation: current plain value.
    hemmer_->tanhVectorInplace(gate_proj, layer_n_);
    // printElapsedTime("tanh");
    // printing(gate_proj);

    // std::cout << "size = " << gate_proj.size();
    // std::cout << "height " << gate_proj[0].getHeight() << ", width = " <<
    // gate_proj[0].getWidth() << std::endl;

    // head linear.
    // hadamardmult >> sum >> rot&sum >> first column (each 2 ctxts): desired
    // values Caution: should generate pre-defined weight value. input: 4 ctxt;

    auto first_col_block = gate_proj;
    // auto second_col_block = gate_proj;

    // hadamardmult
    // TODO: complex packing (for performance)
    for (u64 i = 0; i < 4; ++i) {
        auto idx = 4 * rank + i;
        const auto first_col = getWeightMsg("head_first_col", idx);
        // const auto second_col = getWeightMsg("head_second_col",idx);

        hemmer_->getEval().mult(gate_proj[i].get(), first_col,
                                first_col_block[i].get());
        // hemmer_->getEval().mult(gate_proj[i].get(), second_col,
        // second_col_block[i].get());

        if (i != 0) {
            hemmer_->getEval().add(first_col_block[0].get(),
                                   first_col_block[i].get(),
                                   first_col_block[0].get());
            // hemmer_->getEval().add(second_col_block[0].get(),
            // second_col_block[i].get(), second_col_block[0].get());
        }
    }

    // rot & sum
    Ciphertext tmp{hemmer_->getContext()};
    for (i64 rot = 1; rot < first_col_block[0].getBlockWidth(); rot <<= 1) {
        hemmer_->getEval().leftRotate(first_col_block[0].get(),
                                      static_cast<u64>(rot), tmp);
        hemmer_->getEval().add(first_col_block[0].get(), tmp,
                               first_col_block[0].get());

        // for the stsb task.
        // hemmer_->getEval().mult(first_col_block[0].get(), 1.0/1024,
        // first_col_block[0].gete());

        // hemmer_->getEval().leftRotate(second_col_block[0].get(),
        // static_cast<u64>(rot), tmp);
        // hemmer_->getEval().add(second_col_block[0].get(), tmp,
        // second_col_block[0].get());
    }

    // first cols have an appropriate values!!
    printingfirstCol(first_col_block[0]);
    // printingfirstCol(second_col_block[0]);
}

// output: 9 level
void TransformerBlock::pooling2_bert(std::vector<CtxtTensor> &input,
                                     std::vector<CtxtTensor> &output,
                                     const u64 label) {
    const u64 rank = static_cast<u64>(hemmer_->getRank());
    const u64 max_rank = static_cast<u64>(hemmer_->getMaxRank());

    std::cout << "rank = " << rank << " ,max_rank = " << max_rank << std::endl;

    // const u64 n_iter = 6 / max_rank;
    // const u64 n_iter_half = 3 / max_rank;
    // const u64 n_iter_quarter = 2 / max_rank;

    std::vector<CtxtTensor> cur =
        hemmer_->LayerNorm(input, "final", layer_n_, true); // 12 -> 6
    for (u64 i = 0; i < 3; ++i) {
        auto idx = i;
        const auto &weight = getWeightMsg("final_norm_w", idx);
        const auto &bias = getWeightMsg("final_norm_b", idx);
        hemmer_->hadamardMultInplace(cur[i], weight);             // 6 -> 5
        hemmer_->getEval().add(cur[i].get(), bias, cur[i].get()); // 5 -> 5
    }
    // printElapsedTime("pooler LN");

    // zero row extarction.
    for (u64 i = 0; i < 3; ++i) {
        hemmer_->maskFirstRowInplace(cur[i]); // 5->4
    }
    // std::cout << "pooling ln output" << std::endl;
    // printing(cur);

    // repeated packing
    for (u64 i = 0; i < 3; ++i) {
        for (i64 rot = 1; rot < 128; rot <<= 1) {
            Ciphertext tmp(hemmer_->getContext());
            hemmer_->getEval().rightRotate(cur[i].get(),
                                           static_cast<u64>(rot) * 256, tmp);
            hemmer_->getEval().add(cur[i].get(), tmp, cur[i].get());
        }
    }
    // std::cout << "repeated packing" << std::endl;
    // printing(cur);

    hemmer_->bootstrap2(cur[0], cur[1]); // 12
    hemmer_->bootstrap(cur[2]);

    std::vector<CtxtTensor> temp_gate;
    temp_gate.reserve(8);
    for (u64 i = 0; i < 8; ++i) {
        CtxtTensor tmp{hemmer_->getContext(), cur[0].getShape()};
        for (u64 j = 0; j < 3; ++j) {
            CtxtTensor res{hemmer_->getContext(), cur[0].getShape()};
            auto weight = getWeightMsg("wfinal1", i, j);
            hemmer_->getEval().mult(cur[j].get(), weight, res.get()); // 11

            if (j == 0) {
                tmp = res;
            } else {
                hemmer_->addInplace(tmp, res);
            }
        }

        if (i % 2 == 0) {
            for (i64 rot = 1; rot < 256; rot <<= 1) {
                Ciphertext rot_tmp(hemmer_->getContext());
                hemmer_->getEval().leftRotate(tmp.get(), static_cast<u64>(rot),
                                              rot_tmp);
                hemmer_->getEval().add(tmp.get(), rot_tmp, tmp.get());
            }
            hemmer_->maskFirstColOnlyInplace(tmp); // 10
        } else {
            for (i64 rot = 1; rot < 256; rot <<= 1) {
                Ciphertext rot_tmp(hemmer_->getContext());
                hemmer_->getEval().leftRotate(tmp.get(), static_cast<u64>(rot),
                                              rot_tmp);
                hemmer_->getEval().add(tmp.get(), rot_tmp, tmp.get());
            }
            hemmer_->maskFirstColOnlyInplace(tmp);
            hemmer_->getEval().rightRotate(tmp.get(), 128, tmp.get());
        }

        temp_gate.push_back(tmp);
    }
    // std::cout << "masking" << std::endl;
    // printing(temp_gate);
    // printing_masking(temp_gate[0]);
    // printing_masking(temp_gate[1]);

    std::vector<CtxtTensor> gate_proj; // 10
    gate_proj.reserve(4);
    for (u64 i = 0; i < 4; ++i) {
        hemmer_->addInplace(temp_gate[i * 2], temp_gate[i * 2 + 1]);
        gate_proj.push_back(temp_gate[i * 2]);
    }
    temp_gate.clear();

    for (u64 i = 0; i < 4; ++i) {
        const auto &bias = getWeightMsg("wfinal1_b", i);
        hemmer_->getEval().add(gate_proj[i].get(), bias, gate_proj[i].get());
    }
    // printing(gate_proj);

    //////////////////////////
    // LoRA for head weight //
    //////////////////////////

    CtxtTensor lora_a{cur[0]};
    Ciphertext tmp(hemmer_->getContext());
    // collecting into one ciphertext with appropriate order.
    for (u64 i = 1; i < 3; ++i) {
        hemmer_->getEval().rightRotate(cur[i].get(),
                                       i * ModelArgs::LOW_DIM * 256, tmp);
        hemmer_->getEval().add(lora_a.get(), tmp, lora_a.get());
    }

    // inplace mm with hadamardmult
    // lora_wa_head weight is transposed value.
    auto lora_wa = lora_module_->getCtxtTensor_lora("lora_wa_head", 0, 0, 0);
    hemmer_->getEval().mult(lora_a.get(), lora_wa.get(), lora_a.get()); // 11

    // collecting hadamult results into first col.
    hemmer_->getEval().leftRotate(lora_a.get(), 128, tmp);
    hemmer_->getEval().add(lora_a.get(), tmp, lora_a.get());
    for (u64 i = 1; i < 3; ++i) {
        hemmer_->getEval().leftRotate(lora_a.get(),
                                      i * ModelArgs::LOW_DIM * 256, tmp);
        hemmer_->getEval().add(lora_a.get(), tmp, lora_a.get());
    }

    for (u64 rot = 1; rot < 128; rot <<= 1) {
        hemmer_->getEval().leftRotate(lora_a.get(), rot, tmp);
        hemmer_->getEval().add(lora_a.get(), tmp, lora_a.get());
    }

    hemmer_->maskFirstColPoolingInplace(lora_a); // 10

    // transposition: replacing with rotation ( there migth be faster approach)
    for (u64 i = 1; i < ModelArgs::LOW_DIM; ++i) {
        hemmer_->getEval().leftRotate(lora_a.get(), i * 255, tmp);
        hemmer_->getEval().add(lora_a.get(), tmp, lora_a.get());
    }

    hemmer_->maskFirstRowPoolingInplace(lora_a); // 9

    for (u64 rot = 1; rot < 128; rot <<= 1) {
        hemmer_->getEval().rightRotate(lora_a.get(), rot * 256, tmp);
        hemmer_->getEval().add(lora_a.get(), tmp, lora_a.get());
    }

    hemmer_->getEval().rightRotate(lora_a.get(), 128, tmp);
    hemmer_->getEval().add(lora_a.get(), tmp, lora_a.get());

    lora_module_->saveCtxtTensor(lora_a, "tr_lora_in_b_head", 0, 0, 0);
    for (u64 i = 0; i < 3; ++i) {
        auto tmp_ = cur[i];
        hemmer_->transposeInplace(tmp_);
        lora_module_->saveCtxtTensor(tmp_, "tr_lora_in_a_head", 0, 0, i);
    }

    for (u64 rot = 1; rot < 4; rot <<= 1) {
        hemmer_->getEval().rightRotate(lora_a.get(), rot * ModelArgs::LOW_DIM,
                                       tmp);
        hemmer_->getEval().add(lora_a.get(), tmp, lora_a.get());
    }

    auto lora_wb = lora_module_->getCtxtTensor_lora("lora_wb_head", 0, 0, 0);
    CtxtTensor lora_b{lora_a};
    hemmer_->getEval().mult(lora_b.get(), lora_wb.get(), lora_b.get()); // 8

    for (u64 rot = 1; rot < ModelArgs::LOW_DIM; rot <<= 1) {
        hemmer_->getEval().leftRotate(lora_b.get(), rot, tmp);
        hemmer_->getEval().add(lora_b.get(), tmp, lora_b.get());
    }

    std::vector<CtxtTensor> lora_output;
    lora_output.reserve(4);
    for (u64 i = 0; i < 4; ++i) {
        CtxtTensor tensor_tmp{lora_b};
        if (i != 0) {
            hemmer_->getEval().leftRotate(
                tensor_tmp.get(), i * ModelArgs::LOW_DIM, tensor_tmp.get());
        }
        hemmer_->maskFirstColInplace(tensor_tmp); // 7

        lora_output.push_back(tensor_tmp); // 7
    }

    /* CtxtTensor lora_a{cur[0]};
    auto lora_wa = lora_module_->getCtxtTensor_lora("lora_wa_head", 0,0,0);
    auto lora_a_tmp = hemmer_->matMulHighLow(
        cur[0], hemmer_->getLowColBlock(lora_wa, 0), 0, 5);
    lora_a = lora_a_tmp;
    for (u64 i = 1 ; i < 3 ; ++i) {
        hemmer_->addInplace(
            lora_a,
            hemmer_->matMulHighLow(
                cur[i], hemmer_->getLowColBlock(lora_wa, i), i, 5));
    }

    hemmer_->repackToOneCol(lora_a, 0);
    hemmer_->bootstrap(lora_a);

    std::vector<CtxtTensor> lora_output;
    auto lora_wb = lora_module_->getCtxtTensor_lora("lora_wb_head", 0,0,0);
    for (u64 i = 0; i < 4; ++i) {
        lora_output.emplace_back(hemmer_->matMulLowLow(
            lora_a, hemmer_->getLowRowBlock(lora_wb, i),0,i));
    }

    for (u64 i = 0 ; i < 2 ; ++i)
        hemmer_->bootstrap2(lora_output[i*2], lora_output[i*2+1]);

    for (u64 i = 0 ; i < 4 ; ++i) {
        hemmer_->addInplace(gate_proj[i], lora_output[i]); // 7
    }*/

    // 7 -> 12
    hemmer_->tanhVectorInplace(gate_proj, layer_n_, true);
    // printElapsedTime("tanh");
    // std::cout << "tanh output" << std::endl;
    // printing(gate_proj);

    // Ciphertext tmp(hemmer_->getContext());
    for (u64 i = 0; i < 4; ++i) {
        hemmer_->getEval().rightRotate(gate_proj[i].get(), 1, tmp);
        hemmer_->getEval().add(gate_proj[i].get(), tmp, gate_proj[i].get());
    }
    // std::cout << "repeated packing column-wise" << std::endl;
    // printing(gate_proj);

    // hadamardmult
    // TODO: complex packing (for performance)
    // 12 -> 11
    for (u64 i = 0; i < 4; ++i) {
        const auto weight = getWeightMsg("wfinal2", i);
        hemmer_->getEval().mult(gate_proj[i].get(), weight, gate_proj[i].get());
        if (i != 0) {
            hemmer_->getEval().add(gate_proj[0].get(), gate_proj[i].get(),
                                   gate_proj[0].get());
        }
    }

    // rot & sum
    hemmer_->getEval().leftRotate(gate_proj[0].get(), 128, tmp);
    hemmer_->getEval().add(gate_proj[0].get(), tmp, gate_proj[0].get());

    for (i64 rot = 1; rot < 128; rot <<= 1) {
        hemmer_->getEval().leftRotate(gate_proj[0].get(),
                                      static_cast<u64>(rot) * 256, tmp);
        hemmer_->getEval().add(gate_proj[0].get(), tmp, gate_proj[0].get());
    }

    std::cout << "pooling layer output" << std::endl;
    printing(gate_proj);

    // 11 -> 10
    output = pooling_loss_grad(gate_proj[0], label);
    // std::cout << "pooling forward output level = " << output[0].getLevel() <<
    // std::endl;
    printing(output);

    // printing_masking(output[0]);
}

// output: 9 level
// pooling3 : another weight mm version
// 768 x 32 , 32 x 1024
void TransformerBlock::pooling3_bert(std::vector<CtxtTensor> &input,
                                     CtxtTensor &output, const u64 label) {
    const u64 rank = static_cast<u64>(hemmer_->getRank());
    // const u64 max_rank = static_cast<u64>(hemmer_->getMaxRank());

    std::vector<CtxtTensor> cur =
        hemmer_->LayerNorm(input, "final", layer_n_, true); // 12 -> 6
    for (u64 i = 0; i < 3; ++i) {
        auto idx = i;
        const auto &weight = getWeightMsg("final_norm_w", idx);
        const auto &bias = getWeightMsg("final_norm_b", idx);
        hemmer_->hadamardMultInplace(cur[i], weight);             // 6 -> 5
        hemmer_->getEval().add(cur[i].get(), bias, cur[i].get()); // 5 -> 5
    }
    if (rank == 0) {
        std::cout << "pooling" << std::endl;
        printing(cur);
    }
    // zero index
    for (u64 i = 0; i < 3; ++i) {
        hemmer_->maskFirstRowInplace(cur[i]); // 5->4
    }

    // repacking into one ciphertext
    CtxtTensor cur_repack{cur[0]};
    for (u64 i = 1; i < 3; ++i) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().rightRotate(cur[i].get(),
                                       i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    hemmer_->bootstrap(cur_repack);

    // save for weight a gradient.
    // saved with repeated packing.
    lora_module_->saveCtxtTensor(cur_repack, "tr_pooling_res_ln", 0, 0, 0);

    // W1 MM: (1 x 768) x (768 x 32)
    // W1 is packed row-wise.
    for (u64 rot = 1; rot < ModelArgs::LOW_DIM_HEAD; rot <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().rightRotate(cur_repack.get(), rot * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    auto weight = lora_module_->getCtxtTensor_lora("wfinal1_weight_a", 0, 0, 0);
    // auto weight = weights["wfinal1_weight_a"];

    hemmer_->hadamardMultInplace(cur_repack, weight); // 11

    // collecting
    for (u64 i = 0; i < 1; ++i) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().leftRotate(cur_repack.get(), 128, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    for (u64 i = 1; i < 3; ++i) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().leftRotate(cur_repack.get(),
                                      i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    for (u64 rot = 1; rot < 128; rot <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().leftRotate(cur_repack.get(), rot, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    hemmer_->maskFirstColPoolingInplace(cur_repack); // 10
    // auto bias = getWeightMsg("wfinal1_bias_a");
    // hemmer_->getEval().add(cur_repack.get(), bias, cur_repack.get());

    // for weight b mm, repeated packing
    for (u64 i = 1; i < 4; i <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        if (i == 1) {
            for (u64 rot = 1; rot < 256; rot <<= 1) {
                hemmer_->getEval().rightRotate(cur_repack.get(), rot, tmp);
                hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
            }
        }
        hemmer_->getEval().rightRotate(cur_repack.get(),
                                       i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    // save for weight b gradient.
    lora_module_->saveCtxtTensor(cur_repack, "tr_pooling_res_wa", 0, 0, 0);

    weight = lora_module_->getCtxtTensor_lora("wfinal1_weight_b", 0, 0, 0);
    // weight = weights["wfinal1_weight_b"];

    hemmer_->hadamardMultInplace(cur_repack, weight); // 9

    // collecting results
    for (u64 i = 1; i < ModelArgs::LOW_DIM_HEAD; i <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().leftRotate(cur_repack.get(), i * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    hemmer_->maskFirstRowsPoolingInplace(cur_repack); // 8
    // bias = getWeightMsg("wfinal1_bias_b");
    // hemmer_->getEval().add(cur_repack.get(), bias, cur_repack.get());

    // 7 -> 12
    if (rank == 0) {
        std::cout << "tanh input " << std::endl;
        printing_masking(cur_repack);
    }

    hemmer_->tanhInplace(cur_repack, layer_n_, true);
    if (rank == 0) {
        std::cout << "tanh output " << std::endl;
        printing_masking(cur_repack);
    }
    // printElapsedTime("tanh");
    //  saved with level 12
    //  tanh value position: k-th row , where k = 0, 32, 64 , 96
    lora_module_->saveCtxtTensor(cur_repack, "tr_pooling_res_tanh", 0, 0, 0);

    // repeate packing for head dense
    for (u64 i = 0; i < 1; ++i) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().rightRotate(cur_repack.get(), 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    // hadamardmult
    // 12 -> 11
    weight = lora_module_->getCtxtTensor_lora("wfinal2_weight", 0, 0, 0);
    // weight = weights["wfinal2_weight"];

    hemmer_->hadamardMultInplace(cur_repack, weight);

    // collecting results.
    for (u64 i = 1; i < 4; i <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        if (i == 1) {
            hemmer_->getEval().leftRotate(cur_repack.get(), 128, tmp);
            hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());

            for (u64 rot = 1; rot < 128; rot <<= 1) {
                hemmer_->getEval().leftRotate(cur_repack.get(), rot, tmp);
                hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
            }
        }

        hemmer_->getEval().leftRotate(cur_repack.get(),
                                      i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    // bias = getWeightMsg("wfinal2_bias");
    // hemmer_->getEval().add(cur_repack.get(), bias, cur_repack.get());

    // std::cout << "pooling output" << std::endl;
    // printing_masking(cur_repack);

    // 11 -> 10
    if (rank == 0) {
        std::cout << "pooling loss input " << std::endl;
        printing_masking(cur_repack);
    }
    output = pooling3_loss_grad(cur_repack, label);
    // std::cout << "pooling forward output level = " << output[0].getLevel() <<
    // std::endl;
    if (rank == 0) {
        std::cout << "pooling loss output " << std::endl;
        printing_masking(output);
    }
}

// output: 9 level
// pooling3 : another weight mm version
// 768 x 32 , 32 x 1024
void TransformerBlock::pooling3_bert_sst2(std::vector<CtxtTensor> &input,
                                          CtxtTensor &output, const u64 label) {
    const u64 rank = static_cast<u64>(hemmer_->getRank());
    // const u64 max_rank = static_cast<u64>(hemmer_->getMaxRank());

    std::vector<CtxtTensor> cur =
        hemmer_->LayerNorm(input, "final", layer_n_, true); // 12 -> 6
    for (u64 i = 0; i < 3; ++i) {
        auto idx = i;
        const auto &weight = getWeightMsg("final_norm_w", idx);
        const auto &bias = getWeightMsg("final_norm_b", idx);
        hemmer_->hadamardMultInplace(cur[i], weight);             // 6 -> 5
        hemmer_->getEval().add(cur[i].get(), bias, cur[i].get()); // 5 -> 5
    }
    if (rank == 0) {
        std::cout << "pooling" << std::endl;
        printing(cur);
    }
    // zero index
    for (u64 i = 0; i < 3; ++i) {
        hemmer_->maskFirstRowInplace(cur[i]); // 5->4
    }

    // repacking into one ciphertext
    CtxtTensor cur_repack{cur[0]};
    for (u64 i = 1; i < 3; ++i) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().rightRotate(cur[i].get(),
                                       i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    hemmer_->bootstrap(cur_repack);

    // save for weight a gradient.
    // saved with repeated packing.
    lora_module_->saveCtxtTensor(cur_repack, "tr_pooling_res_ln", 0, 0, 0);

    // W1 MM: (1 x 768) x (768 x 32)
    // W1 is packed row-wise.
    for (u64 rot = 1; rot < ModelArgs::LOW_DIM_HEAD; rot <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().rightRotate(cur_repack.get(), rot * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    auto weight = lora_module_->getCtxtTensor_lora("wfinal1_weight_a", 0, 0, 0);
    // auto weight = weights["wfinal1_weight_a"];

    hemmer_->hadamardMultInplace(cur_repack, weight); // 11

    // collecting
    for (u64 i = 0; i < 1; ++i) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().leftRotate(cur_repack.get(), 128, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    for (u64 i = 1; i < 3; ++i) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().leftRotate(cur_repack.get(),
                                      i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    for (u64 rot = 1; rot < 128; rot <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().leftRotate(cur_repack.get(), rot, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    hemmer_->maskFirstColPoolingInplace(cur_repack); // 10
    // auto bias = getWeightMsg("wfinal1_bias_a");
    // hemmer_->getEval().add(cur_repack.get(), bias, cur_repack.get());

    // for weight b mm, repeated packing
    for (u64 i = 1; i < 4; i <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        if (i == 1) {
            for (u64 rot = 1; rot < 256; rot <<= 1) {
                hemmer_->getEval().rightRotate(cur_repack.get(), rot, tmp);
                hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
            }
        }
        hemmer_->getEval().rightRotate(cur_repack.get(),
                                       i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    // save for weight b gradient.
    lora_module_->saveCtxtTensor(cur_repack, "tr_pooling_res_wa", 0, 0, 0);

    weight = lora_module_->getCtxtTensor_lora("wfinal1_weight_b", 0, 0, 0);
    // weight = weights["wfinal1_weight_b"];

    hemmer_->hadamardMultInplace(cur_repack, weight); // 9

    // collecting results
    for (u64 i = 1; i < ModelArgs::LOW_DIM_HEAD; i <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().leftRotate(cur_repack.get(), i * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    hemmer_->maskFirstRowsPoolingInplace(cur_repack); // 8
    // bias = getWeightMsg("wfinal1_bias_b");
    // hemmer_->getEval().add(cur_repack.get(), bias, cur_repack.get());

    auto tanh_dec = hemmer_->decrypt2(cur_repack);
    auto max = tanh_dec.max().item<double>();
    auto min = tanh_dec.min().item<double>();
    std::cout << "tanh input min: " << min << ", max: " << max << std::endl;

    HEaaN::CudaTools::cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    // 7 -> 12
    if (rank == 0) {
        std::cout << "tanh input " << std::endl;
        printing_masking(cur_repack);
    }

    hemmer_->tanhInplace_SST2(cur_repack, layer_n_, true);
    if (rank == 0) {
        std::cout << "tanh output " << std::endl;
        printing_masking(cur_repack);
    }
    // printElapsedTime("tanh");
    //  saved with level 12
    //  tanh value position: k-th row , where k = 0, 32, 64 , 96
    lora_module_->saveCtxtTensor(cur_repack, "tr_pooling_res_tanh", 0, 0, 0);

    // repeate packing for head dense
    for (u64 i = 0; i < 1; ++i) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().rightRotate(cur_repack.get(), 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    // hadamardmult
    // 12 -> 11
    weight = lora_module_->getCtxtTensor_lora("wfinal2_weight", 0, 0, 0);
    // weight = weights["wfinal2_weight"];

    hemmer_->hadamardMultInplace(cur_repack, weight);

    // collecting results.
    for (u64 i = 1; i < 4; i <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        if (i == 1) {
            hemmer_->getEval().leftRotate(cur_repack.get(), 128, tmp);
            hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());

            for (u64 rot = 1; rot < 128; rot <<= 1) {
                hemmer_->getEval().leftRotate(cur_repack.get(), rot, tmp);
                hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
            }
        }

        hemmer_->getEval().leftRotate(cur_repack.get(),
                                      i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    // bias = getWeightMsg("wfinal2_bias");
    // hemmer_->getEval().add(cur_repack.get(), bias, cur_repack.get());

    // std::cout << "pooling output" << std::endl;
    // printing_masking(cur_repack);

    auto loss_dec = hemmer_->decrypt2(cur_repack);
    max = loss_dec.max().item<double>();
    min = loss_dec.min().item<double>();
    std::cout << "loss input min: " << min << ", max: " << max << std::endl;
    HEaaN::CudaTools::cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    // 11 -> 10
    if (rank == 0) {
        std::cout << "pooling loss input " << std::endl;
        printing_masking(cur_repack);
    }
    output = pooling3_loss_grad_sst2(cur_repack, label);
    // std::cout << "pooling forward output level = " << output[0].getLevel() <<
    // std::endl;
    if (rank == 0) {
        std::cout << "pooling loss output " << std::endl;
        printing_masking(output);
    }
}

// output: 9 level
// pooling3 : another weight mm version
// 768 x 32 , 32 x 1024
void TransformerBlock::pooling3_bert_stsb(std::vector<CtxtTensor> &input,
                                          CtxtTensor &output, const u64 label) {
    const u64 rank = static_cast<u64>(hemmer_->getRank());
    // const u64 max_rank = static_cast<u64>(hemmer_->getMaxRank());

    std::vector<CtxtTensor> cur =
        hemmer_->LayerNorm(input, "final", layer_n_, true); // 12 -> 6
    for (u64 i = 0; i < 3; ++i) {
        auto idx = i;
        const auto &weight = getWeightMsg("final_norm_w", idx);
        const auto &bias = getWeightMsg("final_norm_b", idx);
        hemmer_->hadamardMultInplace(cur[i], weight);             // 6 -> 5
        hemmer_->getEval().add(cur[i].get(), bias, cur[i].get()); // 5 -> 5
    }
    if (rank == 0) {
        std::cout << "pooling" << std::endl;
        printing(cur);
    }
    // zero index
    for (u64 i = 0; i < 3; ++i) {
        hemmer_->maskFirstRowInplace(cur[i]); // 5->4
    }

    // repacking into one ciphertext
    CtxtTensor cur_repack{cur[0]};
    for (u64 i = 1; i < 3; ++i) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().rightRotate(cur[i].get(),
                                       i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    hemmer_->bootstrap(cur_repack);

    // save for weight a gradient.
    // saved with repeated packing.
    lora_module_->saveCtxtTensor(cur_repack, "tr_pooling_res_ln", 0, 0, 0);

    // W1 MM: (1 x 768) x (768 x 32)
    // W1 is packed row-wise.
    for (u64 rot = 1; rot < ModelArgs::LOW_DIM_HEAD; rot <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().rightRotate(cur_repack.get(), rot * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    auto weight = lora_module_->getCtxtTensor_lora("wfinal1_weight_a", 0, 0, 0);
    // auto weight = weights["wfinal1_weight_a"];

    hemmer_->hadamardMultInplace(cur_repack, weight); // 11

    // collecting
    for (u64 i = 0; i < 1; ++i) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().leftRotate(cur_repack.get(), 128, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    for (u64 i = 1; i < 3; ++i) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().leftRotate(cur_repack.get(),
                                      i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    for (u64 rot = 1; rot < 128; rot <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().leftRotate(cur_repack.get(), rot, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    hemmer_->maskFirstColPoolingInplace(cur_repack); // 10
    // auto bias = getWeightMsg("wfinal1_bias_a");
    // hemmer_->getEval().add(cur_repack.get(), bias, cur_repack.get());

    // for weight b mm, repeated packing
    for (u64 i = 1; i < 4; i <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        if (i == 1) {
            for (u64 rot = 1; rot < 256; rot <<= 1) {
                hemmer_->getEval().rightRotate(cur_repack.get(), rot, tmp);
                hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
            }
        }
        hemmer_->getEval().rightRotate(cur_repack.get(),
                                       i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    // save for weight b gradient.
    lora_module_->saveCtxtTensor(cur_repack, "tr_pooling_res_wa", 0, 0, 0);

    weight = lora_module_->getCtxtTensor_lora("wfinal1_weight_b", 0, 0, 0);
    // weight = weights["wfinal1_weight_b"];

    hemmer_->hadamardMultInplace(cur_repack, weight); // 9

    // collecting results
    for (u64 i = 1; i < ModelArgs::LOW_DIM_HEAD; i <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().leftRotate(cur_repack.get(), i * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    hemmer_->maskFirstRowsPoolingInplace(cur_repack); // 8
    // bias = getWeightMsg("wfinal1_bias_b");
    // hemmer_->getEval().add(cur_repack.get(), bias, cur_repack.get());

    // 7 -> 12
    if (rank == 0) {
        std::cout << "tanh input " << std::endl;
        printing_masking(cur_repack);
    }

    hemmer_->tanhInplace(cur_repack, layer_n_, true);
    if (rank == 0) {
        std::cout << "tanh output " << std::endl;
        printing_masking(cur_repack);
    }
    // printElapsedTime("tanh");
    //  saved with level 12
    //  tanh value position: k-th row , where k = 0, 32, 64 , 96
    lora_module_->saveCtxtTensor(cur_repack, "tr_pooling_res_tanh", 0, 0, 0);

    // hadamardmult
    // 12 -> 11
    weight = lora_module_->getCtxtTensor_lora("wfinal2_weight", 0, 0, 0);

    hemmer_->hadamardMultInplace(cur_repack, weight);

    // collecting results.
    for (u64 i = 1; i < 4; i <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        if (i == 1) {
            hemmer_->getEval().leftRotate(cur_repack.get(), 128, tmp);
            hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());

            for (u64 rot = 1; rot < 128; rot <<= 1) {
                hemmer_->getEval().leftRotate(cur_repack.get(), rot, tmp);
                hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
            }
        }

        hemmer_->getEval().leftRotate(cur_repack.get(),
                                      i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    // 11 -> 10
    if (rank == 0) {
        std::cout << "pooling loss input " << std::endl;
        printing_masking(cur_repack);
    }
    output = pooling3_loss_grad_mse(cur_repack, label);
    // std::cout << "pooling forward output level = " << output[0].getLevel() <<
    // std::endl;
    if (rank == 0) {
        std::cout << "pooling loss output " << std::endl;
        printing_masking(output);
    }
}

// output: 9 level
// pooling3 : another weight mm version
// 768 x 32 , 32 x 1024
void TransformerBlock::pooling3_bert_time(std::vector<CtxtTensor> &input,
                                          CtxtTensor &output, const u64 label) {

    // const u64 rank = static_cast<u64>(hemmer_->getRank());

    start = std::chrono::high_resolution_clock::now();

    std::vector<CtxtTensor> cur =
        hemmer_->LayerNorm(input, "final", layer_n_, true); // 12 -> 6
    for (u64 i = 0; i < 3; ++i) {
        auto idx = i;
        const auto &weight = getWeightMsg("final_norm_w", idx);
        const auto &bias = getWeightMsg("final_norm_b", idx);
        hemmer_->hadamardMultInplace(cur[i], weight);             // 6 -> 5
        hemmer_->getEval().add(cur[i].get(), bias, cur[i].get()); // 5 -> 5
    }
    printElapsedTime("pooling LN");

    // zero index
    for (u64 i = 0; i < 3; ++i) {
        hemmer_->maskFirstRowInplace(cur[i]); // 5->4
    }
    printElapsedTime("hadamult");

    // repacking into one ciphertext
    CtxtTensor cur_repack{cur[0]};
    for (u64 i = 1; i < 3; ++i) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().rightRotate(cur[i].get(),
                                       i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }
    printElapsedTime("collecting into one ctxt");

    hemmer_->bootstrap(cur_repack);
    printElapsedTime("BTS");

    // save for weight a gradient.
    // saved with repeated packing.
    lora_module_->saveCtxtTensor(cur_repack, "tr_pooling_res_ln", 0, 0, 0);
    printElapsedTime("saving");

    // W1 MM: (1 x 768) x (768 x 32)
    // W1 is packed row-wise.
    for (u64 rot = 1; rot < ModelArgs::LOW_DIM_HEAD; rot <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().rightRotate(cur_repack.get(), rot * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    auto weight = lora_module_->getCtxtTensor_lora("wfinal1_weight_a", 0, 0, 0);
    // auto weight = weights["wfinal1_weight_a"];

    hemmer_->hadamardMultInplace(cur_repack, weight); // 11
    printElapsedTime("wfinal1_a mm");

    // collecting
    for (u64 i = 0; i < 1; ++i) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().leftRotate(cur_repack.get(), 128, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    for (u64 i = 1; i < 3; ++i) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().leftRotate(cur_repack.get(),
                                      i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    for (u64 rot = 1; rot < 128; rot <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().leftRotate(cur_repack.get(), rot, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    hemmer_->maskFirstColPoolingInplace(cur_repack); // 10
    printElapsedTime("collecting");

    // auto bias = getWeightMsg("wfinal1_bias_a");
    // hemmer_->getEval().add(cur_repack.get(), bias, cur_repack.get());

    // for weight b mm, repeated packing
    for (u64 i = 1; i < 4; i <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        if (i == 1) {
            for (u64 rot = 1; rot < 256; rot <<= 1) {
                hemmer_->getEval().rightRotate(cur_repack.get(), rot, tmp);
                hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
            }
        }
        hemmer_->getEval().rightRotate(cur_repack.get(),
                                       i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    // save for weight b gradient.
    lora_module_->saveCtxtTensor(cur_repack, "tr_pooling_res_wa", 0, 0, 0);

    weight = lora_module_->getCtxtTensor_lora("wfinal1_weight_b", 0, 0, 0);
    // weight = weights["wfinal1_weight_b"];

    hemmer_->hadamardMultInplace(cur_repack, weight); // 9
    printElapsedTime("wfina1l_b mm");

    // collecting results
    for (u64 i = 1; i < ModelArgs::LOW_DIM_HEAD; i <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().leftRotate(cur_repack.get(), i * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    hemmer_->maskFirstRowsPoolingInplace(cur_repack); // 8
    printElapsedTime("collecting");
    // bias = getWeightMsg("wfinal1_bias_b");
    // hemmer_->getEval().add(cur_repack.get(), bias, cur_repack.get());

    // 7 -> 12
    hemmer_->tanhInplace(cur_repack, layer_n_, true);
    printElapsedTime("tanh");

    // saved with level 12
    // tanh value position: k-th row , where k = i*LOW_HEAD_DIM
    lora_module_->saveCtxtTensor(cur_repack, "tr_pooling_res_tanh", 0, 0, 0);
    printElapsedTime("saving");

    // repeate packing for head dense
    for (u64 i = 0; i < 1; ++i) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().rightRotate(cur_repack.get(), 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    // hadamardmult
    // 12 -> 11
    weight = lora_module_->getCtxtTensor_lora("wfinal2_weight", 0, 0, 0);
    // weight = weights["wfinal2_weight"];

    hemmer_->hadamardMultInplace(cur_repack, weight);
    printElapsedTime("wfina2 mm");

    // collecting results.
    for (u64 i = 1; i < 4; i <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        if (i == 1) {
            hemmer_->getEval().leftRotate(cur_repack.get(), 128, tmp);
            hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());

            for (u64 rot = 1; rot < 128; rot <<= 1) {
                hemmer_->getEval().leftRotate(cur_repack.get(), rot, tmp);
                hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
            }
        }

        hemmer_->getEval().leftRotate(cur_repack.get(),
                                      i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }
    printElapsedTime("collecting");

    // 11 -> 10
    output = pooling3_loss_grad(cur_repack, label);
    printElapsedTime("calculate loss grad");
}

// output: 9 level
// pooling3 : another weight mm version
// 768 x 32 , 32 x 1024
void TransformerBlock::pooling3_bert_test(std::vector<CtxtTensor> &input) {
    const u64 rank = static_cast<u64>(hemmer_->getRank());
    // const u64 max_rank = static_cast<u64>(hemmer_->getMaxRank());

    std::vector<CtxtTensor> cur =
        hemmer_->LayerNorm(input, "final", layer_n_, false); // 12 -> 6
    for (u64 i = 0; i < 3; ++i) {
        auto idx = i;
        const auto &weight = getWeightMsg("final_norm_w", idx);
        const auto &bias = getWeightMsg("final_norm_b", idx);
        hemmer_->hadamardMultInplace(cur[i], weight);             // 6 -> 5
        hemmer_->getEval().add(cur[i].get(), bias, cur[i].get()); // 5 -> 5
    }
    // printElapsedTime("LN");
    /* if (rank == 0) {
        std::cout << "pooling LN" << std::endl;
        printing(cur);
    } */

    // zero index
    for (u64 i = 0; i < 3; ++i) {
        hemmer_->maskFirstRowInplace(cur[i]); // 5->4
    }

    // repacking into one ciphertext
    CtxtTensor cur_repack{cur[0]};
    for (u64 i = 1; i < 3; ++i) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().rightRotate(cur[i].get(),
                                       i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }
    // printElapsedTime("collecting into one ctxt");

    hemmer_->bootstrap(cur_repack);
    // printElapsedTime("BTS");

    // W1 MM: (1 x 768) x (768 x 32)
    // W1 is packed row-wise.
    for (u64 rot = 1; rot < ModelArgs::LOW_DIM_HEAD; rot <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().rightRotate(cur_repack.get(), rot * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    auto weight =
        lora_module_->getCtxtTensor_lora_test("wfinal1_weight_a", 0, 0, 0);

    if (rank == 0) {
        auto pth_weight = hemmer_->decrypt2(weight);
        lora_module_->saveTorchTensor(pth_weight, "wfinal1_weight_a", 0);
    }

    // auto weight = getWeightMsg("wfinal1_weight_a");
    hemmer_->hadamardMultInplace(cur_repack, weight); // 11
    // printElapsedTime("w1_a mm");

    // collecting
    for (u64 i = 0; i < 1; ++i) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().leftRotate(cur_repack.get(), 128, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    for (u64 i = 1; i < 3; ++i) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().leftRotate(cur_repack.get(),
                                      i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    for (u64 rot = 1; rot < 128; rot <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().leftRotate(cur_repack.get(), rot, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    hemmer_->maskFirstColPoolingInplace(cur_repack); // 10
    // printElapsedTime("repack (collecting)");

    /* if (rank == 0) {
        std::cout << "pooling weight_a mm " << std::endl;
        printing_masking(cur_repack);
    } */

    // auto bias = getWeightMsg("wfinal1_bias_a");
    // hemmer_->getEval().add(cur_repack.get(), bias, cur_repack.get());

    // for weight b mm, repeated packing
    for (u64 i = 1; i < 4; i <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        if (i == 1) {
            for (u64 rot = 1; rot < 256; rot <<= 1) {
                hemmer_->getEval().rightRotate(cur_repack.get(), rot, tmp);
                hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
            }
        }
        hemmer_->getEval().rightRotate(cur_repack.get(),
                                       i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    weight = lora_module_->getCtxtTensor_lora_test("wfinal1_weight_b", 0, 0, 0);

    if (rank == 0) {
        auto pth_weight = hemmer_->decrypt2(weight);
        lora_module_->saveTorchTensor(pth_weight, "wfinal1_weight_b", 0);
    }

    // weight = getWeightMsg("wfinal1_weight_b");
    hemmer_->hadamardMultInplace(cur_repack, weight); // 9
    // printElapsedTime("w1_b mm");

    // collecting results
    for (u64 i = 1; i < ModelArgs::LOW_DIM_HEAD; i <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().leftRotate(cur_repack.get(), i * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    hemmer_->maskFirstRowsPoolingInplace(cur_repack); // 8
    // printElapsedTime("repack (collecting) ");

    /* if (rank == 0) {
        std::cout << "pooling weight_b mm " << std::endl;
        printing_masking(cur_repack);
    } */

    // bias = getWeightMsg("wfinal1_bias_b");
    // hemmer_->getEval().add(cur_repack.get(), bias, cur_repack.get());

    // 7 -> 12
    hemmer_->tanhInplace(cur_repack, layer_n_, false);
    // printElapsedTime("tanh ");

    /* if (rank == 0) {
        std::cout << "tanh output" << std::endl;
        printing_masking(cur_repack);
    } */

    // repeate packing for head dense
    for (u64 i = 0; i < 1; ++i) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().rightRotate(cur_repack.get(), 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }

    // hadamardmult
    // 12 -> 11
    weight = lora_module_->getCtxtTensor_lora_test("wfinal2_weight", 0, 0, 0);

    if (rank == 0) {
        auto pth_weight = hemmer_->decrypt2(weight);
        lora_module_->saveTorchTensor(pth_weight, "wfinal2_weight", 0);
    }

    // weight = getWeightMsg("wfinal2_weight");
    hemmer_->hadamardMultInplace(cur_repack, weight);
    // printElapsedTime("w2 mm");

    // collecting results.
    for (u64 i = 1; i < 4; i <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        if (i == 1) {
            hemmer_->getEval().leftRotate(cur_repack.get(), 128, tmp);
            hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());

            for (u64 rot = 1; rot < 128; rot <<= 1) {
                hemmer_->getEval().leftRotate(cur_repack.get(), rot, tmp);
                hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
            }
        }

        hemmer_->getEval().leftRotate(cur_repack.get(),
                                      i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
        hemmer_->getEval().add(cur_repack.get(), tmp, cur_repack.get());
    }
    // printElapsedTime("repack (collecting) ");

    /* if (rank == 0) {
        std::cout << "pooling weight2 mm " << std::endl;
        printing_masking(cur_repack);
    } */

    // bias = getWeightMsg("wfinal2_bias");
    // hemmer_->getEval().add(cur_repack.get(), bias, cur_repack.get());

    HEaaN::CudaTools::cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    printingfirstCol(cur_repack);
}

///////////////////////////
////// Bert backward //////
///////////////////////////

// in: 1 x 2 , out: 128 x 768
// in: 9 level -> 12
std::vector<CtxtTensor>
TransformerBlock::backwardpooling2_bert(std::vector<CtxtTensor> &grad_y) {

    // std::cout << "input level " << grad_y[0].get().getLevel() << std::endl;
    //  grad @ W_dout^T
    std::vector<CtxtTensor> tmp_grad;
    // 1024 / 128 / 2 = 4
    tmp_grad.reserve(4);
    // TODO: consider complex packing structure.
    // input gradient
    for (u64 i = 0; i < 4; ++i) {
        Ciphertext tmp(hemmer_->getContext());
        tmp_grad.push_back(grad_y[i]);
        const auto weight = getWeightMsg("tr_wfinal2", i);
        hemmer_->hadamardMultInplace(tmp_grad[i], weight); // 1
        hemmer_->getEval().leftRotate(tmp_grad[i].get(), 1, tmp);
        hemmer_->getEval().add(tmp_grad[i].get(), tmp, tmp_grad[i].get());
        hemmer_->maskFirstColInplace(tmp_grad[i]); // 1
    }
    // std::cout << "wfinal2 grad, output level " <<
    // tmp_grad[0].get().getLevel() << std::endl;

    // grad @ W_dout^T @ tanh'(input)
    // 7
    hemmer_->backwardtanhVectorInplace(tmp_grad, layer_n_);

    // std::cout << "tanh backward" << std::endl;

    ///////////////////
    // LoRA backward //
    ///////////////////
    // LoRA: here, we need unrepeated packing.
    std::vector<CtxtTensor> tmp_grad_;
    tmp_grad_ = tmp_grad;
    auto tr_lora_wb = lora_module_->getCtxtTensor_lora("lora_wb_head", 0, 0, 0);

    CtxtTensor grad_lora_b{tmp_grad_[0]};
    for (u64 i = 1; i < 4; ++i) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().rightRotate(tmp_grad_[i].get(),
                                       i * ModelArgs::LOW_DIM, tmp);
        hemmer_->getEval().add(grad_lora_b.get(), tmp, grad_lora_b.get());
    }

    // LOW_DIM should be a power-of-2 value.
    for (u64 rot = 1; rot < ModelArgs::LOW_DIM; rot <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().rightRotate(grad_lora_b.get(), rot, tmp);
        hemmer_->getEval().add(grad_lora_b.get(), tmp, grad_lora_b.get());
    }

    hemmer_->hadamardMultInplace(grad_lora_b, tr_lora_wb); // 6

    // collecting
    Ciphertext tmp(hemmer_->getContext());
    hemmer_->getEval().leftRotate(grad_lora_b.get(), 128, tmp);
    hemmer_->getEval().add(grad_lora_b.get(), tmp, grad_lora_b.get());

    for (u64 rot = 1; rot < 128; rot <<= 1) {
        hemmer_->getEval().leftRotate(grad_lora_b.get(), rot * 256, tmp);
        hemmer_->getEval().add(grad_lora_b.get(), tmp, grad_lora_b.get());
    }

    // masking: first row (8 elements);
    hemmer_->maskFirstRowPoolingInplace(grad_lora_b); // 5

    hemmer_->bootstrap(grad_lora_b); // 12
    // for lora_a gradient.
    auto grad_lora_a_tmp = grad_lora_b; // 12

    // transposition
    for (u64 i = 1; i < ModelArgs::LOW_DIM; ++i) {
        hemmer_->getEval().rightRotate(grad_lora_b.get(), i * 255, tmp);
        hemmer_->getEval().add(grad_lora_b.get(), tmp, grad_lora_b.get());
    }

    hemmer_->maskFirstColPoolingInplace(grad_lora_b); // 11

    for (u64 rot = 1; rot < 256; rot <<= 1) {
        hemmer_->getEval().rightRotate(grad_lora_b.get(), rot, tmp);
        hemmer_->getEval().add(grad_lora_b.get(), tmp, grad_lora_b.get());
    }

    for (u64 i = 1; i < 3; ++i) {
        hemmer_->getEval().rightRotate(grad_lora_b.get(),
                                       i * ModelArgs::LOW_DIM * 256, tmp);
        hemmer_->getEval().add(grad_lora_b.get(), tmp, grad_lora_b.get());
    }

    // std::cout << "lora wb head" << std::endl;

    auto tr_lora_wa = lora_module_->getCtxtTensor_lora("lora_wa_head", 0, 0, 0);
    hemmer_->hadamardMultInplace(grad_lora_b, tr_lora_wa); // 10

    for (u64 rot = 1; rot < ModelArgs::LOW_DIM; rot <<= 1) {
        hemmer_->getEval().leftRotate(grad_lora_b.get(), rot * 256, tmp);
        hemmer_->getEval().add(grad_lora_b.get(), tmp, grad_lora_b.get());
    }
    // detach and addition part is beyond weight mm

    // std::cout << "lora_wa_head" << std::endl;

    // lora grad w
    auto tr_lora_in_b =
        lora_module_->getCtxtTensor("tr_lora_in_b_head", 0, 0, 0);
    // just initializatoin.
    CtxtTensor grad_lora_wb{grad_lora_b}; // 10
    for (u64 i = 0; i < 4; ++i) {
        CtxtTensor grad_lora_b_tmp{tmp_grad[i]}; // 7
        for (u64 rot = 1; rot < ModelArgs::LOW_DIM; rot <<= 1) {
            hemmer_->getEval().rightRotate(grad_lora_b_tmp.get(), rot, tmp);
            hemmer_->getEval().add(grad_lora_b_tmp.get(), tmp,
                                   grad_lora_b_tmp.get());
        }
        hemmer_->hadamardMultInplace(grad_lora_b_tmp, tr_lora_in_b); // 6

        if (i != 0) {
            hemmer_->getEval().rightRotate(grad_lora_b_tmp.get(),
                                           i * ModelArgs::LOW_DIM,
                                           grad_lora_b_tmp.get());
            hemmer_->addInplace(grad_lora_wb, grad_lora_b_tmp);
        } else {
            grad_lora_wb = grad_lora_b_tmp; // 6
        }
    }

    // repeated packing for grad @ Wb^T
    for (u64 rot = 1; rot < 128; rot <<= 1) {
        hemmer_->getEval().rightRotate(grad_lora_a_tmp.get(), rot * 256, tmp);
        hemmer_->getEval().add(grad_lora_a_tmp.get(), tmp,
                               grad_lora_a_tmp.get()); // 12
    }

    hemmer_->getEval().rightRotate(grad_lora_a_tmp.get(), 128, tmp);
    hemmer_->getEval().add(grad_lora_a_tmp.get(), tmp, grad_lora_a_tmp.get());

    // TODO: checking weight level
    auto grad_lora_wa =
        lora_module_->getCtxtTensor("tr_lora_in_a_head", 0, 0, 0);
    for (u64 rot = 1; rot < ModelArgs::LOW_DIM; rot <<= 1) {
        hemmer_->getEval().rightRotate(grad_lora_wa.get(), rot, tmp);
        hemmer_->getEval().add(grad_lora_wa.get(), tmp, grad_lora_wa.get());
    }
    hemmer_->hadamardMultInplace(grad_lora_wa, grad_lora_a_tmp); // 11

    // repeated packing
    for (u64 i = 1; i < 3; ++i) {
        auto grad_lora_wa_tmp =
            lora_module_->getCtxtTensor("tr_lora_in_a_head", 0, 0, i);
        for (u64 rot = 1; rot < ModelArgs::LOW_DIM; rot <<= 1) {
            hemmer_->getEval().rightRotate(grad_lora_wa_tmp.get(), rot, tmp);
            hemmer_->getEval().add(grad_lora_wa_tmp.get(), tmp,
                                   grad_lora_wa_tmp.get());
        }
        hemmer_->hadamardMultInplace(grad_lora_wa_tmp, grad_lora_a_tmp); // 11
        hemmer_->getEval().rightRotate(grad_lora_wa_tmp.get(),
                                       i * ModelArgs::LOW_DIM,
                                       grad_lora_wa_tmp.get());

        hemmer_->addInplace(grad_lora_wa, grad_lora_wa_tmp);
    }

    hemmer_->transposeInplace(grad_lora_wa); // 10

    if (hemmer_->getMaxRank() > 1) {
        allReduceWrapper(grad_lora_wb);
        allReduceWrapper(grad_lora_wa);

        if (hemmer_->getRank() == 0) {
            hemmer_->getEval().modReduct(grad_lora_wb.get());
            hemmer_->getEval().modReduct(grad_lora_wa.get());

            lora_module_->saveAggGrad(grad_lora_wb, "b_head");
            lora_module_->saveAggGrad(grad_lora_wa, "a_head");
        }
    }

    for (u64 i = 0; i < 4; ++i) {
        for (i64 rot = 1; rot < 128; rot <<= 1) {
            hemmer_->getEval().rightRotate(tmp_grad[i].get(),
                                           static_cast<u64>(rot), tmp);
            hemmer_->getEval().add(tmp_grad[i].get(), tmp, tmp_grad[i].get());
        }
    }
    // std::cout << "after rot&sum " << std::endl;
    // printing(tmp_grad);

    // grad @ W_dout^T @ tanh'(input) @ W_final_in^T
    // 7 -> 5
    std::vector<CtxtTensor> tmp_grad2;
    tmp_grad2.reserve(6);
    for (u64 i = 0; i < 6; ++i) {
        CtxtTensor tmp_{hemmer_->getContext(), tmp_grad[0].getShape()};
        Ciphertext tmp_rot(hemmer_->getContext());
        for (u64 j = 0; j < 4; ++j) {
            // just original packing >> consider a complex packing.
            CtxtTensor res{hemmer_->getContext(), tmp_.getShape()};
            // TODO: should fix a method of saving weight.
            auto weight = getWeightMsg("tr_wfinal1", j, i); // j = row, i = col
            hemmer_->getEval().mult(tmp_grad[j].get(), weight, res.get());
            if (j == 0) {
                tmp_ = res;
            } else {
                hemmer_->addInplace(tmp_, res);
            }
        }

        if (i % 2 == 0) {
            hemmer_->getEval().leftRotate(tmp_.get(), 128, tmp_rot);
            hemmer_->getEval().add(tmp_.get(), tmp_rot, tmp_.get());
            hemmer_->maskLeftHalfInplace(tmp_);
        } else {
            hemmer_->getEval().rightRotate(tmp_.get(), 128, tmp_rot);
            hemmer_->getEval().add(tmp_.get(), tmp_rot, tmp_.get());
            hemmer_->maskRightHalfInplace(tmp_);
        }
        tmp_grad2.push_back(tmp_);
    }
    tmp_grad.clear();

    std::vector<CtxtTensor> output;
    output.reserve(3);
    // repacking
    // 4 -> 3
    for (u64 i = 0; i < 3; ++i) {
        hemmer_->addInplace(tmp_grad2[i * 2], tmp_grad2[i * 2 + 1]);

        // grad_lora_b split and addition
        CtxtTensor lora_tmp{grad_lora_b};
        if (i != 0) {
            hemmer_->getEval().leftRotate(
                lora_tmp.get(), i * ModelArgs::LOW_DIM * 256, lora_tmp.get());
        }
        hemmer_->maskFirstRowInplace(lora_tmp);
        // hemmer_->addInplace(tmp_grad2[i*2], lora_tmp);
        output.push_back(tmp_grad2[i * 2]);

        for (u64 rot = 1; rot < 128; rot <<= 1) {
            hemmer_->getEval().leftRotate(output[i].get(), 256 * rot, tmp);
            hemmer_->getEval().add(output[i].get(), tmp, output[i].get());
        }
    }
    // std::cout << "wfinal1 grad after rot&sum , level = " <<
    // output[0].get().getLevel() << std::endl; std::cout << "pooling backward
    // after weight mm " << std::endl; printing(output);

    /* hemmer_->bootstrap2(output[0], output[1]);
    hemmer_->bootstrap(output[2]); */

    // LN for input gradient.
    // 12 -> 11
    for (u64 i = 0; i < 3; ++i) {
        const auto &weight = getWeightMsg("final_norm_w", i);
        // const auto &bias = getWeightMsg("final_norm_b", i);
        hemmer_->hadamardMultInplace(output[i], weight);
        // hemmer_->getEval().add(output[i].get(), bias, output[i].get());
    }
    // std::cout << "after mutl weight and sum bias" << std::endl;
    // printing(output);

    hemmer_->bootstrap2(output[0], output[1]);
    hemmer_->bootstrap(output[2]);

    // 11 -> 4
    auto cur = hemmer_->backwardLayerNorm(output, "final", layer_n_);
    output.clear();

    hemmer_->bootstrap2(cur[0], cur[1]);
    hemmer_->bootstrap(cur[2]);

    for (u64 i = 0; i < 3; ++i) {
        hemmer_->maskFirstRowInplace(cur[i]);
    }
    /* std::cout << "pooling backward after LN " << std::endl;
    printing(cur); */

    // save output tensor
    /* for (u64 i = 0 ; i < 3 ; ++i) {
        cur[i].get().save("./inputs/backward_input_"+std::to_string(i) +".bin");
    } */

    return cur;
}

// in: 1 x 2 , out: 128 x 768
// in: 9 level -> 12
std::vector<CtxtTensor>
TransformerBlock::backwardpooling3_bert(CtxtTensor &grad_y) {

    // std::cout << "input level " << grad_y[0].get().getLevel() << std::endl;
    //  grad @ W_dout^T
    CtxtTensor grad_head{grad_y};
    // input gradient
    auto weight = lora_module_->getCtxtTensor_lora("wfinal2_weight", 0, 0, 0);
    hemmer_->hadamardMultInplace(grad_head, weight); // 11

    // collecting
    for (u64 i = 0; i < 1; ++i) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().leftRotate(grad_head.get(), 256, tmp);
        hemmer_->getEval().add(grad_head.get(), tmp, grad_head.get());
    }

    // weight gradient
    auto tr_tanh_res =
        lora_module_->getCtxtTensor("tr_pooling_res_tanh", 0, 0, 0);
    CtxtTensor head_grad{grad_y};
    // weight update: combine with other wieghts. (insert after wa gradient)
    hemmer_->hadamardMultInplace(head_grad, tr_tanh_res); // 10

    // Originally, we need to masking.
    // But, becuase we are going through backward tanh, masking comes naturally.

    // grad @ W_dout^T @ tanh'(input)
    // 10
    CtxtTensor tmp_grad{grad_head};
    hemmer_->backwardtanhInplace(tmp_grad, layer_n_); // 9
    // std::cout << "backward tanh " << tmp_grad.get().getLevel() << std::endl;

    // repeated packing
    for (u64 rot = 1; rot < ModelArgs::LOW_DIM_HEAD; rot <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().rightRotate(tmp_grad.get(), rot * 256, tmp);
        hemmer_->getEval().add(tmp_grad.get(), tmp, tmp_grad.get());
    }

    // weight b gradient
    CtxtTensor wb_grad{tmp_grad};
    auto tr_res_wa = lora_module_->getCtxtTensor("tr_pooling_res_wa", 0, 0, 0);
    hemmer_->hadamardMultInplace(wb_grad, tr_res_wa); // 8
    // std::cout << "tr_res_wa " << tr_res_wa.get().getLevel() << std::endl;

    // Wb input gradient
    // We don't need to transpose.
    weight = lora_module_->getCtxtTensor_lora("wfinal1_weight_b", 0, 0, 0);
    hemmer_->hadamardMultInplace(tmp_grad, weight); // 8
    // std::cout << "weight mm " << tmp_grad.get().getLevel() << std::endl;

    // collecting results
    for (u64 i = 1; i < 4; i <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        if (i == 1) {
            hemmer_->getEval().leftRotate(tmp_grad.get(), 128, tmp);
            hemmer_->getEval().add(tmp_grad.get(), tmp, tmp_grad.get());

            for (u64 rot = 1; rot < 128; rot <<= 1) {
                hemmer_->getEval().leftRotate(tmp_grad.get(), rot, tmp);
                hemmer_->getEval().add(tmp_grad.get(), tmp, tmp_grad.get());
            }
        }
        hemmer_->getEval().leftRotate(tmp_grad.get(),
                                      i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
        hemmer_->getEval().add(tmp_grad.get(), tmp, tmp_grad.get());
    }

    hemmer_->maskFirstColPoolingInplace(tmp_grad); // 7

    // repeated packing
    for (u64 rot = 1; rot < 128; rot <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        if (rot == 1) {
            hemmer_->getEval().rightRotate(tmp_grad.get(), 128, tmp);
            hemmer_->getEval().add(tmp_grad.get(), tmp, tmp_grad.get());

            for (u64 i = 1; i < 3; i <<= 1) {
                hemmer_->getEval().rightRotate(
                    tmp_grad.get(), i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
                hemmer_->getEval().add(tmp_grad.get(), tmp, tmp_grad.get());
            }
        }
        hemmer_->getEval().rightRotate(tmp_grad.get(), rot, tmp);
        hemmer_->getEval().add(tmp_grad.get(), tmp, tmp_grad.get());
    }

    // Wa gradient
    // weight gradient
    CtxtTensor wa_grad{tmp_grad};
    auto tr_res_ln = lora_module_->getCtxtTensor("tr_pooling_res_ln", 0, 0, 0);
    hemmer_->hadamardMultInplace(wa_grad, tr_res_ln); // 6

    HEaaN::CudaTools::cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    if (hemmer_->getMaxRank() > 1) {
        allReduceWrapper(head_grad);
        allReduceWrapper(wb_grad);
        allReduceWrapper(wa_grad);

        if (hemmer_->getRank() == 0) {
            hemmer_->getEval().modReduct(head_grad.get());
            hemmer_->getEval().modReduct(wb_grad.get());
            hemmer_->getEval().modReduct(wa_grad.get());

            hemmer_->bootstrap2(wa_grad, wb_grad);
            hemmer_->bootstrap(head_grad);

            // optimizer step
            hemmer_->getEval().mult(head_grad.get(),
                                    1.0 / ModelArgs::BATCH_SIZE,
                                    head_grad.get()); // 10
            hemmer_->getEval().mult(wb_grad.get(), 1.0 / ModelArgs::BATCH_SIZE,
                                    wb_grad.get()); // 7
            hemmer_->getEval().mult(wa_grad.get(), 1.0 / ModelArgs::BATCH_SIZE,
                                    wa_grad.get()); // 5

            lora_module_->saveAggGrad(head_grad, "_head");
            lora_module_->saveAggGrad(wb_grad, "b_head");
            lora_module_->saveAggGrad(wa_grad, "a_head");
        }
    }

    // input gardient
    weight = lora_module_->getCtxtTensor_lora("wfinal1_weight_a", 0, 0, 0);
    hemmer_->hadamardMultInplace(tmp_grad, weight); // 6

    // collecting results
    for (u64 rot = 1; rot < ModelArgs::LOW_DIM_HEAD; rot <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().leftRotate(tmp_grad.get(), rot * 256, tmp);
        hemmer_->getEval().add(tmp_grad.get(), tmp, tmp_grad.get());
    }

    std::vector<CtxtTensor> cur;
    cur.reserve(3);
    for (u64 i = 0; i < 3; ++i) {
        cur.push_back(tmp_grad);
        if (i != 0) {
            hemmer_->getEval().leftRotate(
                cur[i].get(), i * ModelArgs::LOW_DIM_HEAD * 256, cur[i].get());
        }
        // hemmer_->maskFirstRowInplace(cur[i]);
        const auto &weight_ = getWeightMsg("final_norm_w", i);
        hemmer_->hadamardMultInplace(cur[i], weight_); // 5
    }

    // std::cout << "here cur level " << cur[0].get().getLevel() << std::endl;
    hemmer_->bootstrap2(cur[0], cur[1]);
    hemmer_->bootstrap(cur[2]);

    cur = hemmer_->backwardLayerNorm(cur, "final", layer_n_);

    hemmer_->bootstrap2(cur[0], cur[1]);
    hemmer_->bootstrap(cur[2]);

    for (u64 i = 0; i < 3; ++i) {
        hemmer_->maskFirstRowInplace(cur[i]);
    }

    return cur;
}

// in: 1 x 2 , out: 128 x 768
// in: 9 level -> 12
std::vector<CtxtTensor>
TransformerBlock::backwardpooling3_bert_stsb(CtxtTensor &grad_y) {

    // std::cout << "input level " << grad_y[0].get().getLevel() << std::endl;
    //  grad @ W_dout^T
    CtxtTensor grad_head{grad_y};
    // input gradient
    auto weight = lora_module_->getCtxtTensor_lora("wfinal2_weight", 0, 0, 0);
    hemmer_->hadamardMultInplace(grad_head, weight); // 11

    // weight gradient
    auto tr_tanh_res =
        lora_module_->getCtxtTensor("tr_pooling_res_tanh", 0, 0, 0);
    CtxtTensor head_grad{grad_y};
    // weight update: combine with other wieghts. (insert after wa gradient)
    hemmer_->hadamardMultInplace(head_grad, tr_tanh_res); // 10

    // Originally, we need to do masking.
    // But, becuase we will go through backward tanh, masking comes naturally.

    // grad @ W_dout^T @ tanh'(input)
    // 10
    CtxtTensor tmp_grad{grad_head};
    hemmer_->backwardtanhInplace(tmp_grad, layer_n_); // 9
    // std::cout << "backward tanh " << tmp_grad.get().getLevel() << std::endl;

    // repeated packing
    for (u64 rot = 1; rot < ModelArgs::LOW_DIM_HEAD; rot <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().rightRotate(tmp_grad.get(), rot * 256, tmp);
        hemmer_->getEval().add(tmp_grad.get(), tmp, tmp_grad.get());
    }

    // weight b gradient
    CtxtTensor wb_grad{tmp_grad};
    auto tr_res_wa = lora_module_->getCtxtTensor("tr_pooling_res_wa", 0, 0, 0);
    hemmer_->hadamardMultInplace(wb_grad, tr_res_wa); // 8
    // std::cout << "tr_res_wa " << tr_res_wa.get().getLevel() << std::endl;

    // Wb input gradient
    // We don't need to transpose.
    weight = lora_module_->getCtxtTensor_lora("wfinal1_weight_b", 0, 0, 0);
    hemmer_->hadamardMultInplace(tmp_grad, weight); // 8
    // std::cout << "weight mm " << tmp_grad.get().getLevel() << std::endl;

    // collecting results
    for (u64 i = 1; i < 4; i <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        if (i == 1) {
            hemmer_->getEval().leftRotate(tmp_grad.get(), 128, tmp);
            hemmer_->getEval().add(tmp_grad.get(), tmp, tmp_grad.get());

            for (u64 rot = 1; rot < 128; rot <<= 1) {
                hemmer_->getEval().leftRotate(tmp_grad.get(), rot, tmp);
                hemmer_->getEval().add(tmp_grad.get(), tmp, tmp_grad.get());
            }
        }
        hemmer_->getEval().leftRotate(tmp_grad.get(),
                                      i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
        hemmer_->getEval().add(tmp_grad.get(), tmp, tmp_grad.get());
    }

    hemmer_->maskFirstColPoolingInplace(tmp_grad); // 7

    // repeated packing
    for (u64 rot = 1; rot < 128; rot <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        if (rot == 1) {
            hemmer_->getEval().rightRotate(tmp_grad.get(), 128, tmp);
            hemmer_->getEval().add(tmp_grad.get(), tmp, tmp_grad.get());

            for (u64 i = 1; i < 3; i <<= 1) {
                hemmer_->getEval().rightRotate(
                    tmp_grad.get(), i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
                hemmer_->getEval().add(tmp_grad.get(), tmp, tmp_grad.get());
            }
        }
        hemmer_->getEval().rightRotate(tmp_grad.get(), rot, tmp);
        hemmer_->getEval().add(tmp_grad.get(), tmp, tmp_grad.get());
    }

    // Wa gradient
    // weight gradient
    CtxtTensor wa_grad{tmp_grad};
    auto tr_res_ln = lora_module_->getCtxtTensor("tr_pooling_res_ln", 0, 0, 0);
    hemmer_->hadamardMultInplace(wa_grad, tr_res_ln); // 6

    HEaaN::CudaTools::cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    if (hemmer_->getMaxRank() > 1) {
        allReduceWrapper(head_grad);
        allReduceWrapper(wb_grad);
        allReduceWrapper(wa_grad);

        if (hemmer_->getRank() == 0) {
            hemmer_->getEval().modReduct(head_grad.get());
            hemmer_->getEval().modReduct(wb_grad.get());
            hemmer_->getEval().modReduct(wa_grad.get());

            hemmer_->bootstrap2(wa_grad, wb_grad);
            hemmer_->bootstrap(head_grad);

            // optimizer step
            hemmer_->getEval().mult(head_grad.get(),
                                    1.0 / ModelArgs::BATCH_SIZE,
                                    head_grad.get()); // 10
            hemmer_->getEval().mult(wb_grad.get(), 1.0 / ModelArgs::BATCH_SIZE,
                                    wb_grad.get()); // 7
            hemmer_->getEval().mult(wa_grad.get(), 1.0 / ModelArgs::BATCH_SIZE,
                                    wa_grad.get()); // 5

            lora_module_->saveAggGrad(head_grad, "_head");
            lora_module_->saveAggGrad(wb_grad, "b_head");
            lora_module_->saveAggGrad(wa_grad, "a_head");
        }
    }

    // input gardient
    weight = lora_module_->getCtxtTensor_lora("wfinal1_weight_a", 0, 0, 0);
    hemmer_->hadamardMultInplace(tmp_grad, weight); // 6

    // collecting results
    for (u64 rot = 1; rot < ModelArgs::LOW_DIM_HEAD; rot <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().leftRotate(tmp_grad.get(), rot * 256, tmp);
        hemmer_->getEval().add(tmp_grad.get(), tmp, tmp_grad.get());
    }

    std::vector<CtxtTensor> cur;
    cur.reserve(3);
    for (u64 i = 0; i < 3; ++i) {
        cur.push_back(tmp_grad);
        if (i != 0) {
            hemmer_->getEval().leftRotate(
                cur[i].get(), i * ModelArgs::LOW_DIM_HEAD * 256, cur[i].get());
        }
        // hemmer_->maskFirstRowInplace(cur[i]);
        const auto &weight_ = getWeightMsg("final_norm_w", i);
        hemmer_->hadamardMultInplace(cur[i], weight_); // 5
    }

    // std::cout << "here cur level " << cur[0].get().getLevel() << std::endl;
    hemmer_->bootstrap2(cur[0], cur[1]);
    hemmer_->bootstrap(cur[2]);

    cur = hemmer_->backwardLayerNorm(cur, "final", layer_n_);

    hemmer_->bootstrap2(cur[0], cur[1]);
    hemmer_->bootstrap(cur[2]);

    for (u64 i = 0; i < 3; ++i) {
        hemmer_->maskFirstRowInplace(cur[i]);
    }

    return cur;
}

// in: 1 x 2 , out: 128 x 768
// in: 9 level -> 12
std::vector<CtxtTensor>
TransformerBlock::backwardpooling3_bert_time(CtxtTensor &grad_y) {

    // grad @ W_dout^T
    CtxtTensor grad_head{grad_y};
    // input gradient
    auto weight = lora_module_->getCtxtTensor_lora("wfinal2_weight", 0, 0, 0);
    hemmer_->hadamardMultInplace(grad_head, weight); // 11

    // collecting
    for (u64 i = 0; i < 1; ++i) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().leftRotate(grad_head.get(), 256, tmp);
        hemmer_->getEval().add(grad_head.get(), tmp, grad_head.get());
    }
    printElapsedTime("wfinal2 mm");

    // weight gradient
    auto tr_tanh_res =
        lora_module_->getCtxtTensor("tr_pooling_res_tanh", 0, 0, 0);
    CtxtTensor head_grad{grad_y};
    hemmer_->hadamardMultInplace(head_grad, tr_tanh_res); // 10
    printElapsedTime("weight head grad");

    // grad @ W_dout^T @ tanh'(input)
    // 10
    CtxtTensor tmp_grad{grad_head};
    hemmer_->backwardtanhInplace(tmp_grad, layer_n_); // 9
    // std::cout << "backward tanh " << tmp_grad.get().getLevel() << std::endl;

    // repeated packing
    for (u64 rot = 1; rot < ModelArgs::LOW_DIM_HEAD; rot <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().rightRotate(tmp_grad.get(), rot * 256, tmp);
        hemmer_->getEval().add(tmp_grad.get(), tmp, tmp_grad.get());
    }

    // weight b gradient
    CtxtTensor wb_grad{tmp_grad};
    auto tr_res_wa = lora_module_->getCtxtTensor("tr_pooling_res_wa", 0, 0, 0);
    hemmer_->hadamardMultInplace(wb_grad, tr_res_wa); // 8
    // std::cout << "tr_res_wa " << tr_res_wa.get().getLevel() << std::endl;
    printElapsedTime("weight b gradient");

    // Wb input gradient
    // We don't need to transpose.
    weight = lora_module_->getCtxtTensor_lora("wfinal1_weight_b", 0, 0, 0);
    hemmer_->hadamardMultInplace(tmp_grad, weight); // 8
    // std::cout << "weight mm " << tmp_grad.get().getLevel() << std::endl;

    // collecting results
    for (u64 i = 1; i < 4; i <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        if (i == 1) {
            hemmer_->getEval().leftRotate(tmp_grad.get(), 128, tmp);
            hemmer_->getEval().add(tmp_grad.get(), tmp, tmp_grad.get());

            for (u64 rot = 1; rot < 128; rot <<= 1) {
                hemmer_->getEval().leftRotate(tmp_grad.get(), rot, tmp);
                hemmer_->getEval().add(tmp_grad.get(), tmp, tmp_grad.get());
            }
        }
        hemmer_->getEval().leftRotate(tmp_grad.get(),
                                      i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
        hemmer_->getEval().add(tmp_grad.get(), tmp, tmp_grad.get());
    }
    printElapsedTime("winfal1_b mm");

    hemmer_->maskFirstColPoolingInplace(tmp_grad); // 7

    // repeated packing
    for (u64 rot = 1; rot < 128; rot <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        if (rot == 1) {
            hemmer_->getEval().rightRotate(tmp_grad.get(), 128, tmp);
            hemmer_->getEval().add(tmp_grad.get(), tmp, tmp_grad.get());

            for (u64 i = 1; i < 3; i <<= 1) {
                hemmer_->getEval().rightRotate(
                    tmp_grad.get(), i * ModelArgs::LOW_DIM_HEAD * 256, tmp);
                hemmer_->getEval().add(tmp_grad.get(), tmp, tmp_grad.get());
            }
        }
        hemmer_->getEval().rightRotate(tmp_grad.get(), rot, tmp);
        hemmer_->getEval().add(tmp_grad.get(), tmp, tmp_grad.get());
    }

    // Wa gradient
    // weight gradient
    CtxtTensor wa_grad{tmp_grad};
    auto tr_res_ln = lora_module_->getCtxtTensor("tr_pooling_res_ln", 0, 0, 0);
    hemmer_->hadamardMultInplace(wa_grad, tr_res_ln); // 6
    printElapsedTime("wa_grad mm");

    HEaaN::CudaTools::cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    if (hemmer_->getMaxRank() > 1) {
        allReduceWrapper(head_grad);
        allReduceWrapper(wb_grad);
        allReduceWrapper(wa_grad);

        if (hemmer_->getRank() == 0) {
            hemmer_->getEval().modReduct(head_grad.get());
            hemmer_->getEval().modReduct(wb_grad.get());
            hemmer_->getEval().modReduct(wa_grad.get());

            // std::cout << "In backwardpooling3 grad" << std::endl;
            // printing_masking(head_grad);

            // TODO: optimize BTS position.
            // std::cout << "here level " << wa_grad.get().getLevel() <<
            // std::endl; std::cout << "here level " << wb_grad.get().getLevel()
            // << std::endl; std::cout << "here level " <<
            // head_grad.get().getLevel() << std::endl;
            hemmer_->bootstrap2(wa_grad, wb_grad);
            hemmer_->bootstrap(head_grad);

            // optimizer step
            hemmer_->getEval().mult(head_grad.get(),
                                    1.0 / ModelArgs::BATCH_SIZE,
                                    head_grad.get()); // 10
            hemmer_->getEval().mult(wb_grad.get(), 1.0 / ModelArgs::BATCH_SIZE,
                                    wb_grad.get()); // 7
            hemmer_->getEval().mult(wa_grad.get(), 1.0 / ModelArgs::BATCH_SIZE,
                                    wa_grad.get()); // 5

            lora_module_->saveAggGrad(head_grad, "_head");
            lora_module_->saveAggGrad(wb_grad, "b_head");
            lora_module_->saveAggGrad(wa_grad, "a_head");
        }
    }
    printElapsedTime("weight save");

    // input gardient
    weight = lora_module_->getCtxtTensor_lora("wfinal1_weight_a", 0, 0, 0);
    hemmer_->hadamardMultInplace(tmp_grad, weight); // 6

    // collecting results
    for (u64 rot = 1; rot < ModelArgs::LOW_DIM_HEAD; rot <<= 1) {
        Ciphertext tmp(hemmer_->getContext());
        hemmer_->getEval().leftRotate(tmp_grad.get(), rot * 256, tmp);
        hemmer_->getEval().add(tmp_grad.get(), tmp, tmp_grad.get());
    }
    printElapsedTime("(CC) wfinal1_a mm");

    std::vector<CtxtTensor> cur;
    cur.reserve(3);
    for (u64 i = 0; i < 3; ++i) {
        cur.push_back(tmp_grad);
        if (i != 0) {
            hemmer_->getEval().leftRotate(
                cur[i].get(), i * ModelArgs::LOW_DIM_HEAD * 256, cur[i].get());
        }
        // hemmer_->maskFirstRowInplace(cur[i]);
        const auto &weight_ = getWeightMsg("final_norm_w", i);
        hemmer_->hadamardMultInplace(cur[i], weight_); // 5
    }
    printElapsedTime("LN weight hadamult");

    // std::cout << "here cur level " << cur[0].get().getLevel() << std::endl;
    hemmer_->bootstrap2(cur[0], cur[1]);
    hemmer_->bootstrap(cur[2]);
    printElapsedTime("BTS");

    cur = hemmer_->backwardLayerNorm(cur, "final", layer_n_);
    printElapsedTime("back tanh");

    hemmer_->bootstrap2(cur[0], cur[1]);
    hemmer_->bootstrap(cur[2]);
    printElapsedTime("BTS");

    for (u64 i = 0; i < 3; ++i) {
        hemmer_->maskFirstRowInplace(cur[i]);
    }
    printElapsedTime("masking");

    return cur;
}

// in: 128 x 768, out: 128 x 768
void TransformerBlock::backwardfeedForward2_bert(
    std::vector<CtxtTensor> &grad_y) {

    const auto rank = hemmer_->getRank();

    // std::cout << "backward ffn input = " << std::endl;
    // printing(grad_y);
    /* for (u64 i = 0 ; i < 3 ; ++i) {
        hemmer_->backwarddropoutInplace(grad_y[i], "ffn_res",layer_n_, i);
    }
 */

    start = std::chrono::high_resolution_clock::now();

    std::vector<CtxtTensor> tmp_grad;
    tmp_grad.reserve(12);
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(hemmer_->complexPacking(grad_y[0], grad_y[1]),
                               tmp);
        } else {
            hemmer_->matMulPre(grad_y[2], tmp);
        }

        for (u64 j = 0; j < 12; ++j) {
            auto weight = getWeight("tr_wdout", i, j);
            auto result = hemmer_->matMulReUse(tmp, weight);

            if (i == 0) {
                tmp_grad.push_back(result);
            } else {
                hemmer_->addInplace(tmp_grad[j], result);
            }
        }
    }

    std::vector<CtxtTensor> grad_wdout;
    grad_wdout.reserve(6);
    for (u64 i = 0; i < 6; ++i) {
        auto tmp = hemmer_->repack(tmp_grad[i * 2], tmp_grad[i * 2 + 1]);
        grad_wdout.push_back(tmp);
    }
    tmp_grad.clear();
    // std::cout << "backward ffn dense_out mm " << std::endl;
    // printing(grad_wdout);

    std::vector<CtxtTensor> tmp_grad_relu = grad_wdout;
    std::vector<CtxtTensor> tmp_grad_id = grad_wdout;
    // 11 -> 10
    for (u64 i = 0; i < tmp_grad_id.size(); ++i) {
        Ciphertext tmp(hemmer_->getContext());
        tmp.load(hemmer_->getHEPath() + "/relu_forward_" +
                 std::to_string(rank) + "_" + std::to_string(layer_n_) + "_" +
                 std::to_string(i) + ".bin");
        tmp.to(getCurrentCudaDevice());
        hemmer_->getEval().mult(tmp_grad_id[i].get(), tmp,
                                tmp_grad_id[i].get());
    }

    for (u64 i = 0; i < tmp_grad_relu.size(); ++i) {
        Ciphertext tmp(hemmer_->getContext());
        tmp.load(hemmer_->getHEPath() + "/id_forward_" + std::to_string(rank) +
                 "_" + std::to_string(layer_n_) + "_" + std::to_string(i) +
                 ".bin");
        tmp.to(getCurrentCudaDevice());
        hemmer_->getEval().mult(tmp_grad_relu[i].get(), tmp,
                                tmp_grad_relu[i].get());
    }

    // std::cout << "backward ffn relu input " << std::endl;
    // printing(tmp_grad_relu);

    // 10 -> 9
    hemmer_->backwardreluVectorInplace(tmp_grad_relu, layer_n_);
    // std::cout << "backward ffn relu output " << std::endl;
    // printing(tmp_grad_relu);

    std::vector<CtxtTensor> tmp_wdin;
    tmp_wdin.reserve(6);
    // input gradient
    for (u64 i = 0; i < 6; ++i) {
        std::vector<Ciphertext> tmp;
        if (i < 3) {
            hemmer_->matMulPre(hemmer_->complexPacking(tmp_grad_id[i * 2],
                                                       tmp_grad_id[i * 2 + 1]),
                               tmp);
        } else {
            hemmer_->matMulPre(
                hemmer_->complexPacking(tmp_grad_relu[(i - 3) * 2],
                                        tmp_grad_relu[(i - 3) * 2 + 1]),
                tmp);
        }

        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("tr_wdin", i, j);
            auto tmp_res = hemmer_->matMulReUse(tmp, weight);

            if (i == 0) {
                tmp_wdin.push_back(tmp_res);
            } else {
                hemmer_->addInplace(tmp_wdin[i], tmp_res);
            }
        }
    }

    std::vector<CtxtTensor> output;
    output.reserve(3);
    // output: 11
    for (u64 i = 0; i < 3; ++i) {
        auto tmp = hemmer_->repack(tmp_wdin[i * 2], tmp_wdin[i * 2 + 1]);
        output.push_back(tmp);
    }
    tmp_wdin.clear();
    // std::cout << "backward ffn w_in mm" << std::endl;
    // printing(output);

    for (u64 i = 0; i < 3; ++i) {
        const auto &weight = getWeightMsg("norm2_w", i);
        hemmer_->hadamardMultInplace(output[i], weight);
    }

    // input 11 -> 4
    output = hemmer_->backwardLayerNorm(output, "ffn", layer_n_);
    // std::cout << "backward ffn after LN " << std::endl;
    // printing(output);

    for (u64 i = 0; i < 3; ++i) {
        hemmer_->addInplace(grad_y[i], output[i]);
    }
    // std::cout << "backward ffn output level = " << grad_y[0].get().getLevel()
    // << std::endl;

    // TODO: checking a BTS position.
    hemmer_->bootstrap2(grad_y[0], grad_y[1]);
    hemmer_->bootstrap(grad_y[2]);

    /* for (u64 i = 0 ; i < 3 ; ++i) {
        hemmer_->maskFirstRowInplace(grad_y[i]);
    } */

    // std::cout << "backward ffn after residual" << std::endl;
    // printing(grad_y);
}

// in: 128 x 768, out: 128 x 768
void TransformerBlock::backwardfeedForward2_bert_time(
    std::vector<CtxtTensor> &grad_y) {

    const auto rank = hemmer_->getRank();

    start = std::chrono::high_resolution_clock::now();

    for (u64 i = 0; i < 3; ++i) {
        hemmer_->backwarddropoutInplace(grad_y[i], "ffn_res", layer_n_, i);
    }
    printElapsedTime("back_droupout");

    std::vector<CtxtTensor> tmp_grad;
    tmp_grad.reserve(12);
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(hemmer_->complexPacking(grad_y[0], grad_y[1]),
                               tmp);
        } else {
            hemmer_->matMulPre(grad_y[2], tmp);
        }

        for (u64 j = 0; j < 12; ++j) {
            auto weight = getWeight("tr_wdout", i, j);
            auto result = hemmer_->matMulReUse(tmp, weight);

            if (i == 0) {
                tmp_grad.push_back(result);
            } else {
                hemmer_->addInplace(tmp_grad[j], result);
            }
        }
    }
    printElapsedTime("(PC) mm");

    std::vector<CtxtTensor> grad_wdout;
    grad_wdout.reserve(6);
    for (u64 i = 0; i < 6; ++i) {
        auto tmp = hemmer_->repack(tmp_grad[i * 2], tmp_grad[i * 2 + 1]);
        grad_wdout.push_back(tmp);
    }
    tmp_grad.clear();
    printElapsedTime("repack");

    std::vector<CtxtTensor> tmp_grad_relu = grad_wdout;
    std::vector<CtxtTensor> tmp_grad_id = grad_wdout;
    // 11 -> 10
    for (u64 i = 0; i < tmp_grad_id.size(); ++i) {
        Ciphertext tmp(hemmer_->getContext());
        tmp.load(hemmer_->getHEPath() + "/relu_forward_" +
                 std::to_string(rank) + "_" + std::to_string(layer_n_) + "_" +
                 std::to_string(i) + ".bin");
        tmp.to(getCurrentCudaDevice());
        hemmer_->getEval().mult(tmp_grad_id[i].get(), tmp,
                                tmp_grad_id[i].get());
    }
    printElapsedTime("load and mult");

    for (u64 i = 0; i < tmp_grad_relu.size(); ++i) {
        Ciphertext tmp(hemmer_->getContext());
        tmp.load(hemmer_->getHEPath() + "/id_forward_" + std::to_string(rank) +
                 "_" + std::to_string(layer_n_) + "_" + std::to_string(i) +
                 ".bin");
        tmp.to(getCurrentCudaDevice());
        hemmer_->getEval().mult(tmp_grad_relu[i].get(), tmp,
                                tmp_grad_relu[i].get());
    }
    printElapsedTime("load and mult");

    // 10 -> 9
    hemmer_->backwardreluVectorInplace(tmp_grad_relu, layer_n_);
    printElapsedTime("back relu");

    std::vector<CtxtTensor> tmp_wdin;
    tmp_wdin.reserve(6);
    // input gradient
    for (u64 i = 0; i < 6; ++i) {
        std::vector<Ciphertext> tmp;
        if (i < 3) {
            hemmer_->matMulPre(hemmer_->complexPacking(tmp_grad_id[i * 2],
                                                       tmp_grad_id[i * 2 + 1]),
                               tmp);
        } else {
            hemmer_->matMulPre(
                hemmer_->complexPacking(tmp_grad_relu[(i - 3) * 2],
                                        tmp_grad_relu[(i - 3) * 2 + 1]),
                tmp);
        }

        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("tr_wdin", i, j);
            auto tmp_res = hemmer_->matMulReUse(tmp, weight);

            if (i == 0) {
                tmp_wdin.push_back(tmp_res);
            } else {
                hemmer_->addInplace(tmp_wdin[i], tmp_res);
            }
        }
    }
    printElapsedTime("(PC) mm");

    std::vector<CtxtTensor> output;
    output.reserve(3);
    // output: 11
    for (u64 i = 0; i < 3; ++i) {
        auto tmp = hemmer_->repack(tmp_wdin[i * 2], tmp_wdin[i * 2 + 1]);
        output.push_back(tmp);
    }
    tmp_wdin.clear();
    printElapsedTime("repack");

    for (u64 i = 0; i < 3; ++i) {
        const auto &weight = getWeightMsg("norm2_w", i);
        hemmer_->hadamardMultInplace(output[i], weight);
    }
    printElapsedTime("hada");

    // input 11 -> 4
    output = hemmer_->backwardLayerNorm(output, "ffn", layer_n_);
    printElapsedTime("back_LN");

    for (u64 i = 0; i < 3; ++i) {
        hemmer_->addInplace(grad_y[i], output[i]);
    }
    printElapsedTime("add");

    // TODO: checking a BTS position.
    hemmer_->bootstrap2(grad_y[0], grad_y[1]);
    hemmer_->bootstrap(grad_y[2]);
    printElapsedTime("BTS");
}

// in: 12 , out: 5
void TransformerBlock::backwardattention2_bert(std::vector<CtxtTensor> &grad_y,
                                               const std::string &lora_type) {

    auto rank = hemmer_->getRank();
    auto max_rank = hemmer_->getMaxRank();

    start = std::chrono::high_resolution_clock::now();

    std::vector<CtxtTensor> grad_y_tmp;
    grad_y_tmp.reserve(3);
    grad_y_tmp = grad_y;

    for (u64 i = 0; i < 3; ++i) {
        hemmer_->backwarddropoutInplace(grad_y_tmp[i], "atn_output", layer_n_,
                                        i);
    }

    std::vector<CtxtTensor> tmp_grad_;
    tmp_grad_.reserve(6);
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(
                hemmer_->complexPacking(grad_y_tmp[0], grad_y_tmp[1]), tmp);
        } else {
            hemmer_->matMulPre(grad_y_tmp[2], tmp);
        }

        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("tr_wd", i, j);
            auto result = hemmer_->matMulReUse(tmp, weight);

            if (i == 0) {
                tmp_grad_.push_back(result);
            } else {
                hemmer_->addInplace(tmp_grad_[j], result);
            }
        }
    }
    grad_y_tmp.clear();

    std::vector<CtxtTensor> qkv_grad;
    qkv_grad.reserve(3);
    for (u64 i = 0; i < 3; ++i) {
        qkv_grad.push_back(
            hemmer_->repack(tmp_grad_[i * 2], tmp_grad_[i * 2 + 1]));
    }
    tmp_grad_.clear();

    std::vector<CtxtTensor> qkv_grad_split;
    qkv_grad_split.reserve(6);
    // 11 -> 10
    for (u64 i = 0; i < 3; ++i) {
        for (u64 j = 0; j < 2; ++j)
            qkv_grad_split.push_back(qkv_grad[i]);

        hemmer_->maskRightLeft(qkv_grad[i], qkv_grad_split[i * 2],
                               qkv_grad_split[i * 2 + 1]);
    }
    qkv_grad.clear();

    std::vector<CtxtTensor> qkv_grad_split_exp;
    qkv_grad_split_exp = qkv_grad_split;

    // direcition of exp gradient
    // 9 -> 4
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingInplace(qkv_grad_split_exp[i]); // 9
        // should consider saved ctxtTensor structure.
        CtxtTensor tmp =
            lora_module_->getCtxtTensor("forward_res_v", 0, 0, i); // 8
        hemmer_->transposeInplace(tmp);                            // 7
        hemmer_->complexPackingRowInplace(tmp);

        qkv_grad_split_exp[i] =
            hemmer_->packedMatMul(qkv_grad_split_exp[i], tmp);            // 4
        qkv_grad_split_exp[i] = hemmer_->repackCC(qkv_grad_split_exp[i]); // 4
    }

    // std::cout << "attn grad exp direction value mm " << std::endl;
    // printing(qkv_grad_split_exp);
    //  12
    for (u64 i = 0; i < 3; ++i) {
        hemmer_->bootstrap2(qkv_grad_split_exp[i * 2],
                            qkv_grad_split_exp[i * 2 + 1]);
    }

    // backward drop_out = hadamardmult and should follow the same
    // drop_out_masking..
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->backwarddropoutExpInplace(qkv_grad_split_exp[i], "atn_exp",
                                           layer_n_, i); // 12 -> 11
    }

    /* if (layer_n_ == 0) {
        std::cout << "backward exp input " << std::endl;
        printing(qkv_grad_split_exp);
    } */

    // 12 -> 11
    hemmer_->backwardexpVectorInplace(qkv_grad_split_exp, layer_n_);

    /* if (layer_n_ == 0) {
        std::cout << "backward exp output" << std::endl;
        printing(qkv_grad_split_exp);
    } */

    // 11 -> 10
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->getEval().mult(qkv_grad_split_exp[i].get(), -1.0 / 16,
                                qkv_grad_split_exp[i].get());
    }

    std::vector<CtxtTensor> grad_q_tmp;
    std::vector<CtxtTensor> grad_k_tmp;
    grad_q_tmp = qkv_grad_split_exp; // 10
    grad_k_tmp = qkv_grad_split_exp;
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->transposeInplace(grad_k_tmp[i]); // 10 -> 9 (6 -> 5)
    }

    std::vector<CtxtTensor> grad_exp_q;
    std::vector<CtxtTensor> grad_exp_k;
    grad_exp_k = qkv_grad_split_exp; // 10
    grad_exp_q = grad_k_tmp;         // 9
    qkv_grad_split_exp.clear();

    // grad_q = grad@I^t x 2res_q + grad@res_k
    // grad@I^t x 2res_q part
    for (u64 i = 0; i < 3; ++i) {
        hemmer_->tr_oneMatRotSumInplace(grad_q_tmp[i * 2],
                                        grad_q_tmp[i * 2 + 1], 5); // 9-> 8
        hemmer_->tr_oneMatRotSumInplace(
            grad_k_tmp[i * 2], grad_k_tmp[i * 2 + 1], 5); // 9 -> 8 (5 -> 4)
        for (u64 j = 0; j < 2; ++j) {
            auto tmp = lora_module_->getCtxtTensor("forward_res_q", 0, 0,
                                                   i * 2 + j); // 9
            hemmer_->getEval().mult(grad_q_tmp[i * 2 + j].get(), tmp.get(),
                                    grad_q_tmp[i * 2 + j].get()); // 8

            tmp = lora_module_->getCtxtTensor("forward_res_k", 0, 0,
                                              i * 2 + j); // 10
            hemmer_->getEval().mult(grad_k_tmp[i * 2 + j].get(), tmp.get(),
                                    grad_k_tmp[i * 2 + j].get()); // 9 (4)
        }
    }

    // grad@res_k part
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingInplace(grad_exp_k[i]);
        auto tmp = lora_module_->getCtxtTensor("forward_res_k", 0, 0, i); // 10
        hemmer_->complexPackingRowInplace(tmp);                           // 9
        grad_exp_k[i] = hemmer_->packedMatMul(grad_exp_k[i], tmp);        //  4
        grad_exp_k[i] = hemmer_->repackCC(grad_exp_k[i]);

        hemmer_->complexPackingInplace(grad_exp_q[i]);
        tmp = lora_module_->getCtxtTensor("forward_res_q", 0, 0, i);
        grad_exp_q[i] = hemmer_->packedMatMul(grad_exp_q[i], tmp); // 4
        grad_exp_q[i] = hemmer_->repackCC(grad_exp_q[i]);
    }

    // summation part.
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->getEval().sub(grad_q_tmp[i].get(), grad_exp_k[i].get(),
                               grad_q_tmp[i].get());
        hemmer_->getEval().mult(grad_q_tmp[i].get(), 2.0,
                                grad_q_tmp[i].get()); // 4

        hemmer_->getEval().sub(grad_k_tmp[i].get(), grad_exp_q[i].get(),
                               grad_k_tmp[i].get());
        hemmer_->getEval().mult(grad_k_tmp[i].get(), 2.0,
                                grad_k_tmp[i].get()); // 4
    }
    grad_exp_k.clear();
    grad_exp_q.clear();

    // collecting: q1 0 q3 0 + 0 q2 0 q4 >> q1 q2 q3 q4
    // grad_q/k output : 4
    std::vector<CtxtTensor> grad_q;
    std::vector<CtxtTensor> grad_k;
    grad_q.reserve(3);
    grad_k.reserve(3);
    for (u64 i = 0; i < 3; ++i) {
        grad_q.push_back(grad_q_tmp[0]);
        grad_k.push_back(grad_k_tmp[0]);
        hemmer_->getEval().add(grad_q_tmp[i * 2].get(),
                               grad_q_tmp[i * 2 + 1].get(),
                               grad_q[i].get()); // 4
        hemmer_->getEval().add(grad_k_tmp[i * 2].get(),
                               grad_k_tmp[i * 2 + 1].get(),
                               grad_k[i].get()); // 4
    }
    grad_q_tmp.clear();
    grad_k_tmp.clear();

    // direction of value gradient
    std::vector<CtxtTensor> grad_v_tmp;
    grad_v_tmp.reserve(6);
    grad_v_tmp = qkv_grad_split; // 11
    // std::cout << "res_exp mult input " << std::endl;
    // printing(grad_v_tmp);

    // output: 4
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingRowInplace(grad_v_tmp[i]);
        // std::cout << "complex packing done, level = "
        // <<qkv_grad_split[0].get().getLevel() << std::endl;
        CtxtTensor tmp =
            lora_module_->getCtxtTensor("forward_res_exp", 0, 0, i); // 8
        hemmer_->transposeInplace(tmp);                              // 7
        hemmer_->complexPackingInplace(tmp);                         // 6
        grad_v_tmp[i] = hemmer_->packedMatMul(tmp, grad_v_tmp[i]);
        grad_v_tmp[i] = hemmer_->repackCC(grad_v_tmp[i]);
    }
    qkv_grad_split.clear();
    // printing(grad_v_tmp);

    std::vector<CtxtTensor> grad_v;
    grad_v.reserve(3);
    for (u64 i = 0; i < 3; ++i) {
        grad_v.push_back(grad_v_tmp[i]);
        hemmer_->getEval().add(grad_v_tmp[i * 2].get(),
                               grad_v_tmp[i * 2 + 1].get(),
                               grad_v[i].get()); // 4
    }
    grad_v_tmp.clear();

    // 12
    hemmer_->bootstrap2(grad_q[0], grad_q[1]);
    hemmer_->bootstrap2(grad_q[2], grad_k[2]);
    hemmer_->bootstrap2(grad_k[0], grad_k[1]);
    hemmer_->bootstrap2(grad_v[0], grad_v[1]);
    hemmer_->bootstrap(grad_v[2]);

    /* std::cout << "query grad" << std::endl;
    printing(grad_q);

    std::cout << "key grad" << std::endl;
    printing(grad_k);

    std::cout << "value grad" << std::endl;
    printing(grad_v); */

    /* tensor_save(grad_q, "2ly_q_2", layer_n_);
    tensor_save(grad_k, "2ly_k_2", layer_n_);
    tensor_save(grad_v, "2ly_v_2", layer_n_); */

    // input gradient
    std::vector<CtxtTensor> output;
    output.reserve(6);
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(hemmer_->complexPacking(grad_q[0], grad_q[1]),
                               tmp);
        } else {
            hemmer_->matMulPre(grad_q[2], tmp);
        }

        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("tr_wq", i, j);
            auto res = hemmer_->matMulReUse(tmp, weight);
            if (i == 0) {
                output.push_back(res);
            } else {
                hemmer_->addInplace(output[j], res);
            }
        }
    }

    if (lora_type.find('q') == std::string::npos) {
        grad_q.clear();
    }

    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(hemmer_->complexPacking(grad_k[0], grad_k[1]),
                               tmp);
        } else {
            hemmer_->matMulPre(grad_k[2], tmp);
        }
        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("tr_wk", i, j);
            auto res = hemmer_->matMulReUse(tmp, weight);

            hemmer_->addInplace(output[j], res);
        }
    }

    if (lora_type.find('k') == std::string::npos) {
        grad_k.clear();
    }

    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(hemmer_->complexPacking(grad_v[0], grad_v[1]),
                               tmp);
        } else {
            hemmer_->matMulPre(grad_v[2], tmp);
        }

        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("tr_wv", i, j);
            auto res = hemmer_->matMulReUse(tmp, weight);

            hemmer_->addInplace(output[j], res);
        }
    }

    if (lora_type.find('v') == std::string::npos) {
        grad_v.clear();
    }

    std::vector<CtxtTensor> cur;
    cur.reserve(3);
    // output: 11
    for (u64 i = 0; i < 3; ++i) {
        auto repacked = hemmer_->repack(output[i * 2], output[i * 2 + 1]);
        cur.push_back(repacked);
    }
    output.clear();

    /* if(layer_n_ == 0) {
        std::cout << "qkv gradient merge" << std::endl;
        printing(cur);
    } */
    // std::cout << "attn backward weight mm" << std::endl;
    // printing(cur);

    // lora gradient
    // consider accumulation step.
    // input cur level = 11
    for (const char t : lora_type) {
        const std::string lora_t = std::string(1, t);
        std::vector<HELLM::CtxtTensor> tmp_grad;
        switch (t) {
        case 'q':
            tmp_grad = std::move(grad_q);
            break;
        case 'k':
            tmp_grad = std::move(grad_k);
            break;
        case 'v':
            tmp_grad = std::move(grad_v);
            break;
        default:
            std::cout << "error! unsupported lora type" << std::endl;
        }

        // original code
        /* auto tr_lora_wb =
            lora_module_->getCtxtTensor_lora("lora_wb_" + lora_t, 0, 0, 0); //
        need: 9 hemmer_->transposeInplace(tr_lora_wb); // 1 auto grad_lora_b =
        hemmer_->matMulHighLow( tmp_grad[0], hemmer_->getLowColBlock(tr_lora_wb,
        0), 0, 5); //3 for (u64 i = 1; i < 3; ++i) { auto tr_lora_wb_lowcol =
        hemmer_->getLowColBlock(tr_lora_wb, i); auto matmul_res =
                hemmer_->matMulHighLow(tmp_grad[i], tr_lora_wb_lowcol, i, 5);
        //1 hemmer_->addInplace(grad_lora_b, matmul_res);
        }

        grad_lora_b = hemmer_->repackToOneCol(grad_lora_b, 0);
        hemmer_->bootstrap(grad_lora_b); //12
        //printElapsedTime("backward_atn_lora_b_bts"); */

        auto tr_lora_wb =
            lora_module_->getCtxtTensor_lora("lora_wb_" + lora_t, 0, 0, 0);
        hemmer_->getEval().levelDown(tr_lora_wb.get(), 9, tr_lora_wb.get());

        // split weights
        std::vector<std::vector<CtxtTensor>> tr_lora_b_weight;
        tr_lora_b_weight.reserve(3);
        std::vector<CtxtTensor> grad_lora_b_out;
        grad_lora_b_out.reserve(2);

        for (u64 j = 0; j < 2; ++j) {
            grad_lora_b_out.emplace_back(tr_lora_wb);
        }

        auto tmp = tr_lora_wb;

        hemmer_->maskFirstRowInplace(grad_lora_b_out[0]);
        hemmer_->getEval().leftRotate(grad_lora_b_out[1].get(), 1 * 256,
                                      grad_lora_b_out[1].get());
        hemmer_->maskFirstRowInplace(grad_lora_b_out[1]);
        tr_lora_b_weight.emplace_back(grad_lora_b_out);
        for (u64 i = 1; i < 3; ++i) {
            grad_lora_b_out.clear();
            hemmer_->getEval().leftRotate(tr_lora_wb.get(), i * 2 * 256,
                                          tmp.get());
            for (u64 j = 0; j < 2; ++j) {
                grad_lora_b_out.emplace_back(tmp);
            }
            hemmer_->maskFirstRowInplace(grad_lora_b_out[0]);
            hemmer_->getEval().leftRotate(grad_lora_b_out[1].get(), 1 * 256,
                                          grad_lora_b_out[1].get());
            hemmer_->maskFirstRowInplace(grad_lora_b_out[1]);
            tr_lora_b_weight.emplace_back(grad_lora_b_out);
        }

        // repeated packing
        for (u64 i = 0; i < 3; ++i) {
            for (u64 j = 0; j < 2; ++j) {
                for (u64 rot = 1; rot < 128; rot <<= 1) {
                    hemmer_->getEval().rightRotate(tr_lora_b_weight[i][j].get(),
                                                   rot * 256, tmp.get());
                    hemmer_->addInplace(tr_lora_b_weight[i][j], tmp);
                }
            }
        }

        // hadamult
        hemmer_->getEval().mult(tmp_grad[0].get(), tr_lora_b_weight[0][0].get(),
                                grad_lora_b_out[0].get());
        hemmer_->getEval().mult(tmp_grad[0].get(), tr_lora_b_weight[0][1].get(),
                                grad_lora_b_out[1].get());
        for (u64 i = 1; i < 3; ++i) {
            hemmer_->getEval().mult(tmp_grad[i].get(),
                                    tr_lora_b_weight[i][0].get(), tmp.get());
            hemmer_->addInplace(grad_lora_b_out[0], tmp);
            hemmer_->getEval().mult(tmp_grad[i].get(),
                                    tr_lora_b_weight[i][1].get(), tmp.get());
            hemmer_->addInplace(grad_lora_b_out[1], tmp);
        }

        // addition
        for (u64 i = 0; i < 2; ++i) {
            for (u64 rot = 1; rot < 256; rot <<= 1) {
                hemmer_->getEval().leftRotate(grad_lora_b_out[i].get(), rot,
                                              tmp.get());
                hemmer_->addInplace(grad_lora_b_out[i], tmp);
            }
            hemmer_->maskFirstColOnlyInplace(grad_lora_b_out[i]);
        }
        tr_lora_b_weight.clear();

        // origianl code
        /* auto tr_lora_wa =
            lora_module_->getCtxtTensor_lora("lora_wa_" + lora_t, 0, 0, 0);
        hemmer_->transposeInplace(tr_lora_wa); //1
        for (u64 i = 0; i < 3; ++i) {
            auto tr_lora_wa_lowrow = hemmer_->getLowRowBlock(tr_lora_wa, i);
            auto matmul_res =
                hemmer_->matMulLowLow(grad_lora_b, tr_lora_wa_lowrow, 0, i); //4

            hemmer_->addInplace(cur[i], matmul_res); //4
        } */

        // hard coding
        auto tr_lora_wa =
            lora_module_->getCtxtTensor_lora("lora_wa_" + lora_t, 0, 0, 0);
        hemmer_->getEval().levelDown(tr_lora_wa.get(), 6, tr_lora_wa.get());

        // split weights
        std::vector<std::vector<CtxtTensor>> tr_lora_a_weight;
        tr_lora_a_weight.reserve(3);
        std::vector<CtxtTensor> tmp_vector;
        tmp_vector.reserve(2);

        for (u64 i = 0; i < 2; ++i) {
            tmp_vector.emplace_back(tr_lora_wa);
        }
        hemmer_->maskFirstRowInplace(tmp_vector[0]);
        hemmer_->getEval().leftRotate(tmp_vector[1].get(), 1 * 256,
                                      tmp_vector[1].get());
        hemmer_->maskFirstRowInplace(tmp_vector[1]);
        tr_lora_a_weight.emplace_back(tmp_vector);

        for (u64 i = 1; i < 3; ++i) {
            tmp_vector.clear();
            hemmer_->getEval().leftRotate(tr_lora_wa.get(), i * 2 * 256,
                                          tmp.get());
            for (u64 j = 0; j < 2; ++j) {
                tmp_vector.emplace_back(tmp);
            }
            hemmer_->maskFirstRowInplace(tmp_vector[0]);
            hemmer_->getEval().leftRotate(tmp_vector[1].get(), 1 * 256,
                                          tmp.get());
            hemmer_->maskFirstRowInplace(tmp_vector[1]);

            tr_lora_a_weight.emplace_back(tmp_vector);
        }
        tmp_vector.clear();

        // repeated packing
        for (u64 i = 0; i < 2; ++i) {
            for (u64 rot = 1; rot < 256; rot <<= 1) {
                hemmer_->getEval().rightRotate(grad_lora_b_out[i].get(), rot,
                                               tmp.get());
                hemmer_->addInplace(grad_lora_b_out[i], tmp);
            }
        }

        for (u64 i = 0; i < 3; ++i) {
            for (u64 j = 0; j < 2; ++j) {
                for (u64 rot = 1; rot < 128; rot <<= 1) {
                    hemmer_->getEval().rightRotate(tr_lora_a_weight[i][j].get(),
                                                   rot * 256, tmp.get());
                    hemmer_->addInplace(tr_lora_a_weight[i][j], tmp);
                }
            }
        }

        std::vector<CtxtTensor> grad_lora_a_out;
        grad_lora_a_out.reserve(3);
        for (u64 i = 0; i < 3; ++i) {
            grad_lora_a_out.emplace_back(tr_lora_wa);
        }

        // hadamult & addition
        for (u64 i = 0; i < 3; ++i) {
            hemmer_->getEval().mult(grad_lora_b_out[0].get(),
                                    tr_lora_a_weight[i][0].get(),
                                    grad_lora_a_out[i].get());
            hemmer_->getEval().mult(grad_lora_b_out[1].get(),
                                    tr_lora_a_weight[i][1].get(), tmp.get());
            hemmer_->addInplace(grad_lora_a_out[i], tmp);

            hemmer_->addInplace(cur[i], grad_lora_a_out[i]);
        }
        grad_lora_a_out.clear();
        // printElapsedTime("backward_atn_lora_a");

        // compute lora grad w
        // original code
        /* auto tr_lora_in_b = lora_module_->getCtxtTensor(
            "tr_lora_in_b_" + lora_t, 0, 0, 0);
        auto grad_lora_wb = hemmer_->repackToMultiRow(
            hemmer_->matMulLowHigh(tr_lora_in_b, tmp_grad[0], 0, 5), 0); // 5
        for (u64 i = 1; i < 3; ++i) {
            auto matmul_res =
                hemmer_->matMulLowHigh(tr_lora_in_b, tmp_grad[i], 0, 5); //5
            hemmer_->addInplace(grad_lora_wb,
                                hemmer_->repackToMultiRow(matmul_res, i)); //5
        } */

        std::vector<CtxtTensor> tr_lora_in_b;
        tr_lora_in_b.reserve(2);
        for (u64 i = 0; i < 2; ++i) {
            tr_lora_in_b.emplace_back(
                lora_module_->getCtxtTensor("tr_lora_in_b_" + lora_t, 0, 0, i));
        }

        // repeated packing
        for (u64 i = 0; i < 2; ++i) {
            for (u64 rot = 1; rot < 256; rot <<= 1) {
                hemmer_->getEval().rightRotate(tr_lora_in_b[i].get(), rot,
                                               tmp.get());
                hemmer_->addInplace(tr_lora_in_b[i], tmp);
            }
        }

        // hadamult
        std::vector<std::vector<CtxtTensor>> grad_wb_tmp;
        grad_wb_tmp.reserve(2);
        for (u64 i = 0; i < 2; ++i) {
            grad_wb_tmp.emplace_back(tmp_grad);
            for (u64 j = 0; j < 3; ++j) {
                hemmer_->getEval().mult(tr_lora_in_b[i].get(),
                                        tmp_grad[j].get(),
                                        grad_wb_tmp[i][j].get());
            }
        }

        // collect results
        for (u64 i = 0; i < 2; ++i) {
            for (u64 j = 0; j < 3; ++j) {
                for (u64 rot = 1; rot < 128; rot <<= 1) {
                    hemmer_->getEval().leftRotate(grad_wb_tmp[i][j].get(),
                                                  rot * 256, tmp.get());
                    hemmer_->addInplace(grad_wb_tmp[i][j], tmp);
                }
                hemmer_->maskFirstRowInplace(grad_wb_tmp[i][j]);
            }
        }

        auto grad_lora_wb = grad_wb_tmp[0][0];
        hemmer_->getEval().rightRotate(grad_wb_tmp[1][0].get(), 1 * 256,
                                       tmp.get());
        hemmer_->addInplace(grad_lora_wb, tmp);
        for (u64 j = 1; j < 3; ++j) {
            hemmer_->getEval().rightRotate(grad_wb_tmp[1][j].get(), 1 * 256,
                                           tmp.get());
            hemmer_->addInplace(grad_wb_tmp[0][j], tmp);

            hemmer_->getEval().rightRotate(grad_wb_tmp[0][j].get(), 2 * j * 256,
                                           grad_wb_tmp[0][j].get());
            hemmer_->addInplace(grad_lora_wb, grad_wb_tmp[0][j]);
        }
        grad_wb_tmp.clear();

        // original code
        /* auto grad_lora_wa = hemmer_->repackToMultiCol(
            hemmer_->matMulHighLow(lora_module_->getCtxtTensor("tr_lora_in_a",
                                                               0, 0,
                                                               0),
                                   grad_lora_b, 0, 5),
            0); // 5
        for (u64 i = 1; i < 3; ++i) {
            auto ctxt_tensor = lora_module_->getCtxtTensor("tr_lora_in_a",
                                                           0, 0, i);
            auto matmul_res =
                hemmer_->matMulHighLow(ctxt_tensor, grad_lora_b, 0, 5);
            auto repacked_res = hemmer_->repackToMultiCol(matmul_res, i); // 4
            hemmer_->addInplace(grad_lora_wa, repacked_res); //4
        } */

        std::vector<CtxtTensor> tr_lora_in_a;
        tr_lora_in_a.reserve(3);
        for (u64 i = 0; i < 3; ++i) {
            tr_lora_in_a.emplace_back(
                lora_module_->getCtxtTensor("tr_lora_in_a", 0, 0, i));
        }

        // repeated packing -> alreday complted in previous

        // hadamult
        std::vector<std::vector<CtxtTensor>> grad_wa_tmp;
        grad_wa_tmp.reserve(3);
        for (u64 i = 0; i < 3; ++i) {
            grad_wa_tmp.emplace_back(grad_lora_b_out);
            for (u64 j = 0; j < 2; ++j) {
                hemmer_->getEval().mult(tr_lora_in_a[i].get(),
                                        grad_lora_b_out[j].get(),
                                        grad_wa_tmp[i][j].get());
            }
        }

        // collect results
        for (u64 i = 0; i < 3; ++i) {
            for (u64 j = 0; j < 2; ++j) {
                for (u64 rot = 1; rot < 128; rot <<= 1) {
                    hemmer_->getEval().leftRotate(grad_wa_tmp[i][j].get(),
                                                  rot * 256, tmp.get());
                    hemmer_->addInplace(grad_wa_tmp[i][j], tmp);
                }
                hemmer_->maskFirstRowInplace(grad_wa_tmp[i][j]);
            }
        }

        auto grad_lora_wa = grad_wa_tmp[0][0];
        hemmer_->getEval().rightRotate(grad_wa_tmp[0][1].get(), 1 * 256,
                                       tmp.get());
        hemmer_->addInplace(grad_lora_wa, tmp);
        for (u64 j = 1; j < 3; ++j) {
            hemmer_->getEval().rightRotate(grad_wa_tmp[j][1].get(), 1 * 256,
                                           grad_wa_tmp[j][1].get());
            hemmer_->addInplace(grad_wa_tmp[j][0], tmp);

            hemmer_->getEval().rightRotate(grad_wa_tmp[j][0].get(), 2 * j * 256,
                                           grad_wa_tmp[j][0].get());
            hemmer_->addInplace(grad_lora_wa, grad_wa_tmp[j][0]);
        }

        // for multi-GPU
        // in: 4

        HEaaN::CudaTools::cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        if (max_rank > 1) {
            allReduceWrapper(grad_lora_wb);
            allReduceWrapper(grad_lora_wa);

            if (rank == 0) {
                hemmer_->getEval().modReduct(grad_lora_wb.get());
                hemmer_->getEval().modReduct(grad_lora_wa.get());

                hemmer_->bootstrap2(grad_lora_wa, grad_lora_wb);

                hemmer_->getEval().mult(grad_lora_wb.get(),
                                        1.0 / ModelArgs::BATCH_SIZE,
                                        grad_lora_wb.get());
                hemmer_->getEval().mult(grad_lora_wa.get(),
                                        1.0 / ModelArgs::BATCH_SIZE,
                                        grad_lora_wa.get());

                lora_module_->saveAggGrad(grad_lora_wb, "b_" + lora_t);
                lora_module_->saveAggGrad(grad_lora_wa, "a_" + lora_t);
            }
        }
    }
    // output cur level = 4
    // printing(cur);
    hemmer_->bootstrap2(cur[0], cur[1]);
    hemmer_->bootstrap(cur[2]);

    for (u64 i = 0; i < 3; ++i) {
        const auto &weight = getWeight("norm1_w", i);
        hemmer_->hadamardMultInplace(cur[i], weight);
    }

    // printing(cur);
    //  11 -> 4
    cur = hemmer_->backwardLayerNorm(cur, "atn", layer_n_);

    /* if(layer_n_ == 0) {
        std::cout << "backward LN" << std::endl;
        printing(cur);
    } */
    // std::cout << "attn backward LN" << std::endl;
    // printing(cur);

    // 4 -> 4
    for (u64 i = 0; i < 3; ++i) {
        hemmer_->addInplace(grad_y[i], cur[i]);
    }
    cur.clear();
    // std::cout << "attn backward residual" << std::endl;
    // printing(grad_y);
}

// in: 12 , out: 5
// hard coding in 2 layer case
// 2th layer: make sure to keep track of all process
// 1th layer: don't need to go through backward after lora weight gradient.
void TransformerBlock::backwardattention2_bert_loraOpti(
    std::vector<CtxtTensor> &grad_y, const std::string &lora_type) {

    auto rank = hemmer_->getRank();
    auto max_rank = hemmer_->getMaxRank();

    std::vector<CtxtTensor> grad_y_tmp;
    grad_y_tmp.reserve(3);
    grad_y_tmp = grad_y;

    start = std::chrono::high_resolution_clock::now();
    /* for (u64 i = 0 ; i < 3 ; ++i) {
        hemmer_->backwarddropoutInplace(grad_y_tmp[i],"atn_output", layer_n_,
    i);
    } */
    // printElapsedTime("back_dropout");

    start = std::chrono::high_resolution_clock::now();
    std::vector<CtxtTensor> tmp_grad_;
    tmp_grad_.reserve(6);
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(
                hemmer_->complexPacking(grad_y_tmp[0], grad_y_tmp[1]), tmp);
        } else {
            hemmer_->matMulPre(grad_y_tmp[2], tmp);
        }

        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("tr_wd", i, j);
            auto result = hemmer_->matMulReUse(tmp, weight);

            if (i == 0) {
                tmp_grad_.push_back(result);
            } else {
                hemmer_->addInplace(tmp_grad_[j], result);
            }
        }
    }
    grad_y_tmp.clear();
    // printElapsedTime("(PC) tr_wd mm");

    start = std::chrono::high_resolution_clock::now();
    std::vector<CtxtTensor> qkv_grad;
    qkv_grad.reserve(3);
    for (u64 i = 0; i < 3; ++i) {
        qkv_grad.push_back(
            hemmer_->repack(tmp_grad_[i * 2], tmp_grad_[i * 2 + 1]));
    }
    tmp_grad_.clear();
    // printElapsedTime("repack");

    std::vector<CtxtTensor> qkv_grad_split;
    qkv_grad_split.reserve(6);
    // 11 -> 10
    for (u64 i = 0; i < 3; ++i) {
        for (u64 j = 0; j < 2; ++j)
            qkv_grad_split.push_back(qkv_grad[i]);

        hemmer_->maskRightLeft(qkv_grad[i], qkv_grad_split[i * 2],
                               qkv_grad_split[i * 2 + 1]);
    }
    qkv_grad.clear();

    std::vector<CtxtTensor> qkv_grad_split_exp;
    qkv_grad_split_exp = qkv_grad_split;

    // direcition of exp gradient
    // 9 -> 4
    start = std::chrono::high_resolution_clock::now();
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingInplace(qkv_grad_split_exp[i]); // 9
        // should consider saved ctxtTensor structure.
        CtxtTensor tmp =
            lora_module_->getCtxtTensor("forward_res_v", 0, 0, i); // 8
        hemmer_->transposeInplace(tmp);                            // 7
        hemmer_->complexPackingRowInplace(tmp);

        qkv_grad_split_exp[i] =
            hemmer_->packedMatMul(qkv_grad_split_exp[i], tmp); // 4
    }
    // printElapsedTime("(CC) res_v mm");

    for (u64 i = 0; i < 6; ++i) {
        qkv_grad_split_exp[i] = hemmer_->repackCC(qkv_grad_split_exp[i]); // 4
    }
    // printElapsedTime("repack");

    // 12
    for (u64 i = 0; i < 3; ++i) {
        hemmer_->bootstrap2(qkv_grad_split_exp[i * 2],
                            qkv_grad_split_exp[i * 2 + 1]);
    }
    // printElapsedTime("BTS");

    /* for (u64 i = 0 ; i < 6 ; ++i) {
        hemmer_->backwarddropoutExpInplace(qkv_grad_split_exp[i], "atn_exp",
    layer_n_, i); // 12 -> 11
    }  */
    // printElapsedTime("back_dropout");

    // 12 -> 11
    hemmer_->backwardexpVectorInplace(qkv_grad_split_exp, layer_n_);
    // printElapsedTime("back exp");

    // 11 -> 10
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->getEval().mult(qkv_grad_split_exp[i].get(), -1.0 / 16,
                                qkv_grad_split_exp[i].get());
    }

    std::vector<CtxtTensor> grad_q_tmp;
    std::vector<CtxtTensor> grad_k_tmp;
    grad_q_tmp = qkv_grad_split_exp; // 10
    grad_k_tmp = qkv_grad_split_exp;

    start = std::chrono::high_resolution_clock::now();
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->transposeInplace(grad_k_tmp[i]); // 10 -> 9 (6 -> 5)
    }
    // printElapsedTime("transpose");

    std::vector<CtxtTensor> grad_exp_q;
    std::vector<CtxtTensor> grad_exp_k;
    grad_exp_k = qkv_grad_split_exp; // 10
    grad_exp_q = grad_k_tmp;         // 9
    qkv_grad_split_exp.clear();

    // grad_q = grad@I^t x 2res_q + grad@res_k
    // grad@I^t x 2res_q part
    start = std::chrono::high_resolution_clock::now();
    for (u64 i = 0; i < 3; ++i) {
        hemmer_->tr_oneMatRotSumInplace(grad_q_tmp[i * 2],
                                        grad_q_tmp[i * 2 + 1], 5); // 9-> 8
        hemmer_->tr_oneMatRotSumInplace(
            grad_k_tmp[i * 2], grad_k_tmp[i * 2 + 1], 5); // 9 -> 8 (5 -> 4)
        for (u64 j = 0; j < 2; ++j) {
            auto tmp = lora_module_->getCtxtTensor("forward_res_q", 0, 0,
                                                   i * 2 + j); // 9
            hemmer_->getEval().mult(grad_q_tmp[i * 2 + j].get(), tmp.get(),
                                    grad_q_tmp[i * 2 + j].get()); // 8

            tmp = lora_module_->getCtxtTensor("forward_res_k", 0, 0,
                                              i * 2 + j); // 10
            hemmer_->getEval().mult(grad_k_tmp[i * 2 + j].get(), tmp.get(),
                                    grad_k_tmp[i * 2 + j].get()); // 9 (4)
        }
    }
    // printElapsedTime("tr and hada");

    // grad@res_k part
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingInplace(grad_exp_k[i]);
        auto tmp = lora_module_->getCtxtTensor("forward_res_k", 0, 0, i); // 10
        hemmer_->complexPackingRowInplace(tmp);                           // 9
        grad_exp_k[i] = hemmer_->packedMatMul(grad_exp_k[i], tmp);        //  4
    }
    // printElapsedTime("(CC) mm");

    for (u64 i = 0; i < 6; ++i) {
        grad_exp_k[i] = hemmer_->repackCC(grad_exp_k[i]);
    }
    // printElapsedTime("repack");

    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingInplace(grad_exp_q[i]);
        auto tmp = lora_module_->getCtxtTensor("forward_res_q", 0, 0, i);
        grad_exp_q[i] = hemmer_->packedMatMul(grad_exp_q[i], tmp); // 4
    }
    // printElapsedTime("(CC) mm");

    for (u64 i = 0; i < 6; ++i) {
        grad_exp_q[i] = hemmer_->repackCC(grad_exp_q[i]);
    }
    // printElapsedTime("repack");

    // summation part.
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->getEval().sub(grad_q_tmp[i].get(), grad_exp_k[i].get(),
                               grad_q_tmp[i].get());
        hemmer_->getEval().mult(grad_q_tmp[i].get(), 2.0,
                                grad_q_tmp[i].get()); // 4

        hemmer_->getEval().sub(grad_k_tmp[i].get(), grad_exp_q[i].get(),
                               grad_k_tmp[i].get());
        hemmer_->getEval().mult(grad_k_tmp[i].get(), 2.0,
                                grad_k_tmp[i].get()); // 4
    }
    // printElapsedTime("hadamult");
    grad_exp_k.clear();
    grad_exp_q.clear();

    // collecting: q1 0 q3 0 + 0 q2 0 q4 >> q1 q2 q3 q4
    // grad_q/k output : 4
    std::vector<CtxtTensor> grad_q;
    std::vector<CtxtTensor> grad_k;
    grad_q.reserve(3);
    grad_k.reserve(3);
    for (u64 i = 0; i < 3; ++i) {
        grad_q.push_back(grad_q_tmp[0]);
        grad_k.push_back(grad_k_tmp[0]);
        hemmer_->getEval().add(grad_q_tmp[i * 2].get(),
                               grad_q_tmp[i * 2 + 1].get(),
                               grad_q[i].get()); // 4
        hemmer_->getEval().add(grad_k_tmp[i * 2].get(),
                               grad_k_tmp[i * 2 + 1].get(),
                               grad_k[i].get()); // 4
    }
    grad_q_tmp.clear();
    grad_k_tmp.clear();

    // direction of value gradient
    std::vector<CtxtTensor> grad_v_tmp;
    grad_v_tmp.reserve(6);
    grad_v_tmp = qkv_grad_split; // 11

    // output: 4
    start = std::chrono::high_resolution_clock::now();
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingRowInplace(grad_v_tmp[i]);
        CtxtTensor tmp =
            lora_module_->getCtxtTensor("forward_res_exp", 0, 0, i); // 8
        hemmer_->transposeInplace(tmp);                              // 7
        hemmer_->complexPackingInplace(tmp);                         // 6
        grad_v_tmp[i] = hemmer_->packedMatMul(tmp, grad_v_tmp[i]);
    }
    // printElapsedTime("(CC) exp mm");

    for (u64 i = 0; i < 6; ++i) {
        grad_v_tmp[i] = hemmer_->repackCC(grad_v_tmp[i]);
    }
    // printElapsedTime("repackCC");
    qkv_grad_split.clear();

    std::vector<CtxtTensor> grad_v;
    grad_v.reserve(3);
    for (u64 i = 0; i < 3; ++i) {
        grad_v.push_back(grad_v_tmp[i]);
        hemmer_->getEval().add(grad_v_tmp[i * 2].get(),
                               grad_v_tmp[i * 2 + 1].get(),
                               grad_v[i].get()); // 4
    }
    grad_v_tmp.clear();

    // 12
    start = std::chrono::high_resolution_clock::now();
    hemmer_->bootstrap2(grad_q[0], grad_q[1]);
    hemmer_->bootstrap2(grad_q[2], grad_k[2]);
    hemmer_->bootstrap2(grad_k[0], grad_k[1]);
    hemmer_->bootstrap2(grad_v[0], grad_v[1]);
    hemmer_->bootstrap(grad_v[2]);
    // printElapsedTime("BTS");

    // input gradient
    std::vector<CtxtTensor> cur;
    cur.reserve(3);

    if (layer_n_ == 1) {
        start = std::chrono::high_resolution_clock::now();
        std::vector<CtxtTensor> output;
        output.reserve(6);
        for (u64 i = 0; i < 2; ++i) {
            std::vector<Ciphertext> tmp;
            if (i == 0) {
                hemmer_->matMulPre(
                    hemmer_->complexPacking(grad_q[0], grad_q[1]), tmp);
            } else {
                hemmer_->matMulPre(grad_q[2], tmp);
            }

            for (u64 j = 0; j < 6; ++j) {
                auto weight = getWeight("tr_wq", i, j);
                auto res = hemmer_->matMulReUse(tmp, weight);
                if (i == 0) {
                    output.push_back(res);
                } else {
                    hemmer_->addInplace(output[j], res);
                }
            }
        }
        // printElapsedTime("(PC) mm ");

        if (lora_type.find('q') == std::string::npos) {
            grad_q.clear();
        }

        for (u64 i = 0; i < 2; ++i) {
            std::vector<Ciphertext> tmp;
            if (i == 0) {
                hemmer_->matMulPre(
                    hemmer_->complexPacking(grad_k[0], grad_k[1]), tmp);
            } else {
                hemmer_->matMulPre(grad_k[2], tmp);
            }
            for (u64 j = 0; j < 6; ++j) {
                auto weight = getWeight("tr_wk", i, j);
                auto res = hemmer_->matMulReUse(tmp, weight);

                hemmer_->addInplace(output[j], res);
            }
        }
        // printElapsedTime("(PC) mm ");

        if (lora_type.find('k') == std::string::npos) {
            grad_k.clear();
        }

        for (u64 i = 0; i < 2; ++i) {
            std::vector<Ciphertext> tmp;
            if (i == 0) {
                hemmer_->matMulPre(
                    hemmer_->complexPacking(grad_v[0], grad_v[1]), tmp);
            } else {
                hemmer_->matMulPre(grad_v[2], tmp);
            }

            for (u64 j = 0; j < 6; ++j) {
                auto weight = getWeight("tr_wv", i, j);
                auto res = hemmer_->matMulReUse(tmp, weight);

                hemmer_->addInplace(output[j], res);
            }
        }
        // printElapsedTime("(PC) mm ");

        if (lora_type.find('v') == std::string::npos) {
            grad_v.clear();
        }

        /* std::vector<CtxtTensor> cur;
        cur.reserve(3); */

        // output: 11
        for (u64 i = 0; i < 3; ++i) {
            auto repacked = hemmer_->repack(output[i * 2], output[i * 2 + 1]);
            cur.push_back(repacked);
        }
        output.clear();
        // printElapsedTime("repack ");
    }

    // lora gradient
    // consider accumulation step.
    // input cur level = 11
    start = std::chrono::high_resolution_clock::now();
    for (const char t : lora_type) {
        const std::string lora_t = std::string(1, t);
        std::vector<HELLM::CtxtTensor> tmp_grad;
        switch (t) {
        case 'q':
            tmp_grad = std::move(grad_q);
            break;
        case 'k':
            tmp_grad = std::move(grad_k);
            break;
        case 'v':
            tmp_grad = std::move(grad_v);
            break;
        default:
            std::cout << "error! unsupported lora type" << std::endl;
        }

        // input grad B direction
        auto tr_lora_wb =
            lora_module_->getCtxtTensor_lora("lora_wb_" + lora_t, 0, 0, 0);
        hemmer_->getEval().levelDown(tr_lora_wb.get(), 9, tr_lora_wb.get());

        // split weights
        std::vector<std::vector<CtxtTensor>> tr_lora_b_weight;
        tr_lora_b_weight.reserve(3);
        std::vector<CtxtTensor> grad_lora_b_out;
        grad_lora_b_out.reserve(2);

        for (u64 j = 0; j < 2; ++j) {
            grad_lora_b_out.emplace_back(tr_lora_wb);
        }

        auto tmp = tr_lora_wb;

        hemmer_->maskFirstRowInplace(grad_lora_b_out[0]);
        hemmer_->getEval().leftRotate(grad_lora_b_out[1].get(), 1 * 256,
                                      grad_lora_b_out[1].get());
        hemmer_->maskFirstRowInplace(grad_lora_b_out[1]);
        tr_lora_b_weight.emplace_back(grad_lora_b_out);
        for (u64 i = 1; i < 3; ++i) {
            grad_lora_b_out.clear();
            hemmer_->getEval().leftRotate(tr_lora_wb.get(), i * 2 * 256,
                                          tmp.get());
            for (u64 j = 0; j < 2; ++j) {
                grad_lora_b_out.emplace_back(tmp);
            }
            hemmer_->maskFirstRowInplace(grad_lora_b_out[0]);
            hemmer_->getEval().leftRotate(grad_lora_b_out[1].get(), 1 * 256,
                                          grad_lora_b_out[1].get());
            hemmer_->maskFirstRowInplace(grad_lora_b_out[1]);
            tr_lora_b_weight.emplace_back(grad_lora_b_out);
        }

        // repeated packing
        for (u64 i = 0; i < 3; ++i) {
            for (u64 j = 0; j < 2; ++j) {
                for (u64 rot = 1; rot < 128; rot <<= 1) {
                    hemmer_->getEval().rightRotate(tr_lora_b_weight[i][j].get(),
                                                   rot * 256, tmp.get());
                    hemmer_->addInplace(tr_lora_b_weight[i][j], tmp);
                }
            }
        }

        // hadamult
        hemmer_->getEval().mult(tmp_grad[0].get(), tr_lora_b_weight[0][0].get(),
                                grad_lora_b_out[0].get());
        hemmer_->getEval().mult(tmp_grad[0].get(), tr_lora_b_weight[0][1].get(),
                                grad_lora_b_out[1].get());
        for (u64 i = 1; i < 3; ++i) {
            hemmer_->getEval().mult(tmp_grad[i].get(),
                                    tr_lora_b_weight[i][0].get(), tmp.get());
            hemmer_->addInplace(grad_lora_b_out[0], tmp);
            hemmer_->getEval().mult(tmp_grad[i].get(),
                                    tr_lora_b_weight[i][1].get(), tmp.get());
            hemmer_->addInplace(grad_lora_b_out[1], tmp);
        }

        // addition
        for (u64 i = 0; i < 2; ++i) {
            for (u64 rot = 1; rot < 256; rot <<= 1) {
                hemmer_->getEval().leftRotate(grad_lora_b_out[i].get(), rot,
                                              tmp.get());
                hemmer_->addInplace(grad_lora_b_out[i], tmp);
            }
            hemmer_->maskFirstColOnlyInplace(grad_lora_b_out[i]);
        }
        tr_lora_b_weight.clear();
        // printElapsedTime("(CC) lora_wb mm");

        // repeated packing for the following step.
        for (u64 i = 0; i < 2; ++i) {
            for (u64 rot = 1; rot < 256; rot <<= 1) {
                hemmer_->getEval().rightRotate(grad_lora_b_out[i].get(), rot,
                                               tmp.get());
                hemmer_->addInplace(grad_lora_b_out[i], tmp);
            }
        }

        if (layer_n_ == 1) {
            // input grad A direction
            // hard coding
            auto tr_lora_wa =
                lora_module_->getCtxtTensor_lora("lora_wa_" + lora_t, 0, 0, 0);
            hemmer_->getEval().levelDown(tr_lora_wa.get(), 6, tr_lora_wa.get());

            // split weights
            std::vector<std::vector<CtxtTensor>> tr_lora_a_weight;
            tr_lora_a_weight.reserve(3);
            std::vector<CtxtTensor> tmp_vector;
            tmp_vector.reserve(2);

            for (u64 i = 0; i < 2; ++i) {
                tmp_vector.emplace_back(tr_lora_wa);
            }
            hemmer_->maskFirstRowInplace(tmp_vector[0]);
            hemmer_->getEval().leftRotate(tmp_vector[1].get(), 1 * 256,
                                          tmp_vector[1].get());
            hemmer_->maskFirstRowInplace(tmp_vector[1]);
            tr_lora_a_weight.emplace_back(tmp_vector);

            for (u64 i = 1; i < 3; ++i) {
                tmp_vector.clear();
                hemmer_->getEval().leftRotate(tr_lora_wa.get(), i * 2 * 256,
                                              tmp.get());
                for (u64 j = 0; j < 2; ++j) {
                    tmp_vector.emplace_back(tmp);
                }
                hemmer_->maskFirstRowInplace(tmp_vector[0]);
                hemmer_->getEval().leftRotate(tmp_vector[1].get(), 1 * 256,
                                              tmp_vector[1].get());
                hemmer_->maskFirstRowInplace(tmp_vector[1]);

                tr_lora_a_weight.emplace_back(tmp_vector);
            }
            tmp_vector.clear();

            /* // repeated packing (see in the previous step)
            for (u64 i = 0 ; i < 2 ; ++i) {
                for (u64 rot = 1 ; rot < 256 ; rot <<= 1 ){
                    hemmer_->getEval().rightRotate(grad_lora_b_out[i].get(),
            rot, tmp.get()); hemmer_->addInplace(grad_lora_b_out[i], tmp);
                }
            } */

            for (u64 i = 0; i < 3; ++i) {
                for (u64 j = 0; j < 2; ++j) {
                    for (u64 rot = 1; rot < 128; rot <<= 1) {
                        hemmer_->getEval().rightRotate(
                            tr_lora_a_weight[i][j].get(), rot * 256, tmp.get());
                        hemmer_->addInplace(tr_lora_a_weight[i][j], tmp);
                    }
                }
            }

            std::vector<CtxtTensor> grad_lora_a_out;
            grad_lora_a_out.reserve(3);
            for (u64 i = 0; i < 3; ++i) {
                grad_lora_a_out.emplace_back(tr_lora_wa);
            }

            // hadamult & addition
            for (u64 i = 0; i < 3; ++i) {
                hemmer_->getEval().mult(grad_lora_b_out[0].get(),
                                        tr_lora_a_weight[i][0].get(),
                                        grad_lora_a_out[i].get());
                hemmer_->getEval().mult(grad_lora_b_out[1].get(),
                                        tr_lora_a_weight[i][1].get(),
                                        tmp.get());
                hemmer_->addInplace(grad_lora_a_out[i], tmp);

                hemmer_->addInplace(cur[i], grad_lora_a_out[i]);
            }
            grad_lora_a_out.clear();
        }
        // printElapsedTime("backward_atn_lora_a");

        // compute lora grad w
        std::vector<CtxtTensor> tr_lora_in_b;
        tr_lora_in_b.reserve(2);
        for (u64 i = 0; i < 2; ++i) {
            tr_lora_in_b.emplace_back(
                lora_module_->getCtxtTensor("tr_lora_in_b_" + lora_t, 0, 0, i));
        }

        // repeated packing
        for (u64 i = 0; i < 2; ++i) {
            for (u64 rot = 1; rot < 256; rot <<= 1) {
                hemmer_->getEval().rightRotate(tr_lora_in_b[i].get(), rot,
                                               tmp.get());
                hemmer_->addInplace(tr_lora_in_b[i], tmp);
            }
        }

        // hadamult
        std::vector<std::vector<CtxtTensor>> grad_wb_tmp;
        grad_wb_tmp.reserve(2);
        for (u64 i = 0; i < 2; ++i) {
            grad_wb_tmp.emplace_back(tmp_grad);
            for (u64 j = 0; j < 3; ++j) {
                hemmer_->getEval().mult(tr_lora_in_b[i].get(),
                                        tmp_grad[j].get(),
                                        grad_wb_tmp[i][j].get());
            }
        }

        // collect results
        for (u64 i = 0; i < 2; ++i) {
            for (u64 j = 0; j < 3; ++j) {
                for (u64 rot = 1; rot < 128; rot <<= 1) {
                    hemmer_->getEval().leftRotate(grad_wb_tmp[i][j].get(),
                                                  rot * 256, tmp.get());
                    hemmer_->addInplace(grad_wb_tmp[i][j], tmp);
                }
                hemmer_->maskFirstRowInplace(grad_wb_tmp[i][j]);
            }
        }

        auto grad_lora_wb = grad_wb_tmp[0][0];
        hemmer_->getEval().rightRotate(grad_wb_tmp[1][0].get(), 1 * 256,
                                       tmp.get());
        hemmer_->addInplace(grad_lora_wb, tmp);
        for (u64 j = 1; j < 3; ++j) {
            hemmer_->getEval().rightRotate(grad_wb_tmp[1][j].get(), 1 * 256,
                                           tmp.get());
            hemmer_->addInplace(grad_wb_tmp[0][j], tmp);

            hemmer_->getEval().rightRotate(grad_wb_tmp[0][j].get(), 2 * j * 256,
                                           grad_wb_tmp[0][j].get());
            hemmer_->addInplace(grad_lora_wb, grad_wb_tmp[0][j]);
        }
        grad_wb_tmp.clear();
        // printElapsedTime("(CC) lora wegith grad mm");

        std::vector<CtxtTensor> tr_lora_in_a;
        tr_lora_in_a.reserve(3);
        for (u64 i = 0; i < 3; ++i) {
            tr_lora_in_a.emplace_back(
                lora_module_->getCtxtTensor("tr_lora_in_a", 0, 0, i));
        }

        // repeated packing -> alreday complted in previous

        // hadamult
        std::vector<std::vector<CtxtTensor>> grad_wa_tmp;
        grad_wa_tmp.reserve(3);
        for (u64 i = 0; i < 3; ++i) {
            grad_wa_tmp.emplace_back(grad_lora_b_out);
            for (u64 j = 0; j < 2; ++j) {
                hemmer_->getEval().mult(tr_lora_in_a[i].get(),
                                        grad_lora_b_out[j].get(),
                                        grad_wa_tmp[i][j].get());
            }
        }

        // collect results
        for (u64 i = 0; i < 3; ++i) {
            for (u64 j = 0; j < 2; ++j) {
                for (u64 rot = 1; rot < 128; rot <<= 1) {
                    hemmer_->getEval().leftRotate(grad_wa_tmp[i][j].get(),
                                                  rot * 256, tmp.get());
                    hemmer_->addInplace(grad_wa_tmp[i][j], tmp);
                }
                hemmer_->maskFirstRowInplace(grad_wa_tmp[i][j]);
            }
        }

        auto grad_lora_wa = grad_wa_tmp[0][0];
        hemmer_->getEval().rightRotate(grad_wa_tmp[0][1].get(), 1 * 256,
                                       tmp.get());
        hemmer_->addInplace(grad_lora_wa, tmp);
        for (u64 j = 1; j < 3; ++j) {
            hemmer_->getEval().rightRotate(grad_wa_tmp[j][1].get(), 1 * 256,
                                           grad_wa_tmp[j][1].get());
            hemmer_->addInplace(grad_wa_tmp[j][0], tmp);

            hemmer_->getEval().rightRotate(grad_wa_tmp[j][0].get(), 2 * j * 256,
                                           grad_wa_tmp[j][0].get());
            hemmer_->addInplace(grad_lora_wa, grad_wa_tmp[j][0]);
        }
        // printElapsedTime("(CC) lora wegith grad mm and repack");

        HEaaN::CudaTools::cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        if (max_rank > 1) {
            allReduceWrapper(grad_lora_wb);
            allReduceWrapper(grad_lora_wa);

            if (rank == 0) {
                hemmer_->getEval().modReduct(grad_lora_wb.get());
                hemmer_->getEval().modReduct(grad_lora_wa.get());

                hemmer_->bootstrap2(grad_lora_wa, grad_lora_wb);

                hemmer_->getEval().mult(grad_lora_wb.get(),
                                        1.0 / ModelArgs::BATCH_SIZE,
                                        grad_lora_wb.get());
                hemmer_->getEval().mult(grad_lora_wa.get(),
                                        1.0 / ModelArgs::BATCH_SIZE,
                                        grad_lora_wa.get());

                lora_module_->saveAggGrad(grad_lora_wb, "b_" + lora_t);
                lora_module_->saveAggGrad(grad_lora_wa, "a_" + lora_t);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    // printElapsedTime("lora CC backward");

    // output cur level = 4
    // printing(cur);
    if (layer_n_ == 1) {
        start = std::chrono::high_resolution_clock::now();
        hemmer_->bootstrap2(cur[0], cur[1]);
        hemmer_->bootstrap(cur[2]);
        // printElapsedTime("BTS");

        for (u64 i = 0; i < 3; ++i) {
            const auto &weight = getWeight("norm1_w", i);
            hemmer_->hadamardMultInplace(cur[i], weight);
        }

        // printing(cur);
        //  11 -> 4
        start = std::chrono::high_resolution_clock::now();
        cur = hemmer_->backwardLayerNorm(cur, "atn", layer_n_);
        // printElapsedTime("back_LN");

        // 4 -> 4
        for (u64 i = 0; i < 3; ++i) {
            hemmer_->addInplace(grad_y[i], cur[i]);
        }
        cur.clear();
    }
}

// in: 12 , out: 5
void TransformerBlock::backwardattention2_bert_loraOpti_time(
    std::vector<CtxtTensor> &grad_y, const std::string &lora_type) {

    auto rank = hemmer_->getRank();
    auto max_rank = hemmer_->getMaxRank();

    std::vector<CtxtTensor> grad_y_tmp;
    grad_y_tmp.reserve(3);
    grad_y_tmp = grad_y;

    start = std::chrono::high_resolution_clock::now();
    /* for (u64 i = 0 ; i < 3 ; ++i) {
        hemmer_->backwarddropoutInplace(grad_y_tmp[i],"atn_output", layer_n_,
    i);
    }
    printElapsedTime("back_dropout"); */

    start = std::chrono::high_resolution_clock::now();
    std::vector<CtxtTensor> tmp_grad_;
    tmp_grad_.reserve(6);
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(
                hemmer_->complexPacking(grad_y_tmp[0], grad_y_tmp[1]), tmp);
        } else {
            hemmer_->matMulPre(grad_y_tmp[2], tmp);
        }

        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("tr_wd", i, j);
            auto result = hemmer_->matMulReUse(tmp, weight);

            if (i == 0) {
                tmp_grad_.push_back(result);
            } else {
                hemmer_->addInplace(tmp_grad_[j], result);
            }
        }
    }
    grad_y_tmp.clear();
    printElapsedTime("(PC) tr_wd mm");

    start = std::chrono::high_resolution_clock::now();
    std::vector<CtxtTensor> qkv_grad;
    qkv_grad.reserve(3);
    for (u64 i = 0; i < 3; ++i) {
        qkv_grad.push_back(
            hemmer_->repack(tmp_grad_[i * 2], tmp_grad_[i * 2 + 1]));
    }
    tmp_grad_.clear();
    printElapsedTime("repack");

    std::vector<CtxtTensor> qkv_grad_split;
    qkv_grad_split.reserve(6);
    // 11 -> 10
    for (u64 i = 0; i < 3; ++i) {
        for (u64 j = 0; j < 2; ++j)
            qkv_grad_split.push_back(qkv_grad[i]);

        hemmer_->maskRightLeft(qkv_grad[i], qkv_grad_split[i * 2],
                               qkv_grad_split[i * 2 + 1]);
    }
    qkv_grad.clear();

    std::vector<CtxtTensor> qkv_grad_split_exp;
    qkv_grad_split_exp = qkv_grad_split;

    // direcition of exp gradient
    // 9 -> 4
    start = std::chrono::high_resolution_clock::now();
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingInplace(qkv_grad_split_exp[i]); // 9
        // should consider saved ctxtTensor structure.
        CtxtTensor tmp =
            lora_module_->getCtxtTensor("forward_res_v", 0, 0, i); // 8
        hemmer_->transposeInplace(tmp);                            // 7
        hemmer_->complexPackingRowInplace(tmp);

        qkv_grad_split_exp[i] =
            hemmer_->packedMatMul(qkv_grad_split_exp[i], tmp); // 4
    }
    printElapsedTime("(CC) res_v mm");

    for (u64 i = 0; i < 6; ++i) {
        qkv_grad_split_exp[i] = hemmer_->repackCC(qkv_grad_split_exp[i]); // 4
    }
    printElapsedTime("repack");

    // 12
    for (u64 i = 0; i < 3; ++i) {
        hemmer_->bootstrap2(qkv_grad_split_exp[i * 2],
                            qkv_grad_split_exp[i * 2 + 1]);
    }
    printElapsedTime("BTS");

    /* for (u64 i = 0 ; i < 6 ; ++i) {
        hemmer_->backwarddropoutExpInplace(qkv_grad_split_exp[i], "atn_exp",
    layer_n_, i); // 12 -> 11
    }
    printElapsedTime("back_dropout"); */

    // 12 -> 11
    hemmer_->backwardexpVectorInplace(qkv_grad_split_exp, layer_n_);
    printElapsedTime("back exp");

    // 11 -> 10
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->getEval().mult(qkv_grad_split_exp[i].get(), -1.0 / 16,
                                qkv_grad_split_exp[i].get());
    }

    std::vector<CtxtTensor> grad_q_tmp;
    std::vector<CtxtTensor> grad_k_tmp;
    grad_q_tmp = qkv_grad_split_exp; // 10
    grad_k_tmp = qkv_grad_split_exp;

    start = std::chrono::high_resolution_clock::now();
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->transposeInplace(grad_k_tmp[i]); // 10 -> 9 (6 -> 5)
    }
    printElapsedTime("transpose");

    std::vector<CtxtTensor> grad_exp_q;
    std::vector<CtxtTensor> grad_exp_k;
    grad_exp_k = qkv_grad_split_exp; // 10
    grad_exp_q = grad_k_tmp;         // 9
    qkv_grad_split_exp.clear();

    // grad_q = grad@I^t x 2res_q + grad@res_k
    // grad@I^t x 2res_q part
    start = std::chrono::high_resolution_clock::now();
    for (u64 i = 0; i < 3; ++i) {
        hemmer_->tr_oneMatRotSumInplace(grad_q_tmp[i * 2],
                                        grad_q_tmp[i * 2 + 1], 5); // 9-> 8
        hemmer_->tr_oneMatRotSumInplace(
            grad_k_tmp[i * 2], grad_k_tmp[i * 2 + 1], 5); // 9 -> 8 (5 -> 4)
        for (u64 j = 0; j < 2; ++j) {
            auto tmp = lora_module_->getCtxtTensor("forward_res_q", 0, 0,
                                                   i * 2 + j); // 9
            hemmer_->getEval().mult(grad_q_tmp[i * 2 + j].get(), tmp.get(),
                                    grad_q_tmp[i * 2 + j].get()); // 8

            tmp = lora_module_->getCtxtTensor("forward_res_k", 0, 0,
                                              i * 2 + j); // 10
            hemmer_->getEval().mult(grad_k_tmp[i * 2 + j].get(), tmp.get(),
                                    grad_k_tmp[i * 2 + j].get()); // 9 (4)
        }
    }
    printElapsedTime("tr and hada");

    // grad@res_k part
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingInplace(grad_exp_k[i]);
        auto tmp = lora_module_->getCtxtTensor("forward_res_k", 0, 0, i); // 10
        hemmer_->complexPackingRowInplace(tmp);                           // 9
        grad_exp_k[i] = hemmer_->packedMatMul(grad_exp_k[i], tmp);        //  4
    }
    printElapsedTime("(CC) mm");

    for (u64 i = 0; i < 6; ++i) {
        grad_exp_k[i] = hemmer_->repackCC(grad_exp_k[i]);
    }
    printElapsedTime("repack");

    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingInplace(grad_exp_q[i]);
        auto tmp = lora_module_->getCtxtTensor("forward_res_q", 0, 0, i);
        grad_exp_q[i] = hemmer_->packedMatMul(grad_exp_q[i], tmp); // 4
    }
    printElapsedTime("(CC) mm");

    for (u64 i = 0; i < 6; ++i) {
        grad_exp_q[i] = hemmer_->repackCC(grad_exp_q[i]);
    }
    printElapsedTime("repack");

    // summation part.
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->getEval().sub(grad_q_tmp[i].get(), grad_exp_k[i].get(),
                               grad_q_tmp[i].get());
        hemmer_->getEval().mult(grad_q_tmp[i].get(), 2.0,
                                grad_q_tmp[i].get()); // 4

        hemmer_->getEval().sub(grad_k_tmp[i].get(), grad_exp_q[i].get(),
                               grad_k_tmp[i].get());
        hemmer_->getEval().mult(grad_k_tmp[i].get(), 2.0,
                                grad_k_tmp[i].get()); // 4
    }
    printElapsedTime("hadamult");
    grad_exp_k.clear();
    grad_exp_q.clear();

    // collecting: q1 0 q3 0 + 0 q2 0 q4 >> q1 q2 q3 q4
    // grad_q/k output : 4
    std::vector<CtxtTensor> grad_q;
    std::vector<CtxtTensor> grad_k;
    grad_q.reserve(3);
    grad_k.reserve(3);
    for (u64 i = 0; i < 3; ++i) {
        grad_q.push_back(grad_q_tmp[0]);
        grad_k.push_back(grad_k_tmp[0]);
        hemmer_->getEval().add(grad_q_tmp[i * 2].get(),
                               grad_q_tmp[i * 2 + 1].get(),
                               grad_q[i].get()); // 4
        hemmer_->getEval().add(grad_k_tmp[i * 2].get(),
                               grad_k_tmp[i * 2 + 1].get(),
                               grad_k[i].get()); // 4
    }
    grad_q_tmp.clear();
    grad_k_tmp.clear();

    // direction of value gradient
    std::vector<CtxtTensor> grad_v_tmp;
    grad_v_tmp.reserve(6);
    grad_v_tmp = qkv_grad_split; // 11

    // output: 4
    start = std::chrono::high_resolution_clock::now();
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingRowInplace(grad_v_tmp[i]);
        CtxtTensor tmp =
            lora_module_->getCtxtTensor("forward_res_exp", 0, 0, i); // 8
        hemmer_->transposeInplace(tmp);                              // 7
        hemmer_->complexPackingInplace(tmp);                         // 6
        grad_v_tmp[i] = hemmer_->packedMatMul(tmp, grad_v_tmp[i]);
    }
    printElapsedTime("(CC) exp mm");

    for (u64 i = 0; i < 6; ++i) {
        grad_v_tmp[i] = hemmer_->repackCC(grad_v_tmp[i]);
    }
    printElapsedTime("repackCC");
    qkv_grad_split.clear();

    std::vector<CtxtTensor> grad_v;
    grad_v.reserve(3);
    for (u64 i = 0; i < 3; ++i) {
        grad_v.push_back(grad_v_tmp[i]);
        hemmer_->getEval().add(grad_v_tmp[i * 2].get(),
                               grad_v_tmp[i * 2 + 1].get(),
                               grad_v[i].get()); // 4
    }
    grad_v_tmp.clear();

    // 12
    start = std::chrono::high_resolution_clock::now();
    hemmer_->bootstrap2(grad_q[0], grad_q[1]);
    hemmer_->bootstrap2(grad_q[2], grad_k[2]);
    hemmer_->bootstrap2(grad_k[0], grad_k[1]);
    hemmer_->bootstrap2(grad_v[0], grad_v[1]);
    hemmer_->bootstrap(grad_v[2]);
    printElapsedTime("BTS");

    // input gradient
    start = std::chrono::high_resolution_clock::now();
    std::vector<CtxtTensor> output;
    output.reserve(6);
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(hemmer_->complexPacking(grad_q[0], grad_q[1]),
                               tmp);
        } else {
            hemmer_->matMulPre(grad_q[2], tmp);
        }

        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("tr_wq", i, j);
            auto res = hemmer_->matMulReUse(tmp, weight);
            if (i == 0) {
                output.push_back(res);
            } else {
                hemmer_->addInplace(output[j], res);
            }
        }
    }
    printElapsedTime("(PC) mm ");

    if (lora_type.find('q') == std::string::npos) {
        grad_q.clear();
    }

    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(hemmer_->complexPacking(grad_k[0], grad_k[1]),
                               tmp);
        } else {
            hemmer_->matMulPre(grad_k[2], tmp);
        }
        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("tr_wk", i, j);
            auto res = hemmer_->matMulReUse(tmp, weight);

            hemmer_->addInplace(output[j], res);
        }
    }
    printElapsedTime("(PC) mm ");

    if (lora_type.find('k') == std::string::npos) {
        grad_k.clear();
    }

    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(hemmer_->complexPacking(grad_v[0], grad_v[1]),
                               tmp);
        } else {
            hemmer_->matMulPre(grad_v[2], tmp);
        }

        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("tr_wv", i, j);
            auto res = hemmer_->matMulReUse(tmp, weight);

            hemmer_->addInplace(output[j], res);
        }
    }
    printElapsedTime("(PC) mm ");

    if (lora_type.find('v') == std::string::npos) {
        grad_v.clear();
    }

    std::vector<CtxtTensor> cur;
    cur.reserve(3);
    // output: 11
    for (u64 i = 0; i < 3; ++i) {
        auto repacked = hemmer_->repack(output[i * 2], output[i * 2 + 1]);
        cur.push_back(repacked);
    }
    output.clear();
    printElapsedTime("repack ");

    // lora gradient
    // consider accumulation step.
    // input cur level = 11
    start = std::chrono::high_resolution_clock::now();
    for (const char t : lora_type) {
        const std::string lora_t = std::string(1, t);
        std::vector<HELLM::CtxtTensor> tmp_grad;
        switch (t) {
        case 'q':
            tmp_grad = std::move(grad_q);
            break;
        case 'k':
            tmp_grad = std::move(grad_k);
            break;
        case 'v':
            tmp_grad = std::move(grad_v);
            break;
        default:
            std::cout << "error! unsupported lora type" << std::endl;
        }
        auto tr_lora_wb =
            lora_module_->getCtxtTensor_lora("lora_wb_" + lora_t, 0, 0, 0);
        hemmer_->getEval().levelDown(tr_lora_wb.get(), 9, tr_lora_wb.get());

        // split weights
        std::vector<std::vector<CtxtTensor>> tr_lora_b_weight;
        tr_lora_b_weight.reserve(3);
        std::vector<CtxtTensor> grad_lora_b_out;
        grad_lora_b_out.reserve(2);

        for (u64 j = 0; j < 2; ++j) {
            grad_lora_b_out.emplace_back(tr_lora_wb);
        }

        auto tmp = tr_lora_wb;

        hemmer_->maskFirstRowInplace(grad_lora_b_out[0]);
        hemmer_->getEval().leftRotate(grad_lora_b_out[1].get(), 1 * 256,
                                      grad_lora_b_out[1].get());
        hemmer_->maskFirstRowInplace(grad_lora_b_out[1]);
        tr_lora_b_weight.emplace_back(grad_lora_b_out);
        for (u64 i = 1; i < 3; ++i) {
            grad_lora_b_out.clear();
            hemmer_->getEval().leftRotate(tr_lora_wb.get(), i * 2 * 256,
                                          tmp.get());
            for (u64 j = 0; j < 2; ++j) {
                grad_lora_b_out.emplace_back(tmp);
            }
            hemmer_->maskFirstRowInplace(grad_lora_b_out[0]);
            hemmer_->getEval().leftRotate(grad_lora_b_out[1].get(), 1 * 256,
                                          grad_lora_b_out[1].get());
            hemmer_->maskFirstRowInplace(grad_lora_b_out[1]);
            tr_lora_b_weight.emplace_back(grad_lora_b_out);
        }

        // repeated packing
        for (u64 i = 0; i < 3; ++i) {
            for (u64 j = 0; j < 2; ++j) {
                for (u64 rot = 1; rot < 128; rot <<= 1) {
                    hemmer_->getEval().rightRotate(tr_lora_b_weight[i][j].get(),
                                                   rot * 256, tmp.get());
                    hemmer_->addInplace(tr_lora_b_weight[i][j], tmp);
                }
            }
        }

        // hadamult
        hemmer_->getEval().mult(tmp_grad[0].get(), tr_lora_b_weight[0][0].get(),
                                grad_lora_b_out[0].get());
        hemmer_->getEval().mult(tmp_grad[0].get(), tr_lora_b_weight[0][1].get(),
                                grad_lora_b_out[1].get());
        for (u64 i = 1; i < 3; ++i) {
            hemmer_->getEval().mult(tmp_grad[i].get(),
                                    tr_lora_b_weight[i][0].get(), tmp.get());
            hemmer_->addInplace(grad_lora_b_out[0], tmp);
            hemmer_->getEval().mult(tmp_grad[i].get(),
                                    tr_lora_b_weight[i][1].get(), tmp.get());
            hemmer_->addInplace(grad_lora_b_out[1], tmp);
        }

        // addition
        for (u64 i = 0; i < 2; ++i) {
            for (u64 rot = 1; rot < 256; rot <<= 1) {
                hemmer_->getEval().leftRotate(grad_lora_b_out[i].get(), rot,
                                              tmp.get());
                hemmer_->addInplace(grad_lora_b_out[i], tmp);
            }
            hemmer_->maskFirstColOnlyInplace(grad_lora_b_out[i]);
        }
        tr_lora_b_weight.clear();
        // printElapsedTime("(CC) lora_wb mm");

        // hard coding
        auto tr_lora_wa =
            lora_module_->getCtxtTensor_lora("lora_wa_" + lora_t, 0, 0, 0);
        hemmer_->getEval().levelDown(tr_lora_wa.get(), 6, tr_lora_wa.get());

        // split weights
        std::vector<std::vector<CtxtTensor>> tr_lora_a_weight;
        tr_lora_a_weight.reserve(3);
        std::vector<CtxtTensor> tmp_vector;
        tmp_vector.reserve(2);

        for (u64 i = 0; i < 2; ++i) {
            tmp_vector.emplace_back(tr_lora_wa);
        }
        hemmer_->maskFirstRowInplace(tmp_vector[0]);
        hemmer_->getEval().leftRotate(tmp_vector[1].get(), 1 * 256,
                                      tmp_vector[1].get());
        hemmer_->maskFirstRowInplace(tmp_vector[1]);
        tr_lora_a_weight.emplace_back(tmp_vector);

        for (u64 i = 1; i < 3; ++i) {
            tmp_vector.clear();
            hemmer_->getEval().leftRotate(tr_lora_wa.get(), i * 2 * 256,
                                          tmp.get());
            for (u64 j = 0; j < 2; ++j) {
                tmp_vector.emplace_back(tmp);
            }
            hemmer_->maskFirstRowInplace(tmp_vector[0]);
            hemmer_->getEval().leftRotate(tmp_vector[1].get(), 1 * 256,
                                          tmp_vector[1].get());
            hemmer_->maskFirstRowInplace(tmp_vector[1]);

            tr_lora_a_weight.emplace_back(tmp_vector);
        }
        tmp_vector.clear();

        // repeated packing
        for (u64 i = 0; i < 2; ++i) {
            for (u64 rot = 1; rot < 256; rot <<= 1) {
                hemmer_->getEval().rightRotate(grad_lora_b_out[i].get(), rot,
                                               tmp.get());
                hemmer_->addInplace(grad_lora_b_out[i], tmp);
            }
        }

        for (u64 i = 0; i < 3; ++i) {
            for (u64 j = 0; j < 2; ++j) {
                for (u64 rot = 1; rot < 128; rot <<= 1) {
                    hemmer_->getEval().rightRotate(tr_lora_a_weight[i][j].get(),
                                                   rot * 256, tmp.get());
                    hemmer_->addInplace(tr_lora_a_weight[i][j], tmp);
                }
            }
        }

        std::vector<CtxtTensor> grad_lora_a_out;
        grad_lora_a_out.reserve(3);
        for (u64 i = 0; i < 3; ++i) {
            grad_lora_a_out.emplace_back(tr_lora_wa);
        }

        // hadamult & addition
        for (u64 i = 0; i < 3; ++i) {
            hemmer_->getEval().mult(grad_lora_b_out[0].get(),
                                    tr_lora_a_weight[i][0].get(),
                                    grad_lora_a_out[i].get());
            hemmer_->getEval().mult(grad_lora_b_out[1].get(),
                                    tr_lora_a_weight[i][1].get(), tmp.get());
            hemmer_->addInplace(grad_lora_a_out[i], tmp);

            hemmer_->addInplace(cur[i], grad_lora_a_out[i]);
        }
        grad_lora_a_out.clear();
        // printElapsedTime("backward_atn_lora_a");

        // compute lora grad w
        std::vector<CtxtTensor> tr_lora_in_b;
        tr_lora_in_b.reserve(2);
        for (u64 i = 0; i < 2; ++i) {
            tr_lora_in_b.emplace_back(
                lora_module_->getCtxtTensor("tr_lora_in_b_" + lora_t, 0, 0, i));
        }

        // repeated packing
        for (u64 i = 0; i < 2; ++i) {
            for (u64 rot = 1; rot < 256; rot <<= 1) {
                hemmer_->getEval().rightRotate(tr_lora_in_b[i].get(), rot,
                                               tmp.get());
                hemmer_->addInplace(tr_lora_in_b[i], tmp);
            }
        }

        // hadamult
        std::vector<std::vector<CtxtTensor>> grad_wb_tmp;
        grad_wb_tmp.reserve(2);
        for (u64 i = 0; i < 2; ++i) {
            grad_wb_tmp.emplace_back(tmp_grad);
            for (u64 j = 0; j < 3; ++j) {
                hemmer_->getEval().mult(tr_lora_in_b[i].get(),
                                        tmp_grad[j].get(),
                                        grad_wb_tmp[i][j].get());
            }
        }

        // collect results
        for (u64 i = 0; i < 2; ++i) {
            for (u64 j = 0; j < 3; ++j) {
                for (u64 rot = 1; rot < 128; rot <<= 1) {
                    hemmer_->getEval().leftRotate(grad_wb_tmp[i][j].get(),
                                                  rot * 256, tmp.get());
                    hemmer_->addInplace(grad_wb_tmp[i][j], tmp);
                }
                hemmer_->maskFirstRowInplace(grad_wb_tmp[i][j]);
            }
        }

        auto grad_lora_wb = grad_wb_tmp[0][0];
        hemmer_->getEval().rightRotate(grad_wb_tmp[1][0].get(), 1 * 256,
                                       tmp.get());
        hemmer_->addInplace(grad_lora_wb, tmp);
        for (u64 j = 1; j < 3; ++j) {
            hemmer_->getEval().rightRotate(grad_wb_tmp[1][j].get(), 1 * 256,
                                           tmp.get());
            hemmer_->addInplace(grad_wb_tmp[0][j], tmp);

            hemmer_->getEval().rightRotate(grad_wb_tmp[0][j].get(), 2 * j * 256,
                                           grad_wb_tmp[0][j].get());
            hemmer_->addInplace(grad_lora_wb, grad_wb_tmp[0][j]);
        }
        grad_wb_tmp.clear();
        // printElapsedTime("(CC) lora wegith grad mm");

        std::vector<CtxtTensor> tr_lora_in_a;
        tr_lora_in_a.reserve(3);
        for (u64 i = 0; i < 3; ++i) {
            tr_lora_in_a.emplace_back(
                lora_module_->getCtxtTensor("tr_lora_in_a", 0, 0, i));
        }

        // repeated packing -> alreday complted in previous

        // hadamult
        std::vector<std::vector<CtxtTensor>> grad_wa_tmp;
        grad_wa_tmp.reserve(3);
        for (u64 i = 0; i < 3; ++i) {
            grad_wa_tmp.emplace_back(grad_lora_b_out);
            for (u64 j = 0; j < 2; ++j) {
                hemmer_->getEval().mult(tr_lora_in_a[i].get(),
                                        grad_lora_b_out[j].get(),
                                        grad_wa_tmp[i][j].get());
            }
        }

        // collect results
        for (u64 i = 0; i < 3; ++i) {
            for (u64 j = 0; j < 2; ++j) {
                for (u64 rot = 1; rot < 128; rot <<= 1) {
                    hemmer_->getEval().leftRotate(grad_wa_tmp[i][j].get(),
                                                  rot * 256, tmp.get());
                    hemmer_->addInplace(grad_wa_tmp[i][j], tmp);
                }
                hemmer_->maskFirstRowInplace(grad_wa_tmp[i][j]);
            }
        }

        auto grad_lora_wa = grad_wa_tmp[0][0];
        hemmer_->getEval().rightRotate(grad_wa_tmp[0][1].get(), 1 * 256,
                                       tmp.get());
        hemmer_->addInplace(grad_lora_wa, tmp);
        for (u64 j = 1; j < 3; ++j) {
            hemmer_->getEval().rightRotate(grad_wa_tmp[j][1].get(), 1 * 256,
                                           grad_wa_tmp[j][1].get());
            hemmer_->addInplace(grad_wa_tmp[j][0], tmp);

            hemmer_->getEval().rightRotate(grad_wa_tmp[j][0].get(), 2 * j * 256,
                                           grad_wa_tmp[j][0].get());
            hemmer_->addInplace(grad_lora_wa, grad_wa_tmp[j][0]);
        }
        // printElapsedTime("(CC) lora wegith grad mm and repack");

        HEaaN::CudaTools::cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        if (max_rank > 1) {
            allReduceWrapper(grad_lora_wb);
            allReduceWrapper(grad_lora_wa);

            if (rank == 0) {
                hemmer_->getEval().modReduct(grad_lora_wb.get());
                hemmer_->getEval().modReduct(grad_lora_wa.get());

                hemmer_->bootstrap2(grad_lora_wa, grad_lora_wb);

                hemmer_->getEval().mult(grad_lora_wb.get(),
                                        1.0 / ModelArgs::BATCH_SIZE,
                                        grad_lora_wb.get());
                hemmer_->getEval().mult(grad_lora_wa.get(),
                                        1.0 / ModelArgs::BATCH_SIZE,
                                        grad_lora_wa.get());

                lora_module_->saveAggGrad(grad_lora_wb, "b_" + lora_t);
                lora_module_->saveAggGrad(grad_lora_wa, "a_" + lora_t);
            }
        }
    }
    printElapsedTime("lora CC backward");

    // output cur level = 4
    // printing(cur);
    start = std::chrono::high_resolution_clock::now();
    hemmer_->bootstrap2(cur[0], cur[1]);
    hemmer_->bootstrap(cur[2]);
    printElapsedTime("BTS");

    for (u64 i = 0; i < 3; ++i) {
        const auto &weight = getWeight("norm1_w", i);
        hemmer_->hadamardMultInplace(cur[i], weight);
    }

    // printing(cur);
    //  11 -> 4
    start = std::chrono::high_resolution_clock::now();
    cur = hemmer_->backwardLayerNorm(cur, "atn", layer_n_);
    printElapsedTime("back_LN");

    // 4 -> 4
    for (u64 i = 0; i < 3; ++i) {
        hemmer_->addInplace(grad_y[i], cur[i]);
    }
    cur.clear();
}

// in: 12 , out: 5
void TransformerBlock::backwardattention2_bert_time(
    std::vector<CtxtTensor> &grad_y, const std::string &lora_type) {

    auto rank = hemmer_->getRank();
    auto max_rank = hemmer_->getMaxRank();

    std::vector<CtxtTensor> grad_y_tmp;
    grad_y_tmp.reserve(3);
    grad_y_tmp = grad_y;

    start = std::chrono::high_resolution_clock::now();
    for (u64 i = 0; i < 3; ++i) {
        hemmer_->backwarddropoutInplace(grad_y_tmp[i], "atn_output", layer_n_,
                                        i);
    }
    printElapsedTime("back_dropout");

    start = std::chrono::high_resolution_clock::now();
    std::vector<CtxtTensor> tmp_grad_;
    tmp_grad_.reserve(6);
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(
                hemmer_->complexPacking(grad_y_tmp[0], grad_y_tmp[1]), tmp);
        } else {
            hemmer_->matMulPre(grad_y_tmp[2], tmp);
        }

        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("tr_wd", i, j);
            auto result = hemmer_->matMulReUse(tmp, weight);

            if (i == 0) {
                tmp_grad_.push_back(result);
            } else {
                hemmer_->addInplace(tmp_grad_[j], result);
            }
        }
    }
    grad_y_tmp.clear();
    printElapsedTime("(PC) tr_wd mm");

    start = std::chrono::high_resolution_clock::now();
    std::vector<CtxtTensor> qkv_grad;
    qkv_grad.reserve(3);
    for (u64 i = 0; i < 3; ++i) {
        qkv_grad.push_back(
            hemmer_->repack(tmp_grad_[i * 2], tmp_grad_[i * 2 + 1]));
    }
    tmp_grad_.clear();
    printElapsedTime("repack");

    std::vector<CtxtTensor> qkv_grad_split;
    qkv_grad_split.reserve(6);
    // 11 -> 10
    for (u64 i = 0; i < 3; ++i) {
        for (u64 j = 0; j < 2; ++j)
            qkv_grad_split.push_back(qkv_grad[i]);

        hemmer_->maskRightLeft(qkv_grad[i], qkv_grad_split[i * 2],
                               qkv_grad_split[i * 2 + 1]);
    }
    qkv_grad.clear();

    std::vector<CtxtTensor> qkv_grad_split_exp;
    qkv_grad_split_exp = qkv_grad_split;

    // direcition of exp gradient
    // 9 -> 4
    start = std::chrono::high_resolution_clock::now();
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingInplace(qkv_grad_split_exp[i]); // 9
        // should consider saved ctxtTensor structure.
        CtxtTensor tmp =
            lora_module_->getCtxtTensor("forward_res_v", 0, 0, i); // 8
        hemmer_->transposeInplace(tmp);                            // 7
        hemmer_->complexPackingRowInplace(tmp);

        qkv_grad_split_exp[i] =
            hemmer_->packedMatMul(qkv_grad_split_exp[i], tmp); // 4
    }
    printElapsedTime("(CC) res_v mm");

    for (u64 i = 0; i < 6; ++i) {
        qkv_grad_split_exp[i] = hemmer_->repackCC(qkv_grad_split_exp[i]); // 4
    }
    printElapsedTime("repack");

    // 12
    for (u64 i = 0; i < 3; ++i) {
        hemmer_->bootstrap2(qkv_grad_split_exp[i * 2],
                            qkv_grad_split_exp[i * 2 + 1]);
    }
    printElapsedTime("BTS");

    for (u64 i = 0; i < 6; ++i) {
        hemmer_->backwarddropoutExpInplace(qkv_grad_split_exp[i], "atn_exp",
                                           layer_n_, i); // 12 -> 11
    }
    printElapsedTime("back_dropout");

    // 12 -> 11
    hemmer_->backwardexpVectorInplace(qkv_grad_split_exp, layer_n_);
    printElapsedTime("back exp");

    // 11 -> 10
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->getEval().mult(qkv_grad_split_exp[i].get(), -1.0 / 16,
                                qkv_grad_split_exp[i].get());
    }

    std::vector<CtxtTensor> grad_q_tmp;
    std::vector<CtxtTensor> grad_k_tmp;
    grad_q_tmp = qkv_grad_split_exp; // 10
    grad_k_tmp = qkv_grad_split_exp;

    start = std::chrono::high_resolution_clock::now();
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->transposeInplace(grad_k_tmp[i]); // 10 -> 9 (6 -> 5)
    }
    printElapsedTime("transpose");

    std::vector<CtxtTensor> grad_exp_q;
    std::vector<CtxtTensor> grad_exp_k;
    grad_exp_k = qkv_grad_split_exp; // 10
    grad_exp_q = grad_k_tmp;         // 9
    qkv_grad_split_exp.clear();

    // grad_q = grad@I^t x 2res_q + grad@res_k
    // grad@I^t x 2res_q part
    start = std::chrono::high_resolution_clock::now();
    for (u64 i = 0; i < 3; ++i) {
        hemmer_->tr_oneMatRotSumInplace(grad_q_tmp[i * 2],
                                        grad_q_tmp[i * 2 + 1], 5); // 9-> 8
        hemmer_->tr_oneMatRotSumInplace(
            grad_k_tmp[i * 2], grad_k_tmp[i * 2 + 1], 5); // 9 -> 8 (5 -> 4)
        for (u64 j = 0; j < 2; ++j) {
            auto tmp = lora_module_->getCtxtTensor("forward_res_q", 0, 0,
                                                   i * 2 + j); // 9
            hemmer_->getEval().mult(grad_q_tmp[i * 2 + j].get(), tmp.get(),
                                    grad_q_tmp[i * 2 + j].get()); // 8

            tmp = lora_module_->getCtxtTensor("forward_res_k", 0, 0,
                                              i * 2 + j); // 10
            hemmer_->getEval().mult(grad_k_tmp[i * 2 + j].get(), tmp.get(),
                                    grad_k_tmp[i * 2 + j].get()); // 9 (4)
        }
    }
    printElapsedTime("tr and hada");

    // grad@res_k part
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingInplace(grad_exp_k[i]);
        auto tmp = lora_module_->getCtxtTensor("forward_res_k", 0, 0, i); // 10
        hemmer_->complexPackingRowInplace(tmp);                           // 9
        grad_exp_k[i] = hemmer_->packedMatMul(grad_exp_k[i], tmp);        //  4
    }
    printElapsedTime("(CC) mm");

    for (u64 i = 0; i < 6; ++i) {
        grad_exp_k[i] = hemmer_->repackCC(grad_exp_k[i]);
    }
    printElapsedTime("repack");

    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingInplace(grad_exp_q[i]);
        auto tmp = lora_module_->getCtxtTensor("forward_res_q", 0, 0, i);
        grad_exp_q[i] = hemmer_->packedMatMul(grad_exp_q[i], tmp); // 4
    }
    printElapsedTime("(CC) mm");

    for (u64 i = 0; i < 6; ++i) {
        grad_exp_q[i] = hemmer_->repackCC(grad_exp_q[i]);
    }
    printElapsedTime("repack");

    // summation part.
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->getEval().sub(grad_q_tmp[i].get(), grad_exp_k[i].get(),
                               grad_q_tmp[i].get());
        hemmer_->getEval().mult(grad_q_tmp[i].get(), 2.0,
                                grad_q_tmp[i].get()); // 4

        hemmer_->getEval().sub(grad_k_tmp[i].get(), grad_exp_q[i].get(),
                               grad_k_tmp[i].get());
        hemmer_->getEval().mult(grad_k_tmp[i].get(), 2.0,
                                grad_k_tmp[i].get()); // 4
    }
    printElapsedTime("hadamult");
    grad_exp_k.clear();
    grad_exp_q.clear();

    // collecting: q1 0 q3 0 + 0 q2 0 q4 >> q1 q2 q3 q4
    // grad_q/k output : 4
    std::vector<CtxtTensor> grad_q;
    std::vector<CtxtTensor> grad_k;
    grad_q.reserve(3);
    grad_k.reserve(3);
    for (u64 i = 0; i < 3; ++i) {
        grad_q.push_back(grad_q_tmp[0]);
        grad_k.push_back(grad_k_tmp[0]);
        hemmer_->getEval().add(grad_q_tmp[i * 2].get(),
                               grad_q_tmp[i * 2 + 1].get(),
                               grad_q[i].get()); // 4
        hemmer_->getEval().add(grad_k_tmp[i * 2].get(),
                               grad_k_tmp[i * 2 + 1].get(),
                               grad_k[i].get()); // 4
    }
    grad_q_tmp.clear();
    grad_k_tmp.clear();

    // direction of value gradient
    std::vector<CtxtTensor> grad_v_tmp;
    grad_v_tmp.reserve(6);
    grad_v_tmp = qkv_grad_split; // 11

    // output: 4
    start = std::chrono::high_resolution_clock::now();
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingRowInplace(grad_v_tmp[i]);
        CtxtTensor tmp =
            lora_module_->getCtxtTensor("forward_res_exp", 0, 0, i); // 8
        hemmer_->transposeInplace(tmp);                              // 7
        hemmer_->complexPackingInplace(tmp);                         // 6
        grad_v_tmp[i] = hemmer_->packedMatMul(tmp, grad_v_tmp[i]);
    }
    printElapsedTime("(CC) exp mm");

    for (u64 i = 0; i < 6; ++i) {
        grad_v_tmp[i] = hemmer_->repackCC(grad_v_tmp[i]);
    }
    printElapsedTime("repack");
    qkv_grad_split.clear();

    std::vector<CtxtTensor> grad_v;
    grad_v.reserve(3);
    for (u64 i = 0; i < 3; ++i) {
        grad_v.push_back(grad_v_tmp[i]);
        hemmer_->getEval().add(grad_v_tmp[i * 2].get(),
                               grad_v_tmp[i * 2 + 1].get(),
                               grad_v[i].get()); // 4
    }
    grad_v_tmp.clear();

    // 12
    start = std::chrono::high_resolution_clock::now();
    hemmer_->bootstrap2(grad_q[0], grad_q[1]);
    hemmer_->bootstrap2(grad_q[2], grad_k[2]);
    hemmer_->bootstrap2(grad_k[0], grad_k[1]);
    hemmer_->bootstrap2(grad_v[0], grad_v[1]);
    hemmer_->bootstrap(grad_v[2]);
    printElapsedTime("BTS");

    // input gradient
    start = std::chrono::high_resolution_clock::now();
    std::vector<CtxtTensor> output;
    output.reserve(6);
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(hemmer_->complexPacking(grad_q[0], grad_q[1]),
                               tmp);
        } else {
            hemmer_->matMulPre(grad_q[2], tmp);
        }

        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("tr_wq", i, j);
            auto res = hemmer_->matMulReUse(tmp, weight);
            if (i == 0) {
                output.push_back(res);
            } else {
                hemmer_->addInplace(output[j], res);
            }
        }
    }
    printElapsedTime("(PC) mm ");

    if (lora_type.find('q') == std::string::npos) {
        grad_q.clear();
    }

    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(hemmer_->complexPacking(grad_k[0], grad_k[1]),
                               tmp);
        } else {
            hemmer_->matMulPre(grad_k[2], tmp);
        }
        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("tr_wk", i, j);
            auto res = hemmer_->matMulReUse(tmp, weight);

            hemmer_->addInplace(output[j], res);
        }
    }
    printElapsedTime("(PC) mm ");

    if (lora_type.find('k') == std::string::npos) {
        grad_k.clear();
    }

    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(hemmer_->complexPacking(grad_v[0], grad_v[1]),
                               tmp);
        } else {
            hemmer_->matMulPre(grad_v[2], tmp);
        }

        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("tr_wv", i, j);
            auto res = hemmer_->matMulReUse(tmp, weight);

            hemmer_->addInplace(output[j], res);
        }
    }
    printElapsedTime("(PC) mm ");

    if (lora_type.find('v') == std::string::npos) {
        grad_v.clear();
    }

    std::vector<CtxtTensor> cur;
    cur.reserve(3);
    // output: 11
    for (u64 i = 0; i < 3; ++i) {
        auto repacked = hemmer_->repack(output[i * 2], output[i * 2 + 1]);
        cur.push_back(repacked);
    }
    output.clear();
    printElapsedTime("repack ");

    // lora gradient
    // consider accumulation step.
    // input cur level = 11
    for (const char t : lora_type) {
        start = std::chrono::high_resolution_clock::now();
        const std::string lora_t = std::string(1, t);
        std::vector<HELLM::CtxtTensor> tmp_grad;
        switch (t) {
        case 'q':
            tmp_grad = std::move(grad_q);
            break;
        case 'k':
            tmp_grad = std::move(grad_k);
            break;
        case 'v':
            tmp_grad = std::move(grad_v);
            break;
        default:
            std::cout << "error! unsupported lora type" << std::endl;
        }

        start = std::chrono::high_resolution_clock::now();
        auto tr_lora_wb = lora_module_->getCtxtTensor_lora("lora_wb_" + lora_t,
                                                           0, 0, 0); // need: 9
        hemmer_->transposeInplace(tr_lora_wb);                       // 1
        auto grad_lora_b = hemmer_->matMulHighLow(
            tmp_grad[0], hemmer_->getLowColBlock(tr_lora_wb, 0), 0, 5); // 3
        for (u64 i = 1; i < 3; ++i) {
            auto tr_lora_wb_lowcol = hemmer_->getLowColBlock(tr_lora_wb, i);
            auto matmul_res = hemmer_->matMulHighLow(
                tmp_grad[i], tr_lora_wb_lowcol, i, 5); // 1
            hemmer_->addInplace(grad_lora_b, matmul_res);
        }
        printElapsedTime("(CC) lora_wb mm");

        grad_lora_b = hemmer_->repackToOneCol(grad_lora_b, 0);
        hemmer_->bootstrap(grad_lora_b); // 12
        printElapsedTime("lora_wb repack");

        /* if ( lora_t == "q") {
            std::cout << "backward lora_wb mm res" << std::endl;
            printing_masking(grad_lora_b);
        } */

        auto tr_lora_wa =
            lora_module_->getCtxtTensor_lora("lora_wa_" + lora_t, 0, 0, 0);
        hemmer_->transposeInplace(tr_lora_wa); // 1
        for (u64 i = 0; i < 3; ++i) {
            auto tr_lora_wa_lowrow = hemmer_->getLowRowBlock(tr_lora_wa, i);
            auto matmul_res = hemmer_->matMulLowLow(
                grad_lora_b, tr_lora_wa_lowrow, 0, i); // 4

            /* if (lora_t == "q" && i == 0) {
                std::cout << "backward lora_wa mm res" << std::endl;
                printing_masking(matmul_res);
            } */
            hemmer_->addInplace(cur[i], matmul_res); // 4
        }
        printElapsedTime("(CC) lora_wa mm");

        // compute lora grad w
        auto tr_lora_in_b =
            lora_module_->getCtxtTensor("tr_lora_in_b_" + lora_t, 0, 0, 0);
        auto grad_lora_wb = hemmer_->repackToMultiRow(
            hemmer_->matMulLowHigh(tr_lora_in_b, tmp_grad[0], 0, 5), 0); // 5
        for (u64 i = 1; i < 3; ++i) {
            auto matmul_res =
                hemmer_->matMulLowHigh(tr_lora_in_b, tmp_grad[i], 0, 5); // 5
            hemmer_->addInplace(grad_lora_wb,
                                hemmer_->repackToMultiRow(matmul_res, i)); // 5
        }

        /* if( lora_t == "q") {
            std::cout << "lora weight grad wb mm res" << std::endl;
            printing_masking(grad_lora_wb);
        } */

        printElapsedTime("(CC) lora wegith grad mm");

        auto grad_lora_wa = hemmer_->repackToMultiCol(
            hemmer_->matMulHighLow(
                lora_module_->getCtxtTensor("tr_lora_in_a", 0, 0, 0),
                grad_lora_b, 0, 5),
            0); // 5
        for (u64 i = 1; i < 3; ++i) {
            auto ctxt_tensor =
                lora_module_->getCtxtTensor("tr_lora_in_a", 0, 0, i);
            auto matmul_res =
                hemmer_->matMulHighLow(ctxt_tensor, grad_lora_b, 0, 5);
            auto repacked_res = hemmer_->repackToMultiCol(matmul_res, i); // 4
            hemmer_->addInplace(grad_lora_wa, repacked_res);              // 4
        }

        /* if (lora_t == "q") {
            auto tmp = grad_lora_wa;
            hemmer_->transposeInplace(tmp);
            std::cout << "lora weight grad wa mm res" << std::endl;

            printing_masking(tmp);
        } */

        printElapsedTime("(CC) lora wegith grad mm and repack");

        HEaaN::CudaTools::cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        if (max_rank > 1) {
            allReduceWrapper(grad_lora_wb);
            allReduceWrapper(grad_lora_wa);

            if (rank == 0) {
                hemmer_->getEval().modReduct(grad_lora_wb.get());
                hemmer_->getEval().modReduct(grad_lora_wa.get());

                hemmer_->bootstrap2(grad_lora_wa, grad_lora_wb);

                hemmer_->getEval().mult(grad_lora_wb.get(),
                                        1.0 / ModelArgs::BATCH_SIZE,
                                        grad_lora_wb.get());
                hemmer_->getEval().mult(grad_lora_wa.get(),
                                        1.0 / ModelArgs::BATCH_SIZE,
                                        grad_lora_wa.get());

                lora_module_->saveAggGrad(grad_lora_wb, "b_" + lora_t);
                lora_module_->saveAggGrad(grad_lora_wa, "a_" + lora_t);
            }
        }
    }
    // output cur level = 4
    // printing(cur);
    start = std::chrono::high_resolution_clock::now();
    hemmer_->bootstrap2(cur[0], cur[1]);
    hemmer_->bootstrap(cur[2]);
    printElapsedTime("BTS");

    for (u64 i = 0; i < 3; ++i) {
        const auto &weight = getWeight("norm1_w", i);
        hemmer_->hadamardMultInplace(cur[i], weight);
    }

    // printing(cur);
    //  11 -> 4
    start = std::chrono::high_resolution_clock::now();
    cur = hemmer_->backwardLayerNorm(cur, "atn", layer_n_);
    printElapsedTime("back_LN");

    // 4 -> 4
    for (u64 i = 0; i < 3; ++i) {
        hemmer_->addInplace(grad_y[i], cur[i]);
    }
    cur.clear();
}

// in: 12 , out: 5
void TransformerBlock::backwardattention2_bert_SM(
    std::vector<CtxtTensor> &grad_y, const std::string &lora_type) {

    auto rank = hemmer_->getRank();
    auto max_rank = hemmer_->getMaxRank();

    std::vector<CtxtTensor> grad_y_tmp;
    grad_y_tmp.reserve(3);
    grad_y_tmp = grad_y;

    start = std::chrono::high_resolution_clock::now();
    for (u64 i = 0; i < 3; ++i) {
        hemmer_->backwarddropoutInplace(grad_y_tmp[i], "atn_output", layer_n_,
                                        i);
    }
    printElapsedTime("back_dropout");

    start = std::chrono::high_resolution_clock::now();
    std::vector<CtxtTensor> tmp_grad_;
    tmp_grad_.reserve(6);
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(
                hemmer_->complexPacking(grad_y_tmp[0], grad_y_tmp[1]), tmp);
        } else {
            hemmer_->matMulPre(grad_y_tmp[2], tmp);
        }

        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("tr_wd", i, j);
            auto result = hemmer_->matMulReUse(tmp, weight);

            if (i == 0) {
                tmp_grad_.push_back(result);
            } else {
                hemmer_->addInplace(tmp_grad_[j], result);
            }
        }
    }
    grad_y_tmp.clear();
    printElapsedTime("(PC) tr_wd mm");

    start = std::chrono::high_resolution_clock::now();
    std::vector<CtxtTensor> qkv_grad;
    qkv_grad.reserve(3);
    for (u64 i = 0; i < 3; ++i) {
        qkv_grad.push_back(
            hemmer_->repack(tmp_grad_[i * 2], tmp_grad_[i * 2 + 1]));
    }
    tmp_grad_.clear();
    printElapsedTime("repack");

    std::vector<CtxtTensor> qkv_grad_split;
    qkv_grad_split.reserve(6);
    // 11 -> 10
    for (u64 i = 0; i < 3; ++i) {
        for (u64 j = 0; j < 2; ++j)
            qkv_grad_split.push_back(qkv_grad[i]);

        hemmer_->maskRightLeft(qkv_grad[i], qkv_grad_split[i * 2],
                               qkv_grad_split[i * 2 + 1]);
    }
    qkv_grad.clear();

    std::vector<CtxtTensor> qkv_grad_split_exp;
    qkv_grad_split_exp = qkv_grad_split;

    // direcition of exp gradient
    // 9 -> 4
    start = std::chrono::high_resolution_clock::now();
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingInplace(qkv_grad_split_exp[i]); // 9
        // should consider saved ctxtTensor structure.
        CtxtTensor tmp =
            lora_module_->getCtxtTensor("forward_res_v", 0, 0, i); // 8
        hemmer_->transposeInplace(tmp);                            // 7
        hemmer_->complexPackingRowInplace(tmp);

        qkv_grad_split_exp[i] =
            hemmer_->packedMatMul(qkv_grad_split_exp[i], tmp); // 4
    }
    printElapsedTime("(CC) res_v mm");

    for (u64 i = 0; i < 6; ++i) {
        qkv_grad_split_exp[i] = hemmer_->repackCC(qkv_grad_split_exp[i]); // 4
    }
    printElapsedTime("repack");

    // 12
    for (u64 i = 0; i < 3; ++i) {
        hemmer_->bootstrap2(qkv_grad_split_exp[i * 2],
                            qkv_grad_split_exp[i * 2 + 1]);
    }
    printElapsedTime("BTS");

    for (u64 i = 0; i < 6; ++i) {
        hemmer_->backwarddropoutExpInplace(qkv_grad_split_exp[i], "atn_exp",
                                           layer_n_, i); // 12 -> 11
    }
    printElapsedTime("back_dropout");

    // 12 -> 10
    hemmer_->backwardsoftmaxVectorInplaceHETAL(qkv_grad_split_exp, layer_n_);
    printElapsedTime("back SM");

    std::vector<CtxtTensor> grad_q_tmp;
    std::vector<CtxtTensor> grad_k_tmp;
    grad_q_tmp = qkv_grad_split_exp; // 10
    grad_k_tmp = qkv_grad_split_exp;

    start = std::chrono::high_resolution_clock::now();
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->transposeInplace(grad_k_tmp[i]); // 10 -> 9 (6 -> 5)
    }
    printElapsedTime("transpose");

    // grad@res_k part
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingInplace(grad_k_tmp[i]);
        auto tmp = lora_module_->getCtxtTensor("forward_res_k", 0, 0, i); // 10
        hemmer_->complexPackingRowInplace(tmp);                           // 9
        grad_k_tmp[i] = hemmer_->packedMatMul(grad_k_tmp[i], tmp);        //  4
        grad_k_tmp[i] = hemmer_->repackCC(grad_k_tmp[i]);
    }
    printElapsedTime("(CC) res_k mm");

    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingInplace(grad_q_tmp[i]);
        auto tmp = lora_module_->getCtxtTensor("forward_res_q", 0, 0, i);
        grad_q_tmp[i] = hemmer_->packedMatMul(grad_q_tmp[i], tmp); // 4
        grad_q_tmp[i] = hemmer_->repackCC(grad_q_tmp[i]);
    }
    printElapsedTime("(CC) res_q mm");

    // collecting: q1 0 q3 0 + 0 q2 0 q4 >> q1 q2 q3 q4
    // grad_q/k output : 4
    std::vector<CtxtTensor> grad_q;
    std::vector<CtxtTensor> grad_k;
    grad_q.reserve(3);
    grad_k.reserve(3);
    for (u64 i = 0; i < 3; ++i) {
        grad_q.push_back(grad_q_tmp[0]);
        grad_k.push_back(grad_k_tmp[0]);
        hemmer_->getEval().add(grad_q_tmp[i * 2].get(),
                               grad_q_tmp[i * 2 + 1].get(),
                               grad_q[i].get()); // 4
        hemmer_->getEval().add(grad_k_tmp[i * 2].get(),
                               grad_k_tmp[i * 2 + 1].get(),
                               grad_k[i].get()); // 4
    }
    grad_q_tmp.clear();
    grad_k_tmp.clear();

    // direction of value gradient
    std::vector<CtxtTensor> grad_v_tmp;
    grad_v_tmp.reserve(6);
    grad_v_tmp = qkv_grad_split; // 11

    // output: 4
    start = std::chrono::high_resolution_clock::now();
    for (u64 i = 0; i < 6; ++i) {
        hemmer_->complexPackingRowInplace(grad_v_tmp[i]);
        CtxtTensor tmp =
            lora_module_->getCtxtTensor("forward_res_exp", 0, 0, i); // 8
        hemmer_->transposeInplace(tmp);                              // 7
        hemmer_->complexPackingInplace(tmp);                         // 6
        grad_v_tmp[i] = hemmer_->packedMatMul(tmp, grad_v_tmp[i]);
    }
    printElapsedTime("(CC) exp mm");

    for (u64 i = 0; i < 6; ++i) {
        grad_v_tmp[i] = hemmer_->repackCC(grad_v_tmp[i]);
    }
    printElapsedTime("repack");
    qkv_grad_split.clear();

    std::vector<CtxtTensor> grad_v;
    grad_v.reserve(3);
    for (u64 i = 0; i < 3; ++i) {
        grad_v.push_back(grad_v_tmp[i]);
        hemmer_->getEval().add(grad_v_tmp[i * 2].get(),
                               grad_v_tmp[i * 2 + 1].get(),
                               grad_v[i].get()); // 4
    }
    grad_v_tmp.clear();

    // 12
    start = std::chrono::high_resolution_clock::now();
    hemmer_->bootstrap2(grad_q[0], grad_q[1]);
    hemmer_->bootstrap2(grad_q[2], grad_k[2]);
    hemmer_->bootstrap2(grad_k[0], grad_k[1]);
    hemmer_->bootstrap2(grad_v[0], grad_v[1]);
    hemmer_->bootstrap(grad_v[2]);
    printElapsedTime("BTS");

    // input gradient
    start = std::chrono::high_resolution_clock::now();
    std::vector<CtxtTensor> output;
    output.reserve(6);
    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(hemmer_->complexPacking(grad_q[0], grad_q[1]),
                               tmp);
        } else {
            hemmer_->matMulPre(grad_q[2], tmp);
        }

        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("tr_wq", i, j);
            auto res = hemmer_->matMulReUse(tmp, weight);
            if (i == 0) {
                output.push_back(res);
            } else {
                hemmer_->addInplace(output[j], res);
            }
        }
    }
    printElapsedTime("(PC) mm ");

    if (lora_type.find('q') == std::string::npos) {
        grad_q.clear();
    }

    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(hemmer_->complexPacking(grad_k[0], grad_k[1]),
                               tmp);
        } else {
            hemmer_->matMulPre(grad_k[2], tmp);
        }
        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("tr_wk", i, j);
            auto res = hemmer_->matMulReUse(tmp, weight);

            hemmer_->addInplace(output[j], res);
        }
    }
    printElapsedTime("(PC) mm ");

    if (lora_type.find('k') == std::string::npos) {
        grad_k.clear();
    }

    for (u64 i = 0; i < 2; ++i) {
        std::vector<Ciphertext> tmp;
        if (i == 0) {
            hemmer_->matMulPre(hemmer_->complexPacking(grad_v[0], grad_v[1]),
                               tmp);
        } else {
            hemmer_->matMulPre(grad_v[2], tmp);
        }

        for (u64 j = 0; j < 6; ++j) {
            auto weight = getWeight("tr_wv", i, j);
            auto res = hemmer_->matMulReUse(tmp, weight);

            hemmer_->addInplace(output[j], res);
        }
    }
    printElapsedTime("(PC) mm ");

    if (lora_type.find('v') == std::string::npos) {
        grad_v.clear();
    }

    std::vector<CtxtTensor> cur;
    cur.reserve(3);
    // output: 11
    for (u64 i = 0; i < 3; ++i) {
        auto repacked = hemmer_->repack(output[i * 2], output[i * 2 + 1]);
        cur.push_back(repacked);
    }
    output.clear();
    printElapsedTime("repack ");

    // lora gradient
    // consider accumulation step.
    // input cur level = 11
    for (const char t : lora_type) {
        start = std::chrono::high_resolution_clock::now();
        const std::string lora_t = std::string(1, t);
        std::vector<HELLM::CtxtTensor> tmp_grad;
        switch (t) {
        case 'q':
            tmp_grad = std::move(grad_q);
            break;
        case 'k':
            tmp_grad = std::move(grad_k);
            break;
        case 'v':
            tmp_grad = std::move(grad_v);
            break;
        default:
            std::cout << "error! unsupported lora type" << std::endl;
        }

        start = std::chrono::high_resolution_clock::now();
        auto tr_lora_wb = lora_module_->getCtxtTensor_lora("lora_wb_" + lora_t,
                                                           0, 0, 0); // need: 9
        hemmer_->transposeInplace(tr_lora_wb);                       // 1
        auto grad_lora_b = hemmer_->matMulHighLow(
            tmp_grad[0], hemmer_->getLowColBlock(tr_lora_wb, 0), 0, 5); // 3
        for (u64 i = 1; i < 3; ++i) {
            auto tr_lora_wb_lowcol = hemmer_->getLowColBlock(tr_lora_wb, i);
            auto matmul_res = hemmer_->matMulHighLow(
                tmp_grad[i], tr_lora_wb_lowcol, i, 5); // 1
            hemmer_->addInplace(grad_lora_b, matmul_res);
        }
        printElapsedTime("(CC) lora_wb mm");

        grad_lora_b = hemmer_->repackToOneCol(grad_lora_b, 0);
        hemmer_->bootstrap(grad_lora_b); // 12
        printElapsedTime("lora_wb repack");

        auto tr_lora_wa =
            lora_module_->getCtxtTensor_lora("lora_wa_" + lora_t, 0, 0, 0);
        hemmer_->transposeInplace(tr_lora_wa); // 1
        for (u64 i = 0; i < 3; ++i) {
            auto tr_lora_wa_lowrow = hemmer_->getLowRowBlock(tr_lora_wa, i);
            auto matmul_res = hemmer_->matMulLowLow(
                grad_lora_b, tr_lora_wa_lowrow, 0, i); // 4

            hemmer_->addInplace(cur[i], matmul_res); // 4
        }
        printElapsedTime("(CC) lora_wa mm");

        // compute lora grad w
        auto tr_lora_in_b =
            lora_module_->getCtxtTensor("tr_lora_in_b_" + lora_t, 0, 0, 0);
        auto grad_lora_wb = hemmer_->repackToMultiRow(
            hemmer_->matMulLowHigh(tr_lora_in_b, tmp_grad[0], 0, 5), 0); // 5
        for (u64 i = 1; i < 3; ++i) {
            auto matmul_res =
                hemmer_->matMulLowHigh(tr_lora_in_b, tmp_grad[i], 0, 5); // 5
            hemmer_->addInplace(grad_lora_wb,
                                hemmer_->repackToMultiRow(matmul_res, i)); // 5
        }
        printElapsedTime("(CC) lora wegith grad mm");

        auto grad_lora_wa = hemmer_->repackToMultiCol(
            hemmer_->matMulHighLow(
                lora_module_->getCtxtTensor("tr_lora_in_a", 0, 0, 0),
                grad_lora_b, 0, 5),
            0); // 5
        for (u64 i = 1; i < 3; ++i) {
            auto ctxt_tensor =
                lora_module_->getCtxtTensor("tr_lora_in_a", 0, 0, i);
            auto matmul_res =
                hemmer_->matMulHighLow(ctxt_tensor, grad_lora_b, 0, 5);
            auto repacked_res = hemmer_->repackToMultiCol(matmul_res, i); // 4
            hemmer_->addInplace(grad_lora_wa, repacked_res);              // 4
        }
        printElapsedTime("(CC) lora wegith grad mm and repack");

        HEaaN::CudaTools::cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        if (max_rank > 1) {
            allReduceWrapper(grad_lora_wb);
            allReduceWrapper(grad_lora_wa);

            if (rank == 0) {
                hemmer_->getEval().modReduct(grad_lora_wb.get());
                hemmer_->getEval().modReduct(grad_lora_wa.get());

                hemmer_->bootstrap2(grad_lora_wa, grad_lora_wb);

                hemmer_->getEval().mult(grad_lora_wb.get(),
                                        1.0 / ModelArgs::BATCH_SIZE,
                                        grad_lora_wb.get());
                hemmer_->getEval().mult(grad_lora_wa.get(),
                                        1.0 / ModelArgs::BATCH_SIZE,
                                        grad_lora_wa.get());

                lora_module_->saveAggGrad(grad_lora_wb, "b_" + lora_t);
                lora_module_->saveAggGrad(grad_lora_wa, "a_" + lora_t);
            }
        }
    }
    // output cur level = 4
    // printing(cur);
    start = std::chrono::high_resolution_clock::now();
    hemmer_->bootstrap2(cur[0], cur[1]);
    hemmer_->bootstrap(cur[2]);
    printElapsedTime("BTS");

    for (u64 i = 0; i < 3; ++i) {
        const auto &weight = getWeight("norm1_w", i);
        hemmer_->hadamardMultInplace(cur[i], weight);
    }

    // printing(cur);
    //  11 -> 4
    start = std::chrono::high_resolution_clock::now();
    cur = hemmer_->backwardLayerNorm(cur, "atn", layer_n_);
    printElapsedTime("back_LN");

    // 4 -> 4
    for (u64 i = 0; i < 3; ++i) {
        hemmer_->addInplace(grad_y[i], cur[i]);
    }
    cur.clear();
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//////////////////////////////// BOUNDARY /////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

PtxtTensor TransformerBlock::getWeight(const std::string &name, u64 w_index) {
    std::string weight_name = name + "_" + std::to_string(w_index);
    Message msg{getLogFullSlots(hemmer_->getContext())};
    msg.load(path_ + weight_name + ".msg");
    msg.to(getCurrentCudaDevice());
    auto ptxt = hemmer_->getEnDec().encode(msg, ModelArgs::MATRIX_WEIGHT_LEVEL);

    return PtxtTensor{ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM, ptxt,
                      static_cast<HEaaN::i64>(ModelArgs::HEAD_DIM * 2)};
}

PtxtTensor TransformerBlock::getWeight(const std::string &name, u64 h_index,
                                       u64 w_index) {
    std::string weight_name =
        name + "_" + std::to_string(h_index) + "_" + std::to_string(w_index);
    Message msg{getLogFullSlots(hemmer_->getContext())};
    msg.load(path_ + weight_name + ".msg");
    msg.to(getCurrentCudaDevice());
    auto ptxt = hemmer_->getEnDec().encode(msg, ModelArgs::MATRIX_WEIGHT_LEVEL);

    return PtxtTensor{ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM, ptxt,
                      static_cast<HEaaN::i64>(ModelArgs::HEAD_DIM * 2)};
}

const Message &TransformerBlock::getWeightMsg(const std::string &name,
                                              u64 w_index) {
    std::string weight_name = name + "_" + std::to_string(w_index);
    if (weights_.find(weight_name) == weights_.end()) {
        Message msg{getLogFullSlots(hemmer_->getContext())};
        msg.load(path_ + weight_name + ".msg");
        msg.to(getCurrentCudaDevice());
        weights_.insert(std::make_pair(weight_name, msg));
    }
    return weights_[weight_name];
}

const Message &TransformerBlock::getWeightMsg(const std::string &name,
                                              u64 h_index, u64 w_index) {
    std::string weight_name =
        name + "_" + std::to_string(h_index) + "_" + std::to_string(w_index);
    if (weights_.find(weight_name) == weights_.end()) {
        Message msg{getLogFullSlots(hemmer_->getContext())};
        msg.load(path_ + weight_name + ".msg");
        msg.to(getCurrentCudaDevice());
        weights_.insert(std::make_pair(weight_name, msg));
    }
    return weights_[weight_name];
}

// TODO: check path_ and get_path_
void TransformerBlock::saveCtxtTensor_bert(const CtxtTensor &tensor,
                                           const std::string &name, u64 row,
                                           u64 column) {

    tensor.get().save(path_ + "/" + name + "_" + std::to_string(layer_n_) +
                      "_" + std::to_string(row) + "_" + std::to_string(column) +
                      ".bin");
}

CtxtTensor TransformerBlock::getCtxtTensor_bert(const std::string &name,
                                                u64 row, u64 column) {

    CtxtTensor tensor(hemmer_->getContext(), ModelArgs::MAX_SEQ_LEN,
                      ModelArgs::HEAD_DIM,
                      static_cast<HEaaN::i64>(ModelArgs::HEAD_DIM * 2));
    tensor.get().load(path_ + "/" + name + "_" + std::to_string(layer_n_) +
                      "_" + std::to_string(row) + "_" + std::to_string(column) +
                      ".bin");
    tensor.get().to(getCurrentCudaDevice());
    return tensor;
}

#ifdef HELLM_MULTIGPU
void TransformerBlock::reduceWrapperHidden(
    std::vector<CtxtTensor> &ctxt_tensor_vec, const int layer_n) const {
    ncclReduceWrapperHidden(ctxt_tensor_vec, layer_n);
}

void TransformerBlock::reduceWrapper(
    std::vector<CtxtTensor> &ctxt_tensor_vec) const {
    ncclReduceWrapper(ctxt_tensor_vec);
}

void TransformerBlock::reduceWrapper_mult(CtxtTensor &ctxt_tensor) const {
    ncclAllReduceWrapper(ctxt_tensor);
}

void TransformerBlock::allReduceWrapper(CtxtTensor &ctxt_tensor) const {
    ncclAllReduceWrapper(ctxt_tensor);
}

void TransformerBlock::ncclReduceWrapperHidden(
    const std::vector<CtxtTensor> &ctxt_tensor_vec, const int layer_n) const {
    const auto size = ctxt_tensor_vec.size();
    const auto degree = getDegree(this->hemmer_->getContext());
    const auto max_rank = this->hemmer_->getMaxRank();

    NCCLCHECK(ncclGroupStart());
    for (u64 i = 0; i < size; ++i) {
        const auto root = getFFNMatrixDevice(layer_n, i / 2, max_rank);

        for (u64 j = 0; j < 2; ++j) {
            NCCLCHECK(ncclReduce(ctxt_tensor_vec[i].get().getPolyData(j, 0),
                                 ctxt_tensor_vec[i].get().getPolyData(j, 0),
                                 (ctxt_tensor_vec[i].getLevel() + 1) * degree,
                                 ncclUint64, ncclSum, root,
                                 hemmer_->getNcclComm(), nullptr));
        }
    }
    NCCLCHECK(ncclGroupEnd());
}

void TransformerBlock::ncclReduceWrapper(
    const std::vector<CtxtTensor> &ctxt_tensor_vec) const {
    const auto size = ctxt_tensor_vec.size();
    const auto degree = getDegree(this->hemmer_->getContext());
    const auto max_rank = static_cast<u64>(this->hemmer_->getMaxRank());

    NCCLCHECK(ncclGroupStart());
    for (u64 i = 0; i < size; ++i) {
        const auto root = static_cast<int>(i / (size / max_rank));

        for (u64 j = 0; j < 2; ++j) {
            NCCLCHECK(ncclReduce(ctxt_tensor_vec[i].get().getPolyData(j, 0),
                                 ctxt_tensor_vec[i].get().getPolyData(j, 0),
                                 (ctxt_tensor_vec[i].getLevel() + 1) * degree,
                                 ncclUint64, ncclSum, root,
                                 hemmer_->getNcclComm(), nullptr));
        }
    }
    NCCLCHECK(ncclGroupEnd());
}

void TransformerBlock::ncclAllReduceWrapper(
    const CtxtTensor &ctxt_tensor) const {
    const auto degree = getDegree(this->hemmer_->getContext());
    NCCLCHECK(ncclGroupStart());
    for (u64 i = 0; i < 2; ++i) {
        NCCLCHECK(ncclAllReduce(ctxt_tensor.get().getPolyData(i, 0),
                                ctxt_tensor.get().getPolyData(i, 0),
                                (ctxt_tensor.getLevel() + 1) * degree,
                                ncclUint64, ncclSum, hemmer_->getNcclComm(),
                                nullptr));
    }
    NCCLCHECK(ncclGroupEnd());
}
#endif

} // namespace HELLM
