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

#include "HELLM/HEMMer.hpp"

#include "torch/script.h"
#include "torch/torch.h"
#include <string.h>

namespace HELLM::LoRA {

class LoraModule {
public:
    LoraModule(HEMMer *hemmer, int layer_n)
        : hemmer_(hemmer), layer_n_(layer_n) {}

    void generateInitialLoraWeight(const std::string &lora_type) const;
    void compareLoraWeight(const std::string &lora_type) const;

    void zeroGrad(const std::string &lora_type) const;
    void zeroAggGrad_head() const;
    void zeroAggGrad_head2() const;

    // For gathering the gradients in each sequence block
    void zeroAggGrad(const std::string &lora_type) const;
    void saveAggGrad(const CtxtTensor &grad,
                     const std::string &weight_name) const;

    void printing(const CtxtTensor &tensor_vec) const;

    void optimizerStep(const std::string &lora_type) const;

    void AdamW(const std::string &lora_type, const char *task, int step) const;
    void AdamW_head(const char *task, int step) const;
    void AdamW_head2(const char *task, int step) const;
    // void optimizerStep_bert(const std::string &lora_type, const char *task,
    // int step) const;
    void optimizerStep_bert(const char *task, int step) const;
    void optimizerStep_head_bert(const char *task, int step) const;
    void optimizerStep_head2_bert(const char *task, int step) const;

    void updateWeight(torch::Tensor &weight, const torch::Tensor &grad) const;

    CtxtTensor getCtxtTensor(const std::string &name, u64 sbi, u64 sbj,
                             u64 index) const;

    CtxtTensor getCtxtTensor_lora(const std::string &name, u64 sbi, u64 sbj,
                                  u64 index) const;
    CtxtTensor getCtxtTensor_lora_test(const std::string &name, u64 sbi,
                                       u64 sbj, u64 index) const;

    void saveCtxtTensor(const CtxtTensor &tensor, const std::string &name,
                        u64 sbi, u64 sbj, u64 index) const;

    void saveCtxtTensor_lora(const CtxtTensor &tensor, const std::string &name,
                             u64 sbi, u64 sbj, u64 index) const;

    torch::Tensor getTorchTensor(const std::string &name, u32 index) const;
    void saveTorchTensor(const torch::Tensor &tensor, const std::string &name,
                         u32 index) const;

private:
    HEMMer *hemmer_;
    const int layer_n_;

#ifdef HELLM_MULTIGPU
    void allReduceWrapper(CtxtTensor &ctxt_tensor) const;
    void ncclAllReduceWrapper(const CtxtTensor &ctxt_tensor) const;
#endif
};

void approxInvSqrt_adamw(const HEaaN::HomEvaluator &eval,
                         const HEaaN::Bootstrapper &btp,
                         const HEaaN::Ciphertext &ctxt,
                         HEaaN::Ciphertext &ctxt_out, const HEaaN::Real initial,
                         const HEaaN::u64 num_iter);

void approxInverseSqrtNewton_TS(const HEaaN::HomEvaluator &eval,
                                const HEaaN::Bootstrapper &btp,
                                const HEaaN::Ciphertext &ctxt,
                                HEaaN::Ciphertext &ctxt_out,
                                const HEaaN::Ciphertext &ctxt_init,
                                const HEaaN::u64 num_iter);

void approxInverseSqrt_COLA(const HEaaN::HomEvaluator &eval,
                            const HEaaN::Bootstrapper &btp,
                            const HEaaN::Ciphertext &op, HEaaN::Ciphertext &res,
                            const HEaaN::u64 num_iter);

void approxInverseSqrt_MRPC(const HEaaN::HomEvaluator &eval,
                            const HEaaN::Bootstrapper &btp,
                            const HEaaN::Ciphertext &op, HEaaN::Ciphertext &res,
                            const HEaaN::u64 num_iter);

void approxInverseSqrt_RTE(const HEaaN::HomEvaluator &eval,
                           const HEaaN::Bootstrapper &btp,
                           const HEaaN::Ciphertext &op, HEaaN::Ciphertext &res,
                           const HEaaN::u64 num_iter);

void approxInverseSqrt_SST2(const HEaaN::HomEvaluator &eval,
                            const HEaaN::Bootstrapper &btp,
                            const HEaaN::Ciphertext &op, HEaaN::Ciphertext &res,
                            const HEaaN::u64 num_iter);

void approxInverseSqrt_STSB(const HEaaN::HomEvaluator &eval,
                            const HEaaN::Bootstrapper &btp,
                            const HEaaN::Ciphertext &op, HEaaN::Ciphertext &res,
                            const HEaaN::u64 num_iter);

} // namespace HELLM::LoRA
