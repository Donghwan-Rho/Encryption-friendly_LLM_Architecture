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
#include "HETensor.hpp"
#include "HEaaN/Message.hpp"
#include "LoRA.hpp"
#include <memory>
#include <string>
#include <vector>

namespace HELLM {

class TransformerBlock {
public:
    TransformerBlock(HEMMer *hemmer, const std::string &weight_dir,
                     const std::string &weight_gen_dir, int layer_n,
                     bool benchmark = true)
        : hemmer_{hemmer}, layer_n_{layer_n}, path_{weight_dir +
                                                    std::to_string(layer_n_) +
                                                    "_"},
          gen_path_{weight_gen_dir + std::to_string(layer_n_) + "_"},
          benchmark_{benchmark} {
        lora_module_ = std::make_shared<LoRA::LoraModule>(hemmer_, layer_n_);
    }

    // bert forward
    std::vector<CtxtTensor> forward_bert(std::vector<CtxtTensor> &input,
                                         const Message &exp_message);
    std::vector<CtxtTensor> forward2_bert(std::vector<CtxtTensor> &input,
                                          const Message &exp_message,
                                          const std::string &lora_type);
    std::vector<CtxtTensor> forward2_bert_eval(std::vector<CtxtTensor> &input,
                                               const Message &exp_message,
                                               const std::string &lora_type);
    std::vector<CtxtTensor> forward2_bert_test(std::vector<CtxtTensor> &input,
                                               const Message &exp_message,
                                               const std::string &lora_type);
    std::vector<CtxtTensor> forward2_bert_time(std::vector<CtxtTensor> &input,
                                               const Message &exp_message,
                                               const std::string &lora_type);
    std::vector<CtxtTensor> forward2_bert_SM(std::vector<CtxtTensor> &input,
                                             const std::string &lora_type);
    std::vector<CtxtTensor> forward_bert_final(std::vector<CtxtTensor> &input);
    void forward2_pooling_bert(std::vector<CtxtTensor> &input,
                               std::vector<CtxtTensor> &output,
                               const u64 label);
    void forward3_pooling_bert(std::vector<CtxtTensor> &input,
                               CtxtTensor &output, const u64 label);
    void forward3_pooling_bert_stsb(std::vector<CtxtTensor> &input,
                                    CtxtTensor &output, const u64 label);
    void forward3_pooling_bert_sst2(std::vector<CtxtTensor> &input,
                                    CtxtTensor &output, const u64 label);
    void forward3_pooling_bert_time(std::vector<CtxtTensor> &input,
                                    CtxtTensor &output, const u64 label);
    void forward3_pooling_bert_test(std::vector<CtxtTensor> &input);
    std::vector<CtxtTensor> forward_bert_multi(std::vector<CtxtTensor> &input,
                                               const Message &exp_message);

    // bert backward
    std::vector<CtxtTensor> backward2_bert(std::vector<CtxtTensor> &grad_y,
                                           const std::string &lora_type);
    std::vector<CtxtTensor> backward2_bert_time(std::vector<CtxtTensor> &grad_y,
                                                const std::string &lora_type);
    std::vector<CtxtTensor> backward2_bert_SM(std::vector<CtxtTensor> &grad_y,
                                              const std::string &lora_type);
    std::vector<CtxtTensor>
    backward2_pooling_bert(std::vector<CtxtTensor> &grad_y);
    std::vector<CtxtTensor> backward3_pooling_bert(CtxtTensor &grad_y);
    std::vector<CtxtTensor> backward3_pooling_bert_stsb(CtxtTensor &grad_y);
    std::vector<CtxtTensor> backward3_pooling_bert_time(CtxtTensor &grad_y);

    void generateInitialLoraWeight(const std::string &lora_type) const;
    void compareLoraWeight(const std::string &lora_type) const;
    void zeroGrad(const std::string &lora_type) const;
    void printing(const std::vector<CtxtTensor> &tensor_vec) const;
    void printing_whole(const std::vector<CtxtTensor> &tensor_vec) const;
    void printing_exp(const std::vector<CtxtTensor> &tensor_vec) const;
    void printing_masking(const CtxtTensor &tensor_vec) const;
    void printingfirstCol(const CtxtTensor &tensor_vec) const;
    void tensor_save(const std::vector<CtxtTensor> &tensor_vec,
                     const std::string &name, const int layer_n) const;

    const Message &getWeightMsg(const std::string &name, u64 w_index);
    const Message &getWeightMsg(const std::string &name, u64 h_index,
                                u64 w_index);

    // Updated
    std::vector<CtxtTensor>
    forward2_bert_loraOpti(std::vector<CtxtTensor> &input,
                           const Message &exp_message,
                           const std::string &lora_type);
    std::vector<CtxtTensor>
    backward2_bert_loraOpti(std::vector<CtxtTensor> &grad_y,
                            const std::string &lora_type);
    std::vector<CtxtTensor>
    forward2_bert_loraOpti_time(std::vector<CtxtTensor> &input,
                                const Message &exp_message,
                                const std::string &lora_type);
    std::vector<CtxtTensor>
    backward2_bert_loraOpti_time(std::vector<CtxtTensor> &grad_y,
                                 const std::string &lora_type);

private:
    HEMMer *hemmer_;
    const int layer_n_;
    const std::string path_;
    const std::string gen_path_;
    const bool benchmark_;
    std::shared_ptr<LoRA::LoraModule> lora_module_;

    std::map<std::string, Message> weights_;

    // BERT //
    void attention_bert(std::vector<CtxtTensor> &input,
                        const Message &exp_message);
    void attention2_bert(std::vector<CtxtTensor> &input,
                         const Message &exp_message,
                         const std::string &lora_type);
    void attention2_bert_test(std::vector<CtxtTensor> &input,
                              const Message &exp_message,
                              const std::string &lora_type);
    void attention2_bert_time(std::vector<CtxtTensor> &input,
                              const Message &exp_message,
                              const std::string &lora_type);
    void attention_bert_SM(std::vector<CtxtTensor> &input,
                           const std::string &lora_type);
    void attention_bert_multi(std::vector<CtxtTensor> &input,
                              const Message &exp_message);
    void feedForward_bert(std::vector<CtxtTensor> &input);
    void feedForward2_bert(std::vector<CtxtTensor> &input);
    void feedForward2_bert_test(std::vector<CtxtTensor> &input);
    void feedForward2_bert_time(std::vector<CtxtTensor> &input);
    void feedForward_bert_multi(std::vector<CtxtTensor> &input);
    void pooling_bert(std::vector<CtxtTensor> &input);
    void pooling2_bert(std::vector<CtxtTensor> &input,
                       std::vector<CtxtTensor> &output, const u64 label);
    void pooling3_bert(std::vector<CtxtTensor> &input, CtxtTensor &output,
                       const u64 label);
    void pooling3_bert_stsb(std::vector<CtxtTensor> &input, CtxtTensor &output,
                            const u64 label);
    void pooling3_bert_sst2(std::vector<CtxtTensor> &input, CtxtTensor &output,
                            const u64 label);
    void pooling3_bert_time(std::vector<CtxtTensor> &input, CtxtTensor &output,
                            const u64 label);
    void pooling3_bert_test(std::vector<CtxtTensor> &input);
    std::vector<CtxtTensor> pooling_res_repack(CtxtTensor &tensor);
    std::vector<CtxtTensor> pooling_loss_grad(CtxtTensor &tensor,
                                              const u64 label);
    CtxtTensor pooling3_loss_grad(CtxtTensor &tensor, const u64 label);
    CtxtTensor pooling3_loss_grad_sst2(CtxtTensor &tensor, const u64 label);
    CtxtTensor pooling3_loss_grad_mse(CtxtTensor &tensor, const u64 label);

    std::vector<CtxtTensor>
    backwardpooling2_bert(std::vector<CtxtTensor> &grad_y);
    std::vector<CtxtTensor> backwardpooling3_bert(CtxtTensor &grad_y);
    std::vector<CtxtTensor> backwardpooling3_bert_stsb(CtxtTensor &grad_y);
    std::vector<CtxtTensor> backwardpooling3_bert_time(CtxtTensor &grad_y);
    void backwardfeedForward2_bert(std::vector<CtxtTensor> &grad_y);
    void backwardfeedForward2_bert_time(std::vector<CtxtTensor> &grad_y);
    void backwardattention2_bert(std::vector<CtxtTensor> &grad_y,
                                 const std::string &lora_type);
    void backwardattention2_bert_time(std::vector<CtxtTensor> &grad_y,
                                      const std::string &lora_type);
    void backwardattention2_bert_SM(std::vector<CtxtTensor> &grad_y,
                                    const std::string &lora_type);

    // Updated
    void attention2_bert_loraOpti(std::vector<CtxtTensor> &input,
                                  const Message &exp_message,
                                  const std::string &lora_type);
    void attention2_bert_loraOpti_eval(std::vector<CtxtTensor> &input,
                                       const Message &exp_message,
                                       const std::string &lora_type);
    void attention2_bert_loraOpti_time(std::vector<CtxtTensor> &input,
                                       const Message &exp_message,
                                       const std::string &lora_type);
    void backwardattention2_bert_loraOpti(std::vector<CtxtTensor> &grad_y,
                                          const std::string &lora_type);
    void backwardattention2_bert_loraOpti_time(std::vector<CtxtTensor> &grad_y,
                                               const std::string &lora_type);

    PtxtTensor getWeight(const std::string &name, u64 w_index);
    PtxtTensor getWeight(const std::string &name, u64 h_index, u64 w_index);

    void printElapsedTime(const char *str) const;

    // bert version get/saveCtxtTensor
    // Caution: the followings are same with ones in loar_module.
    void saveCtxtTensor_bert(const CtxtTensor &tensor, const std::string &name,
                             u64 row, u64 column);
    CtxtTensor getCtxtTensor_bert(const std::string &name, u64 row, u64 column);

#ifdef HELLM_MULTIGPU
    void reduceWrapperHidden(std::vector<CtxtTensor> &ctxt_tensor_vec,
                             const int layer_n) const;
    void reduceWrapper(std::vector<CtxtTensor> &ctxt_tensor_vec) const;
    void reduceWrapper_mult(CtxtTensor &ctxt_tensor) const;
    void allReduceWrapper(CtxtTensor &ctxt_tensor) const;

    void ncclReduceWrapperHidden(const std::vector<CtxtTensor> &ctxt_tensor_vec,
                                 const int layer_n) const;
    void
    ncclReduceWrapper(const std::vector<CtxtTensor> &ctxt_tensor_vec) const;
    void ncclAllReduceWrapper(const CtxtTensor &ctxt_tensor) const;
#endif
};

} // namespace HELLM
