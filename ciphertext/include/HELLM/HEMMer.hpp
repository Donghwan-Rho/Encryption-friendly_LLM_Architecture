////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2024 Crypto Lab Inc. All rights reserved.               //
//                                                                            //
// This source code is protected under international copyright law.           //
// All rights reserved and protected by the copyright holders.                //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "HELLM/MatrixUtils.hpp"
#include "HELLM/ModelArgs.hpp"
#include "HEaaN/Integers.hpp"
#include "HEaaN/Message.hpp"
#include <vector>

#include "HEaaN/HEaaN.hpp"
#include "torch/torch.h"

#ifdef HELLM_MULTIGPU
#include "nccl.h"
#endif

#include "HETensor.hpp"

namespace HELLM {

using namespace HEaaN;

void cleanUpMatrixTransformer();

class HEMMer final {
public:
    static HEMMer genHEMMer(int device_id = 0,
                            ParameterPreset preset = ParameterPreset::FGb,
                            const bool is_multi_gpu = true);
#ifdef HELLM_MULTIGPU
    static HEMMer
    genHEMMerMultiGPU(ParameterPreset preset = ParameterPreset::FGb);
    static void ncclDestroy();
#endif

    HEMMer(int device_id = 0, ParameterPreset preset = ParameterPreset::FGb);
    HEMMer(const std::string &key_path, int device_id = 0,
           ParameterPreset preset = ParameterPreset::FGb);
    ~HEMMer();

    const Context &getContext() const { return context_; }
    const HomEvaluator &getEval() const { return eval_; }
    const Bootstrapper &getBtp() const { return btp_; }
    const EnDecoder &getEnDec() const { return endec_; }
    const Decryptor &getDec() const { return dec_; }
    const SecretKey &getsk() const { return sk_; }
#ifdef HELLM_MULTIGPU
    const int &getRank() const { return device_id_; }
    const int &getMaxRank() const { return max_rank_; }
    const ncclComm_t &getNcclComm() const { return comm_; }
#endif

    void save(const std::string &keys_path);

    Message message2(const torch::Tensor &data1,
                     const torch::Tensor &data2) const;
    PtxtTensor encode(const torch::Tensor &data, u64 level = 0) const;
    PtxtTensor encode2(const torch::Tensor &data1, const torch::Tensor &data2,
                       u64 level = 0) const;
    Message msg4096Vector(const torch::Tensor &data) const;
    PtxtTensor encode4096Vector(const torch::Tensor &data, u64 level = 0) const;
    PtxtTensor encode4096VectorRowwiseRepeat(const torch::Tensor &data,
                                             u64 level = 0) const;
    Message msgDiagonalToRow4(const torch::Tensor &data1,
                              const torch::Tensor &data2,
                              const torch::Tensor &data3,
                              const torch::Tensor &data4) const;
    PtxtTensor encodeDiagonalToRow4(const torch::Tensor &data1,
                                    const torch::Tensor &data2,
                                    const torch::Tensor &data3,
                                    const torch::Tensor &data4,
                                    u64 level = 0) const;
    void encodeWeight(const torch::Tensor &tensor, const std::string &save_path,
                      bool is_generation = false) const;
    void encodeWeight_bert(const torch::Tensor &tensor,
                           const std::string &save_path,
                           bool is_generation = false) const;
    void encodeWeight_bert2(const torch::Tensor &tensor,
                            const std::string &save_path,
                            bool is_generation = false) const;
    void encodeWeight_pooling(const torch::Tensor &tensor,
                              const std::string &save_path,
                              bool is_generation = false) const;

    CtxtTensor encrypt(const torch::Tensor &data) const;
    CtxtTensor encrypt2(const torch::Tensor &data1,
                        const torch::Tensor &data2) const;
    CtxtTensor encrypt4096Vector(const torch::Tensor &vector) const;

    torch::Tensor decrypt(const CtxtTensor &tensor) const;
    torch::Tensor decrypt4096Vector(const CtxtTensor &tensor) const;
    torch::Tensor decrypt2(const CtxtTensor &tensor) const;

    void addInplace(torch::Tensor &tensor_a,
                    const torch::Tensor &tensor_b) const;
    template <class T>
    void addInplace(CtxtTensor &tensor_a, const HETensor<T> &tensor_b) const;

    void hadamardMultInplace(torch::Tensor &tensor_a,
                             const torch::Tensor &tensor_b) const;
    template <class T>
    void hadamardMultInplace(CtxtTensor &tensor_a,
                             const HETensor<T> &tensor_b) const;

    void hadamardMultInplace(CtxtTensor &tensor_a,
                             const Message &tensor_b) const;

    void divInplace(torch::Tensor &tensor, double num) const;
    void divInplace(CtxtTensor &tensor, double num) const;
    void divPtxtInplace(PtxtTensor &ptxt, double val) const;

    void bootstrap(CtxtTensor &tensor) const;
    void bootstrap2(CtxtTensor &tensor1, CtxtTensor &tensor2) const;
    void bootstrap2_exponly(CtxtTensor &tensor1, CtxtTensor &tensor2) const;
    void bootstrapExtendedWithMultRange(u64 range,
                                        const HEaaN::Ciphertext &input,
                                        HEaaN::Ciphertext &res) const;
    void bootstrapUnitRange(CtxtTensor &tensor) const;

    void transposeInplace(torch::Tensor &tensor) const;
    void transposeInplace(CtxtTensor &tensor, u64 target_level = 0) const;

    torch::Tensor matMul(const torch::Tensor &tensor_a,
                         const torch::Tensor &tensor_b) const;

    void complexPackingInplace(CtxtTensor &tensor) const;
    void complexPackingRowInplace(CtxtTensor &tensor) const;
    void transposeComplexPackingInplace(CtxtTensor &tensor,
                                        u64 target_level = 0) const;

    void splitInTwo(CtxtTensor &tensor, CtxtTensor &tensor1,
                    CtxtTensor &tensor2) const;

    CtxtTensor packedMatMul(const CtxtTensor &tensor_a,
                            const CtxtTensor &tensor_b,
                            u64 target_level = 4) const;

    CtxtTensor packedMatMulPre(const CtxtTensor &tensor_a,
                               u64 target_level = 4) const;

    void packedMatMulPreRot(const CtxtTensor &tensor_a,
                            std::vector<Ciphertext> &tmp,
                            u64 target_level = 4) const;

    CtxtTensor packedMatMulPreRev(const CtxtTensor &tensor_a,
                                  u64 target_level = 4) const;

    CtxtTensor packedMatMulCCReuse(const std::vector<Ciphertext> &tmp,
                                   const CtxtTensor &tensor_b,
                                   u64 target_level = 4) const;

    CtxtTensor singleMatMul(const CtxtTensor &tensor_a,
                            const PtxtTensor &tensor_b,
                            u64 target_level = 4) const;

    CtxtTensor singleCCMatMul(const CtxtTensor &tensor_a,
                              const CtxtTensor &tensor_b,
                              u64 target_level = 4) const;

    void oneMatRotSumInplace(CtxtTensor &tensor_a, CtxtTensor &tensor_b) const;
    void tr_oneMatRotSumInplace(CtxtTensor &tensor_a, CtxtTensor &tensor_b,
                                u64 target_level = 0) const;

    CtxtTensor repackCC(CtxtTensor &tensor_1) const;

    PtxtTensor complexPacking(const PtxtTensor &tensor_1,
                              const PtxtTensor &tensor_2) const;

    CtxtTensor complexPacking(const CtxtTensor &tensor_1,
                              const CtxtTensor &tensor_2) const;

    CtxtTensor complexPackingRev(const CtxtTensor &tensor_1,
                                 const CtxtTensor &tensor_2) const;

    CtxtTensor repack(CtxtTensor &tensor_1, CtxtTensor &tensor_2) const;
    CtxtTensor repack_notBTS(CtxtTensor &tensor_1, CtxtTensor &tensor_2) const;
    CtxtTensor repack_loraOpti(CtxtTensor &tensor_1,
                               CtxtTensor &tensor_2) const;

    void repackVector(std::vector<CtxtTensor> &vector) const;

    void matMulPre(const CtxtTensor &tensor_a, std::vector<Ciphertext> &tmp,
                   u64 target_level = 4) const;
    CtxtTensor matMulReUse(const std::vector<Ciphertext> &tmp,
                           const PtxtTensor &tensor_b,
                           u64 target_level = 4) const;

    CtxtTensor matMulPreRev(const CtxtTensor &tensor_a,
                            u64 target_level = 4) const;
    CtxtTensor matMulCCReUse(const std::vector<Ciphertext> &tmp,
                             const CtxtTensor &tensor_b,
                             u64 target_level = 4) const;

    std::vector<Plaintext> encodeConcatMasks(u64 col_level,
                                             u64 row_level) const;
    void columnConcatInplace(CtxtTensor &tensor, const CtxtTensor &vectors,
                             const Plaintext &mask_col_cat_gen_1,
                             const Plaintext &mask_col_cat_gen_2,
                             const u64 idx) const;
    void rowConcatInplace(CtxtTensor &tensor, const CtxtTensor &vectors,
                          const Plaintext &mask_row_cat_gen_1,
                          const Plaintext &mask_row_cat_gen_2,
                          const u64 idx) const;

    void multVecPre(const CtxtTensor &vector,
                    std::vector<Ciphertext> &tmp_vectors,
                    u64 target_level = 0) const;
    CtxtTensor vec4096MatMul(const Ciphertext &tmp_vector,
                             const PtxtTensor &weight,
                             u64 target_level = 0) const;
    CtxtTensor vec4096MatMul(const Ciphertext &tmp_vector,
                             const Message &weight, u64 target_level = 0) const;
    CtxtTensor vec4096MatMul(const CtxtTensor &vector,
                             const std::vector<CtxtTensor> &mat,
                             u64 target_level = 0) const;
    void multVecPost(std::vector<CtxtTensor>::const_iterator begin,
                     std::vector<CtxtTensor>::const_iterator end,
                     CtxtTensor &res) const;

    // bert
    void maskRightLeft(const CtxtTensor &tensor, CtxtTensor &tensor_out1,
                       CtxtTensor &tensor_out2) const;
    void maskFirstColInplace(CtxtTensor &tensor) const;
    void maskFirstColOnlyInplace(CtxtTensor &tensor) const;
    void maskFirstRowInplace(CtxtTensor &tensor) const;
    void maskLeftHalfInplace(CtxtTensor &tensor) const;
    void maskRightHalfInplace(CtxtTensor &tensor) const;
    void maskFirsteleInplace(CtxtTensor &tensor) const;
    void maskFirstele1OnlyInplace(CtxtTensor &tensor) const;
    void maskFirstele2Inplace(CtxtTensor &tensor) const;
    void maskFirstele2InplaceSST2(CtxtTensor &tensor) const;
    void maskFirsteleOnlyInplace(CtxtTensor &tensor) const;
    void maskFirsteleOnlyInplaceSST2(CtxtTensor &tensor) const;
    void maskFirstColPoolingInplace(CtxtTensor &tensor) const;
    void maskFirstRowPoolingInplace(CtxtTensor &tensor) const;
    void maskFirstRowsPoolingInplace(CtxtTensor &tensor) const;

    void dropoutInplace(CtxtTensor &tensor, const std::string &name,
                        const int layer_n, const u64 idx) const;
    void backwarddropoutInplace(CtxtTensor &tensor, const std::string &name,
                                const int layer_n, const u64 idx) const;
    void dropoutExpInplace(CtxtTensor &tensor, const std::string &name,
                           const int layer_n, const u64 idx) const;
    void backwarddropoutExpInplace(CtxtTensor &tensor, const std::string &name,
                                   const int layer_n, const u64 idx) const;

    // forward nonlinear
    void expInplace(torch::Tensor &tensor, const int layer_n,
                    const bool train_mode) const;
    void lossExpInplace(CtxtTensor &tensor);
    void lossExpInplaceSST2(CtxtTensor &tensor);
    void lossInvInplace(CtxtTensor &tensor);
    void lossInvInplaceSST2(CtxtTensor &tensor);
    void lossMax(CtxtTensor &tensor);
    void expVectorInplace(std::vector<CtxtTensor> &tensor_vec,
                          const int layer_n, const bool train_mode) const;

    void expParallelInplace(std::vector<CtxtTensor> &tensor_vec,
                            const int layer_n, const bool train_mode) const;
    void softmaxVectorInplaceHETAL(std::vector<CtxtTensor> &tensor_vec,
                                   const int layer_n, const u64 base_idx,
                                   const bool train_mode, const Decryptor &dec,
                                   const SecretKey &sk) const;

    void softmaxVectorInplaceCCS(std::vector<CtxtTensor> &tensor_vec,
                                 const int layer_n,
                                 const bool train_mode) const;

    void reluInplace(torch::Tensor &tensor, const int layer_n,
                     const bool train_bode) const;
    void reluVectorInplace(std::vector<CtxtTensor> &tensor_vec,
                           const int layer_n,
                           const bool train_mode = false) const;

    void tanhInplace(CtxtTensor &tensor, const int layer_n,
                     const bool train_mode = false) const;
    void tanhInplace_SST2(CtxtTensor &tensor, const int layer_n,
                          const bool train_mode = false) const;
    void tanhVectorInplace(std::vector<CtxtTensor> &tensor_vec,
                           const int layer_n,
                           const bool train_mode = false) const;

    // bert //
    std::vector<torch::Tensor>
    LayerNorm(const std::vector<torch::Tensor> &tensor_vec,
              const std::string &module_name, const int layer_n,
              const bool train_mode = false) const;

    std::vector<CtxtTensor> LayerNorm(const std::vector<CtxtTensor> &tensor_vec,
                                      const std::string &module_name,
                                      const int layer_n,
                                      const bool train_mode = false) const;

    std::vector<CtxtTensor>
    LayerNorm_multi(const std::vector<CtxtTensor> &tensor_vec,
                    const std::string &module_name, const int layer_n,
                    const bool train_mode = false) const;

    // backward BERT
    std::vector<CtxtTensor>
    backwardLayerNorm(const std::vector<CtxtTensor> &tensor_vec,
                      const std::string &module_name, const int layer_n) const;

    void backwardexpVectorInplace(std::vector<CtxtTensor> &tensor_vec,
                                  const int layer_n) const;

    void backwardsoftmaxVectorInplaceHETAL(std::vector<CtxtTensor> &grad_y,
                                           const int layer_n) const;

    void backwardreluVectorInplace(std::vector<CtxtTensor> &tensor_vec,
                                   const int layer_n) const;

    void backwardtanhVectorInplace(std::vector<CtxtTensor> &tensor_vec,
                                   const int layer_n) const;
    void backwardtanhInplace(CtxtTensor &tensor, const int layer_n) const;

    // lora
    CtxtTensor matMulHighLow(const CtxtTensor &tensor_a,
                             const CtxtTensor &tensor_b, u64 in_col_block,
                             u64 target_level = 4) const;
    CtxtTensor matMulLowHigh(const CtxtTensor &tensor_a,
                             const CtxtTensor &tensor_b, u64 in_row_block,
                             u64 target_level = 4) const;
    CtxtTensor matMulLowLow(const CtxtTensor &tensor_a,
                            const CtxtTensor &tensor_b, u64 in_col_block,
                            u64 in_row_block, u64 target_level = 4) const;

    CtxtTensor repackToOneCol(const CtxtTensor &tensor,
                              u64 out_col_block) const;
    CtxtTensor repackToMultiCol(const CtxtTensor &tensor,
                                u64 out_col_block) const;
    CtxtTensor repackToOneRow(const CtxtTensor &tensor,
                              u64 out_row_block) const;
    CtxtTensor repackToMultiRow(const CtxtTensor &tensor,
                                u64 out_row_block) const;

    CtxtTensor getLowColBlock(const CtxtTensor &tensor, u64 col_block) const;
    CtxtTensor getLowRowBlock(const CtxtTensor &tensor, u64 row_block) const;

    std::string getTorchPath() { return torch_path_; }
    std::string getHEPath() { return he_path_; }
    std::string getWeightPath() { return weight_path_; }
    std::string getWeightTestPath() { return weight_test_path_; }

private:
    int device_id_;

    Context context_;
    u64 log_slots_;

    SecretKey sk_;
    Encryptor enc_;
    Decryptor dec_;
    EnDecoder endec_;
    KeyPack pack_;
    HomEvaluator eval_;
    Bootstrapper btp_;

    Message rms_mask_;

    Message col_mask_even_;
    Message row_mask_even_;

    // bert
    // upper :
    //  1 1 1 ...
    //  0 0 0 ...
    // lower:
    //  0 0 0 ...
    //  1 1 1 ...
    Message mask_one_right_;
    Message mask_one_left_;
    Message mask_first_col_;
    Message mask_first_col_only_;
    Message mask_first_row_;
    Message mask_left_half_;
    Message mask_right_half_;
    Message mask_upper_half_;
    Message mask_lower_half_;
    Message mask_first_ele_;
    Message mask_first_ele2_;
    Message mask_first_ele2_sst2_;
    Message mask_first_ele_only_;
    Message mask_first_ele_1_only_;
    Message mask_first_ele_only_sst2_;
    Message mask_first_col_pool_;
    Message mask_first_row_pool_;
    Message mask_first_rows_pool_;
    std::vector<Message> mask_drop_out_;
    std::vector<Message> mask_drop_out_exp_;

    Plaintext mask_matmul_;
    Plaintext mask_matmul_half_;

    const std::string he_path_ = "./backward/he";
    const std::string torch_path_ = "./backward/torch";
    const std::string weight_path_ = "./data_2ly_mrpc/";
    const std::string weight_test_path_ = "./data_2ly_mrpc/";

#ifdef HELLM_MULTIGPU
    int max_rank_;
    inline static ncclComm_t comm_;
#endif

    void genMask();

    void generateAxisMasks();

    // bert
    void generateRightLeftMasks();
    void generateDropoutMasks();

    // row sum for RMSNorm and SoftmaxFinal
    void squareSum(const std::vector<CtxtTensor> &tensor_vec,
                   CtxtTensor &tensor) const;
    void reduceRowSumWithScaledMask(CtxtTensor &tensor,
                                    const HEaaN::Message &mask) const;

    void inverseSqrtLN(CtxtTensor &tensor, const std::string &module_name,
                       const int layer_n) const;

    void inverseSqrtLNFT(CtxtTensor &tensor, const std::string &module_name,
                         const int layer_n) const;

#ifdef HELLM_MULTIGPU
    void allReduceWrapper(CtxtTensor &ctxt_tensor) const;

    void ncclAllReduceWrapper(const CtxtTensor &ctxt_tensor) const;
#endif
};

} // namespace HELLM
