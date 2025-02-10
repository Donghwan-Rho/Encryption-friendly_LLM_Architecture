////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2024 Crypto Lab Inc. All rights reserved.               //
//                                                                            //
// This source code is protected under international copyright law.           //
// All rights reserved and protected by the copyright holders.                //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "HELLM/HEMMer.hpp"
#include "HELLM/ModelArgs.hpp"
#include "HELLM/TorchTransformerBlock.hpp"
#include "HELLM/TransformerBlock.hpp"

#include "HEaaN/device/CudaTools.hpp"

#include <cstddef>
#include <sentencepiece_processor.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <mpi.h>

int main() {
    static const std::string LORA_TYPE = "qkv";
    // static const std::string LORA_TYPE = "";

    static const std::string data_path = "./data_2ly_mrpc/";

    auto *hemmer = new HELLM::HEMMer{HELLM::HEMMer::genHEMMerMultiGPU()};
    MPI_Barrier(MPI_COMM_WORLD);

    HELLM::TransformerBlock block{hemmer, data_path, data_path, 0};
    int rank = hemmer->getRank();
    int size = hemmer->getMaxRank();
    std::cout << "rank " << rank << " max rank " << size << std::endl;

    int prompt_len = 128;
    auto container = torch::jit::load(
        data_path + std::string("converted_weights_mrpc_eval.pth"));

    /* if (rank == 0) {
        for (int i = 0 ; i < 2 ; ++i) {
            std::shared_ptr<HELLM::TorchTransformerBlock> torch_block;
            torch_block = std::make_shared<HELLM::TorchTransformerBlock>(
                    hemmer, container, i);
            torch_block->generateInitialLoraWeight(LORA_TYPE);
        }
    }
    HEaaN::CudaTools::cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD); */

    // MRPC:51
    // RTE: 34 + 5
    // COLA: 130 + 3
    // QNLI: 682 + 7
    for (int j = 0; j < 51; ++j) {
        auto inp = torch::empty({prompt_len, HELLM::ModelArgs::DIM});
        auto input_name = "input_" + std::to_string(rank + size * j);
        auto tok_emb_w = container.attr(input_name).toTensor().transpose(0, 1);
        // std::cout << "dim = " << tok_emb_w.sizes() << std::endl;

        auto mask_name = "mask_" + std::to_string(rank + size * j);
        auto mask = container.attr(mask_name).toTensor();
        mask = mask.squeeze().slice(0, 0).unsqueeze(0).repeat({128, 1});
        for (int i = 0; i < prompt_len; ++i) {
            inp[i] = tok_emb_w[0][i];
        }
        inp = inp.view({prompt_len, HELLM::ModelArgs::N_HEAD,
                        HELLM::ModelArgs::HEAD_DIM})
                  .transpose(0, 1);
        std::vector<torch::Tensor> cur{HELLM::ModelArgs::N_HEAD};
        std::vector<HELLM::CtxtTensor> ctxt_cur;
        ctxt_cur.reserve(HELLM::ModelArgs::N_HEAD / 2);

        for (long i = 0; i < HELLM::ModelArgs::N_HEAD / 2; ++i) {
            auto input_ctxt = hemmer->encrypt2(inp[i * 2], inp[i * 2 + 1]);
            ctxt_cur.push_back(input_ctxt);
        }

        HEaaN::CudaTools::cudaDeviceSynchronize();

        auto exp_mask = hemmer->message2(mask, mask);

        for (int i = 0; i < 2; ++i) {

            HELLM::TransformerBlock block_tmp{hemmer, data_path, data_path, i};

            ctxt_cur =
                block_tmp.forward2_bert_eval(ctxt_cur, exp_mask, LORA_TYPE);
            // ctxt_cur = block_tmp.forward2_bert_SM(ctxt_cur, LORA_TYPE);
            // ctxt_cur = block_tmp.forward2_bert_test(ctxt_cur, exp_mask,
            // LORA_TYPE);
            HEaaN::CudaTools::cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);
        }
        block.forward3_pooling_bert_test(ctxt_cur);
        HEaaN::CudaTools::cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
    }
    delete hemmer;
    HELLM::cleanUpMatrixTransformer();
}
