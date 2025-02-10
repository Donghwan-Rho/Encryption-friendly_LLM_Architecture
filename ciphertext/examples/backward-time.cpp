////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2024 Crypto Lab Inc. All rights reserved.               //
//                                                                            //
// This source code is protected under international copyright law.           //
// All rights reserved and protected by the copyright holders.                //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "HELLM/HEMMer.hpp"
#include "HELLM/LoRA.hpp"
#include "HELLM/ModelArgs.hpp"
#include "HELLM/TorchTransformerBlock.hpp"
#include "HELLM/TransformerBlock.hpp"

#include "HEaaN/device/CudaTools.hpp"

#include <cstddef>
#include <sentencepiece_processor.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <mpi.h>

#include <algorithm>
#include <ctime>
#include <filesystem>
#include <random>

namespace fs = std::filesystem;

void deleteFolder(const std::string &folderPath) {
    if (fs::exists(folderPath) && fs::is_directory(folderPath)) {
        std::error_code ec;
        fs::remove_all(folderPath, ec);
        if (ec) {
            std::cerr << "Error deleting folder: " << folderPath << " ("
                      << ec.message() << ")" << std::endl;
        } else {
            std::cout << "Folder deleted successfully: " << folderPath
                      << std::endl;
        }
    } else {
        std::cerr << "Folder does not exist: " << folderPath << std::endl;
    }
}

int main() {
    static const std::string LORA_TYPE = "qkv";
    static const std::string weight_dir = "./data_2ly_mrpc/";
    static const std::string weight_path =
        weight_dir + "/converted_weights_mrpc.pth";

    auto *hemmer = new HELLM::HEMMer{HELLM::HEMMer::genHEMMerMultiGPU()};
    MPI_Barrier(MPI_COMM_WORLD);
    HELLM::TransformerBlock block{hemmer, std::string(weight_dir),
                                  std::string(weight_dir), 0};

    int rank = hemmer->getRank();
    int size = hemmer->getMaxRank();

    int prompt_len = 128;
    auto container = torch::jit::load(weight_path);

    auto start = std::chrono::high_resolution_clock::now();

    // RTE
    std::vector<HEaaN::u64> labels = HELLM::ModelArgs::RTE_LABELS;

    // 8
    const int num_gpu = 8;
    const int batch_size = static_cast<int>(16.0 / num_gpu);

    for (HEaaN::u64 epo = 0; epo < 1; epo++) {

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<int> rand_numbers(30);
        if (rank == 0) {
            for (int i = 0; i < 30; ++i) {
                rand_numbers[i] = i;
            }
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(rand_numbers.begin(), rand_numbers.end(), g);
        }
        MPI_Bcast(rand_numbers.data(), static_cast<int>(rand_numbers.size()),
                  MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "initialization Setting time: " << elapsed.count()
                      << " s" << std::endl;
        }

        start = std::chrono::high_resolution_clock::now();
        for (HEaaN::u64 step = 0; step < 2; ++step) {
            auto inp = torch::empty({prompt_len, HELLM::ModelArgs::DIM});
            // auto input_name = "input_" + std::to_string(rand_numbers[rank +
            // num_gpu*step]);
            auto input_name = "input_" + std::to_string(0);
            auto tok_emb_w =
                container.attr(input_name).toTensor().transpose(0, 1);
            // auto mask_name = "mask_" + std::to_string(rand_numbers[rank +
            // num_gpu*step]);
            auto mask_name = "mask_" + std::to_string(0);
            auto mask = container.attr(mask_name).toTensor();
            mask = mask.squeeze().slice(0, 0).unsqueeze(0).repeat({128, 1});

            for (int i = 0; i < prompt_len; ++i) {
                inp[i] = tok_emb_w[0][i];
            }

            if (rank == 0) {
                deleteFolder("./backward");
                deleteFolder("./mask");
                std::error_code ec;
                fs::create_directory("./mask", ec);
                fs::create_directory("./backward", ec);
                fs::create_directory("./backward/he", ec);
            }

            inp = inp.view({prompt_len, HELLM::ModelArgs::N_HEAD,
                            HELLM::ModelArgs::HEAD_DIM})
                      .transpose(0, 1);
            std::vector<torch::Tensor> cur{HELLM::ModelArgs::N_HEAD};
            std::vector<HELLM::CtxtTensor> ctxt_cur;
            ctxt_cur.reserve(HELLM::ModelArgs::N_HEAD / 2);
            for (long i = 0; i < HELLM::ModelArgs::N_HEAD / 2; ++i) {
                ctxt_cur.push_back(
                    hemmer->encrypt2(inp[i * 2], inp[i * 2 + 1]));
            }
            auto exp_mask = hemmer->message2(mask, mask);

            HEaaN::CudaTools::cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);

            auto start = std::chrono::high_resolution_clock::now();

            // forward
            for (int i = 0; i < 1; ++i) {
                HELLM::TransformerBlock block_tmp{hemmer,
                                                  std::string(weight_dir),
                                                  std::string(weight_dir), i};
                auto start_ = std::chrono::high_resolution_clock::now();

                // ctxt_cur = block_tmp.forward2_bert_SM(ctxt_cur, LORA_TYPE);
                // ctxt_cur = block_tmp.forward2_bert_time(ctxt_cur, exp_mask,
                // LORA_TYPE);
                ctxt_cur = block_tmp.forward2_bert_loraOpti_time(
                    ctxt_cur, exp_mask, LORA_TYPE);

                HEaaN::CudaTools::cudaDeviceSynchronize();
                MPI_Barrier(MPI_COMM_WORLD);

                if (rank == 0) {
                    auto end_ = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed = end_ - start_;
                    std::cout << "forward time: " << elapsed.count() << " s"
                              << std::endl;
                }
            }

            /* HELLM::CtxtTensor forward{ctxt_cur[0]};
            // pooling forward

            auto start_ = std::chrono::high_resolution_clock::now();
            //block.forward3_pooling_bert_time(ctxt_cur, forward,
            labels[rand_numbers[rank + num_gpu*step]]);
            block.forward3_pooling_bert_time(ctxt_cur, forward, 0);

            HEaaN::CudaTools::cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == 0) {
                auto end_ = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end_ - start_;
                std::cout << "pooling forward time: " << elapsed.count() << " s"
                        << std::endl;
            }

            start_ = std::chrono::high_resolution_clock::now();
            ctxt_cur = block.backward3_pooling_bert_time(forward);
            HEaaN::CudaTools::cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == 0) {
                auto end_ = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end_ - start_;
                std::cout << "pooling backward time: " << elapsed.count() << "
            s"
                        << std::endl;
            }

            for(int i = 1 ; i > -1 ; --i) {
                HELLM::TransformerBlock block_tmp{hemmer,
            std::string(weight_dir), std::string(weight_dir), i};
                // attn, ffn

                start_ = std::chrono::high_resolution_clock::now();
                //ctxt_cur = block_tmp.backward2_bert_SM(ctxt_cur, LORA_TYPE);
                //ctxt_cur = block_tmp.backward2_bert_time(ctxt_cur, LORA_TYPE);
                ctxt_cur = block_tmp.backward2_bert_loraOpti_time(ctxt_cur,
            LORA_TYPE);

                HEaaN::CudaTools::cudaDeviceSynchronize();
                MPI_Barrier(MPI_COMM_WORLD);
                if (rank == 0) {
                    auto end_ = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed = end_ - start_;
                    std::cout << "backward time: " << elapsed.count() << " s"
                            << std::endl;
                }
            }

            if (rank == 0) {
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                std::cout << "total 1 setp time: " << elapsed.count() << " s"
                              << std::endl;
            } */

            // optimizer step
            HEaaN::CudaTools::cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    delete hemmer;
    HELLM::cleanUpMatrixTransformer();
}
