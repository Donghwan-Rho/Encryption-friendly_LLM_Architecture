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
    const std::string weight_pth = "./data_2ly_mrpc/"; // Put the HE weight path used when running convert.py.
    const int num_gpu = 8;
    const int batch_size = static_cast<int>(16.0 / num_gpu);
    // { R : RTE, C : COLA, M : MRPC, S : STSB , T: SST2}
    auto task = "M";
    // {RTE: 310, COLA: 1068, MRPC: 458, STSB: 718, SST2: 8418, QNLI: 13092 }
    const HELLM::u64 one_epo_step = 458; // temp
    // {RTE: 2490, COLA: 8551, MRPC: 3668, STSB: 5749, SST2: 67349, QNLI:
    // 104743}
    const int num_data = 3668; // temp

    auto *hemmer = new HELLM::HEMMer{HELLM::HEMMer::genHEMMerMultiGPU()};
    MPI_Barrier(MPI_COMM_WORLD);

    HELLM::TransformerBlock block{hemmer, weight_pth, weight_pth, 0};

    int rank = hemmer->getRank();
    int size = hemmer->getMaxRank();

    int prompt_len = 128;
    auto container = torch::jit::load(
        weight_pth + std::string("converted_weights_mrpc.pth"));

    auto start = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        for (int i = 0; i < 2; ++i) {
            std::shared_ptr<HELLM::TorchTransformerBlock> torch_block;
            torch_block = std::make_shared<HELLM::TorchTransformerBlock>(
                hemmer, container, i);
            torch_block->generateInitialLoraWeight(LORA_TYPE);
        }
    }
    HEaaN::CudaTools::cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "LoRA weight initialization time: " << elapsed.count()
                  << " s" << std::endl;
    }

    if (rank == 0) {
        auto start = std::chrono::high_resolution_clock::now();
        for (HEaaN::u64 layer_ = 0; layer_ < 2; ++layer_) {
            std::shared_ptr<HELLM::LoRA::LoraModule> lora_module_ =
                std::make_shared<HELLM::LoRA::LoraModule>(hemmer, layer_);
            lora_module_->zeroAggGrad(LORA_TYPE);

            if (layer_ == 0) {
                lora_module_->zeroAggGrad_head();
                lora_module_->zeroAggGrad_head2();
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "zeroAggGrad initialization time: " << elapsed.count()
                  << " s" << std::endl;
    }
    HEaaN::CudaTools::cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    // MRPC
    std::vector<HEaaN::u64> labels = HELLM::ModelArgs::MRPC_LABELS;
    // RTE
    // std::vector<HEaaN::u64> labels = HELLM::ModelArgs::RTE_LABELS;
    //  COLA
    // std::vector<HEaaN::u64> labels = HELLM::ModelArgs::COLA_LABELS;
    // STSB
    // std::vector<double> labels = HELLM::ModelArgs::STSB_LABELS;
    // SST2
    // std::vector<HEaaN::u64> labels = HELLM::ModelArgs::SST2_LABELS;
    // QNLI
    // std::vector<HEaaN::u64> labels = HELLM::ModelArgs::QNLI_LABELS;

    for (HEaaN::u64 epo = 0; epo < 5; epo++) {
        if (rank == 0) {
            std::string path =
                hemmer->getWeightPath() + std::to_string(epo) + "epo";
            deleteFolder(path);
            std::error_code ec;
            fs::create_directory(path, ec);
        }

        // MRPC: 3668
        // RTE: 2490
        // COLA: 8551
        // STSB: 5749
        // SST2: 67349
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<int> rand_numbers(num_data);

        if (rank == 0) {
            for (int i = 0; i < num_data; ++i) {
                rand_numbers[i] = i;
            }
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(rand_numbers.begin(), rand_numbers.end(), g);

            for (int i = 0; i < num_data; ++i) {
                std::cout << rand_numbers[i] << ", ";
                if ((i + 1) % 20 == 0) {
                    std::cout << std::endl;
                }
            }
            std::cout << std::endl;
        }
        MPI_Bcast(rand_numbers.data(), static_cast<int>(rand_numbers.size()),
                  MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "initialization Setting time: " << elapsed.count()
                      << " s" << std::endl;
        }

        // MRPC: 458
        // RTE: 311 -> 310
        // COLA: 1068.875 -> 1068
        // STSB: 718
        start = std::chrono::high_resolution_clock::now();

        for (HEaaN::u64 step = 0; step < one_epo_step; ++step) {
            auto start_ = std::chrono::high_resolution_clock::now();

            if (rank == 0)
                std::cout << "step: " << step + 1 << std::endl;

            auto inp = torch::empty({prompt_len, HELLM::ModelArgs::DIM});
            auto input_name =
                "input_" + std::to_string(rand_numbers[rank + num_gpu * step]);
            auto tok_emb_w =
                container.attr(input_name).toTensor().transpose(0, 1);
            auto mask_name =
                "mask_" + std::to_string(rand_numbers[rank + num_gpu * step]);
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

            if (rank == 0) {
                std::cout << "Construct Done" << std::endl;
            }
            HEaaN::CudaTools::cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);

            // forward
            for (int i = 0; i < 2; ++i) {
                HELLM::TransformerBlock block_tmp{hemmer, weight_pth,
                                                  weight_pth, i};
                ctxt_cur = block_tmp.forward2_bert_loraOpti(ctxt_cur, exp_mask,
                                                            LORA_TYPE);
                HEaaN::CudaTools::cudaDeviceSynchronize();
                MPI_Barrier(MPI_COMM_WORLD);
            }

            if (rank == 0) {
                std::cout << "forward Done" << std::endl;
            }

            HELLM::CtxtTensor forward{ctxt_cur[0]};
            // pooling forward
            if (std::strcmp(task, "S") == 0) {

                block.forward3_pooling_bert_stsb(
                    ctxt_cur, forward,
                    labels[rand_numbers[rank + num_gpu * step]]);
                HEaaN::CudaTools::cudaDeviceSynchronize();
                MPI_Barrier(MPI_COMM_WORLD);

                if (rank == 0) {
                    std::cout << "pooling forward Done" << std::endl;
                }

                // pooling backward
                ctxt_cur = block.backward3_pooling_bert_stsb(forward);
                HEaaN::CudaTools::cudaDeviceSynchronize();
                MPI_Barrier(MPI_COMM_WORLD);

                if (rank == 0) {
                    std::cout << "pooling backward Done" << std::endl;
                }
            } else if (std::strcmp(task, "T") == 0 ||
                       std::strcmp(task, "Q") == 0) {
                block.forward3_pooling_bert_sst2(
                    ctxt_cur, forward,
                    labels[rand_numbers[rank + num_gpu * step]]);
                HEaaN::CudaTools::cudaDeviceSynchronize();
                MPI_Barrier(MPI_COMM_WORLD);

                if (rank == 0) {
                    std::cout << "pooling forward Done" << std::endl;
                }

                // pooling backward
                ctxt_cur = block.backward3_pooling_bert(forward);
                HEaaN::CudaTools::cudaDeviceSynchronize();
                MPI_Barrier(MPI_COMM_WORLD);

                if (rank == 0) {
                    std::cout << "pooling backward Done" << std::endl;
                }

            } else {

                block.forward3_pooling_bert(
                    ctxt_cur, forward,
                    labels[rand_numbers[rank + num_gpu * step]]);
                HEaaN::CudaTools::cudaDeviceSynchronize();
                MPI_Barrier(MPI_COMM_WORLD);

                if (rank == 0) {
                    std::cout << "pooling forward Done" << std::endl;
                }

                // pooling backward
                ctxt_cur = block.backward3_pooling_bert(forward);
                HEaaN::CudaTools::cudaDeviceSynchronize();
                MPI_Barrier(MPI_COMM_WORLD);

                if (rank == 0) {
                    std::cout << "pooling backward Done" << std::endl;
                }
            }

            for (int i = 1; i > -1; --i) {
                HELLM::TransformerBlock block_tmp{hemmer, weight_pth,
                                                  weight_pth, i};
                // attn, ffn
                ctxt_cur =
                    block_tmp.backward2_bert_loraOpti(ctxt_cur, LORA_TYPE);
                HEaaN::CudaTools::cudaDeviceSynchronize();
                MPI_Barrier(MPI_COMM_WORLD);
            }

            if (rank == 0) {
                std::cout << "backward Done" << std::endl;
            }

            // optimizer step
            HEaaN::CudaTools::cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);

            auto start_opti = std::chrono::high_resolution_clock::now();

            if ((step + 1) % batch_size == 0) {
                auto start_opti = std::chrono::high_resolution_clock::now();
                auto num_step = static_cast<int>(
                    one_epo_step * epo / batch_size + (step) / batch_size);

                if (rank == 0) {
                    std::cout << "num_step: " << num_step << std::endl;
                }
                // hard coding
                if (rank < 3) {
                    std::shared_ptr<HELLM::LoRA::LoraModule> lora_module_ =
                        std::make_shared<HELLM::LoRA::LoraModule>(hemmer, 0);
                    lora_module_->optimizerStep_bert(task, num_step);
                } else if (rank >= 3 && rank < 6) {
                    std::shared_ptr<HELLM::LoRA::LoraModule> lora_module_ =
                        std::make_shared<HELLM::LoRA::LoraModule>(hemmer, 1);
                    lora_module_->optimizerStep_bert(task, num_step);
                } else if (rank == 6) {
                    std::shared_ptr<HELLM::LoRA::LoraModule> lora_module_ =
                        std::make_shared<HELLM::LoRA::LoraModule>(hemmer, 0);
                    lora_module_->optimizerStep_head_bert(task, num_step);
                } else {
                    std::shared_ptr<HELLM::LoRA::LoraModule> lora_module_ =
                        std::make_shared<HELLM::LoRA::LoraModule>(hemmer, 0);
                    lora_module_->optimizerStep_head2_bert(task, num_step);
                }

                HEaaN::CudaTools::cudaDeviceSynchronize();
                MPI_Barrier(MPI_COMM_WORLD);
                if (rank == 0) {
                    /* for (HEaaN::u64 layer_ = 0 ; layer_ < 2 ; ++layer_){
                        std::shared_ptr<HELLM::LoRA::LoraModule> lora_module_ =
                    std::make_shared<HELLM::LoRA::LoraModule>(hemmer, layer_);
                        //lora_module_->zeroAggGrad(LORA_TYPE);

                        //((step+1) ==
                    static_cast<HEaaN::u64>(one_epo_step/batch_size)) if (
                    ((step+1)%1000) == 0  && ( (step+1) != one_epo_step ) ) {
                            for (const char t : LORA_TYPE){
                                const std::string lora_t = std::string(1,t);
                                auto weight_a =
                    lora_module_->getCtxtTensor_lora("lora_wa_" + lora_t ,
                    0,0,0); auto weight_b =
                    lora_module_->getCtxtTensor_lora("lora_wb_" + lora_t ,
                    0,0,0); auto momentum_ma =
                    lora_module_->getCtxtTensor_lora("momentum_ma_"+lora_t,
                    0,0,0); auto momentum_va =
                    lora_module_->getCtxtTensor_lora("momentum_va_"+lora_t,
                    0,0,0); auto momentum_mb =
                    lora_module_->getCtxtTensor_lora("momentum_mb_"+lora_t,
                    0,0,0); auto momentum_vb =
                    lora_module_->getCtxtTensor_lora("momentum_vb_"+lora_t,
                    0,0,0); lora_module_->saveCtxtTensor_lora(weight_a,
                    std::to_string((step+1)) + "step/lora_wa_" + lora_t ,
                    0,0,0); lora_module_->saveCtxtTensor_lora(weight_b,
                    std::to_string((step+1)) + "step/lora_wb_" + lora_t ,
                    0,0,0); lora_module_->saveCtxtTensor_lora(momentum_ma,
                    std::to_string((step+1)) + "step/momentum_ma_" + lora_t  ,
                    0,0,0); lora_module_->saveCtxtTensor_lora(momentum_va,
                    std::to_string((step+1)) + "step/momentum_va_" + lora_t  ,
                    0,0,0); lora_module_->saveCtxtTensor_lora(momentum_mb,
                    std::to_string((step+1)) + "step/momentum_mb_" + lora_t  ,
                    0,0,0); lora_module_->saveCtxtTensor_lora(momentum_vb,
                    std::to_string((step+1)) + "step/momentum_vb_" + lora_t  ,
                    0,0,0);
                            }
                            if (layer_ == 0) {
                                auto wfinal1_a =
                    lora_module_->getCtxtTensor_lora("wfinal1_weight_a", 0,0,0);
                                auto wfinal1_b =
                    lora_module_->getCtxtTensor_lora("wfinal1_weight_b", 0,0,0);
                                auto weight =
                    lora_module_->getCtxtTensor_lora("wfinal2_weight",0,0,0);
                                auto momentum_ma_head =
                    lora_module_->getCtxtTensor_lora("momentum_ma_head" ,
                    0,0,0); auto momentum_va_head =
                    lora_module_->getCtxtTensor_lora("momentum_va_head" ,
                    0,0,0); auto momentum_mb_head =
                    lora_module_->getCtxtTensor_lora("momentum_mb_head" ,
                    0,0,0); auto momentum_vb_head =
                    lora_module_->getCtxtTensor_lora("momentum_vb_head" ,
                    0,0,0); auto momentum_m_head =
                    lora_module_->getCtxtTensor_lora("momentum_m_head" , 0,0,0);
                                auto momentum_v_head =
                    lora_module_->getCtxtTensor_lora("momentum_v_head" , 0,0,0);
                                lora_module_->saveCtxtTensor_lora(wfinal1_a,
                    std::to_string((step+1)) + "step/wfinal1_weight_a" , 0, 0,
                    0); lora_module_->saveCtxtTensor_lora(wfinal1_b,
                    std::to_string((step+1)) + "step/wfinal1_weight_b" , 0, 0,
                    0); lora_module_->saveCtxtTensor_lora(weight,
                    std::to_string((step+1)) + "step/wfinal2_weight", 0, 0, 0);
                                lora_module_->saveCtxtTensor_lora(momentum_ma_head,
                    std::to_string((step+1)) + "step/momentum_ma_head"  ,
                    0,0,0); lora_module_->saveCtxtTensor_lora(momentum_va_head,
                    std::to_string((step+1)) + "step/momentum_va_head"  ,
                    0,0,0); lora_module_->saveCtxtTensor_lora(momentum_mb_head,
                    std::to_string((step+1)) + "step/momentum_mb_head"  ,
                    0,0,0); lora_module_->saveCtxtTensor_lora(momentum_vb_head,
                    std::to_string((step+1)) + "step/momentum_vb_head"  ,
                    0,0,0); lora_module_->saveCtxtTensor_lora(momentum_m_head,
                    std::to_string((step+1)) + "step/momentum_m_head"  , 0,0,0);
                                lora_module_->saveCtxtTensor_lora(momentum_v_head,
                    std::to_string((step+1)) + "step/momentum_v_head"  , 0,0,0);
                            }
                        }
                    } */

                    auto end_opti = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed_opti =
                        end_opti - start_opti;
                    std::cout << "weight saving time: " << elapsed_opti.count()
                              << " s" << std::endl;
                }
                HEaaN::CudaTools::cudaDeviceSynchronize();
                MPI_Barrier(MPI_COMM_WORLD);
            }

            if (rank == 0) {
                auto end_ = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_ = end_ - start_;
                std::cout << "1 step running time: " << elapsed_.count() << " s"
                          << std::endl;
            }
        }
        if (rank == 0) {
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "Total running time: " << elapsed.count() << " s"
                      << std::endl;
        }

        if (rank == 0) {
            for (HEaaN::u64 layer_ = 0; layer_ < 2; ++layer_) {
                std::shared_ptr<HELLM::LoRA::LoraModule> lora_module_ =
                    std::make_shared<HELLM::LoRA::LoraModule>(hemmer, layer_);
                const std::string cases = "ab";
                for (const char t : LORA_TYPE) {
                    const std::string lora_t = std::string(1, t);
                    const std::string layer = std::to_string(layer_);
                    for (const char a : cases) {
                        const std::string case_ = std::string(1, a);
                        const std::string case_type = case_ + "_" + lora_t;
                        auto weight_ = lora_module_->getCtxtTensor_lora(
                            "lora_w" + case_type, 0, 0, 0);
                        auto momentum_m = lora_module_->getCtxtTensor_lora(
                            "momentum_m" + case_type, 0, 0, 0);
                        auto momentum_v = lora_module_->getCtxtTensor_lora(
                            "momentum_v" + case_type, 0, 0, 0);
                        lora_module_->saveCtxtTensor_lora(
                            weight_,
                            std::to_string(epo) + "epo/lora_w" + case_type, 0,
                            0, 0);
                        lora_module_->saveCtxtTensor_lora(
                            momentum_m,
                            std::to_string(epo) + "epo/momentum_m" + case_type,
                            0, 0, 0);
                        lora_module_->saveCtxtTensor_lora(
                            momentum_v,
                            std::to_string(epo) + "epo/momentum_v" + case_type,
                            0, 0, 0);
                    }
                }
                if (layer_ == 0) {
                    auto weight = lora_module_->getCtxtTensor_lora(
                        "wfinal2_weight", 0, 0, 0);
                    auto momentum_m_head = lora_module_->getCtxtTensor_lora(
                        "momentum_m_head", 0, 0, 0);
                    auto momentum_v_head = lora_module_->getCtxtTensor_lora(
                        "momentum_v_head", 0, 0, 0);
                    lora_module_->saveCtxtTensor_lora(
                        weight, std::to_string(epo) + "epo/wfinal2_weight", 0,
                        0, 0);
                    lora_module_->saveCtxtTensor_lora(
                        momentum_m_head,
                        std::to_string(epo) + "epo/momentum_m_head", 0, 0, 0);
                    lora_module_->saveCtxtTensor_lora(
                        momentum_v_head,
                        std::to_string(epo) + "epo/momentum_v_head", 0, 0, 0);

                    for (const char a : cases) {
                        const std::string case_ = std::string(1, a);
                        auto wfinal1_ = lora_module_->getCtxtTensor_lora(
                            "wfinal1_weight_" + case_, 0, 0, 0);
                        auto momentum_m_head = lora_module_->getCtxtTensor_lora(
                            "momentum_m" + case_ + "_head", 0, 0, 0);
                        auto momentum_v_head = lora_module_->getCtxtTensor_lora(
                            "momentum_v" + case_ + "_head", 0, 0, 0);
                        lora_module_->saveCtxtTensor_lora(
                            wfinal1_,
                            std::to_string(epo) + "epo/wfinal1_weight_" + case_,
                            0, 0, 0);
                        lora_module_->saveCtxtTensor_lora(momentum_m_head,
                                                          std::to_string(epo) +
                                                              "epo/momentum_m" +
                                                              case_ + "_head",
                                                          0, 0, 0);
                        lora_module_->saveCtxtTensor_lora(momentum_v_head,
                                                          std::to_string(epo) +
                                                              "epo/momentum_v" +
                                                              case_ + "_head",
                                                          0, 0, 0);
                    }
                }
            }
        }
    }

    delete hemmer;
    HELLM::cleanUpMatrixTransformer();
}
