////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2024 Crypto Lab Inc. All rights reserved.               //
//                                                                            //
// This source code is protected under international copyright law.           //
// All rights reserved and protected by the copyright holders.                //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "HELLM/HEMMer.hpp"
#include "HELLM/HETensor.hpp"
#include "HELLM/ModelArgs.hpp"
#include "HEaaN/Ciphertext.hpp"

#include <ATen/core/TensorBody.h>

#include <torch/script.h>
#include <torch/torch.h>

const int max_seq_len = HELLM::ModelArgs::MAX_SEQ_LEN;
const int dim_in = HELLM::ModelArgs::DIM;
const int dim_out = HELLM::ModelArgs::DIM;
const int head_dim = HELLM::ModelArgs::HEAD_DIM;
const int n_heads_in = dim_in / head_dim;
const int half_heads_in = n_heads_in / 2;
const int n_iter_quarter_in = (half_heads_in + 1) / 2;
const int n_heads_out = dim_out / head_dim;
const int half_heads_out = n_heads_out / 2;
const int n_iter_quarter_out = (half_heads_out + 1) / 2;

static void compareTensor2(const HELLM::HEMMer *hemmer,
                           const torch::Tensor &tensor,
                           const HELLM::CtxtTensor &ctxt_tensor,
                           double max_error = std::pow(2.0, -10)) {
    auto dec_tensor = hemmer->decrypt2(ctxt_tensor);

    auto diff = torch::max(torch::abs(tensor - dec_tensor)).item<double>();
    if (diff > max_error) {
        std::cout << diff << std::endl;
    }
}

static void compareOutput(const HELLM::HEMMer *hemmer,
                          const torch::Tensor &tensor,
                          const torch::Tensor &weight,
                          const std::vector<HELLM::CtxtTensor> &res) {
    auto tensor_splitted = torch::mm(tensor, weight)
                               .view({max_seq_len, n_heads_out, head_dim})
                               .transpose(0, 1);
    for (int i = 0; i < half_heads_out; ++i) {
        compareTensor2(
            hemmer,
            torch::stack({tensor_splitted[i * 2], tensor_splitted[i * 2 + 1]}),
            res[i]);
    }
}

std::vector<HELLM::CtxtTensor> encryptInput(const HELLM::HEMMer *hemmer,
                                            const torch::Tensor &tensor) {

    auto tensor_splitted =
        tensor.view({max_seq_len, n_heads_in, head_dim}).transpose(0, 1);
    std::vector<HELLM::CtxtTensor> ctxt_tensor;
    ctxt_tensor.reserve(half_heads_in);
    for (int i = 0; i < half_heads_in; ++i) {
        ctxt_tensor.push_back(hemmer->encrypt2(tensor_splitted[i * 2],
                                               tensor_splitted[i * 2 + 1]));
    }

    return ctxt_tensor;
}

std::vector<std::vector<HELLM::PtxtTensor>>
encodeWeight(const HELLM::HEMMer *hemmer, const torch::Tensor &weight) {

    auto tensor_splitted =
        weight.view({n_heads_in, head_dim, n_heads_out, head_dim})
            .transpose(1, 2);
    std::vector<std::vector<HELLM::PtxtTensor>> ptxt_weight(n_iter_quarter_in);
    for (int i = 0; i < half_heads_in / 2; ++i) {
        ptxt_weight[i].reserve(n_heads_out);
        for (int j = 0; j < n_heads_out; ++j) {
            ptxt_weight[i].push_back(hemmer->encodeDiagonalToRow4(
                tensor_splitted[i * 4][j], tensor_splitted[i * 4 + 1][j],
                tensor_splitted[i * 4 + 2][j], tensor_splitted[i * 4 + 3][j],
                5));
        }
    }
    if (half_heads_in % 2 != 0) {
        int idx = n_iter_quarter_in - 1;
        for (int j = 0; j < n_heads_out; ++j) {
            ptxt_weight[idx].push_back(hemmer->encodeDiagonalToRow4(
                tensor_splitted[idx * 4][j], tensor_splitted[idx * 4 + 1][j],
                torch::zeros({head_dim, head_dim}),
                torch::zeros({head_dim, head_dim}), 5));
        }
    }

    return ptxt_weight;
}

std::vector<std::vector<HELLM::CtxtTensor>>
encryptWeight(const HELLM::HEMMer *hemmer, const torch::Tensor &weight) {

    auto tensor_splitted =
        weight.view({n_heads_in, head_dim, n_heads_out, head_dim})
            .transpose(1, 2);
    std::vector<std::vector<HELLM::CtxtTensor>> ctxt_weight(half_heads_in);
    for (int i = 0; i < half_heads_in; ++i) {
        ctxt_weight[i].reserve(n_heads_out);
        for (int j = 0; j < n_heads_out; ++j) {
            ctxt_weight[i].push_back(hemmer->encrypt2(
                tensor_splitted[i * 2][j], tensor_splitted[i * 2 + 1][j]));
        }
    }

    return ctxt_weight;
}

int main(int argc, char *argv[]) {
    std::string lora_type = "qkv";
    bool pcmm_test = true;
    bool ccmm_test = true;
    if (argc > 1) {
        switch (argc) {
        case 4:
            ccmm_test = std::stoi(argv[3]) == 1;
        case 3:
            pcmm_test = std::stoi(argv[2]) == 1;
        case 2:
            lora_type = argv[1];
        }
    }

    HEaaN::setCurrentCudaDevice(0);
    auto *hemmer = new HELLM::HEMMer{HELLM::HEMMer::genHEMMer(0)};

    std::cout << std::endl << "Lora Type: " << lora_type << std::endl;
    std::cout << "Weight Dimension: (" << dim_in << ", " << dim_out << ")"
              << std::endl;

    auto tensor_x = torch::rand({max_seq_len, dim_in});
    auto weight_q = torch::rand({dim_in, dim_out});
    auto weight_k = torch::rand({dim_in, dim_out});
    auto weight_v = torch::rand({dim_in, dim_out});

    auto ctxt_tensor_x = encryptInput(hemmer, tensor_x);

    std::vector<HELLM::CtxtTensor> temp_q, temp_k, temp_v;
    temp_q.reserve(n_heads_out);
    temp_k.reserve(n_heads_out);
    temp_v.reserve(n_heads_out);
    for (int i = 0; i < n_heads_out; ++i) {
        temp_q.emplace_back(ctxt_tensor_x[0]);
        temp_k.emplace_back(ctxt_tensor_x[0]);
        temp_v.emplace_back(ctxt_tensor_x[0]);
    }

    if (pcmm_test) {
        std::cout << std::endl << "PCMM Test" << std::endl;

        auto ptxt_weight_q = encodeWeight(hemmer, weight_q);
        auto ptxt_weight_k = encodeWeight(hemmer, weight_k);
        auto ptxt_weight_v = encodeWeight(hemmer, weight_v);
        {
            for (int i = 0; i < n_iter_quarter_in; ++i) {
                auto packed_x =
                    (i == n_iter_quarter_in - 1 && half_heads_in % 2 != 0)
                        ? ctxt_tensor_x[i * 2]
                        : hemmer->complexPacking(ctxt_tensor_x[i * 2],
                                                 ctxt_tensor_x[i * 2 + 1]);
                std::vector<HEaaN::Ciphertext> tmp;
                hemmer->matMulPre(packed_x, tmp);
                for (int j = 0; j < n_heads_out; ++j) {
                    if (lora_type.find('q') != std::string::npos) {
                        if (i == 0) {
                            temp_q[j] =
                                hemmer->matMulReUse(tmp, ptxt_weight_q[i][j]);
                        } else {
                            hemmer->addInplace(
                                temp_q[j],
                                hemmer->matMulReUse(tmp, ptxt_weight_q[i][j]));
                        }
                    }

                    if (lora_type.find('k') != std::string::npos) {
                        if (i == 0) {
                            temp_k[j] =
                                hemmer->matMulReUse(tmp, ptxt_weight_k[i][j]);
                        } else {
                            hemmer->addInplace(
                                temp_k[j],
                                hemmer->matMulReUse(tmp, ptxt_weight_k[i][j]));
                        }
                    }

                    if (lora_type.find('v') != std::string::npos) {
                        if (i == 0) {
                            temp_v[j] =
                                hemmer->matMulReUse(tmp, ptxt_weight_v[i][j]);
                        } else {
                            hemmer->addInplace(
                                temp_v[j],
                                hemmer->matMulReUse(tmp, ptxt_weight_v[i][j]));
                        }
                    }
                }
            }

            if (lora_type.find('q') != std::string::npos) {
                std::vector<HELLM::CtxtTensor> query;
                query.reserve(half_heads_out);
                for (int i = 0; i < half_heads_out; ++i) {
                    query.push_back(
                        hemmer->repack(temp_q[i * 2], temp_q[i * 2 + 1]));
                }
                compareOutput(hemmer, tensor_x, weight_q, query);
            }

            if (lora_type.find('k') != std::string::npos) {
                std::vector<HELLM::CtxtTensor> key;
                key.reserve(half_heads_out);
                for (int i = 0; i < half_heads_out; ++i) {
                    key.push_back(
                        hemmer->repack(temp_k[i * 2], temp_k[i * 2 + 1]));
                }
                compareOutput(hemmer, tensor_x, weight_k, key);
            }

            if (lora_type.find('v') != std::string::npos) {
                std::vector<HELLM::CtxtTensor> value;
                value.reserve(half_heads_out);
                for (int i = 0; i < half_heads_out; ++i) {
                    value.push_back(
                        hemmer->repack(temp_v[i * 2], temp_v[i * 2 + 1]));
                }
                compareOutput(hemmer, tensor_x, weight_v, value);
            }

            std::size_t free = 0, total = 0;
            cudaMemGetInfo(&free, &total);
            std::cout << "(GPU) " << (total - free) / 1000000 << "MiB / "
                      << total / 1000000 << "MiB" << std::endl;
        }

        {
            HEaaN::CudaTools::cudaDeviceSynchronize();
            std::cout << "Construct Done" << std::endl;
            auto start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < n_iter_quarter_in; ++i) {
                auto packed_x =
                    (i == n_iter_quarter_in - 1 && half_heads_in % 2 != 0)
                        ? ctxt_tensor_x[i * 2]
                        : hemmer->complexPacking(ctxt_tensor_x[i * 2],
                                                 ctxt_tensor_x[i * 2 + 1]);
                std::vector<HEaaN::Ciphertext> tmp;
                hemmer->matMulPre(packed_x, tmp);
                for (int j = 0; j < n_heads_out; ++j) {
                    if (lora_type.find('q') != std::string::npos) {
                        if (i == 0) {
                            temp_q[j] =
                                hemmer->matMulReUse(tmp, ptxt_weight_q[i][j]);
                        } else {
                            hemmer->addInplace(
                                temp_q[j],
                                hemmer->matMulReUse(tmp, ptxt_weight_q[i][j]));
                        }
                    }

                    if (lora_type.find('k') != std::string::npos) {
                        if (i == 0) {
                            temp_k[j] =
                                hemmer->matMulReUse(tmp, ptxt_weight_k[i][j]);
                        } else {
                            hemmer->addInplace(
                                temp_k[j],
                                hemmer->matMulReUse(tmp, ptxt_weight_k[i][j]));
                        }
                    }

                    if (lora_type.find('v') != std::string::npos) {
                        if (i == 0) {
                            temp_v[j] =
                                hemmer->matMulReUse(tmp, ptxt_weight_v[i][j]);
                        } else {
                            hemmer->addInplace(
                                temp_v[j],
                                hemmer->matMulReUse(tmp, ptxt_weight_v[i][j]));
                        }
                    }
                }
            }
            HEaaN::CudaTools::cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "PCMM Elapsed time: " << elapsed.count() << " s"
                      << std::endl;
            start = std::chrono::high_resolution_clock::now();

            if (lora_type.find('q') != std::string::npos) {
                std::vector<HELLM::CtxtTensor> query;
                query.reserve(half_heads_out);
                for (int i = 0; i < half_heads_out; ++i) {
                    query.push_back(
                        hemmer->repack(temp_q[i * 2], temp_q[i * 2 + 1]));
                }
            }

            if (lora_type.find('k') != std::string::npos) {
                std::vector<HELLM::CtxtTensor> key;
                key.reserve(half_heads_out);
                for (int i = 0; i < half_heads_out; ++i) {
                    key.push_back(
                        hemmer->repack(temp_k[i * 2], temp_k[i * 2 + 1]));
                }
            }

            if (lora_type.find('v') != std::string::npos) {
                std::vector<HELLM::CtxtTensor> value;
                value.reserve(half_heads_out);
                for (int i = 0; i < half_heads_out; ++i) {
                    value.push_back(
                        hemmer->repack(temp_v[i * 2], temp_v[i * 2 + 1]));
                }
            }
            HEaaN::CudaTools::cudaDeviceSynchronize();
            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            std::cout << "PCMM repack Elapsed time: " << elapsed.count() << " s"
                      << std::endl;

            std::size_t free = 0, total = 0;
            cudaMemGetInfo(&free, &total);
            std::cout << "(GPU) " << (total - free) / 1000000 << "MiB / "
                      << total / 1000000 << "MiB" << std::endl;
        }
    }

    if (ccmm_test) {
        std::cout << std::endl << "CCMM Test" << std::endl;

        auto ctxt_weight_q = encryptWeight(hemmer, weight_q);
        auto ctxt_weight_k = encryptWeight(hemmer, weight_k);
        auto ctxt_weight_v = encryptWeight(hemmer, weight_v);
        {
            for (int i = 0; i < n_iter_quarter_in; ++i) {
                auto packed_x =
                    (i == n_iter_quarter_in - 1 && half_heads_in % 2 != 0)
                        ? ctxt_tensor_x[i * 2]
                        : hemmer->complexPacking(ctxt_tensor_x[i * 2],
                                                 ctxt_tensor_x[i * 2 + 1]);
                std::vector<HEaaN::Ciphertext> tmp;
                hemmer->matMulPre(packed_x, tmp);

                for (int j = 0; j < n_heads_out; ++j) {
                    if (lora_type.find('q') != std::string::npos) {
                        auto packed_wq = (i == n_iter_quarter_in - 1 &&
                                          half_heads_in % 2 != 0)
                                             ? ctxt_weight_q[i * 2][j]
                                             : hemmer->complexPackingRev(
                                                   ctxt_weight_q[i * 2][j],
                                                   ctxt_weight_q[i * 2 + 1][j]);
                        hemmer->getEval().mult(packed_wq.get(), 0.5,
                                               packed_wq.get()); // FIX
                        auto pre_wq = hemmer->matMulPreRev(packed_wq);

                        if (i == 0) {
                            temp_q[j] = hemmer->matMulCCReUse(tmp, pre_wq);
                        } else {
                            hemmer->addInplace(
                                temp_q[j], hemmer->matMulCCReUse(tmp, pre_wq));
                        }
                    }

                    if (lora_type.find('k') != std::string::npos) {
                        auto packed_wk = (i == n_iter_quarter_in - 1 &&
                                          half_heads_in % 2 != 0)
                                             ? ctxt_weight_k[i * 2][j]
                                             : hemmer->complexPackingRev(
                                                   ctxt_weight_k[i * 2][j],
                                                   ctxt_weight_k[i * 2 + 1][j]);
                        hemmer->getEval().mult(packed_wk.get(), 0.5,
                                               packed_wk.get()); // FIX
                        auto pre_wk = hemmer->matMulPreRev(packed_wk);

                        if (i == 0) {
                            temp_k[j] = hemmer->matMulCCReUse(tmp, pre_wk);
                        } else {
                            hemmer->addInplace(
                                temp_k[j], hemmer->matMulCCReUse(tmp, pre_wk));
                        }
                    }

                    if (lora_type.find('v') != std::string::npos) {
                        auto packed_wv = (i == n_iter_quarter_in - 1 &&
                                          half_heads_in % 2 != 0)
                                             ? ctxt_weight_v[i * 2][j]
                                             : hemmer->complexPackingRev(
                                                   ctxt_weight_v[i * 2][j],
                                                   ctxt_weight_v[i * 2 + 1][j]);
                        hemmer->getEval().mult(packed_wv.get(), 0.5,
                                               packed_wv.get()); // FIX
                        auto pre_wv = hemmer->matMulPreRev(packed_wv);

                        if (i == 0) {
                            temp_v[j] = hemmer->matMulCCReUse(tmp, pre_wv);
                        } else {
                            hemmer->addInplace(
                                temp_v[j], hemmer->matMulCCReUse(tmp, pre_wv));
                        }
                    }
                }
            }

            if (lora_type.find('q') != std::string::npos) {
                std::vector<HELLM::CtxtTensor> query;
                query.reserve(half_heads_out);
                for (int i = 0; i < half_heads_out; ++i) {
                    query.push_back(
                        hemmer->repack(temp_q[i * 2], temp_q[i * 2 + 1]));
                }
                compareOutput(hemmer, tensor_x, weight_q, query);
            }

            if (lora_type.find('k') != std::string::npos) {
                std::vector<HELLM::CtxtTensor> key;
                key.reserve(half_heads_out);
                for (int i = 0; i < half_heads_out; ++i) {
                    key.push_back(
                        hemmer->repack(temp_k[i * 2], temp_k[i * 2 + 1]));
                }
                compareOutput(hemmer, tensor_x, weight_k, key);
            }

            if (lora_type.find('v') != std::string::npos) {
                std::vector<HELLM::CtxtTensor> value;
                value.reserve(half_heads_out);
                for (int i = 0; i < half_heads_out; ++i) {
                    value.push_back(
                        hemmer->repack(temp_v[i * 2], temp_v[i * 2 + 1]));
                }
                compareOutput(hemmer, tensor_x, weight_v, value);
            }

            std::size_t free = 0, total = 0;
            cudaMemGetInfo(&free, &total);
            std::cout << "(GPU) " << (total - free) / 1000000 << "MiB / "
                      << total / 1000000 << "MiB" << std::endl;
        }

        {
            HEaaN::CudaTools::cudaDeviceSynchronize();
            std::cout << "Construct Done" << std::endl;
            auto start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < n_iter_quarter_in; ++i) {
                auto packed_x =
                    (i == n_iter_quarter_in - 1 && half_heads_in % 2 != 0)
                        ? ctxt_tensor_x[i * 2]
                        : hemmer->complexPacking(ctxt_tensor_x[i * 2],
                                                 ctxt_tensor_x[i * 2 + 1]);
                std::vector<HEaaN::Ciphertext> tmp;
                hemmer->matMulPre(packed_x, tmp);

                for (int j = 0; j < n_heads_out; ++j) {
                    if (lora_type.find('q') != std::string::npos) {
                        auto packed_wq = (i == n_iter_quarter_in - 1 &&
                                          half_heads_in % 2 != 0)
                                             ? ctxt_weight_q[i * 2][j]
                                             : hemmer->complexPackingRev(
                                                   ctxt_weight_q[i * 2][j],
                                                   ctxt_weight_q[i * 2 + 1][j]);
                        hemmer->getEval().mult(packed_wq.get(), 0.5,
                                               packed_wq.get()); // FIX
                        auto pre_wq = hemmer->matMulPreRev(packed_wq);

                        if (i == 0) {
                            temp_q[j] = hemmer->matMulCCReUse(tmp, pre_wq);
                        } else {
                            hemmer->addInplace(
                                temp_q[j], hemmer->matMulCCReUse(tmp, pre_wq));
                        }
                    }

                    if (lora_type.find('k') != std::string::npos) {
                        auto packed_wk = (i == n_iter_quarter_in - 1 &&
                                          half_heads_in % 2 != 0)
                                             ? ctxt_weight_k[i * 2][j]
                                             : hemmer->complexPackingRev(
                                                   ctxt_weight_k[i * 2][j],
                                                   ctxt_weight_k[i * 2 + 1][j]);
                        hemmer->getEval().mult(packed_wk.get(), 0.5,
                                               packed_wk.get()); // FIX
                        auto pre_wk = hemmer->matMulPreRev(packed_wk);

                        if (i == 0) {
                            temp_k[j] = hemmer->matMulCCReUse(tmp, pre_wk);
                        } else {
                            hemmer->addInplace(
                                temp_k[j], hemmer->matMulCCReUse(tmp, pre_wk));
                        }
                    }

                    if (lora_type.find('v') != std::string::npos) {
                        auto packed_wv = (i == n_iter_quarter_in - 1 &&
                                          half_heads_in % 2 != 0)
                                             ? ctxt_weight_v[i * 2][j]
                                             : hemmer->complexPackingRev(
                                                   ctxt_weight_v[i * 2][j],
                                                   ctxt_weight_v[i * 2 + 1][j]);
                        hemmer->getEval().mult(packed_wv.get(), 0.5,
                                               packed_wv.get()); // FIX
                        auto pre_wv = hemmer->matMulPreRev(packed_wv);

                        if (i == 0) {
                            temp_v[j] = hemmer->matMulCCReUse(tmp, pre_wv);
                        } else {
                            hemmer->addInplace(
                                temp_v[j], hemmer->matMulCCReUse(tmp, pre_wv));
                        }
                    }
                }
            }
            HEaaN::CudaTools::cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "CCMM Elapsed time: " << elapsed.count() << " s"
                      << std::endl;
            start = std::chrono::high_resolution_clock::now();

            if (lora_type.find('q') != std::string::npos) {
                std::vector<HELLM::CtxtTensor> query;
                query.reserve(half_heads_out);
                for (int i = 0; i < half_heads_out; ++i) {
                    query.push_back(
                        hemmer->repack(temp_q[i * 2], temp_q[i * 2 + 1]));
                }
            }

            if (lora_type.find('k') != std::string::npos) {
                std::vector<HELLM::CtxtTensor> key;
                key.reserve(half_heads_out);
                for (int i = 0; i < half_heads_out; ++i) {
                    key.push_back(
                        hemmer->repack(temp_k[i * 2], temp_k[i * 2 + 1]));
                }
            }

            if (lora_type.find('v') != std::string::npos) {
                std::vector<HELLM::CtxtTensor> value;
                value.reserve(half_heads_out);
                for (int i = 0; i < half_heads_out; ++i) {
                    value.push_back(
                        hemmer->repack(temp_v[i * 2], temp_v[i * 2 + 1]));
                }
            }
            HEaaN::CudaTools::cudaDeviceSynchronize();
            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            std::cout << "CCMM repack Elapsed time: " << elapsed.count() << " s"
                      << std::endl;

            std::size_t free = 0, total = 0;
            cudaMemGetInfo(&free, &total);
            std::cout << "(GPU) " << (total - free) / 1000000 << "MiB / "
                      << total / 1000000 << "MiB" << std::endl;
        }
    }

    delete hemmer;
    HELLM::cleanUpMatrixTransformer();
    return 0;
}
