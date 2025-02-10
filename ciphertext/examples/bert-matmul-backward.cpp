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

#include <ATen/core/TensorBody.h>

#include <string>
#include <torch/csrc/autograd/generated/variable_factories.h>
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

static void
compareGradWeight(const HELLM::HEMMer *hemmer, const torch::Tensor &tensor,
                  const torch::Tensor &grad_y,
                  const std::vector<std::vector<HELLM::CtxtTensor>> &res) {
    auto tensor_splitted =
        torch::mm(tensor, grad_y)
            .view({n_heads_in, head_dim, n_heads_out, head_dim})
            .transpose(1, 2);
    for (int i = 0; i < half_heads_in; ++i) {
        for (int j = 0; j < n_heads_out; ++j) {
            compareTensor2(hemmer,
                           torch::stack({tensor_splitted[i * 2][j],
                                         tensor_splitted[i * 2 + 1][j]}),
                           res[i][j]);
        }
    }
}

static void compareGradInput(const HELLM::HEMMer *hemmer,
                             const torch::Tensor &tensor,
                             const torch::Tensor &weight,
                             const std::vector<HELLM::CtxtTensor> &res) {
    auto tensor_splitted = torch::mm(tensor, weight)
                               .view({max_seq_len, n_heads_in, head_dim})
                               .transpose(0, 1);
    for (int i = 0; i < half_heads_in; ++i) {
        compareTensor2(
            hemmer,
            torch::stack({tensor_splitted[i * 2], tensor_splitted[i * 2 + 1]}),
            res[i]);
    }
}

std::vector<HELLM::CtxtTensor> encryptOutput(const HELLM::HEMMer *hemmer,
                                             const torch::Tensor &tensor) {

    auto tensor_splitted =
        tensor.view({max_seq_len, n_heads_out, head_dim}).transpose(0, 1);
    std::vector<HELLM::CtxtTensor> ctxt_tensor;
    ctxt_tensor.reserve(half_heads_out);
    for (int i = 0; i < half_heads_out; ++i) {
        ctxt_tensor.push_back(hemmer->encrypt2(tensor_splitted[i * 2],
                                               tensor_splitted[i * 2 + 1]));
    }

    return ctxt_tensor;
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
    bool grad_weight = true;
    bool grad_input = true;
    if (argc > 1) {
        switch (argc) {
        case 4:
            grad_input = std::stoi(argv[3]) == 1;
        case 3:
            grad_weight = std::stoi(argv[2]) == 1;
        case 2:
            lora_type = argv[1];
        }
    }

    HEaaN::setCurrentCudaDevice(0);
    auto *hemmer = new HELLM::HEMMer{HELLM::HEMMer::genHEMMer(0)};

    std::cout << std::endl << "Lora Type: " << lora_type << std::endl;
    std::cout << "Weight Dimension: (" << dim_in << ", " << dim_out << ")"
              << std::endl;

    auto grad_q = torch::rand({max_seq_len, dim_out});
    auto grad_k = torch::rand({max_seq_len, dim_out});
    auto grad_v = torch::rand({max_seq_len, dim_out});

    auto ctxt_grad_q = encryptOutput(hemmer, grad_q);
    auto ctxt_grad_k = encryptOutput(hemmer, grad_k);
    auto ctxt_grad_v = encryptOutput(hemmer, grad_v);

    std::vector<HELLM::CtxtTensor> temp_q, temp_k, temp_v;
    temp_q.reserve(n_heads_out);
    temp_k.reserve(n_heads_out);
    temp_v.reserve(n_heads_out);
    for (int i = 0; i < n_heads_out; ++i) {
        temp_q.emplace_back(ctxt_grad_q[0]);
        temp_k.emplace_back(ctxt_grad_k[0]);
        temp_v.emplace_back(ctxt_grad_v[0]);
    }

    if (grad_weight) {
        std::cout << std::endl << "Grad Weight Test" << std::endl;

        auto tensor_x = torch::rand({max_seq_len, dim_in});
        auto transpose_x = tensor_x.transpose(0, 1);

        auto ctxt_tensor_x = encryptInput(hemmer, tensor_x);

        auto grad_wq = torch::zeros({dim_in, dim_out});
        auto grad_wk = torch::zeros({dim_in, dim_out});
        auto grad_wv = torch::zeros({dim_in, dim_out});

        auto ctxt_grad_wq = encryptWeight(hemmer, grad_wq);
        auto ctxt_grad_wk = encryptWeight(hemmer, grad_wk);
        auto ctxt_grad_wv = encryptWeight(hemmer, grad_wv);

        {
            for (auto &tensor : ctxt_tensor_x) {
                hemmer->transposeInplace(tensor);
            }

            if (lora_type.find('q') != std::string::npos) {
                for (int i = 0; i < half_heads_out; ++i) {
                    auto tmp = ctxt_grad_q[i];
                    hemmer->complexPackingRowInplace(tmp);
                    tmp = hemmer->packedMatMulPreRev(tmp, 5);
                    hemmer->splitInTwo(tmp, temp_q[i * 2], temp_q[i * 2 + 1]);
                }
            }

            if (lora_type.find('k') != std::string::npos) {
                for (int i = 0; i < half_heads_out; ++i) {
                    auto tmp = ctxt_grad_k[i];
                    hemmer->complexPackingRowInplace(tmp);
                    tmp = hemmer->packedMatMulPreRev(tmp, 5);
                    hemmer->splitInTwo(tmp, temp_k[i * 2], temp_k[i * 2 + 1]);
                }
            }

            if (lora_type.find('v') != std::string::npos) {
                for (int i = 0; i < half_heads_out; ++i) {
                    auto tmp = ctxt_grad_v[i];
                    hemmer->complexPackingRowInplace(tmp);
                    tmp = hemmer->packedMatMulPreRev(tmp, 5);
                    hemmer->splitInTwo(tmp, temp_v[i * 2], temp_v[i * 2 + 1]);
                }
            }

            for (int i = 0; i < half_heads_in; ++i) {
                auto &tmp_x = ctxt_tensor_x[i];
                hemmer->complexPackingInplace(tmp_x);
                tmp_x = hemmer->packedMatMulPre(tmp_x);

                std::vector<HEaaN::Ciphertext> tmp;
                hemmer->packedMatMulPreRot(tmp_x, tmp);

                for (int j = 0; j < n_heads_out; ++j) {
                    if (lora_type.find('q') != std::string::npos) {
                        ctxt_grad_wq[i][j] =
                            hemmer->packedMatMulCCReuse(tmp, temp_q[j]);
                    }

                    if (lora_type.find('k') != std::string::npos) {
                        ctxt_grad_wk[i][j] =
                            hemmer->packedMatMulCCReuse(tmp, temp_k[j]);
                    }

                    if (lora_type.find('v') != std::string::npos) {
                        ctxt_grad_wv[i][j] =
                            hemmer->packedMatMulCCReuse(tmp, temp_v[j]);
                    }
                }
            }

            if (lora_type.find('q') != std::string::npos) {
                for (auto &row : ctxt_grad_wq) {
                    for (auto &tensor : row) {
                        tensor = hemmer->repackCC(tensor);
                    }
                    for (int i = 0; i < half_heads_out; ++i) {
                        hemmer->bootstrap2(row[i * 2], row[i * 2 + 1]);
                    }
                }
                compareGradWeight(hemmer, transpose_x, grad_q, ctxt_grad_wq);
            }

            if (lora_type.find('k') != std::string::npos) {
                for (auto &row : ctxt_grad_wk) {
                    for (auto &tensor : row) {
                        tensor = hemmer->repackCC(tensor);
                    }
                    for (int i = 0; i < half_heads_out; ++i) {
                        hemmer->bootstrap2(row[i * 2], row[i * 2 + 1]);
                    }
                }
                compareGradWeight(hemmer, transpose_x, grad_k, ctxt_grad_wk);
            }

            if (lora_type.find('v') != std::string::npos) {
                for (auto &row : ctxt_grad_wv) {
                    for (auto &tensor : row) {
                        tensor = hemmer->repackCC(tensor);
                    }
                    for (int i = 0; i < half_heads_out; ++i) {
                        hemmer->bootstrap2(row[i * 2], row[i * 2 + 1]);
                    }
                }
                compareGradWeight(hemmer, transpose_x, grad_v, ctxt_grad_wv);
            }

            std::size_t free = 0, total = 0;
            cudaMemGetInfo(&free, &total);
            std::cout << "(GPU) " << (total - free) / 1000000 << "MiB / "
                      << total / 1000000 << "MiB" << std::endl;
        }

        ctxt_tensor_x = encryptInput(hemmer, tensor_x);

        {
            HEaaN::CudaTools::cudaDeviceSynchronize();
            std::cout << "Construct Done" << std::endl;
            auto start = std::chrono::high_resolution_clock::now();

            for (auto &tensor : ctxt_tensor_x) {
                hemmer->transposeInplace(tensor);
            }
            HEaaN::CudaTools::cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "Grad weight transpose Elapsed time: "
                      << elapsed.count() << " s" << std::endl;
            start = std::chrono::high_resolution_clock::now();

            if (lora_type.find('q') != std::string::npos) {
                for (int i = 0; i < half_heads_out; ++i) {
                    auto tmp = ctxt_grad_q[i];
                    hemmer->complexPackingRowInplace(tmp);
                    tmp = hemmer->packedMatMulPreRev(tmp, 5);
                    hemmer->splitInTwo(tmp, temp_q[i * 2], temp_q[i * 2 + 1]);
                }
            }

            if (lora_type.find('k') != std::string::npos) {
                for (int i = 0; i < half_heads_out; ++i) {
                    auto tmp = ctxt_grad_k[i];
                    hemmer->complexPackingRowInplace(tmp);
                    tmp = hemmer->packedMatMulPreRev(tmp, 5);
                    hemmer->splitInTwo(tmp, temp_k[i * 2], temp_k[i * 2 + 1]);
                }
            }

            if (lora_type.find('v') != std::string::npos) {
                for (int i = 0; i < half_heads_out; ++i) {
                    auto tmp = ctxt_grad_v[i];
                    hemmer->complexPackingRowInplace(tmp);
                    tmp = hemmer->packedMatMulPreRev(tmp, 5);
                    hemmer->splitInTwo(tmp, temp_v[i * 2], temp_v[i * 2 + 1]);
                }
            }
            HEaaN::CudaTools::cudaDeviceSynchronize();
            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            std::cout << "Grad weight pre reverse Elapsed time: "
                      << elapsed.count() << " s" << std::endl;
            start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < half_heads_in; ++i) {
                auto &tmp_x = ctxt_tensor_x[i];
                hemmer->complexPackingInplace(tmp_x);
                tmp_x = hemmer->packedMatMulPre(tmp_x);

                std::vector<HEaaN::Ciphertext> tmp;
                hemmer->packedMatMulPreRot(tmp_x, tmp);

                for (int j = 0; j < n_heads_out; ++j) {
                    if (lora_type.find('q') != std::string::npos) {
                        ctxt_grad_wq[i][j] =
                            hemmer->packedMatMulCCReuse(tmp, temp_q[j]);
                    }

                    if (lora_type.find('k') != std::string::npos) {
                        ctxt_grad_wk[i][j] =
                            hemmer->packedMatMulCCReuse(tmp, temp_k[j]);
                    }

                    if (lora_type.find('v') != std::string::npos) {
                        ctxt_grad_wv[i][j] =
                            hemmer->packedMatMulCCReuse(tmp, temp_v[j]);
                    }
                }
            }
            HEaaN::CudaTools::cudaDeviceSynchronize();
            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            std::cout << "Grad weight Elapsed time: " << elapsed.count() << " s"
                      << std::endl;
            start = std::chrono::high_resolution_clock::now();

            if (lora_type.find('q') != std::string::npos) {
                for (auto &row : ctxt_grad_wq) {
                    for (auto &tensor : row) {
                        tensor = hemmer->repackCC(tensor);
                    }
                    for (int i = 0; i < half_heads_out; ++i) {
                        hemmer->bootstrap2(row[i * 2], row[i * 2 + 1]);
                    }
                }
            }

            if (lora_type.find('k') != std::string::npos) {
                for (auto &row : ctxt_grad_wk) {
                    for (auto &tensor : row) {
                        tensor = hemmer->repackCC(tensor);
                    }
                    for (int i = 0; i < half_heads_out; ++i) {
                        hemmer->bootstrap2(row[i * 2], row[i * 2 + 1]);
                    }
                }
            }

            if (lora_type.find('v') != std::string::npos) {
                for (auto &row : ctxt_grad_wv) {
                    for (auto &tensor : row) {
                        tensor = hemmer->repackCC(tensor);
                    }
                    for (int i = 0; i < half_heads_out; ++i) {
                        hemmer->bootstrap2(row[i * 2], row[i * 2 + 1]);
                    }
                }
            }
            HEaaN::CudaTools::cudaDeviceSynchronize();
            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            std::cout << "Grad weight repack Elapsed time: " << elapsed.count()
                      << " s" << std::endl;

            std::size_t free = 0, total = 0;
            cudaMemGetInfo(&free, &total);
            std::cout << "(GPU) " << (total - free) / 1000000 << "MiB / "
                      << total / 1000000 << "MiB" << std::endl;
        }
    }

    if (grad_input) {
        std::cout << std::endl << "Grad Input Test" << std::endl;

        auto weight_q = torch::rand({dim_in, dim_out});
        auto weight_k = torch::rand({dim_in, dim_out});
        auto weight_v = torch::rand({dim_in, dim_out});

        auto transpose_wq = weight_q.transpose(0, 1);
        auto transpose_wk = weight_k.transpose(0, 1);
        auto transpose_wv = weight_v.transpose(0, 1);

        auto ctxt_weight_q = encryptWeight(hemmer, weight_q);
        auto ctxt_weight_k = encryptWeight(hemmer, weight_k);
        auto ctxt_weight_v = encryptWeight(hemmer, weight_v);

        auto grad_xq = torch::zeros({max_seq_len, dim_in});
        auto grad_xk = torch::zeros({max_seq_len, dim_in});
        auto grad_xv = torch::zeros({max_seq_len, dim_in});

        auto ctxt_grad_xq = encryptInput(hemmer, grad_xq);
        auto ctxt_grad_xk = encryptInput(hemmer, grad_xk);
        auto ctxt_grad_xv = encryptInput(hemmer, grad_xv);

        {
            if (lora_type.find('q') != std::string::npos) {
                for (auto &row : ctxt_weight_q) {
                    for (auto &tensor : row) {
                        hemmer->transposeInplace(tensor, 8);
                    }
                }
            }

            if (lora_type.find('k') != std::string::npos) {
                for (auto &row : ctxt_weight_k) {
                    for (auto &tensor : row) {
                        hemmer->transposeInplace(tensor, 8);
                    }
                }
            }

            if (lora_type.find('v') != std::string::npos) {
                for (auto &row : ctxt_weight_v) {
                    for (auto &tensor : row) {
                        hemmer->transposeInplace(tensor, 8);
                    }
                }
            }

            if (lora_type.find('q') != std::string::npos) {
                for (auto &row : ctxt_weight_q) {
                    for (auto &tensor : row) {
                        hemmer->complexPackingRowInplace(tensor);
                        tensor = hemmer->packedMatMulPreRev(tensor);
                    }
                }
            }

            if (lora_type.find('k') != std::string::npos) {
                for (auto &row : ctxt_weight_k) {
                    for (auto &tensor : row) {
                        hemmer->complexPackingRowInplace(tensor);
                        tensor = hemmer->packedMatMulPreRev(tensor);
                    }
                }
            }

            if (lora_type.find('v') != std::string::npos) {
                for (auto &row : ctxt_weight_v) {
                    for (auto &tensor : row) {
                        hemmer->complexPackingRowInplace(tensor);
                        tensor = hemmer->packedMatMulPreRev(tensor);
                    }
                }
            }

            for (int i = 0; i < half_heads_out; ++i) {
                if (lora_type.find('q') != std::string::npos) {
                    auto tmp_pre = ctxt_grad_q[i];
                    hemmer->complexPackingInplace(tmp_pre);
                    tmp_pre = hemmer->packedMatMulPre(tmp_pre, 5);
                    hemmer->splitInTwo(tmp_pre, temp_q[i * 2],
                                       temp_q[i * 2 + 1]);

                    std::vector<HEaaN::Ciphertext> tmp;
                    hemmer->packedMatMulPreRot(temp_q[i * 2], tmp);

                    for (int j = 0; j < half_heads_in; ++j) {
                        if (i == 0) {
                            ctxt_grad_xq[j] = hemmer->packedMatMulCCReuse(
                                tmp, ctxt_weight_q[j][i * 2]);
                        } else {
                            hemmer->addInplace(
                                ctxt_grad_xq[j],
                                hemmer->packedMatMulCCReuse(
                                    tmp, ctxt_weight_q[j][i * 2]));
                        }
                    }

                    hemmer->packedMatMulPreRot(temp_q[i * 2 + 1], tmp);
                    for (int j = 0; j < half_heads_in; ++j) {
                        hemmer->addInplace(
                            ctxt_grad_xq[j],
                            hemmer->packedMatMulCCReuse(
                                tmp, ctxt_weight_q[j][i * 2 + 1]));
                    }
                }

                if (lora_type.find('k') != std::string::npos) {
                    auto tmp_pre = ctxt_grad_k[i];
                    hemmer->complexPackingInplace(tmp_pre);
                    tmp_pre = hemmer->packedMatMulPre(tmp_pre, 5);
                    hemmer->splitInTwo(tmp_pre, temp_k[i * 2],
                                       temp_k[i * 2 + 1]);

                    std::vector<HEaaN::Ciphertext> tmp;
                    hemmer->packedMatMulPreRot(temp_k[i * 2], tmp);
                    for (int j = 0; j < half_heads_in; ++j) {
                        if (i == 0) {
                            ctxt_grad_xk[j] = hemmer->packedMatMulCCReuse(
                                tmp, ctxt_weight_k[j][i * 2]);
                        } else {
                            hemmer->addInplace(
                                ctxt_grad_xk[j],
                                hemmer->packedMatMulCCReuse(
                                    tmp, ctxt_weight_k[j][i * 2]));
                        }
                    }

                    hemmer->packedMatMulPreRot(temp_k[i * 2 + 1], tmp);
                    for (int j = 0; j < half_heads_in; ++j) {
                        hemmer->addInplace(
                            ctxt_grad_xk[j],
                            hemmer->packedMatMulCCReuse(
                                tmp, ctxt_weight_k[j][i * 2 + 1]));
                    }
                }

                if (lora_type.find('v') != std::string::npos) {
                    auto tmp_pre = ctxt_grad_v[i];
                    hemmer->complexPackingInplace(tmp_pre);
                    tmp_pre = hemmer->packedMatMulPre(tmp_pre, 5);
                    hemmer->splitInTwo(tmp_pre, temp_v[i * 2],
                                       temp_v[i * 2 + 1]);

                    std::vector<HEaaN::Ciphertext> tmp;
                    hemmer->packedMatMulPreRot(temp_v[i * 2], tmp);
                    for (int j = 0; j < half_heads_in; ++j) {
                        if (i == 0) {
                            ctxt_grad_xv[j] = hemmer->packedMatMulCCReuse(
                                tmp, ctxt_weight_v[j][i * 2]);
                        } else {
                            hemmer->addInplace(
                                ctxt_grad_xv[j],
                                hemmer->packedMatMulCCReuse(
                                    tmp, ctxt_weight_v[j][i * 2]));
                        }
                    }

                    hemmer->packedMatMulPreRot(temp_v[i * 2 + 1], tmp);
                    for (int j = 0; j < half_heads_in; ++j) {
                        hemmer->addInplace(
                            ctxt_grad_xv[j],
                            hemmer->packedMatMulCCReuse(
                                tmp, ctxt_weight_v[j][i * 2 + 1]));
                    }
                }
            }

            if (lora_type.find('q') != std::string::npos) {
                for (auto &tensor : ctxt_grad_xq) {
                    tensor = hemmer->repackCC(tensor);
                }
                for (int i = 0; i < n_iter_quarter_in - 1; ++i) {
                    hemmer->bootstrap2(ctxt_grad_xq[i * 2],
                                       ctxt_grad_xq[i * 2 + 1]);
                }
                if (half_heads_in % 2 != 0) {
                    hemmer->bootstrap(ctxt_grad_xq[half_heads_in - 1]);
                }
                compareGradInput(hemmer, grad_q, transpose_wq, ctxt_grad_xq);
            }

            if (lora_type.find('k') != std::string::npos) {
                for (auto &tensor : ctxt_grad_xk) {
                    tensor = hemmer->repackCC(tensor);
                }
                for (int i = 0; i < n_iter_quarter_in - 1; ++i) {
                    hemmer->bootstrap2(ctxt_grad_xk[i * 2],
                                       ctxt_grad_xk[i * 2 + 1]);
                }
                if (half_heads_in % 2 != 0) {
                    hemmer->bootstrap(ctxt_grad_xk[half_heads_in - 1]);
                }
                compareGradInput(hemmer, grad_k, transpose_wk, ctxt_grad_xk);
            }

            if (lora_type.find('v') != std::string::npos) {
                for (auto &tensor : ctxt_grad_xv) {
                    tensor = hemmer->repackCC(tensor);
                }
                for (int i = 0; i < n_iter_quarter_in - 1; ++i) {
                    hemmer->bootstrap2(ctxt_grad_xv[i * 2],
                                       ctxt_grad_xv[i * 2 + 1]);
                }
                if (half_heads_in % 2 != 0) {
                    hemmer->bootstrap(ctxt_grad_xv[half_heads_in - 1]);
                }
                compareGradInput(hemmer, grad_v, transpose_wv, ctxt_grad_xv);
            }

            std::size_t free = 0, total = 0;
            cudaMemGetInfo(&free, &total);
            std::cout << "(GPU) " << (total - free) / 1000000 << "MiB / "
                      << total / 1000000 << "MiB" << std::endl;
        }

        ctxt_weight_q = encryptWeight(hemmer, weight_q);
        ctxt_weight_k = encryptWeight(hemmer, weight_k);
        ctxt_weight_v = encryptWeight(hemmer, weight_v);

        {
            HEaaN::CudaTools::cudaDeviceSynchronize();
            std::cout << "Construct Done" << std::endl;
            auto start = std::chrono::high_resolution_clock::now();

            if (lora_type.find('q') != std::string::npos) {
                for (auto &row : ctxt_weight_q) {
                    for (auto &tensor : row) {
                        hemmer->transposeInplace(tensor, 8);
                    }
                }
            }

            if (lora_type.find('k') != std::string::npos) {
                for (auto &row : ctxt_weight_k) {
                    for (auto &tensor : row) {
                        hemmer->transposeInplace(tensor, 8);
                    }
                }
            }

            if (lora_type.find('v') != std::string::npos) {
                for (auto &row : ctxt_weight_v) {
                    for (auto &tensor : row) {
                        hemmer->transposeInplace(tensor, 8);
                    }
                }
            }
            HEaaN::CudaTools::cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "Grad input transpose Elapsed time: "
                      << elapsed.count() << " s" << std::endl;
            start = std::chrono::high_resolution_clock::now();

            if (lora_type.find('q') != std::string::npos) {
                for (auto &row : ctxt_weight_q) {
                    for (auto &tensor : row) {
                        hemmer->complexPackingRowInplace(tensor);
                        tensor = hemmer->packedMatMulPreRev(tensor);
                    }
                }
            }

            if (lora_type.find('k') != std::string::npos) {
                for (auto &row : ctxt_weight_k) {
                    for (auto &tensor : row) {
                        hemmer->complexPackingRowInplace(tensor);
                        tensor = hemmer->packedMatMulPreRev(tensor);
                    }
                }
            }

            if (lora_type.find('v') != std::string::npos) {
                for (auto &row : ctxt_weight_v) {
                    for (auto &tensor : row) {
                        hemmer->complexPackingRowInplace(tensor);
                        tensor = hemmer->packedMatMulPreRev(tensor);
                    }
                }
            }
            HEaaN::CudaTools::cudaDeviceSynchronize();
            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            std::cout << "Grad input pre reverse Elapsed time: "
                      << elapsed.count() << " s" << std::endl;
            start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < half_heads_out; ++i) {
                if (lora_type.find('q') != std::string::npos) {
                    auto tmp_pre = ctxt_grad_q[i];
                    hemmer->complexPackingInplace(tmp_pre);
                    tmp_pre = hemmer->packedMatMulPre(tmp_pre, 5);
                    hemmer->splitInTwo(tmp_pre, temp_q[i * 2],
                                       temp_q[i * 2 + 1]);

                    std::vector<HEaaN::Ciphertext> tmp;
                    hemmer->packedMatMulPreRot(temp_q[i * 2], tmp);

                    for (int j = 0; j < half_heads_in; ++j) {
                        if (i == 0) {
                            ctxt_grad_xq[j] = hemmer->packedMatMulCCReuse(
                                tmp, ctxt_weight_q[j][i * 2]);
                        } else {
                            hemmer->addInplace(
                                ctxt_grad_xq[j],
                                hemmer->packedMatMulCCReuse(
                                    tmp, ctxt_weight_q[j][i * 2]));
                        }
                    }

                    hemmer->packedMatMulPreRot(temp_q[i * 2 + 1], tmp);
                    for (int j = 0; j < half_heads_in; ++j) {
                        hemmer->addInplace(
                            ctxt_grad_xq[j],
                            hemmer->packedMatMulCCReuse(
                                tmp, ctxt_weight_q[j][i * 2 + 1]));
                    }
                }

                if (lora_type.find('k') != std::string::npos) {
                    auto tmp_pre = ctxt_grad_k[i];
                    hemmer->complexPackingInplace(tmp_pre);
                    tmp_pre = hemmer->packedMatMulPre(tmp_pre, 5);
                    hemmer->splitInTwo(tmp_pre, temp_k[i * 2],
                                       temp_k[i * 2 + 1]);

                    std::vector<HEaaN::Ciphertext> tmp;
                    hemmer->packedMatMulPreRot(temp_k[i * 2], tmp);
                    for (int j = 0; j < half_heads_in; ++j) {
                        if (i == 0) {
                            ctxt_grad_xk[j] = hemmer->packedMatMulCCReuse(
                                tmp, ctxt_weight_k[j][i * 2]);
                        } else {
                            hemmer->addInplace(
                                ctxt_grad_xk[j],
                                hemmer->packedMatMulCCReuse(
                                    tmp, ctxt_weight_k[j][i * 2]));
                        }
                    }

                    hemmer->packedMatMulPreRot(temp_k[i * 2 + 1], tmp);
                    for (int j = 0; j < half_heads_in; ++j) {
                        hemmer->addInplace(
                            ctxt_grad_xk[j],
                            hemmer->packedMatMulCCReuse(
                                tmp, ctxt_weight_k[j][i * 2 + 1]));
                    }
                }

                if (lora_type.find('v') != std::string::npos) {
                    auto tmp_pre = ctxt_grad_v[i];
                    hemmer->complexPackingInplace(tmp_pre);
                    tmp_pre = hemmer->packedMatMulPre(tmp_pre, 5);
                    hemmer->splitInTwo(tmp_pre, temp_v[i * 2],
                                       temp_v[i * 2 + 1]);

                    std::vector<HEaaN::Ciphertext> tmp;
                    hemmer->packedMatMulPreRot(temp_v[i * 2], tmp);
                    for (int j = 0; j < half_heads_in; ++j) {
                        if (i == 0) {
                            ctxt_grad_xv[j] = hemmer->packedMatMulCCReuse(
                                tmp, ctxt_weight_v[j][i * 2]);
                        } else {
                            hemmer->addInplace(
                                ctxt_grad_xv[j],
                                hemmer->packedMatMulCCReuse(
                                    tmp, ctxt_weight_v[j][i * 2]));
                        }
                    }

                    hemmer->packedMatMulPreRot(temp_v[i * 2 + 1], tmp);
                    for (int j = 0; j < half_heads_in; ++j) {
                        hemmer->addInplace(
                            ctxt_grad_xv[j],
                            hemmer->packedMatMulCCReuse(
                                tmp, ctxt_weight_v[j][i * 2 + 1]));
                    }
                }
            }
            HEaaN::CudaTools::cudaDeviceSynchronize();
            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            std::cout << "Grad input Elapsed time: " << elapsed.count() << " s"
                      << std::endl;
            start = std::chrono::high_resolution_clock::now();

            if (lora_type.find('q') != std::string::npos) {
                for (auto &tensor : ctxt_grad_xq) {
                    tensor = hemmer->repackCC(tensor);
                }
                for (int i = 0; i < n_iter_quarter_in - 1; ++i) {
                    hemmer->bootstrap2(ctxt_grad_xq[i * 2],
                                       ctxt_grad_xq[i * 2 + 1]);
                }
                if (half_heads_in % 2 != 0) {
                    hemmer->bootstrap(ctxt_grad_xq[half_heads_in - 1]);
                }
            }

            if (lora_type.find('k') != std::string::npos) {
                for (auto &tensor : ctxt_grad_xk) {
                    tensor = hemmer->repackCC(tensor);
                }
                for (int i = 0; i < n_iter_quarter_in - 1; ++i) {
                    hemmer->bootstrap2(ctxt_grad_xk[i * 2],
                                       ctxt_grad_xk[i * 2 + 1]);
                }
                if (half_heads_in % 2 != 0) {
                    hemmer->bootstrap(ctxt_grad_xk[half_heads_in - 1]);
                }
            }

            if (lora_type.find('v') != std::string::npos) {
                for (auto &tensor : ctxt_grad_xv) {
                    tensor = hemmer->repackCC(tensor);
                }
                for (int i = 0; i < n_iter_quarter_in - 1; ++i) {
                    hemmer->bootstrap2(ctxt_grad_xv[i * 2],
                                       ctxt_grad_xv[i * 2 + 1]);
                }
                if (half_heads_in % 2 != 0) {
                    hemmer->bootstrap(ctxt_grad_xv[half_heads_in - 1]);
                }
            }
            HEaaN::CudaTools::cudaDeviceSynchronize();
            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            std::cout << "Grad input repack Elapsed time: " << elapsed.count()
                      << " s" << std::endl;

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
