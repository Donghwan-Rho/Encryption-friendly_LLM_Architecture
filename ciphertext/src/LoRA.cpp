////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2024 Crypto Lab Inc. All rights reserved.               //
//                                                                            //
// This source code is protected under international copyright law.           //
// All rights reserved and protected by the copyright holders.                //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "HELLM/LoRA.hpp"
#include "HELLM/LoRARemezCoeff.hpp"

#include "HELLM/TransformerBlock.hpp"

#include "HEaaN-math/tools/PolynomialEvaluator.hpp"

#include "HELLM/utils/check_macros.hpp"

#ifdef HELLM_MULTIGPU
#include <mpi.h>
#endif

namespace HELLM::LoRA {

using namespace HEaaN;
using namespace Math;

void LoraModule::generateInitialLoraWeight(const std::string &lora_type) const {
    for (const char t : lora_type) {
        const std::string lora_t = std::string(1, t);
        auto lora_wa = torch::empty({2, 128, 128});
        lora_wa = torch::nn::init::kaiming_uniform_(
            lora_wa, std::sqrt(5), torch::kFanIn, torch::kReLU);

        auto lora_wb =
            torch::zeros({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});

        if (hemmer_->getRank() == 0) {
            std::cout << "lora_wa sampling " << t << std::endl;
            printing(hemmer_->encrypt2(lora_wa[0], lora_wa[1]));
        }

        saveCtxtTensor_lora(hemmer_->encrypt2(lora_wa[0], lora_wa[1]),
                            "lora_wa_" + lora_t, 0, 0, 0);
        saveCtxtTensor_lora(hemmer_->encrypt2(lora_wb[0], lora_wb[1]),
                            "lora_wb_" + lora_t, 0, 0, 0);

        saveCtxtTensor_lora(hemmer_->encrypt2(lora_wb[0], lora_wb[1]),
                            "momentum_ma_" + lora_t, 0, 0, 0);
        saveCtxtTensor_lora(hemmer_->encrypt2(lora_wb[0], lora_wb[1]),
                            "momentum_mb_" + lora_t, 0, 0, 0);
        saveCtxtTensor_lora(hemmer_->encrypt2(lora_wb[0], lora_wb[1]),
                            "momentum_va_" + lora_t, 0, 0, 0);
        saveCtxtTensor_lora(hemmer_->encrypt2(lora_wb[0], lora_wb[1]),
                            "momentum_vb_" + lora_t, 0, 0, 0);
    }

    if (layer_n_ == 0) {
        /* auto container =
        torch::jit::load("./data_2ly_mrpc/converted_weights.pth"); auto weight_a
        = container.attr("wfinal1_weight_a").toTensor(); auto weight_b =
        container.attr("wfinal1_weight_b").toTensor(); auto head =
        container.attr("wfinal2_weight").toTensor(); */
        auto lora_wa = torch::normal(
            ModelArgs::LORA_MEAN, std::sqrt(ModelArgs::LORA_SQUARE_STD),
            {2, ModelArgs::HEAD_DIM, ModelArgs::MAX_SEQ_LEN});

        auto lora_wb =
            torch::zeros({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});

        lora_wa =
            torch::normal(ModelArgs::LORA_MEAN, 0.02,
                          {2, ModelArgs::HEAD_DIM, ModelArgs::MAX_SEQ_LEN});
        saveCtxtTensor_lora(hemmer_->encrypt2(lora_wa[0], lora_wa[1]),
                            "wfinal1_weight_a", 0, 0, 0);
        lora_wa =
            torch::normal(ModelArgs::LORA_MEAN, 0.02,
                          {2, ModelArgs::HEAD_DIM, ModelArgs::MAX_SEQ_LEN});
        saveCtxtTensor_lora(hemmer_->encrypt2(lora_wa[0], lora_wa[1]),
                            "wfinal1_weight_b", 0, 0, 0);
        lora_wa =
            torch::normal(ModelArgs::LORA_MEAN, 0.02,
                          {2, ModelArgs::HEAD_DIM, ModelArgs::MAX_SEQ_LEN});
        saveCtxtTensor_lora(hemmer_->encrypt2(lora_wa[0], lora_wa[1]),
                            "wfinal2_weight", 0, 0, 0);

        // append momentum initialization.
        auto zeros =
            torch::zeros({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});
        saveCtxtTensor_lora(hemmer_->encrypt2(zeros[0], zeros[1]),
                            "momentum_ma_head", 0, 0, 0);
        saveCtxtTensor_lora(hemmer_->encrypt2(zeros[0], zeros[1]),
                            "momentum_mb_head", 0, 0, 0);
        saveCtxtTensor_lora(hemmer_->encrypt2(zeros[0], zeros[1]),
                            "momentum_va_head", 0, 0, 0);
        saveCtxtTensor_lora(hemmer_->encrypt2(zeros[0], zeros[1]),
                            "momentum_vb_head", 0, 0, 0);
        saveCtxtTensor_lora(hemmer_->encrypt2(zeros[0], zeros[1]),
                            "momentum_m_head", 0, 0, 0);
        saveCtxtTensor_lora(hemmer_->encrypt2(zeros[0], zeros[1]),
                            "momentum_v_head", 0, 0, 0);
    }
    // zeroAggGrad(lora_type);
}

void LoraModule::compareLoraWeight(const std::string &lora_type) const {
    std::cout << layer_n_ << "-th compare LoRA weight" << std::endl;

    for (const char t : lora_type) {
        const std::string lora_t = std::string(1, t);
        // hegith = 128, LOW_DIM = 8
        std::vector<torch::Tensor> tmp;
        for (u32 i = 0; i < ModelArgs::N_HEAD; ++i) {
            tmp.push_back(getTorchTensor("lora_wa_" + lora_t, i));
        }
        // 32 * [height, LOW_DIM] -> [32 * height, LOW_DIM]
        auto lora_wa = torch::cat(tmp, 0).contiguous();

        auto dec_lora_wa =
            hemmer_->decrypt2(getCtxtTensor("lora_wa_" + lora_t, 0, 0, 0));
        // [2, height, 16 * LOW_DIM] -> [2 * height, 16 * LOW_DIM]
        dec_lora_wa = torch::cat({dec_lora_wa[0], dec_lora_wa[1]}, 0);
        // [2 * height, 16 * LOW_DIM] -> 16 * [2 * height, LOW_DIM]
        //                            -> [32 * height, LOW_DIM]
        dec_lora_wa =
            torch::cat(torch::split(dec_lora_wa, ModelArgs::LOW_DIM, 1), 0);
        dec_lora_wa = dec_lora_wa.contiguous();

        auto max_err = (dec_lora_wa - lora_wa).abs().max().item<double>();
        std::cout << "lora weight_a (type " << lora_t
                  << ") max_err = " << max_err << std::endl;

        tmp.clear();
        for (u32 i = 0; i < ModelArgs::N_HEAD; ++i) {
            tmp.push_back(getTorchTensor("lora_wb_" + lora_t, i));
        }
        // 32 * [LOW_DIM, height] -> [LOW_DIM, 32 * height]
        auto lora_wb = torch::cat(tmp, 1).contiguous();

        auto dec_lora_wb =
            hemmer_->decrypt2(getCtxtTensor("lora_wb_" + lora_t, 0, 0, 0));
        // [2, 16 * LOW_DIM, height] -> [16 * LOW_DIM, 2 * height]
        dec_lora_wb = torch::cat({dec_lora_wb[0], dec_lora_wb[1]}, 1);
        // [16 * LOW_DIM, 2 * height] -> 16 * [LOW_DIM, 2 * height]
        //                            -> [LOW_DIM, 32 * height]
        dec_lora_wb =
            torch::cat(torch::split(dec_lora_wb, ModelArgs::LOW_DIM, 0), 1);
        dec_lora_wb = dec_lora_wb.contiguous();

        max_err = (dec_lora_wb - lora_wb).abs().max().item<double>();
        std::cout << "lora weight_b (type " << lora_t
                  << ") max_err = " << max_err << std::endl;
    }
    std::cout << std::endl;
}

void LoraModule::zeroGrad(const std::string &lora_type) const {
    auto grad_wa = torch::zeros({ModelArgs::HEAD_DIM, ModelArgs::MAX_SEQ_LEN});
    auto grad_wb = torch::zeros({ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});

    // auto grad_wa_split = torch::split(grad_wa, ModelArgs::LOW_DIM, 1);
    // auto grad_wb_split = torch::split(grad_wb, ModelArgs::LOW_DIM, 0);

    for (const char t : lora_type) {
        const std::string lora_t = std::string(1, t);
        // momentum gradients
        saveCtxtTensor(hemmer_->encrypt2(grad_wa, grad_wa), "grad_wa_" + lora_t,
                       0, 0, 0);
        saveCtxtTensor(hemmer_->encrypt2(grad_wb, grad_wb), "grad_wb_" + lora_t,
                       0, 0, 0);

        /* for (u32 i = 0; i < ModelArgs::N_HEAD / 2; ++i) {
            saveTorchTensor(grad_wa_split[i], "grad_wa_" + lora_t, i);
            saveTorchTensor(grad_wb_split[i], "grad_wb_" + lora_t, i);
        } */
    }
    // for head weight
    saveCtxtTensor(hemmer_->encrypt2(grad_wa, grad_wa), "grad_wa_head", 0, 0,
                   0);
    saveCtxtTensor(hemmer_->encrypt2(grad_wb, grad_wb), "grad_wb_head", 0, 0,
                   0);
}

void LoraModule::zeroAggGrad(const std::string &lora_type) const {
    auto grad_wa = torch::zeros({ModelArgs::HEAD_DIM, ModelArgs::MAX_SEQ_LEN});
    auto grad_wb = torch::zeros({ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});

    for (const char t : lora_type) {
        const std::string lora_t = std::string(1, t);
        // aggregate gradients
        saveCtxtTensor_lora(hemmer_->encrypt2(grad_wa, grad_wa),
                            "agg_grad_wa_" + lora_t, 0, 0, 0);
        saveCtxtTensor_lora(hemmer_->encrypt2(grad_wb, grad_wb),
                            "agg_grad_wb_" + lora_t, 0, 0, 0);
    }

    // if (layer_n_ == 0) {
    //     saveCtxtTensor_lora(hemmer_->encrypt2(grad_wa, grad_wa),
    //                         "agg_grad_wa_head", 0, 0, 0); // wfinal1_a
    //     saveCtxtTensor_lora(hemmer_->encrypt2(grad_wb, grad_wb),
    //                         "agg_grad_wb_head", 0, 0, 0); // wfinal1_b
    //     saveCtxtTensor_lora(hemmer_->encrypt2(grad_wa, grad_wb),
    //                         "agg_grad_w_head", 0, 0, 0); // wfinal2
    // }
}

void LoraModule::zeroAggGrad_head() const {
    auto grad_wa = torch::zeros({ModelArgs::HEAD_DIM, ModelArgs::MAX_SEQ_LEN});
    auto grad_wb = torch::zeros({ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});

    saveCtxtTensor_lora(hemmer_->encrypt2(grad_wa, grad_wa), "agg_grad_wa_head",
                        0, 0, 0); // wfinal1_a
    saveCtxtTensor_lora(hemmer_->encrypt2(grad_wb, grad_wb), "agg_grad_wb_head",
                        0, 0, 0); // wfinal1_b
}

void LoraModule::zeroAggGrad_head2() const {
    auto grad_wa = torch::zeros({ModelArgs::HEAD_DIM, ModelArgs::MAX_SEQ_LEN});
    auto grad_wb = torch::zeros({ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});

    saveCtxtTensor_lora(hemmer_->encrypt2(grad_wa, grad_wb), "agg_grad_w_head",
                        0, 0, 0); // wfinal2
}

void LoraModule::saveAggGrad(const CtxtTensor &grad,
                             const std::string &weight_name) const {
    auto agg_grad_lora =
        getCtxtTensor_lora("agg_grad_w" + weight_name, 0, 0, 0);
    hemmer_->getEval().add(agg_grad_lora.get(), grad.get(),
                           agg_grad_lora.get());
    saveCtxtTensor_lora(agg_grad_lora, "agg_grad_w" + weight_name, 0, 0, 0);
}

void LoraModule::printing(const CtxtTensor &tensor_vec) const {

    auto dec_tensor = hemmer_->decrypt2(tensor_vec);

    for (HELLM::i64 k = 0; k < 1; ++k) {
        for (HELLM::i64 i = 0; i < 1; ++i) {
            for (HELLM::i64 j = 0; j < 0 + 4; ++j) {
                std::cout.precision(10);
                std::cout << dec_tensor[k].index({i, j}).item<double>() << ", ";
            }
            std::cout << std::endl;
        }
        // std::cout << std::endl;
    }
    // std::cout << std::endl;
}

void LoraModule::optimizerStep(const std::string &lora_type) const {
    for (const char t : lora_type) {
        const std::string lora_t = std::string(1, t);
        auto weight = getCtxtTensor_lora("lora_wa_" + lora_t, 0, 0, 0);
        auto grad = getCtxtTensor("grad_wa_" + lora_t, 0, 0, 0);
        auto agg_grad = getCtxtTensor("agg_grad_wa_" + lora_t, 0, 0, 0);

        // TODO: ADAMW
        hemmer_->getEval().add(grad.get(), agg_grad.get(), grad.get());
        hemmer_->bootstrap2(weight, grad);
        hemmer_->getEval().mult(grad.get(), -ModelArgs::LEARNING_RATE,
                                grad.get());
        hemmer_->getEval().add(weight.get(), grad.get(), weight.get());
        saveCtxtTensor_lora(weight, "lora_wa_" + lora_t, 0, 0, 0);
        saveCtxtTensor(grad, "grad_wa_" + lora_t, 0, 0, 0);

        weight = getCtxtTensor_lora("lora_wb_" + lora_t, 0, 0, 0);
        grad = getCtxtTensor("grad_wb_" + lora_t, 0, 0, 0);
        agg_grad = getCtxtTensor("agg_grad_wb_" + lora_t, 0, 0, 0);

        // TODO: ADAMW
        hemmer_->getEval().add(grad.get(), agg_grad.get(), grad.get());
        hemmer_->bootstrap2(weight, grad);
        hemmer_->getEval().mult(grad.get(), -ModelArgs::LEARNING_RATE,
                                grad.get());
        hemmer_->getEval().add(weight.get(), grad.get(), weight.get());
        saveCtxtTensor_lora(weight, "lora_wb_" + lora_t, 0, 0, 0);
        saveCtxtTensor(grad, "grad_wb_" + lora_t, 0, 0, 0);
    }
    zeroAggGrad(lora_type);
}

void LoraModule::AdamW(const std::string &lora_t, const char *task,
                       int step) const {

    const std::string cases = "ab";
    const auto rank = hemmer_->getRank();

    auto lr = 0.0;
    if (strcmp(task, "R") == 0) {
        lr = ModelArgs::RTE_LRS[static_cast<HEaaN::u64>(step)];
    } else if (strcmp(task, "C") == 0) {
        lr = ModelArgs::COLA_LRS[static_cast<HEaaN::u64>(step)];
    } else if (strcmp(task, "M") == 0) {
        lr = ModelArgs::MRPC_LRS[static_cast<HEaaN::u64>(step)];
    } else if (strcmp(task, "S") == 0) {
        lr = ModelArgs::STSB_LRS[static_cast<HEaaN::u64>(step)];
    } else if (strcmp(task, "T") == 0) {
        lr = ModelArgs::SST2_LRS[static_cast<HEaaN::u64>(step)];
    } else if (strcmp(task, "Q") == 0) {
        lr = ModelArgs::QNLI_LRS[static_cast<HEaaN::u64>(step)];
    } else {
        std::cout << "error! unsupported task type" << std::endl;
        return;
    }

    if (rank == 0) {
        std::cout << "lr: " << lr << std::endl;
    }

    for (const char a : cases) {
        const std::string case_ = std::string(1, a);
        const std::string case_type_rank =
            case_ + "_" + lora_t + "_" + std::to_string(layer_n_);
        // std::cout << "rank = " << hemmer_->getRank() << ", case_type : " <<
        // case_type_rank << std::endl;
        auto weight =
            getCtxtTensor_lora("lora_w" + case_ + "_" + lora_t, 0, 0, 0);
        // Consider picking agg_grad.
        auto agg_grad = getCtxtTensor_lora("agg_grad_w" + case_ + "_" + lora_t,
                                           0, 0, 0); // if 12
        CtxtTensor theta{weight};                    // 12
        auto momentum_m = getCtxtTensor_lora(
            "momentum_m" + case_ + "_" + lora_t, 0, 0, 0); // if 8
        auto momentum_v = getCtxtTensor_lora(
            "momentum_v" + case_ + "_" + lora_t, 0, 0, 0); // if 7

        // theta_t = (1-learning_rate*weight_deacy)*grad
        hemmer_->getEval().mult(theta.get(), 1 - ModelArgs::WEIGHT_DECAY * lr,
                                theta.get()); // 11

        // 4 = minimum ExtBTS level
        if (momentum_m.get().getLevel() < 4 + 3 ||
            momentum_v.get().getLevel() < 4 + 4) {
            hemmer_->bootstrap2(momentum_m, momentum_v);
        }

        // momentum_m = beta_1*momoentum_m + (1-beta_1)*grad
        hemmer_->getEval().mult(momentum_m.get(), ModelArgs::BETA_1,
                                momentum_m.get()); // 11 // 7
        CtxtTensor grad_tmp{agg_grad};
        hemmer_->getEval().mult(grad_tmp.get(), 1 - ModelArgs::BETA_1,
                                grad_tmp.get());   // 11 // 11
        hemmer_->addInplace(momentum_m, grad_tmp); // 11 // 7

        // momentum_v = beta_2*momoentum_v + (1-beta_2)*grad^2
        hemmer_->getEval().mult(momentum_v.get(), ModelArgs::BETA_2,
                                momentum_v.get()); // 11 // 6
        CtxtTensor grad_square{agg_grad};
        hemmer_->getEval().square(grad_square.get(),
                                  grad_square.get()); // 11 // 11
        hemmer_->getEval().mult(grad_square.get(), 1 - ModelArgs::BETA_2,
                                grad_square.get());   // 10 // 10
        hemmer_->addInplace(momentum_v, grad_square); // 10 // 6

        CtxtTensor m_hat{momentum_m}; // 11 // 7
        CtxtTensor v_hat{momentum_v}; // 11 // 6
        // hemmer_->getEval().mult(m_hat.get(), ModelArgs::LEARNING_RATE*(1.0/(1
        // - std::pow(ModelArgs::BETA_1, step))), m_hat.get()); //10 // 6
        hemmer_->getEval().mult(
            m_hat.get(), lr * (1.0 / (1 - std::pow(ModelArgs::BETA_1, step))),
            m_hat.get()); // 10 // 6
        hemmer_->getEval().mult(v_hat.get(),
                                1.0 / (1 - std::pow(ModelArgs::BETA_2, step)),
                                v_hat.get()); // 10 // 5

        // inv_sqrt input
        hemmer_->getEval().add(v_hat.get(), ModelArgs::OPTI_EPS,
                               v_hat.get()); // 10
        /* if (hemmer_->getRank() == 0) {
            std::cout << "adam head invSqrt input level " <<
        v_hat.get().getLevel() << std::endl; // 8
        }
        */

        auto v_hat_dec = hemmer_->decrypt2(v_hat);
        auto max = v_hat_dec.max().item<double>();
        auto min = v_hat_dec.min().item<double>();
        std::cout << "lora_" << lora_t << "_" << case_ << " min = " << min
                  << ", max = " << max << std::endl;

        if (strcmp(task, "C") == 0) {
            // input: 10, output: 5
            hemmer_->getEval().mult(v_hat.get(), 5, v_hat.get()); // 10
            approxInverseSqrt_COLA(hemmer_->getEval(), hemmer_->getBtp(),
                                   v_hat.get(), v_hat.get(), 3);
            hemmer_->getEval().mult(v_hat.get(), std::sqrt(5), v_hat.get());
        } else if (strcmp(task, "M") == 0) {
            hemmer_->getEval().mult(v_hat.get(), 3, v_hat.get()); // 7
            approxInverseSqrt_MRPC(hemmer_->getEval(), hemmer_->getBtp(),
                                   v_hat.get(), v_hat.get(), 3);
            hemmer_->getEval().mult(v_hat.get(), std::sqrt(3), v_hat.get());
        } else if (strcmp(task, "R") == 0) {
            hemmer_->getEval().mult(v_hat.get(), 10, v_hat.get());
            approxInverseSqrt_RTE(hemmer_->getEval(), hemmer_->getBtp(),
                                  v_hat.get(), v_hat.get(), 3);
            hemmer_->getEval().mult(v_hat.get(), std::sqrt(10), v_hat.get());
        } else if (strcmp(task, "S") == 0) {
            hemmer_->getEval().mult(v_hat.get(), 1.0 / 100, v_hat.get());
            approxInverseSqrt_STSB(hemmer_->getEval(), hemmer_->getBtp(),
                                   v_hat.get(), v_hat.get(), 3);
            hemmer_->getEval().mult(v_hat.get(), 1.0 / std::sqrt(100),
                                    v_hat.get());
        } else if (strcmp(task, "T") == 0) {
            // hemmer_->getEval().mult(v_hat.get(), 2, v_hat.get()); // 10
            approxInverseSqrt_SST2(hemmer_->getEval(), hemmer_->getBtp(),
                                   v_hat.get(), v_hat.get(), 3);
            // hemmer_->getEval().mult(v_hat.get(), std::sqrt(2), v_hat.get());
        } else if (strcmp(task, "Q") == 0) {
            // hemmer_->getEval().mult(v_hat.get(), 2, v_hat.get()); // 10
            approxInverseSqrt_SST2(hemmer_->getEval(), hemmer_->getBtp(),
                                   v_hat.get(), v_hat.get(), 3);
            // hemmer_->getEval().mult(v_hat.get(), std::sqrt(2), v_hat.get());
        } else {
            std::cout << "There is no such task!" << std::endl;
            return;
        }

        hemmer_->hadamardMultInplace(m_hat, v_hat);
        hemmer_->getEval().sub(theta.get(), m_hat.get(), theta.get());

        // weight BTS
        hemmer_->bootstrap(theta);
        saveCtxtTensor_lora(theta, "lora_w" + case_ + "_" + lora_t, 0, 0, 0);
        saveCtxtTensor_lora(momentum_m, "momentum_m" + case_ + "_" + lora_t, 0,
                            0, 0);
        saveCtxtTensor_lora(momentum_v, "momentum_v" + case_ + "_" + lora_t, 0,
                            0, 0);
    }
}

void LoraModule::AdamW_head(const char *task, int step) const {

    const std::string cases = "ab";

    auto lr = 0.0;

    if (strcmp(task, "R") == 0) {
        lr = ModelArgs::RTE_LRS[static_cast<HEaaN::u64>(step)];
    } else if (strcmp(task, "C") == 0) {
        lr = ModelArgs::COLA_LRS[static_cast<HEaaN::u64>(step)];
    } else if (strcmp(task, "M") == 0) {
        lr = ModelArgs::MRPC_LRS[static_cast<HEaaN::u64>(step)];
    } else if (strcmp(task, "S") == 0) {
        lr = ModelArgs::STSB_LRS[static_cast<HEaaN::u64>(step)];
    } else if (strcmp(task, "T") == 0) {
        lr = ModelArgs::SST2_LRS[static_cast<HEaaN::u64>(step)];
    } else if (strcmp(task, "Q") == 0) {
        lr = ModelArgs::QNLI_LRS[static_cast<HEaaN::u64>(step)];
    } else {
        std::cout << "error! unsupported task type" << std::endl;
        return;
    }

    for (const char a : cases) {
        const std::string case_ = std::string(1, a);

        auto weight = getCtxtTensor_lora("wfinal1_weight_" + case_, 0, 0, 0);
        auto agg_grad =
            getCtxtTensor_lora("agg_grad_w" + case_ + "_head", 0, 0, 0); // 5
        CtxtTensor theta{weight};                                        // 12
        auto momentum_m =
            getCtxtTensor_lora("momentum_m" + case_ + "_head", 0, 0, 0);
        auto momentum_v =
            getCtxtTensor_lora("momentum_v" + case_ + "_head", 0, 0, 0);

        // theta_t = (1-learning_rate*weight_deacy)*grad
        hemmer_->getEval().mult(theta.get(), 1 - ModelArgs::WEIGHT_DECAY * lr,
                                theta.get());

        if (momentum_m.get().getLevel() < 4 + 3 ||
            momentum_v.get().getLevel() < 4 + 4) {
            hemmer_->bootstrap2(momentum_m, momentum_v);
        }

        // momentum_m = beta_1*momoentum_m + (1-beta_1)*grad
        hemmer_->getEval().mult(momentum_m.get(), ModelArgs::BETA_1,
                                momentum_m.get());
        CtxtTensor grad_tmp{agg_grad}; // 5
        hemmer_->getEval().mult(grad_tmp.get(), 1 - ModelArgs::BETA_1,
                                grad_tmp.get()); // 4
        hemmer_->addInplace(momentum_m, grad_tmp);

        // momentum_v = beta_2*momoentum_v + (1-beta_2)*grad^2
        hemmer_->getEval().mult(momentum_v.get(), ModelArgs::BETA_2,
                                momentum_v.get());
        CtxtTensor grad_square{agg_grad};
        hemmer_->getEval().square(grad_square.get(), grad_square.get());
        hemmer_->getEval().mult(grad_square.get(), 1 - ModelArgs::BETA_2,
                                grad_square.get());
        hemmer_->addInplace(momentum_v, grad_square);

        CtxtTensor m_hat{momentum_m};
        CtxtTensor v_hat{momentum_v};
        // hemmer_->getEval().mult(m_hat.get(), ModelArgs::LEARNING_RATE*(1.0/(1
        // - std::pow(ModelArgs::BETA_1, step))), m_hat.get());
        hemmer_->getEval().mult(
            m_hat.get(), lr * (1.0 / (1 - std::pow(ModelArgs::BETA_1, step))),
            m_hat.get());
        hemmer_->getEval().mult(v_hat.get(),
                                1.0 / (1 - std::pow(ModelArgs::BETA_2, step)),
                                v_hat.get());

        // inv_sqrt input
        hemmer_->getEval().add(v_hat.get(), ModelArgs::OPTI_EPS, v_hat.get());

        auto v_hat_dec = hemmer_->decrypt2(v_hat);
        auto max = v_hat_dec.max().item<double>();
        auto min = v_hat_dec.min().item<double>();
        std::cout << "final head_" << case_ << " min = " << min
                  << ", max = " << max << std::endl;

        if (strcmp(task, "C") == 0) {
            // input: 10, output:
            hemmer_->getEval().mult(v_hat.get(), 5, v_hat.get()); // 10
            approxInverseSqrt_COLA(hemmer_->getEval(), hemmer_->getBtp(),
                                   v_hat.get(), v_hat.get(), 3);
            hemmer_->getEval().mult(v_hat.get(), std::sqrt(5), v_hat.get());
        } else if (strcmp(task, "M") == 0) {
            hemmer_->getEval().mult(v_hat.get(), 3, v_hat.get()); // 7
            approxInverseSqrt_MRPC(hemmer_->getEval(), hemmer_->getBtp(),
                                   v_hat.get(), v_hat.get(), 3);
            hemmer_->getEval().mult(v_hat.get(), std::sqrt(3), v_hat.get());
        } else if (strcmp(task, "R") == 0) {
            hemmer_->getEval().mult(v_hat.get(), 10, v_hat.get());
            approxInverseSqrt_RTE(hemmer_->getEval(), hemmer_->getBtp(),
                                  v_hat.get(), v_hat.get(), 3);
            hemmer_->getEval().mult(v_hat.get(), std::sqrt(10), v_hat.get());
        } else if (strcmp(task, "S") == 0) {
            hemmer_->getEval().mult(v_hat.get(), 1.0 / 100, v_hat.get());
            approxInverseSqrt_STSB(hemmer_->getEval(), hemmer_->getBtp(),
                                   v_hat.get(), v_hat.get(), 3);
            hemmer_->getEval().mult(v_hat.get(), 1.0 / std::sqrt(100),
                                    v_hat.get());
        } else if (strcmp(task, "T") == 0) {
            // hemmer_->getEval().mult(v_hat.get(), 2, v_hat.get()); // 10
            approxInverseSqrt_SST2(hemmer_->getEval(), hemmer_->getBtp(),
                                   v_hat.get(), v_hat.get(), 3);
            // hemmer_->getEval().mult(v_hat.get(), std::sqrt(2), v_hat.get());
        } else if (strcmp(task, "Q") == 0) {
            // hemmer_->getEval().mult(v_hat.get(), 2, v_hat.get()); // 10
            approxInverseSqrt_SST2(hemmer_->getEval(), hemmer_->getBtp(),
                                   v_hat.get(), v_hat.get(), 3);
            // hemmer_->getEval().mult(v_hat.get(), std::sqrt(2), v_hat.get());
        } else {
            std::cout << "There is no such task!" << std::endl;
            return;
        }

        hemmer_->hadamardMultInplace(m_hat, v_hat);
        hemmer_->getEval().sub(theta.get(), m_hat.get(), theta.get());
        // theta is an updated weight.

        // weight BTS
        hemmer_->bootstrap(theta);
        saveCtxtTensor_lora(theta, "wfinal1_weight_" + case_, 0, 0, 0);
        saveCtxtTensor_lora(momentum_m, "momentum_m" + case_ + "_head", 0, 0,
                            0);
        saveCtxtTensor_lora(momentum_v, "momentum_v" + case_ + "_head", 0, 0,
                            0);
    }
}

void LoraModule::AdamW_head2(const char *task, int step) const {

    auto lr = 0.0;

    if (strcmp(task, "R") == 0) {
        lr = ModelArgs::RTE_LRS[static_cast<HEaaN::u64>(step)];
    } else if (strcmp(task, "C") == 0) {
        lr = ModelArgs::COLA_LRS[static_cast<HEaaN::u64>(step)];
    } else if (strcmp(task, "M") == 0) {
        lr = ModelArgs::MRPC_LRS[static_cast<HEaaN::u64>(step)];
    } else if (strcmp(task, "S") == 0) {
        lr = ModelArgs::STSB_LRS[static_cast<HEaaN::u64>(step)];
    } else if (strcmp(task, "T") == 0) {
        lr = ModelArgs::SST2_LRS[static_cast<HEaaN::u64>(step)];
    } else if (strcmp(task, "Q") == 0) {
        lr = ModelArgs::QNLI_LRS[static_cast<HEaaN::u64>(step)];
    } else {
        std::cout << "error! unsupported task type" << std::endl;
        return;
    }

    auto weight = getCtxtTensor_lora("wfinal2_weight", 0, 0, 0);
    auto agg_grad = getCtxtTensor_lora("agg_grad_w_head", 0, 0, 0);
    CtxtTensor theta{weight}; // 12
    auto momentum_m = getCtxtTensor_lora("momentum_m_head", 0, 0, 0);
    auto momentum_v = getCtxtTensor_lora("momentum_v_head", 0, 0, 0);

    // theta_t = (1-learning_rate*weight_deacy)*grad
    hemmer_->getEval().mult(theta.get(), 1 - ModelArgs::WEIGHT_DECAY * lr,
                            theta.get());

    if (momentum_m.get().getLevel() < 4 + 3 ||
        momentum_v.get().getLevel() < 4 + 4) {
        hemmer_->bootstrap2(momentum_m, momentum_v);
    }

    // momentum_m = beta_1*momoentum_m + (1-beta_1)*grad
    hemmer_->getEval().mult(momentum_m.get(), ModelArgs::BETA_1,
                            momentum_m.get());
    CtxtTensor grad_tmp{agg_grad};
    hemmer_->getEval().mult(grad_tmp.get(), 1 - ModelArgs::BETA_1,
                            grad_tmp.get());
    hemmer_->addInplace(momentum_m, grad_tmp);

    // momentum_v = beta_2*momoentum_v + (1-beta_2)*grad^2
    hemmer_->getEval().mult(momentum_v.get(), ModelArgs::BETA_2,
                            momentum_v.get());
    CtxtTensor grad_square{agg_grad};
    hemmer_->getEval().square(grad_square.get(), grad_square.get());
    hemmer_->getEval().mult(grad_square.get(), 1 - ModelArgs::BETA_2,
                            grad_square.get());
    hemmer_->addInplace(momentum_v, grad_square);

    CtxtTensor m_hat{momentum_m};
    CtxtTensor v_hat{momentum_v};
    hemmer_->getEval().mult(
        m_hat.get(), lr * (1.0 / (1 - std::pow(ModelArgs::BETA_1, step))),
        m_hat.get());
    hemmer_->getEval().mult(v_hat.get(),
                            1.0 / (1 - std::pow(ModelArgs::BETA_2, step)),
                            v_hat.get());

    // inv_sqrt input
    hemmer_->getEval().add(v_hat.get(), ModelArgs::OPTI_EPS, v_hat.get());

    auto v_hat_dec = hemmer_->decrypt2(v_hat);
    auto max = v_hat_dec.max().item<double>();
    auto min = v_hat_dec.min().item<double>();
    std::cout << "final head2  min = " << min << ", max = " << max << std::endl;

    if (strcmp(task, "C") == 0) {
        // input: 10, output:
        // TODO: put a dividing uppder bound value.
        hemmer_->getEval().mult(v_hat.get(), 5, v_hat.get()); // 10
        approxInverseSqrt_COLA(hemmer_->getEval(), hemmer_->getBtp(),
                               v_hat.get(), v_hat.get(), 3);
        hemmer_->getEval().mult(v_hat.get(), std::sqrt(5), v_hat.get());
    } else if (strcmp(task, "M") == 0) {
        hemmer_->getEval().mult(v_hat.get(), 3, v_hat.get()); // 7
        approxInverseSqrt_MRPC(hemmer_->getEval(), hemmer_->getBtp(),
                               v_hat.get(), v_hat.get(), 3);
        hemmer_->getEval().mult(v_hat.get(), std::sqrt(3), v_hat.get());
    } else if (strcmp(task, "R") == 0) {
        hemmer_->getEval().mult(v_hat.get(), 10, v_hat.get());
        approxInverseSqrt_RTE(hemmer_->getEval(), hemmer_->getBtp(),
                              v_hat.get(), v_hat.get(), 3);
        hemmer_->getEval().mult(v_hat.get(), std::sqrt(10), v_hat.get());
    } else if (strcmp(task, "S") == 0) {
        hemmer_->getEval().mult(v_hat.get(), 1.0 / 100, v_hat.get());
        approxInverseSqrt_STSB(hemmer_->getEval(), hemmer_->getBtp(),
                               v_hat.get(), v_hat.get(), 3);
        hemmer_->getEval().mult(v_hat.get(), 1.0 / std::sqrt(100), v_hat.get());
    } else if (strcmp(task, "T") == 0) {
        // hemmer_->getEval().mult(v_hat.get(), 2, v_hat.get()); // 10
        approxInverseSqrt_SST2(hemmer_->getEval(), hemmer_->getBtp(),
                               v_hat.get(), v_hat.get(), 3);
        // hemmer_->getEval().mult(v_hat.get(), std::sqrt(2), v_hat.get());
    } else if (strcmp(task, "Q") == 0) {
        // hemmer_->getEval().mult(v_hat.get(), 2, v_hat.get()); // 10
        approxInverseSqrt_SST2(hemmer_->getEval(), hemmer_->getBtp(),
                               v_hat.get(), v_hat.get(), 3);
        // hemmer_->getEval().mult(v_hat.get(), std::sqrt(2), v_hat.get());
    } else {
        std::cout << "There is no such task!" << std::endl;
        return;
    }

    hemmer_->hadamardMultInplace(m_hat, v_hat);
    hemmer_->getEval().sub(theta.get(), m_hat.get(), theta.get());
    // theta is an updated weight.

    hemmer_->bootstrap(theta);

    saveCtxtTensor_lora(theta, "wfinal2_weight", 0, 0, 0);
    saveCtxtTensor_lora(momentum_m, "momentum_m_head", 0, 0, 0);
    saveCtxtTensor_lora(momentum_v, "momentum_v_head", 0, 0, 0);
}

// lr: customized learning rate.
// void LoraModule::optimizerStep_bert(const std::string &lora_type, const char
// *task, int step) const {
void LoraModule::optimizerStep_bert(const char *task, int step) const {

    // origin version
    /* for (const char t : lora_type) {
        const std::string lora_t = std::string(1, t);
        AdamW(lora_t, task, step);
    } */

    // hard coding for GPU rank and qkv
    auto rank = hemmer_->getRank();
    if (rank == 0) {
        AdamW("q", task, step);
        zeroAggGrad("q");
    } else if (rank == 1) {
        AdamW("k", task, step);
        zeroAggGrad("k");
    } else if (rank == 2) {
        AdamW("v", task, step);
        zeroAggGrad("v");
    } else if (rank == 3) {
        AdamW("q", task, step);
        zeroAggGrad("q");
    } else if (rank == 4) {
        AdamW("k", task, step);
        zeroAggGrad("k");
    } else if (rank == 5) {
        AdamW("v", task, step);
        zeroAggGrad("v");
    }
}

void LoraModule::optimizerStep_head_bert(const char *task, int step) const {
    AdamW_head(task, step);
    zeroAggGrad_head();
}

void LoraModule::optimizerStep_head2_bert(const char *task, int step) const {
    AdamW_head2(task, step);
    zeroAggGrad_head2();
}

// TODO: torch version of optimizerStep
void LoraModule::updateWeight(torch::Tensor &weight,
                              const torch::Tensor &grad) const {
    hemmer_->addInplace(weight, -ModelArgs::LEARNING_RATE * grad);
}

CtxtTensor LoraModule::getCtxtTensor(const std::string &name, u64 sbi, u64 sbj,
                                     u64 index) const {
    const auto rank = hemmer_->getRank();
    // std::cout << "current rank: " << rank << std::endl;

    CtxtTensor tensor(hemmer_->getContext(), ModelArgs::MAX_SEQ_LEN,
                      ModelArgs::HEAD_DIM,
                      static_cast<HEaaN::i64>(ModelArgs::HEAD_DIM * 2));

    tensor.get().load(hemmer_->getHEPath() + "/" + name + "_" +
                      std::to_string(rank) + "_" + std::to_string(layer_n_) +
                      "_" + std::to_string(sbi) + "_" + std::to_string(sbj) +
                      "_" + std::to_string(index) + ".bin");
    tensor.get().to(getCurrentCudaDevice());
    return tensor;
}

CtxtTensor LoraModule::getCtxtTensor_lora(const std::string &name, u64 sbi,
                                          u64 sbj, u64 index) const {
    // const auto rank = hemmer_->getRank();

    CtxtTensor tensor(hemmer_->getContext(), ModelArgs::MAX_SEQ_LEN,
                      ModelArgs::HEAD_DIM,
                      static_cast<HEaaN::i64>(ModelArgs::HEAD_DIM * 2));

    tensor.get().load(hemmer_->getWeightPath() + name + "_" +
                      std::to_string(layer_n_) + "_" + std::to_string(sbi) +
                      "_" + std::to_string(sbj) + "_" + std::to_string(index) +
                      ".bin");
    tensor.get().to(getCurrentCudaDevice());
    return tensor;
}

CtxtTensor LoraModule::getCtxtTensor_lora_test(const std::string &name, u64 sbi,
                                               u64 sbj, u64 index) const {
    // const auto rank = hemmer_->getRank();

    CtxtTensor tensor(hemmer_->getContext(), ModelArgs::MAX_SEQ_LEN,
                      ModelArgs::HEAD_DIM,
                      static_cast<HEaaN::i64>(ModelArgs::HEAD_DIM * 2));

    tensor.get().load(hemmer_->getWeightTestPath() + name + "_" +
                      std::to_string(layer_n_) + "_" + std::to_string(sbi) +
                      "_" + std::to_string(sbj) + "_" + std::to_string(index) +
                      ".bin");
    tensor.get().to(getCurrentCudaDevice());
    return tensor;
}

void LoraModule::saveCtxtTensor(const CtxtTensor &tensor,
                                const std::string &name, u64 sbi, u64 sbj,
                                u64 index) const {
    const auto rank = hemmer_->getRank();

    tensor.get().save(hemmer_->getHEPath() + "/" + name + "_" +
                      std::to_string(rank) + "_" + std::to_string(layer_n_) +
                      "_" + std::to_string(sbi) + "_" + std::to_string(sbj) +
                      "_" + std::to_string(index) + ".bin");
}

void LoraModule::saveCtxtTensor_lora(const CtxtTensor &tensor,
                                     const std::string &name, u64 sbi, u64 sbj,
                                     u64 index) const {
    // const auto rank = hemmer_->getRank();

    tensor.get().save(hemmer_->getWeightPath() + name + "_" +
                      std::to_string(layer_n_) + "_" + std::to_string(sbi) +
                      "_" + std::to_string(sbj) + "_" + std::to_string(index) +
                      ".bin");
}

torch::Tensor LoraModule::getTorchTensor(const std::string &name,
                                         u32 index) const {
    torch::Tensor tensor;
    torch::load(tensor, hemmer_->getTorchPath() + "/" + name + "_" +
                            std::to_string(layer_n_) + "_" +
                            std::to_string(index) + ".pth");
    return tensor;
}

void LoraModule::saveTorchTensor(const torch::Tensor &tensor,
                                 const std::string &name, u32 index) const {
    torch::save(tensor, hemmer_->getTorchPath() + "/" + name + "_" +
                            std::to_string(layer_n_) + "_" +
                            std::to_string(index) + ".pth");
}

void approxInverseSqrtNewton_TS(const HomEvaluator &eval,
                                const Bootstrapper &btp, const Ciphertext &ctxt,
                                Ciphertext &ctxt_out,
                                const Ciphertext &ctxt_init, u64 num_iter) {
    Ciphertext ctxt_x(eval.getContext());
    Ciphertext tmp1(eval.getContext());
    Ciphertext tmp2(eval.getContext());

    eval.mult(ctxt, -0.5, ctxt_x); // COLA: 9

    ctxt_out = ctxt_init; // COLA: 5

    for (u64 i = 0; i < num_iter; i++) {

        // Find the posisition.
        if (ctxt_out.getLevel() < btp.getMinLevelForBootstrap() + 2 + 1)
            btp.bootstrapExtended(ctxt_out, ctxt_out); // 12 // x
        // shodul consider precision problem. extBTS -> BTS
        if (ctxt_x.getLevel() < ctxt_out.getLevel())
            btp.bootstrapExtended(ctxt_x, ctxt_x); // 12 // x

        eval.mult(ctxt_out, 1.5, tmp1);      // 11 // 9
        eval.mult(ctxt_out, ctxt_x, tmp2);   // 11 // 9
        eval.square(ctxt_out, ctxt_out);     // 11 // 9
        eval.mult(ctxt_out, tmp2, ctxt_out); // 10 // 8
        eval.add(ctxt_out, tmp1, ctxt_out);  // 10 // 8
    }
}

void bootstrapIfNecessary(const Bootstrapper &btp, Ciphertext &ctxt,
                          const u64 depth) {
    if (ctxt.getLevel() < depth + btp.getMinLevelForBootstrap()) {
        btp.bootstrap(ctxt, ctxt);
    }
}

void approxInvSqrt_adamw(const HomEvaluator &eval, const Bootstrapper &btp,
                         const Ciphertext &ctxt, Ciphertext &ctxt_out,
                         const Real initial, const u64 num_iter) {

    Ciphertext ctxt_x(eval.getContext());
    Ciphertext tmp1(eval.getContext());
    Ciphertext tmp2(eval.getContext());

    eval.mult(ctxt, -0.5, ctxt_x);

    for (u64 i = 0; i < num_iter; i++) {
        if (i == 0) {
            eval.mult(ctxt, -initial * initial, tmp1);
            eval.add(tmp1, 3, tmp1);
            eval.mult(tmp1, 0.5 * initial, ctxt_out);
        } else {

            // TOODO: check BTS position.
            if (ctxt_out.getLevel() < 2 + btp.getMinLevelForBootstrap()) {
                btp.bootstrapExtended(ctxt_out, ctxt_out);

                eval.mult(ctxt_out, 1.5, tmp1);
                eval.mult(ctxt_out, ctxt_x, tmp2);
                eval.square(ctxt_out, ctxt_out);
                eval.mult(ctxt_out, tmp2, ctxt_out);
                eval.add(ctxt_out, tmp1, ctxt_out);
            }
        }
    }
}

// in: 10, out:
void approxInverseSqrt_COLA(const HomEvaluator &eval, const Bootstrapper &btp,
                            const Ciphertext &op, Ciphertext &res,
                            const u64 num_iter) {
    // std::cout << "COLA input level " << op.getLevel() << std::endl;
    const ChebyshevCoefficients &cheby_coeffs = COLA_INVERSE_SQRT_127;

    Ciphertext ctxt_x =
        linearTransform(eval, btp, op, InputInterval(0, 1)); // 9

    // bootstrapIfNecessary(btp, ctxt_x, 1 + cheby_coeffs.level_cost);
    bootstrapIfNecessary(btp, ctxt_x, 1 + cheby_coeffs.level_cost);

    Ciphertext ctxt_y(eval.getContext());
    ctxt_y = evaluateChebyshevExpansion(eval, btp, ctxt_x, cheby_coeffs,
                                        1.0); // 12 -> 5

    if (num_iter > 0)
        approxInverseSqrtNewton_TS(eval, btp, op, ctxt_y, ctxt_y, num_iter);

    res = ctxt_y;
}

void approxInverseSqrt_MRPC(const HomEvaluator &eval, const Bootstrapper &btp,
                            const Ciphertext &op, Ciphertext &res,
                            const u64 num_iter) {

    const ChebyshevCoefficients &cheby_coeffs = COLA_INVERSE_SQRT_127;

    Ciphertext ctxt_x =
        linearTransform(eval, btp, op, InputInterval(0, 1)); // 9

    // bootstrapIfNecessary(btp, ctxt_x, 1 + cheby_coeffs.level_cost);
    bootstrapIfNecessary(btp, ctxt_x, 1 + cheby_coeffs.level_cost);

    Ciphertext ctxt_y(eval.getContext());
    ctxt_y = evaluateChebyshevExpansion(eval, btp, ctxt_x, cheby_coeffs,
                                        1.0); // 12 -> 5

    if (num_iter > 0)
        approxInverseSqrtNewton_TS(eval, btp, op, ctxt_y, ctxt_y, num_iter);

    res = ctxt_y;
}

void approxInverseSqrt_RTE(const HomEvaluator &eval, const Bootstrapper &btp,
                           const Ciphertext &op, Ciphertext &res,
                           const u64 num_iter) {

    const ChebyshevCoefficients &cheby_coeffs = RTE_INVERSE_SQRT_127;

    Ciphertext ctxt_x = linearTransform(eval, btp, op, InputInterval(0, 1));

    bootstrapIfNecessary(btp, ctxt_x, cheby_coeffs.level_cost + 1);

    Ciphertext ctxt_y(eval.getContext());
    ctxt_y = evaluateChebyshevExpansion(eval, btp, ctxt_x, cheby_coeffs, 1.0);

    if (num_iter > 0)
        approxInverseSqrtNewton_TS(eval, btp, op, ctxt_y, ctxt_y, num_iter);

    res = ctxt_y;
}

void approxInverseSqrt_SST2(const HomEvaluator &eval, const Bootstrapper &btp,
                            const Ciphertext &op, Ciphertext &res,
                            const u64 num_iter) {

    const ChebyshevCoefficients &cheby_coeffs = COLA_INVERSE_SQRT_127;

    Ciphertext ctxt_x =
        linearTransform(eval, btp, op, InputInterval(0, 1)); // 9

    bootstrapIfNecessary(btp, ctxt_x, 1 + cheby_coeffs.level_cost);

    Ciphertext ctxt_y(eval.getContext());
    ctxt_y = evaluateChebyshevExpansion(eval, btp, ctxt_x, cheby_coeffs,
                                        1.0); // 12 -> 5

    if (num_iter > 0)
        approxInverseSqrtNewton_TS(eval, btp, op, ctxt_y, ctxt_y, num_iter);

    res = ctxt_y;
}

void approxInverseSqrt_STSB(const HomEvaluator &eval, const Bootstrapper &btp,
                            const Ciphertext &op, Ciphertext &res,
                            const u64 num_iter) {

    // const ChebyshevCoefficients &cheby_coeffs = STSB_INVERSE_SQRT_63;
    const ChebyshevCoefficients &cheby_coeffs = COLA_INVERSE_SQRT_127;

    Ciphertext ctxt_x =
        linearTransform(eval, btp, op, InputInterval(0, 1)); // if 4 -> 3

    // bootstrapIfNecessary(btp, ctxt_x, 1 + cheby_coeffs.level_cost);
    bootstrapIfNecessary(btp, ctxt_x, cheby_coeffs.level_cost); // 12

    Ciphertext ctxt_y(eval.getContext());
    ctxt_y =
        evaluateChebyshevExpansion(eval, btp, ctxt_x, cheby_coeffs, 1.0); // 6

    if (num_iter > 0) {
        // std::cout << "newton input " << ctxt_y.getLevel() << std::endl; // 6
        approxInverseSqrtNewton_TS(eval, btp, op, ctxt_y, ctxt_y,
                                   num_iter); // op = 4 , ctxt_y = 6
        // std::cout << "newton output " << ctxt_y.getLevel() << std::endl;
    }
    res = ctxt_y;
}

// private
#ifdef HELLM_MULTIGPU
void LoraModule::allReduceWrapper(CtxtTensor &ctxt_tensor) const {
    ncclAllReduceWrapper(ctxt_tensor);
}

void LoraModule::ncclAllReduceWrapper(const CtxtTensor &ctxt_tensor) const {
    const auto degree = getDegree(hemmer_->getContext());
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
} // namespace HELLM::LoRA
