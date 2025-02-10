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
#include "HEaaN/HEaaN.hpp"
#include <cstring>

#include "torch/script.h"
#include "torch/torch.h"

#include <cstddef>
#include <cstring>
#include <filesystem>

int main(int argc, char *argv[]) {
    bool convert_generation = false;
    if (argc == 2 && std::strcmp(argv[1], "-g") == 0)
        convert_generation = true;

    HELLM::HEMMer hemmer = HELLM::HEMMer::genHEMMer();

    // HELLM_PTH_PATH >> "./data/converted_weights.pth"
    // auto container = torch::jit::load(std::getenv("HELLM_PTH_PATH"));
    auto container =
        torch::jit::load("./data_2ly_mrpc/converted_weights_mrpc.pth");

    // HELLM_GEN_PTXT_PATH >> "./data/ptxt/"
    // std::string ptxt_path{std::getenv("HELLM_PTXT_PATH")};
    std::string ptxt_path = "./data_2ly_mrpc/";
    std::filesystem::create_directory(ptxt_path);

    /*  for (int i = 0 ; i < 1 ; ++i){
         auto data = "input_" + std::to_string(i);
         auto file_name = "input" + std::to_string(i) + "_2ly_rte_lora.pth";
         torch::save(container.attr(data).toTensor(), ptxt_path + file_name);
     }

     for (int i = 0 ; i < 1 ; ++i) {
         auto data = "mask_" + std::to_string(i);
         auto file_name = "mask" + std::to_string(i) + "_2ly_rte_lora.pth";
         torch::save(container.attr(data).toTensor(), ptxt_path + file_name);
     }
  */
    // torch::save(container.attr("attngrad").toTensor(), ptxt_path +
    // "attngrad.pth");

    // Q. What is the purpose?
    // torch::save(container.attr("decoder_w").toTensor(),
    //            ptxt_path + "decoder_w.pth");
    torch::save(container.attr("emb_word_w").toTensor(),
                ptxt_path + "emb_word_w.pth");
    torch::save(container.attr("emb_norm_w").toTensor(),
                ptxt_path + "emb_norm_w.pth");
    torch::save(container.attr("emb_norm_b").toTensor(),
                ptxt_path + "emb_norm_b.pth");

    torch::save(container.attr("encover_emb_pos_scale_factor").toTensor(),
                ptxt_path + "encover_emb_pos_scale_factor.pth");
    torch::save(container.attr("final_norm_w").toTensor(),
                ptxt_path + "final_norm_w.pth");
    torch::save(container.attr("final_norm_b").toTensor(),
                ptxt_path + "final_norm_b.pth");

#pragma omp parallel for
    // Caution: N_LAYER >> NUM_LAYER;
    // 10 layers case = NUM_LAYER10
    // 20 layers case = NUM_LAYER20
    // 40 layers case = NUM_LAYER40
    for (int i = 0; i < 2; ++i) {
        HEaaN::CudaTools::cudaSetDevice(i % HELLM::ModelArgs::N_DEVICE);
        auto prefix = std::to_string(i) + "_";

        // BERT forward
        hemmer.encodeWeight(container.attr(prefix + "wq").toTensor(),
                            ptxt_path + prefix + "wq_");
        hemmer.encodeWeight(container.attr(prefix + "wk").toTensor(),
                            ptxt_path + prefix + "wk_");
        hemmer.encodeWeight(container.attr(prefix + "wv").toTensor(),
                            ptxt_path + prefix + "wv_");
        hemmer.encodeWeight(container.attr(prefix + "wd").toTensor(),
                            ptxt_path + prefix + "wd_");
        hemmer.encodeWeight(container.attr(prefix + "wdin").toTensor(),
                            ptxt_path + prefix + "wdin_");
        hemmer.encodeWeight(container.attr(prefix + "wdout").toTensor(),
                            ptxt_path + prefix + "wdout_");
        hemmer.encodeWeight(container.attr(prefix + "norm1_b").toTensor(),
                            ptxt_path + prefix + "norm1_b_");
        hemmer.encodeWeight(container.attr(prefix + "norm1_w").toTensor(),
                            ptxt_path + prefix + "norm1_w_");
        hemmer.encodeWeight(container.attr(prefix + "norm2_b").toTensor(),
                            ptxt_path + prefix + "norm2_b_");
        hemmer.encodeWeight(container.attr(prefix + "norm2_w").toTensor(),
                            ptxt_path + prefix + "norm2_w_");

        hemmer.encodeWeight(container.attr(prefix + "final_norm_w").toTensor(),
                            ptxt_path + prefix + "final_norm_w_");
        hemmer.encodeWeight(container.attr(prefix + "final_norm_b").toTensor(),
                            ptxt_path + prefix + "final_norm_b_");

        /* hemmer.encodeWeight_bert2(container.attr(prefix +
        "wfinal1").toTensor(), ptxt_path + prefix + "wfinal1_");
        hemmer.encodeWeight_bert2(container.attr(prefix +
        "wfinal1_b").toTensor(), ptxt_path + prefix + "wfinal1_b_");
        hemmer.encodeWeight_bert(container.attr(prefix + "wfinal2").toTensor(),
                            ptxt_path + prefix + "wfinal2_"); */

        // train mode
        hemmer.encodeWeight(container.attr(prefix + "wq")
                                .toTensor()
                                .transpose(0, 1)
                                .transpose(2, 3),
                            ptxt_path + prefix + "tr_wq_");

        hemmer.encodeWeight(container.attr(prefix + "wk")
                                .toTensor()
                                .transpose(0, 1)
                                .transpose(2, 3),
                            ptxt_path + prefix + "tr_wk_");

        hemmer.encodeWeight(container.attr(prefix + "wv")
                                .toTensor()
                                .transpose(0, 1)
                                .transpose(2, 3),
                            ptxt_path + prefix + "tr_wv_");

        hemmer.encodeWeight(container.attr(prefix + "wd")
                                .toTensor()
                                .transpose(0, 1)
                                .transpose(2, 3),
                            ptxt_path + prefix + "tr_wd_");

        hemmer.encodeWeight(container.attr(prefix + "wdin")
                                .toTensor()
                                .transpose(0, 1)
                                .transpose(2, 3),
                            ptxt_path + prefix + "tr_wdin_");

        hemmer.encodeWeight(container.attr(prefix + "wdout")
                                .toTensor()
                                .transpose(0, 1)
                                .transpose(2, 3),
                            ptxt_path + prefix + "tr_wdout_");

        /* hemmer.encodeWeight_bert(container.attr(prefix + "wfinal1")
                                .toTensor(),
                            ptxt_path + prefix + "tr_wfinal1_");

        hemmer.encodeWeight_bert(container.attr(prefix + "wfinal2")
                                .toTensor(),
                            ptxt_path + prefix + "tr_wfinal2_"); */
    }

    // hemmer.encodeWewight_pooling(container.attr("wfinal1_weight_a").toTensor(),
    //                     ptxt_path + "0_wfinal1_weight_a");
    // hemmer.encodeWeight_pooling(container.attr("wfinal1_weight_b").toTensor(),
    //                     ptxt_path +  "0_wfinal1_weight_b");
    // hemmer.encodeWeight_pooling(container.attr("wfinal1_bias_a").toTensor(),
    //                     ptxt_path + "0_wfinal1_bias_a");
    // hemmer.encodeWeight_pooling(container.attr("wfinal1_bias_b").toTensor(),
    //                     ptxt_path + "0_wfinal1_bias_b");
    // hemmer.encodeWeight_pooling(container.attr("wfinal2_weight").toTensor(),
    //                     ptxt_path + "0_wfinal2_weight");
    // hemmer.encodeWeight_pooling(container.attr("wfinal2_bias").toTensor(),
    //                     ptxt_path + "0_wfinal2_bias");
}
