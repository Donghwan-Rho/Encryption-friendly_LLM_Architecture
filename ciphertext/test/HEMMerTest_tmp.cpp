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
#include "HELLM/Loss.hpp"
#include "HELLM/ModelArgs.hpp"
#include "HELLM/ReLU.hpp"
#include "HELLM/Softmax.hpp"
#include "HELLM/Tanh.hpp"
#include "HEaaN/HEaaN.hpp"

#include "HELLMTestBase.hpp"
#include <ATen/TensorIndexing.h>

#include <mpi.h>

namespace HELLM {

class HEMMerTest : public HELLMTestBase {};

/* TEST_F(HEMMerTest, loadtest) {
    std::shared_ptr<HELLM::LoRA::LoraModule> lora_module_ =
std::make_shared<HELLM::LoRA::LoraModule>(getHemmer(), 0);

    torch::Tensor tensor = torch::zeros({2, 128, 128});
    auto ctxt = getHemmer()->encrypt2(tensor[0], tensor[1]);


    if (getHemmer()->getRank() == 1) {
        //torch::Tensor tensor = torch::rand({2, 128, 128});
        //auto ctxt = getHemmer()->encrypt2(tensor[0], tensor[1]);
        //std::cout << "input: " << std::endl;
        //printing(ctxt);
        ctxt = lora_module_->getCtxtTensor("tr_lora_in_a",0,0,0);
    } else {
        ctxt = lora_module_->getCtxtTensor("tr_lora_in_a",0,0,0);
    }
    MPI_Barrier(MPI_COMM_WORLD);


    std::cout << "rank: " << getHemmer()->getRank() << std::endl;
    //auto read = lora_module_->getCtxtTensor_lora("test2",0,0,0);

    printing(ctxt);

} */

/* TEST_F(HEMMerTest, CCMM) {

   double avg = 0.0;
   int len = 10;

   for (int i = 0 ; i < len ; ++i) {

        auto tensor_a =
            torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});
        auto tensor_b =
            torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});


        auto ctxt_tensor_a = getHemmer()->encrypt2(tensor_a[0], tensor_a[1]);
        auto ctxt_tensor_b = getHemmer()->encrypt2(tensor_b[0], tensor_b[1]);

        getHemmer()->complexPackingInplace(ctxt_tensor_a);
        getHemmer()->transposeComplexPackingInplace(ctxt_tensor_b);

        auto start = std::chrono::high_resolution_clock::now();
        //getHemmer()->singleCCMatMul(ctxt_tensor_a, ctxt_tensor_b, 9);
        ctxt_tensor_a = getHemmer()->packedMatMul(ctxt_tensor_a, ctxt_tensor_b,
8); ctxt_tensor_a = getHemmer()->repackCC(ctxt_tensor_a);
        //getHemmer()->bootstrap(ctxt_tensor_a);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;


        if (i != 0) {
            avg += elapsed.count();
        }
   }

   std::cout << "avg latency CCMM: " << avg/(len-1) << std::endl;

} */

/* TEST_F(HEMMerTest, PCMM) {

    double avg = 0.0;
    int len = 20;

    for (int i = 0 ; i < len ; ++i) {
        auto tensor_a =
            torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});

        auto tensor_c =
            torch::rand({128,128});

        auto ctxt_tensor_a = getHemmer()->encrypt2(tensor_a[0], tensor_a[1]);


        auto ptxt = getHemmer()->encode2(tensor_c, tensor_c);

        auto start = std::chrono::high_resolution_clock::now();

        auto output = getHemmer()->singleMatMul(ctxt_tensor_a, ptxt, 9);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        if (i == 0) {
            std::cout << "output level: " << output.get().getLevel() <<
std::endl;
        }
        //std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;

        if (i != 0) {
            avg += elapsed.count();
        }
    }

    std::cout << "avg latency (PCMM): " << avg/(len-1) << " s" << std::endl;

} */

/* TEST_F(HEMMerTest, MatMulPtxt4) {

    auto avg = 0.0;
    int len = 1;

    for (int i = 0 ; i < len ; ++i) {

        auto tensor_a =
            torch::rand({1, 4, ModelArgs::MAX_SEQ_LEN,
ModelArgs::HEAD_DIM}).mul(0.25); auto tensor_b = torch::rand({4, 2,
ModelArgs::HEAD_DIM, ModelArgs::HEAD_DIM}).mul(0.25);

        auto ctxt_tensor_a = getHemmer()->complexPacking(
            getHemmer()->encrypt2(tensor_a[0][0], tensor_a[0][1]),
            getHemmer()->encrypt2(tensor_a[0][2], tensor_a[0][3]));
        auto ptxt_tensor_b_0 = getHemmer()->encodeDiagonalToRow4(
            tensor_b[0][0], tensor_b[1][0], tensor_b[2][0], tensor_b[3][0]);
        auto ptxt_tensor_b_1 = getHemmer()->encodeDiagonalToRow4(
            tensor_b[0][1], tensor_b[1][1], tensor_b[2][1], tensor_b[3][1]);

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<HEaaN::Ciphertext> tmp;
        getHemmer()->matMulPre(ctxt_tensor_a, tmp, 10);
        std::cout << "pre output " << tmp[0].getLevel() << std::endl;
        auto temp_0 = getHemmer()->matMulReUse(tmp, ptxt_tensor_b_0, 10);
        std::cout << "reuse output: " << temp_0.get().getLevel() << std::endl;
        auto temp_1 = getHemmer()->matMulReUse(tmp, ptxt_tensor_b_1, 10);
        temp_0 = getHemmer()->repack(temp_0, temp_1);
        std::cout << "repack output: " << temp_0.get().getLevel() << std::endl;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        if (i != 0) {
            avg += elapsed.count();
        }
    }

    //std::cout << "avg PCMM latency: " << avg/(len-1) << " s" << std::endl;
} */

/* TEST_F(HEMMerTest, Enc) {

    double avg = 0.0;

    for (int i = 0 ; i < 10 ; ++i) {
        auto tensor = torch::rand({2, ModelArgs::MAX_SEQ_LEN,
ModelArgs::HEAD_DIM});

        auto start = std::chrono::high_resolution_clock::now();
        auto ctxt_tensor = getHemmer()->encrypt2(tensor[0], tensor[1]);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        //std::cout << " Enc Latency: " << elapsed.count() << " s" << std::endl;

        if (i != 0) {
            avg += elapsed.count();
        }
    }
    std::cout << "avg latency: " << avg/9 << " s" << std::endl;
} */

/* TEST_F(HEMMerTest, Dec) {

    double avg = 0.0;
    int len = 20;


    for (int i = 0 ; i < len ; ++i) {
        auto tensor = torch::rand({2, ModelArgs::MAX_SEQ_LEN,
ModelArgs::HEAD_DIM}); auto ctxt_tensor = getHemmer()->encrypt2(tensor[0],
tensor[1]);

        auto start = std::chrono::high_resolution_clock::now();
        auto dec_tensor = getHemmer()->decrypt2(ctxt_tensor);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        //std::cout << " Dec Latency: " << elapsed.count() << " s" << std::endl;

        if (i != 0) {
            avg += elapsed.count();
        }
    }

    std::cout << "avg time: " << avg/(len-1) << " s" << std::endl;
} */

TEST_F(HEMMerTest, Add) {

    double avg = 0.0;
    int len = 20;

    for (int i = 0; i < len; ++i) {
        auto tensor_a =
            torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});
        auto tensor_b =
            torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});

        auto ctxt_tensor_a = getHemmer()->encrypt2(tensor_a[0], tensor_a[1]);
        auto ctxt_tensor_b = getHemmer()->encrypt2(tensor_b[0], tensor_b[1]);

        CudaTools::cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        getHemmer()->addInplace(ctxt_tensor_a, ctxt_tensor_b);
        CudaTools::cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        // std::cout << " Add Latency: " << elapsed.count() << " s" <<
        // std::endl;

        if (i != 0) {
            avg += elapsed.count();
        }
    }

    std::cout << "avg latency: " << avg / (len - 1) << " s" << std::endl;
}

TEST_F(HEMMerTest, Mult) {

    double avg = 0.0;
    int len = 20;

    for (int i = 0; i < len; ++i) {
        auto tensor_a =
            torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});
        auto tensor_b =
            torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});

        auto ctxt_tensor_a = getHemmer()->encrypt2(tensor_a[0], tensor_a[1]);
        auto ctxt_tensor_b = getHemmer()->encrypt2(tensor_b[0], tensor_b[1]);
        // getHemmer()->getEval().levelDown(ctxt_tensor_a.get(), 4,
        // ctxt_tensor_a.get());
        // getHemmer()->getEval().levelDown(ctxt_tensor_b.get(), 4,
        // ctxt_tensor_b.get());

        CudaTools::cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        getHemmer()->hadamardMultInplace(ctxt_tensor_a, ctxt_tensor_b);
        CudaTools::cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        if (i != 0) {
            avg += elapsed.count();
        }
        // std::cout << " Mult Latency: " << elapsed.count() << " s" <<
        // std::endl;
    }

    std::cout << "avg latency: " << avg / (len - 1) << " s" << std::endl;
}

TEST_F(HEMMerTest, Rot) {

    double avg = 0.0;
    int len = 20;

    for (int i = 0; i < len; ++i) {
        auto tensor_a =
            torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});
        auto tensor_b =
            torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});

        auto ctxt_tensor_a = getHemmer()->encrypt2(tensor_a[0], tensor_a[1]);
        auto ptxt_tensor_b = getHemmer()->encode2(tensor_b[0], tensor_b[1]);
        // getHemmer()->getEval().levelDown(ctxt_tensor_a.get(), 4,
        // ctxt_tensor_a.get());

        CudaTools::cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        getHemmer()->getEval().rightRotate(ptxt_tensor_b.get(), 1,
                                           ptxt_tensor_b.get());
        CudaTools::cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        // std::cout << " Rot Latency: " << elapsed.count() << " s" <<
        // std::endl;

        if (i != 0) {
            avg += elapsed.count();
        }
    }

    std::cout << "avg latency: " << avg / (len - 1) << " s" << std::endl;
}

TEST_F(HEMMerTest, CMult) {

    double avg = 0.0;
    int len = 20;

    for (int i = 0; i < len; ++i) {
        auto tensor_a =
            torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});
        auto tensor_b =
            torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});

        auto ctxt_tensor_a = getHemmer()->encrypt2(tensor_a[0], tensor_a[1]);
        auto ctxt_tensor_b = getHemmer()->encode2(tensor_b[0], tensor_b[1]);

        CudaTools::cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        getHemmer()->getEval().mult(ctxt_tensor_a.get(), ctxt_tensor_b.get(),
                                    ctxt_tensor_a.get());
        CudaTools::cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        // std::cout << " CMult Latency: " << elapsed.count() << " s" <<
        // std::endl;

        if (i != 0) {
            avg += elapsed.count();
        }
    }
    std::cout << "avg latency: " << avg / (len - 1) << " s" << std::endl;
}

TEST_F(HEMMerTest, ExtBTS) {

    double avg = 0.0;
    int len = 20;

    for (int i = 0; i < len; ++i) {
        auto tensor_a =
            torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});

        auto ctxt_tensor_a = getHemmer()->encrypt2(tensor_a[0], tensor_a[1]);

        getHemmer()->getEval().levelDown(ctxt_tensor_a.get(), 4,
                                         ctxt_tensor_a.get());

        CudaTools::cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        getHemmer()->bootstrap(ctxt_tensor_a);
        CudaTools::cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        // std::cout << " ExtBTS Latency: " << elapsed.count() << " s" <<
        // std::endl;

        if (i != 0) {
            avg += elapsed.count();
        }
    }

    std::cout << "avg latency: " << avg / (len - 1) << std::endl;
}

/* TEST_F(HEMMerTest, BTS) {

    double avg = 0.0;
    int len = 10;

    for (int i = 0 ; i < 10 ; ++i) {
        auto tensor_a =
            torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});

        auto ctxt_tensor_a = getHemmer()->encrypt2(tensor_a[0], tensor_a[1]);

        //getHemmer()->getEval().levelDown(ctxt_tensor_a.get(),3,
ctxt_tensor_a.get());


        CudaTools::cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        getHemmer()->bootstrapUnitRange(ctxt_tensor_a);
        CudaTools::cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        //std::cout << " BTS Latency: " << elapsed.count() << " s" << std::endl;

        if (i != 0) {
            avg += elapsed.count();
        }
    }

    std::cout << "avg latency: " << avg/(len-1) << " s" << std::endl;
} */

/* TEST_F(HEMMerTest, InvSqrt_LoRA) {
    const double prec = std::pow(2.0, -18);

    const auto max = 1.0;
    const auto min = -1.0;

    for (u64 i = 0 ; i < 5 ; ++i) {
        auto tensor_a =
            torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM})*(max -
min) + min;

        auto ctxt_tensor_a = getHemmer()->encrypt2(tensor_a[0], tensor_a[1]);
        auto tmp = ctxt_tensor_a;

        //getHemmer()->getEval().mult(ctxt_tensor_a.get(), 1.0/40,
ctxt_tensor_a.get());
        //HELLM::LoRA::approxInverseSqrt_COLA(getHemmer()->getEval(),
getHemmer()->getBtp(),
        //                ctxt_tensor_a.get(), ctxt_tensor_a.get(), 3);
        //getHemmer()->getEval().mult(ctxt_tensor_a.get(), 1.0/std::sqrt(40),
ctxt_tensor_a.get()); HELLM::ReLU::ApproxReLU(getHemmer()->getEval(),
getHemmer()->getBtp(), ctxt_tensor_a.get(), ctxt_tensor_a.get(), tmp.get());


        //tensor_a = 1.0/tensor_a;
        //tensor_a = torch::rsqrt(tensor_a);
        //tensor_a = torch::tanh(tensor_a);
        tensor_a = torch::relu(tensor_a);

        reportCompareTensor3dim(tensor_a, ctxt_tensor_a);
    }
} */

/* TEST_F(HEMMerTest, InvSqrt_LoRA) {
    const double prec = std::pow(2.0, -18);

    const auto max = 40;
    const auto min = 0.4;

    for (u64 i = 0 ; i < 10 ; ++i) {
        auto tensor_a =
            torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM})*(max -
min) + min;

        auto ctxt_tensor_a = getHemmer()->encrypt2(tensor_a[0], tensor_a[1]);

        getHemmer()->getEval().mult(ctxt_tensor_a.get(), 1.0/40,
ctxt_tensor_a.get());
        HELLM::LoRA::approxInverseSqrt_STSB(getHemmer()->getEval(),
getHemmer()->getBtp(), ctxt_tensor_a.get(), ctxt_tensor_a.get(), 2);
        getHemmer()->getEval().mult(ctxt_tensor_a.get(), 1.0/std::sqrt(40),
ctxt_tensor_a.get());

        tensor_a = torch::rsqrt(tensor_a);

        reportCompareTensor3dim(tensor_a, ctxt_tensor_a);
    }
} */

/* TEST_F(HEMMerTest, MatMulCtxtHighLow) {
    const double prec = std::pow(2.0, -20);
    const u64 low_dim = 2;
    const u64 in_col_block = 0;
    const u64 out_col_block = 0;
    const u64 in_row_block = 0;

    const auto max = 1;
    const auto min = -1;

    auto tensor_a =
        torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM})*(max -
min) + min; auto tensor_b = torch::zeros({2, ModelArgs::MAX_SEQ_LEN,
ModelArgs::HEAD_DIM}); auto tensor_c = torch::zeros({2, ModelArgs::MAX_SEQ_LEN,
ModelArgs::HEAD_DIM});

    tensor_b.slice(2, in_col_block * low_dim, (in_col_block + 1) * low_dim) =
        torch::rand({2, ModelArgs::HEAD_DIM, low_dim})*(max - min) + min;
    tensor_c.slice(1, in_row_block * low_dim, (in_row_block + 1) * low_dim) =
        torch::rand({2, low_dim, ModelArgs::HEAD_DIM})*(max - min) + min;


    auto tr_tensor_b = tensor_b.transpose(1,2);

    std::cout << "tensor shape: " << tensor_b.sizes() << std::endl;

    auto ctxt_tensor_a = getHemmer()->encrypt2(tensor_a[0], tensor_a[1]);
    auto ctxt_tensor_b = getHemmer()->encrypt2(tensor_b[0], tensor_b[1]);
    auto ctxt_tensor_c = getHemmer()->encrypt2(tensor_c[0], tensor_c[1]);
    auto ctxt_tr_tensor_b = getHemmer()->encrypt2(tr_tensor_b[0],
tr_tensor_b[1]);


    // hard coding
    getHemmer()->getEval().levelDown(ctxt_tensor_a.get(), 9,
ctxt_tensor_a.get()); getHemmer()->getEval().levelDown(ctxt_tensor_c.get(), 6,
ctxt_tensor_c.get()); getHemmer()->getEval().levelDown(ctxt_tr_tensor_b.get(),
9, ctxt_tr_tensor_b.get());

    ctxt_tensor_a =
        getHemmer()->matMulHighLow(ctxt_tensor_a, ctxt_tensor_b, in_col_block,
5); ctxt_tensor_a = getHemmer()->repackToMultiCol(ctxt_tensor_a, out_col_block);

    getHemmer()->bootstrap(ctxt_tensor_a);

    ctxt_tensor_a =
        getHemmer()->matMulLowLow(ctxt_tensor_a, ctxt_tensor_c, out_col_block,
in_row_block);


    // hard coding
    std::vector<CtxtTensor> weights;
    weights.reserve(2);
    for (u64 i = 0 ; i < 2 ; ++i) {
        weights.emplace_back(ctxt_tr_tensor_b);

        if (i == 0) {
            getHemmer()->maskFirstRowInplace(weights[i]);
        } else {
            getHemmer()->getEval().leftRotate(weights[i].get(), i*256,
weights[i].get()); getHemmer()->maskFirstRowInplace(weights[i]);
        }
    }

    Ciphertext tmp{ctxt_tensor_a.get()};
    for (u64 i = 0 ; i < 2 ; ++i) {
        for (u64 rot = 1 ; rot < 128 ; rot <<= 1 ) {
            getHemmer()->getEval().rightRotate(weights[i].get(), rot*256, tmp);
            getHemmer()->getEval().add(weights[i].get(), tmp, weights[i].get());
        }
    }

    getHemmer()->getEval().mult(ctxt_tensor_a.get(), weights[0].get(),
weights[0].get()); getHemmer()->getEval().mult(ctxt_tensor_a.get(),
weights[1].get(), weights[1].get());

    for (u64 i = 0 ; i < 2 ; ++i) {
        for (u64 rot = 1 ; rot < 128 ; rot <<= 1) {
            getHemmer()->getEval().leftRotate(weights[i].get(), rot, tmp);
            getHemmer()->getEval().add(weights[i].get(), tmp, weights[i].get());
        }
        getHemmer()->maskFirstColInplace(weights[i]);
    }

    //getHemmer()->getEval().rightRotate(weights[1].get(), 1, weights[1].get());
    //getHemmer()->getEval().add(weights[0].get(), weights[1].get(),
weights[0].get());

    std::vector<CtxtTensor> weights_b;
    weights_b.reserve(2);
    for (u64 i = 0 ; i < 2 ; ++i) {
        weights_b.emplace_back(ctxt_tensor_c);

        if (i == 0) {
            getHemmer()->maskFirstRowInplace(weights_b[i]);
        } else {
            getHemmer()->getEval().leftRotate(weights_b[i].get(), i*256,
weights_b[i].get()); getHemmer()->maskFirstRowInplace(weights_b[i]);
        }
    }

    for (u64 i = 0 ; i < 2 ; ++i) {
        for (u64 rot = 1 ; rot < 128 ; rot <<= 1 ) {
            getHemmer()->getEval().rightRotate(weights[i].get(), rot, tmp);
            getHemmer()->getEval().add(weights[i].get(), tmp, weights[i].get());

            getHemmer()->getEval().rightRotate(weights_b[i].get(), rot*256,
tmp); getHemmer()->getEval().add(weights_b[i].get(), tmp, weights_b[i].get());
        }

        getHemmer()->getEval().mult(weights[i].get(), weights_b[i].get(),
weights[i].get());
    }

    getHemmer()->getEval().add(weights[0].get(), weights[1].get(),
weights[0].get());


    tensor_a = tensor_a.matmul(tensor_b);
    tensor_c = tensor_a.matmul(tensor_c);

    compareTensor2(tensor_c, ctxt_tensor_a, prec);
} */

/* TEST_F(HEMMerTest, LoRA_w_grad_A_time) {

    for (u64 time = 0 ; time < 3 ; ++time) {


        auto tensor = torch::rand({2, 128,128});
        auto tensor_zero = torch::zeros({2,128,128});
        auto ctxt_tensor = getHemmer()->encrypt2(tensor[0], tensor[1]);
        auto grad_lora_wa = getHemmer()->encrypt2(tensor_zero[0],
tensor_zero[1]); getHemmer()->getEval().levelDown(ctxt_tensor.get(), 6,
ctxt_tensor.get());


        std::vector<CtxtTensor> grad_lora_b; // 2개로 들어옴.
        grad_lora_b.reserve(2);
        for (u64 i = 0 ; i < 2 ; ++i) {
            grad_lora_b.emplace_back(ctxt_tensor);
        }

        std::vector<CtxtTensor> tr_lora_in_a;
        tr_lora_in_a.reserve(3);
        for (u64 i = 0 ; i < 3 ; ++i) {
            tr_lora_in_a.emplace_back(ctxt_tensor);
        }

        auto start = std::chrono::high_resolution_clock::now();

        auto tmp = ctxt_tensor;
        //repeated pack
        for (u64 i = 0 ; i < 2 ; ++i) {
            for (u64 rot = 1 ; rot < 256 ; rot <<= 1) {
                getHemmer()->getEval().rightRotate(grad_lora_b[i].get(), rot,
tmp.get()); getHemmer()->getEval().add(grad_lora_b[i].get(), tmp.get(),
grad_lora_b[i].get());
            }
        }

        //hada
        std::vector<std::vector<CtxtTensor>> tmps;
        tmps.reserve(3);
        for (u64 i = 0 ; i < 3 ; ++i) {
            tmps.emplace_back(grad_lora_b);
            for (u64 j = 0 ; j < 2 ; ++j) {
                getHemmer()->getEval().mult(tr_lora_in_a[i].get(),
grad_lora_b[j].get(), tmps[i][j].get());
            }
        }

        // collect data
        for (u64 j = 0 ; j < 2; ++j) {
            for (u64 i = 0 ; i < 3; ++i) {
                for (u64 rot = 1 ; rot < 128 ; rot <<= 1 ) {
                    getHemmer()->getEval().leftRotate(tmps[i][j].get(), rot,
tmp.get()); getHemmer()->getEval().add(tmps[i][j].get(), tmp.get(),
tmps[i][j].get());
                }
                getHemmer()->maskFirstRowInplace(tmps[i][j]);
            }
        }

        grad_lora_wa = tmps[0][0];
        getHemmer()->getEval().rightRotate(tmps[0][1].get(), 1*256, tmp.get());
        getHemmer()->getEval().add(grad_lora_wa.get(), tmp.get(),
grad_lora_wa.get()); for (u64 j = 1 ; j < 3 ; ++j) {
            getHemmer()->getEval().rightRotate(tmps[j][1].get(), 1*256,
tmp.get()); getHemmer()->getEval().add(tmps[j][0].get(), tmp.get(),
tmps[j][0].get());

            getHemmer()->getEval().rightRotate(tmps[j][0].get(), 2*j*256,
tmps[j][0].get()); getHemmer()->getEval().add(grad_lora_wa.get(),
tmps[j][0].get(), grad_lora_wa.get());
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Latency: " << elapsed.count() << " s" << std::endl;
        std::cout << "output level " << grad_lora_wa.get().getLevel() <<
std::endl;


    }
} */

/* TEST_F(HEMMerTest, LoRA_w_grad_B_time) {

    for (u64 time = 0 ; time < 3 ; ++time) {


        auto tensor = torch::rand({2, 128,128});
        auto tensor_zero = torch::zeros({2,128,128});
        auto ctxt_tensor = getHemmer()->encrypt2(tensor[0], tensor[1]);
        auto grad_lora_wb = getHemmer()->encrypt2(tensor_zero[0],
tensor_zero[1]); getHemmer()->getEval().levelDown(ctxt_tensor.get(), 6,
ctxt_tensor.get());


        std::vector<CtxtTensor> tr_lora_in_b; // 2개로 들어옴.
        tr_lora_in_b.reserve(2);
        for (u64 i = 0 ; i < 2 ; ++i) {
            tr_lora_in_b.emplace_back(ctxt_tensor);
        }

        std::vector<CtxtTensor> tmp_grad;
        tmp_grad.reserve(3);
        for (u64 i = 0 ; i < 3 ; ++i) {
            tmp_grad.emplace_back(ctxt_tensor);
        }

        auto start = std::chrono::high_resolution_clock::now();

        auto tmp = ctxt_tensor;
        //repeated pack
        for (u64 i = 0 ; i < 2 ; ++i) {
            for (u64 rot = 1 ; rot < 256 ; rot <<= 1) {
                getHemmer()->getEval().rightRotate(tr_lora_in_b[i].get(), rot,
tmp.get()); getHemmer()->getEval().add(tr_lora_in_b[i].get(), tmp.get(),
tr_lora_in_b[i].get());
            }
        }

        //hada
        std::vector<std::vector<CtxtTensor>> tmps;
        tmps.reserve(2);
        for (u64 i = 0 ; i < 2 ; ++i) {
            tmps.emplace_back(tmp_grad);
            for (u64 j = 0 ; j < 3 ; ++j) {
                getHemmer()->getEval().mult(tr_lora_in_b[i].get(),
tmp_grad[j].get(), tmps[i][j].get());
            }
        }

        // collect data
        for (u64 j = 0 ; j < 3; ++j) {
            for (u64 i = 0 ; i < 2; ++i) {
                for (u64 rot = 1 ; rot < 128 ; rot <<= 1 ) {
                    getHemmer()->getEval().leftRotate(tmps[i][j].get(), rot,
tmp.get()); getHemmer()->getEval().add(tmps[i][j].get(), tmp.get(),
tmps[i][j].get());
                }
                getHemmer()->maskFirstRowInplace(tmps[i][j]);
            }
        }

        grad_lora_wb = tmps[0][0];
        getHemmer()->getEval().rightRotate(tmps[1][0].get(), 1*256, tmp.get());
        getHemmer()->getEval().add(grad_lora_wb.get(), tmp.get(),
grad_lora_wb.get()); for (u64 j = 1 ; j < 3 ; ++j) {
            getHemmer()->getEval().rightRotate(tmps[1][j].get(), 1*256,
tmp.get()); getHemmer()->getEval().add(tmps[0][j].get(), tmp.get(),
tmps[0][j].get());

            getHemmer()->getEval().rightRotate(tmps[0][j].get(), 2*j*256,
tmps[0][j].get()); getHemmer()->getEval().add(grad_lora_wb.get(),
tmps[0][j].get(), grad_lora_wb.get());
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Latency: " << elapsed.count() << " s" << std::endl;
        std::cout << "output level " << grad_lora_wb.get().getLevel() <<
std::endl;


    }
} */

/* TEST_F(HEMMerTest, LN_weight_grad) {

    for (u64 time = 0 ; time < 12 ; ++time) {

        auto tensor = torch::rand({2, 128,128});
        auto ctxt_tensor = getHemmer()->encrypt2(tensor[0], tensor[1]);
        auto weight = getHemmer()->encrypt2(tensor[0], tensor[1]);

        std::vector<CtxtTensor> cur;
        cur.reserve(3);
        for (u64 i = 0 ; i < 3 ; ++i) {
            cur.emplace_back(ctxt_tensor);
        }

        auto start = std::chrono::high_resolution_clock::now();

        getHemmer()->getEval().leftRotate(weight.get(), 256, weight.get());
        getHemmer()->getEval().leftRotate(weight.get(), 256, weight.get());
        getHemmer()->maskFirstRowInplace(weight);

        for (u64 i = 0 ; i < 3 ; ++i) {

            getHemmer()->getEval().mult(cur[i].get(), weight.get(),
cur[i].get());


            CtxtTensor tmp{cur[i]};
            for (u64 rot = 1 ; rot < 128 ; rot <<= 1) {
                getHemmer()->getEval().leftRotate(cur[i].get(), rot*256,
tmp.get()); getHemmer()->addInplace(cur[i], tmp);
            }

            getHemmer()->maskFirstRowInplace(cur[i]);

            for (u64 rot = 1 ; rot < 128 ; rot <<= 1) {
                getHemmer()->getEval().rightRotate(cur[i].get(), rot*256,
tmp.get()); getHemmer()->addInplace(cur[i], tmp);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Latency: " << elapsed.count() << " s" << std::endl;
    }
} */

/* TEST_F(HEMMerTest, LoRA_A_time) {

    for (u64 time = 0 ; time < 3 ; ++time) {
        auto tensor = torch::rand({2, 128,128});
        auto ctxt_tensor = getHemmer()->encrypt2(tensor[0], tensor[1]);
        std::vector<CtxtTensor> cur;
        cur.reserve(3);
        for (u64 i = 0 ; i < 3 ; ++i) {
            cur.emplace_back(ctxt_tensor);
        }

        std::vector<std::vector<CtxtTensor>> lora_a_weight;
        lora_a_weight.reserve(3);
        std::vector<CtxtTensor> lora_a_output;
        lora_a_output.reserve(2);

        auto start = std::chrono::high_resolution_clock::now();

        auto lora_wa = ctxt_tensor;
        getHemmer()->getEval().levelDown(lora_wa.get(), 9, lora_wa.get());

        auto tmp = lora_wa;

        // init
        for (u64 j = 0 ; j < 2 ; ++j) {
            lora_a_output.emplace_back(lora_wa);
        }


        // weight split
        std::vector<CtxtTensor> tmps;
        tmps.reserve(2);
        for (u64 i = 0 ; i < 2 ; ++i) {
            tmps.emplace_back(lora_wa);
        }
        getHemmer()->maskFirstRowInplace(tmps[0]);
        getHemmer()->getEval().leftRotate(tmps[1].get(), 1*256, tmps[1].get());
        getHemmer()->maskFirstRowInplace(tmps[1]);
        lora_a_weight.emplace_back(tmps);

        for (u64 i = 1 ; i < 3 ; ++i) {
            tmps.clear();
            getHemmer()->getEval().leftRotate(lora_wa.get(), i*2*256,
tmp.get()); for (u64 j = 0 ; j < 2 ; ++j) { tmps.emplace_back(tmp);
            }
            getHemmer()->maskFirstRowInplace(tmps[0]);
            getHemmer()->getEval().leftRotate(tmps[1].get(), 1*256,
tmps[1].get()); getHemmer()->maskFirstRowInplace(tmps[1]);
            lora_a_weight.emplace_back(tmps);
        }

        //repeated packing
        for (u64 i = 0 ; i < 3 ; ++i) {
            for (u64 j = 0 ; j < 2 ; ++j) {
                for (u64 rot = 1 ; rot < 128 ; rot <<= 1) {
                    getHemmer()->getEval().rightRotate(lora_a_weight[i][j].get(),
rot , tmp.get()); getHemmer()->getEval().add(lora_a_weight[i][j].get(),
tmp.get(), lora_a_weight[i][j].get());
                }
            }
        }

        // hadamult
        getHemmer()->getEval().mult(cur[0].get(), lora_a_weight[0][0].get(),
lora_a_output[0].get()); getHemmer()->getEval().mult(cur[0].get(),
lora_a_weight[0][1].get(), lora_a_output[1].get()); for (u64 i = 1 ; i < 3 ;
++i) { getHemmer()->getEval().mult(cur[i].get(), lora_a_weight[i][0].get(),
tmp.get()); getHemmer()->getEval().add(lora_a_output[0].get(), tmp.get(),
lora_a_output[0].get()); getHemmer()->getEval().mult(cur[i].get(),
lora_a_weight[i][1].get(), tmp.get());
            getHemmer()->getEval().add(lora_a_output[1].get(), tmp.get(),
lora_a_output[1].get());
        }

        // collect
        for (u64 i = 0 ; i < 2 ;++i) {
            for (u64 rot = 1 ; rot < 256 ; rot <<= 1) {
                getHemmer()->getEval().leftRotate(lora_a_output[i].get(), rot,
tmp.get()); getHemmer()->getEval().add(lora_a_output[i].get(), tmp.get(),
lora_a_output[i].get());
            }
            getHemmer()->maskFirstColOnlyInplace(lora_a_output[i]);

        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Latency: " << elapsed.count() << " s" << std::endl;
        std::cout << "output level " << lora_a_output[0].get().getLevel() <<
std::endl;
    }
} */

/* TEST_F(HEMMerTest, LoRA_B_time) {

    for (u64 time = 0 ; time < 3 ; ++time) {
        auto tensor = torch::rand({2, 128,128});
        auto ctxt_tensor = getHemmer()->encrypt2(tensor[0], tensor[1]);
        getHemmer()->getEval().levelDown(ctxt_tensor.get(), 6,
ctxt_tensor.get()); auto lora_wb = ctxt_tensor;

        std::vector<CtxtTensor> lora_a_output;
        lora_a_output.reserve(2);
        for (u64 i = 0 ; i < 2 ; ++i) {
            lora_a_output.emplace_back(ctxt_tensor);
        }

        auto start = std::chrono::high_resolution_clock::now();


        // extract weights
        std::vector<std::vector<CtxtTensor>> lora_b_weight;
        lora_b_weight.reserve(3);


        std::vector<CtxtTensor> tmps;
        tmps.reserve(2);
        for (u64 j = 0 ; j < 2; ++j) {
            tmps.emplace_back(lora_wb);
        }
        getHemmer()->maskFirstRowInplace(tmps[0]);
        getHemmer()->getEval().rightRotate(tmps[1].get(), 1*256, tmps[1].get());
        getHemmer()->maskFirstRowInplace(tmps[1]);
        lora_b_weight.emplace_back(tmps);

        for (u64 i = 1 ; i < 3 ; ++i) {
            tmps.clear();
            for (u64 j = 0 ; j < 2; ++j) {
                tmps.emplace_back(lora_wb);
            }
            getHemmer()->maskFirstRowInplace(tmps[0]);
            getHemmer()->getEval().rightRotate(tmps[1].get(), i*2*256,
tmps[1].get()); getHemmer()->maskFirstRowInplace(tmps[1]);

            lora_b_weight.emplace_back(tmps);
        }


        // repeated packing
        auto tmp = lora_wb;
        for (u64 i = 0 ; i < 2 ; ++i) {
            for (u64 rot = 1 ; rot < 256 ; rot <<= 1) {
                getHemmer()->getEval().rightRotate(lora_a_output[i].get(), rot,
tmp.get() ); getHemmer()->getEval().add(lora_a_output[i].get(), tmp.get(),
lora_a_output[i].get());
            }
        }

        for (u64 i = 0 ; i < 3 ; ++i) {
            for (u64 j = 0 ; j < 2 ; ++j) {
                for (u64 rot = 1 ; rot < 128 ; rot <<= 1) {
                    getHemmer()->getEval().rightRotate(lora_b_weight[i][j].get(),
rot, tmp.get()); getHemmer()->getEval().add(lora_b_weight[i][j].get(),
tmp.get(), lora_b_weight[i][j].get());
                }
            }
        }

        std::vector<CtxtTensor> lora_b_output;
        lora_b_output.reserve(3);
        for (u64 i = 0 ; i < 3 ;++i) {
            lora_b_output.emplace_back(lora_wb);
        }

        for (u64 i = 0 ; i < 3 ; ++i) {
            getHemmer()->getEval().mult(lora_a_output[0].get(),lora_b_weight[0][0].get(),
lora_b_output[i].get()); getHemmer()->getEval().mult(lora_a_output[1].get(),
lora_b_weight[0][1].get(), tmp.get());
            getHemmer()->getEval().add(lora_b_output[i].get(), tmp.get(),
lora_b_output[i].get());
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Latency: " << elapsed.count() << " s" << std::endl;
        std::cout << "output level " << lora_b_output[0].get().getLevel() <<
std::endl;
    }
} */

/*TEST_F(HEMMerTest, SqrtInverse_RTE) {
    const double prec = std::pow(2.0, -20);

    const auto min = 0.01/3;
    const auto max = 0.1;

    auto tensor =
        torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM})*(max -
min) + min ;

    //auto tensor = torch::ones({2,128,128});

    auto tensor_out = torch::rsqrt(tensor);
    auto ctxt_tensor = getHemmer()->encrypt2(tensor[0], tensor[1]);
    auto ctxt_tensor_out1{ctxt_tensor};
    LoRA::approxInverseSqrt_RTE(getHemmer()->getEval(), getHemmer()->getBtp(),
ctxt_tensor.get(), ctxt_tensor_out1.get(), 3); compareTensor2(tensor_out,
ctxt_tensor_out1,prec);
}*/

/* TEST_F(HEMMerTest, Softmax_CCS) {
    const double prec = std::pow(2.0, -13);

    const auto min = 0.0;
    const auto max = 1024.0;

    auto tensor =
        torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM})*(max -
min) + min ;
    //auto tensor = torch::ones({2, 128,128});

    auto tensor_out = torch::softmax(tensor, 2);
    auto ctxt_tensor = getHemmer()->encrypt2(tensor[0], tensor[1]);
    auto ctxt_tensor_out1{ctxt_tensor};

    for (int i = 0 ; i < 2 ; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        if ( i == 1) {
            for (int j = 0 ; j < 6 ; ++j) {
                Softmax::Softmax_128_512(getHemmer()->getEval(),
getHemmer()->getBtp(), ctxt_tensor.get(), ctxt_tensor_out1.get());
            }
        } else {
            Softmax::Softmax_128_512(getHemmer()->getEval(),
getHemmer()->getBtp(), ctxt_tensor.get(), ctxt_tensor_out1.get());
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Softmax: " << elapsed.count() << " s" << std::endl;
    }

    std::vector<HELLM::CtxtTensor> ctxt_tensors;
    ctxt_tensors.reserve(6);
    for (u64 i = 0 ; i < 6 ; ++i) {
        ctxt_tensors.emplace_back(ctxt_tensor);
    }
    std::vector<HELLM::CtxtTensor> out_tensors;
    out_tensors = ctxt_tensors;

    for (int i = 0 ; i < 2 ; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        Softmax::Softmax_128_512_Parallel(getHemmer()->getEval(),
getHemmer()->getBtp(), ctxt_tensors, out_tensors); auto end =
std::chrono::high_resolution_clock::now(); std::chrono::duration<double> elapsed
= end - start; std::cout << "Prallel Softmax: " << elapsed.count() << " s" <<
std::endl;
    }


    compareTensor2(tensor_out, out_tensors[0], prec);
} */

/* TEST_F(HEMMerTest, Softmax_HETAL) {
    const double prec = std::pow(2.0, -13);

    const auto min = 0.0;
    const auto max = 1024.0;


    for (int i = 0 ; i < 2 ; ++i ) {

        //auto tensor =
        //    torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM})*(max
- min) + min ;

        auto tensor = torch::rand({256,128})*(max - min) + min ;

        auto tensor_out = torch::softmax(tensor, 1);
        auto ctxt_tensor = getHemmer()->encrypt(tensor);

        //auto tensor_out = torch::softmax(tensor, 2);
        //auto ctxt_tensor = getHemmer()->encrypt2(tensor[0], tensor[1]);


        std::vector<HELLM::CtxtTensor> ctxt_tensor_vec;
        for (int i = 0 ; i < 1 ; ++i)
            ctxt_tensor_vec.emplace_back(ctxt_tensor);

        auto start = std::chrono::high_resolution_clock::now();
        getHemmer()->softmaxVectorInplaceHETAL(ctxt_tensor_vec, 0,0,false,
getHemmer()->getDec(), getHemmer()->getsk()); auto end =
std::chrono::high_resolution_clock::now(); std::chrono::duration<double> elapsed
= end - start; std::cout << "Softmax eval time: " << elapsed.count() << " s" <<
std::endl;

        reportCompareTensor(tensor_out, ctxt_tensor_vec[0]);
        //reportCompareTensor3dim(tensor_out, ctxt_tensor_vec[0]);
    }
} */

/* TEST_F(HEMMerTest, MatMulCtxtHighLow) {
    const double prec = std::pow(2.0, -15);
    const u64 low_dim = ModelArgs::LOW_DIM;
    const u64 in_col_block = 0;
    const u64 out_col_block = 0;

    auto tensor_a =
        torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});
    auto tensor_b =
        torch::zeros({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});

    tensor_b.slice(2, in_col_block * low_dim, (in_col_block + 1) * low_dim) =
        torch::rand({2, ModelArgs::HEAD_DIM, low_dim});

    auto ctxt_tensor_a1 = getHemmer()->encrypt2(tensor_a[0], tensor_a[1]);
    auto ctxt_tensor_a2 = getHemmer()->encrypt2(tensor_a[0], tensor_a[1]);
    auto ctxt_tensor_b = getHemmer()->encrypt2(tensor_b[0], tensor_b[1]);
    auto ptxt_tensor_b = getHemmer()->encode2(tensor_b[0], tensor_b[1]);
    auto ptxt_tensor_c = getHemmer()->encode2(tensor_b[0], tensor_b[1]);
    ptxt_tensor_b.get().setLevel(7);

    for (int j = 0 ; j < 2; ++j) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0 ; i < 128 ; ++i) {
            //auto tmp =
                getHemmer()->getEval().leftRotate(ctxt_tensor_a1.get(), 1,
ctxt_tensor_a2.get());
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "C rotation: " << elapsed.count() << " s" << std::endl;

        start = std::chrono::high_resolution_clock::now();
        for (int i = 0 ; i < 128 ; ++i) {
            //tmp =
            getHemmer()->getEval().leftRotate(ptxt_tensor_b.get(), 1,
ptxt_tensor_c.get());

        }
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "P rotation: " << elapsed.count() << " s" << std::endl;

        start = std::chrono::high_resolution_clock::now();

        std::vector<HELLM::Ciphertext> tmp;
        getHemmer()->matMulPre(ctxt_tensor_a2, tmp);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "matmulpre: " << elapsed.count() << " s" << std::endl;
    }


    //ctxt_tensor_a = getHemmer()->repackToMultiCol(ctxt_tensor_a,
out_col_block);



    for (int i = 0 ; i < 2 ; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        getHemmer()->complexPackingInplace(ctxt_tensor_a1);
        getHemmer()->complexPackingRowInplace(ctxt_tensor_b);
        ctxt_tensor_a1 = getHemmer()->packedMatMul(ctxt_tensor_a, ctxt_tensor_b,
7); ctxt_tensor_a = getHemmer()->repackCC(ctxt_tensor_a);
        //tensor_a = tensor_a.matmul(tensor_b);
    }

    //compareTensor2(tensor_a, ctxt_tensor_a, prec);
} */

/* TEST_F(HEMMerTest, MatMulCtxtLowHigh) {
    const double prec = std::pow(2.0, -15);
    const u64 low_dim = 2;
    const u64 in_row_block = 0;
    const u64 out_row_block = 0;

    auto tensor_a =
        torch::zeros({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});
    auto tensor_b =
        torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});

    tensor_a.slice(1, in_row_block * low_dim, (in_row_block + 1) * low_dim) =
        torch::rand({2, low_dim, ModelArgs::HEAD_DIM});

    auto ctxt_tensor_a = getHemmer()->encrypt2(tensor_a[0], tensor_a[1]);
    auto ctxt_tensor_b = getHemmer()->encrypt2(tensor_b[0], tensor_b[1]);

    ctxt_tensor_a =
        getHemmer()->matMulLowHigh(ctxt_tensor_a, ctxt_tensor_b, in_row_block);
    ctxt_tensor_a = getHemmer()->repackToMultiRow(ctxt_tensor_a, out_row_block);

    tensor_a = tensor_a.matmul(tensor_b);

    compareTensor2(tensor_a, ctxt_tensor_a, prec);
} */

/* TEST_F(HEMMerTest, MatMulCtxtLowLow) {
    const double prec = std::pow(2.0, -15);
    const u64 low_dim = ModelArgs::LOW_DIM;
    const u64 in_col_block = 0;
    const u64 in_row_block = 1;

    auto tensor_a =
        torch::zeros({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});
    auto tensor_b =
        torch::zeros({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});

    tensor_a.slice(2, in_col_block * low_dim, (in_col_block + 1) * low_dim) =
        torch::rand({2, ModelArgs::HEAD_DIM, low_dim});
    tensor_b.slice(1, in_row_block * low_dim, (in_row_block + 1) * low_dim) =
        torch::rand({2, low_dim, ModelArgs::HEAD_DIM});

    auto ctxt_tensor_a = getHemmer()->encrypt2(tensor_a[0], tensor_a[1]);
    auto ctxt_tensor_b = getHemmer()->encrypt2(tensor_b[0], tensor_b[1]);

    ctxt_tensor_a = getHemmer()->matMulLowLow(ctxt_tensor_a, ctxt_tensor_b,
                                              in_col_block, in_row_block);

    if (in_row_block != in_col_block) {
        tensor_a.slice(2, in_row_block * low_dim,
                       (in_row_block + 1) * low_dim) =
            tensor_a.slice(2, in_col_block * low_dim,
                           (in_col_block + 1) * low_dim);

        tensor_a.slice(2, in_col_block * low_dim,
                       (in_col_block + 1) * low_dim) =
            torch::zeros({2, ModelArgs::HEAD_DIM, low_dim});
    }

    tensor_a = tensor_a.matmul(tensor_b);

    compareTensor2(tensor_a, ctxt_tensor_a, prec);
} */

/*
TEST_F(HEMMerTest, SqrtInverse_Newton) {
    const double prec = std::pow(2.0, -20);

    const auto min = 0.01/10;
    const auto max = 0.01;

    //auto tensor =
    //    torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM})*(max -
min) + min ;

    auto tensor =
        torch::ones({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM})*min ;


    auto tensor_out = torch::rsqrt(tensor);
    auto ctxt_tensor = getHemmer()->encrypt2(tensor[0], tensor[1]);
    auto ctxt_tensor_out1{ctxt_tensor};


    LoRA::approxInvSqrt_adamw(getHemmer()->getEval(), getHemmer()->getBtp(),
             ctxt_tensor.get(), ctxt_tensor_out1.get(), 20.0, 40);
    reportCompareTensor3dim(tensor_out, ctxt_tensor_out1);
}
*/

/*
TEST_F(HEMMerTest, Rot) {
    const double prec = std::pow(2.0, -15);

    auto tensor_a =
        torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});
    auto tensor_b =
        torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});
    auto tensor_c =
        torch::rand({128,128});

    auto ctxt_tensor_a = getHemmer()->encrypt2(tensor_a[0], tensor_a[1]);
    auto ctxt_tensor_b = getHemmer()->encrypt2(tensor_b[0], tensor_b[1]);

    auto ptxt = getHemmer()->encode2(tensor_c, tensor_c);

    for (u64 i = 0 ; i < 2 ; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        getHemmer()->getEval().leftRotate(ctxt_tensor_a.get(), 4,
ctxt_tensor_a.get()); auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;
    }


    for (u64 i = 0 ; i < 2 ; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        getHemmer()->getEval().leftRotate(ctxt_tensor_a.get(), 1,
ctxt_tensor_a.get()); getHemmer()->getEval().leftRotate(ctxt_tensor_a.get(), 4,
ctxt_tensor_a.get()); auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;
    }
    //tensor_a = tensor_a.matmul(tensor_b.transpose(1, 2));

    //compareTensor2(tensor_a, ctxt_tensor_a, prec);
} */

/* TEST_F(HEMMerTest, BackwardMatMulCtxtPacked1) {
    const double prec = std::pow(2.0, -15);

    auto tensor_a =
        torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});
    auto tensor_b =
        torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});

    auto ctxt_tensor_a = getHemmer()->encrypt2(tensor_a[0], tensor_a[1]);
    auto ctxt_tensor_b = getHemmer()->encrypt2(tensor_b[0], tensor_b[1]);

    auto start = std::chrono::high_resolution_clock::now();
    for (u64 i = 0 ; i < 1; ++i) {
        for (u64 j = 0 ; j < 6; ++j) {
            auto tmp = ctxt_tensor_a;
            tmp.get().setLevel(8);
            auto tmp2 = ctxt_tensor_b;
            tmp2.get().setLevel(9);
            getHemmer()->complexPackingInplace(tmp);
            getHemmer()->transposeComplexPackingInplace(tmp2);
            getHemmer()->packedMatMul(tmp, tmp2);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (u64 i = 0 ; i < 1 ; ++i) {
        for (u64 j = 0 ; j < 6; ++j) {
            getHemmer()->repackCC(ctxt_tensor_a);
        }
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "repack time: " << elapsed.count() << " s" << std::endl;


    //tensor_a = tensor_a.matmul(tensor_b.transpose(1, 2));

    //compareTensor2(tensor_a, ctxt_tensor_a, prec);
}
 */

/* TEST_F(HEMMerTest, BackwardMatMulCtxtPacked2) {
    const double prec = std::pow(2.0, -15);

    auto tensor_a =
        torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});
    auto tensor_b =
        torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});

    auto ctxt_tensor_a = getHemmer()->encrypt2(tensor_a[0], tensor_a[1]);
    auto ctxt_tensor_b = getHemmer()->encrypt2(tensor_b[0], tensor_b[1]);

    auto start = std::chrono::high_resolution_clock::now();
    for (u64 i = 0 ; i < 6; ++i) {
        for (u64 j = 0 ; j < 12; ++j) {
            auto tmp = ctxt_tensor_a;
            tmp.get().setLevel(8);
            auto tmp2 = ctxt_tensor_b;
            tmp2.get().setLevel(9);
            getHemmer()->complexPackingInplace(tmp);
            getHemmer()->transposeComplexPackingInplace(tmp2);
            getHemmer()->packedMatMul(tmp, tmp2);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (u64 i = 0 ; i < 6 ; ++i) {
        for (u64 j = 0 ; j < 12; ++j) {
            getHemmer()->repackCC(ctxt_tensor_a);
        }
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "repack time: " << elapsed.count() << " s" << std::endl;


    //tensor_a = tensor_a.matmul(tensor_b.transpose(1, 2));

    //compareTensor2(tensor_a, ctxt_tensor_a, prec);
} */

/* TEST_F(HEMMerTest, matmulPre) {
    const double prec = std::pow(2.0, -15);

    auto tensor_a =
        torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});
    auto tensor_b =
        torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});
    auto tensor_c =
        torch::rand({128,128});

    auto ctxt_tensor_a = getHemmer()->encrypt2(tensor_a[0], tensor_a[1]);
    auto ctxt_tensor_b = getHemmer()->encrypt2(tensor_b[0], tensor_b[1]);

    auto ptxt = getHemmer()->encode2(tensor_c, tensor_c);

    auto start = std::chrono::high_resolution_clock::now();


    for (u64 i = 0 ; i < 1; ++i) {
        std::vector<Ciphertext> tmp;
        auto res = ctxt_tensor_a;
        if (i == 0) {
            getHemmer()->matMulPre(getHemmer()->complexPacking(ctxt_tensor_a,
ctxt_tensor_a), tmp);
        }
        for (u64 j = 0 ; j < 3; ++j) {
            auto res_tmp = getHemmer()->matMulReUse(tmp, ptxt);
            if( j == 0) {
                res = res_tmp;
            } else {
                getHemmer()->addInplace(res, res_tmp);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;

    //tensor_a = tensor_a.matmul(tensor_b.transpose(1, 2));

    //compareTensor2(tensor_a, ctxt_tensor_a, prec);
} */

/* TEST_F(HEMMerTest, CCMM) {
    const double prec = std::pow(2.0, -15);

    auto tensor_a =
        torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});
    auto tensor_b =
        torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});
    auto tensor_c =
        torch::rand({128,128});

    auto ctxt_tensor_a = getHemmer()->encrypt2(tensor_a[0], tensor_a[1]);
    auto ctxt_tensor_b = getHemmer()->encrypt2(tensor_b[0], tensor_b[1]);

    auto ptxt = getHemmer()->encode2(tensor_c, tensor_c);

    auto start = std::chrono::high_resolution_clock::now();
    getHemmer()->singleCCMatMul(ctxt_tensor_a, ctxt_tensor_b);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;

    //tensor_a = tensor_a.matmul(tensor_b.transpose(1, 2));

    //compareTensor2(tensor_a, ctxt_tensor_a, prec);
} */

/* TEST_F(HEMMerTest, PCMM) {
    const double prec = std::pow(2.0, -15);

    auto tensor_a =
        torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});
    auto tensor_b =
        torch::rand({2, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});
    auto tensor_c =
        torch::rand({128,128});

    auto ctxt_tensor_a = getHemmer()->encrypt2(tensor_a[0], tensor_a[1]);
    auto ctxt_tensor_b = getHemmer()->encrypt2(tensor_b[0], tensor_b[1]);

    auto ptxt = getHemmer()->encode2(tensor_c, tensor_c);

    auto start = std::chrono::high_resolution_clock::now();

    getHemmer()->singleMatMul(ctxt_tensor_a, ptxt);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;

    //tensor_a = tensor_a.matmul(tensor_b.transpose(1, 2));

    //compareTensor2(tensor_a, ctxt_tensor_a, prec);
} */

/*
TEST_F(HEMMerTest, MatMulPtxt4) {
    const double prec = std::pow(2.0, -15);

    auto tensor_a =
        torch::rand({1, 4, ModelArgs::MAX_SEQ_LEN,
ModelArgs::HEAD_DIM}).mul(0.25); auto tensor_b = torch::rand({4, 2,
ModelArgs::HEAD_DIM, ModelArgs::HEAD_DIM}).mul(0.25);

    auto ctxt_tensor_a = getHemmer()->complexPacking(
        getHemmer()->encrypt2(tensor_a[0][0], tensor_a[0][1]),
        getHemmer()->encrypt2(tensor_a[0][2], tensor_a[0][3]));
    auto ptxt_tensor_b_0 = getHemmer()->encodeDiagonalToRow4(
        tensor_b[0][0], tensor_b[1][0], tensor_b[2][0], tensor_b[3][0]);
    auto ptxt_tensor_b_1 = getHemmer()->encodeDiagonalToRow4(
        tensor_b[0][1], tensor_b[1][1], tensor_b[2][1], tensor_b[3][1]);

    std::vector<HEaaN::Ciphertext> tmp;
    getHemmer()->matMulPre(ctxt_tensor_a, tmp);
    auto temp_0 = getHemmer()->matMulReUse(tmp, ptxt_tensor_b_0);
    auto temp_1 = getHemmer()->matMulReUse(tmp, ptxt_tensor_b_1);
    ctxt_tensor_a = getHemmer()->repack(temp_0, temp_1);

    tensor_a =
        tensor_a.transpose(1, 2).reshape({1L * ModelArgs::MAX_SEQ_LEN, -1});
    tensor_b =
        tensor_b.transpose(1, 2).reshape({4L * ModelArgs::MAX_SEQ_LEN, -1});
    tensor_a = tensor_a.matmul(tensor_b)
                   .view({ModelArgs::MAX_SEQ_LEN, 2, ModelArgs::HEAD_DIM})
                   .transpose(0, 1);
    std::cout << "PCMM" << std::endl;
    reportCompareTensor3dim(tensor_a, ctxt_tensor_a);

    auto tensor_c =
        torch::rand({1, 4, ModelArgs::MAX_SEQ_LEN,
ModelArgs::HEAD_DIM}).mul(0.25); auto tensor_d = torch::rand({4, 2,
ModelArgs::HEAD_DIM, ModelArgs::HEAD_DIM}).mul(0.25);

    auto ctxt_tensor_c = getHemmer()->complexPacking(
        getHemmer()->encrypt2(tensor_c[0][0], tensor_c[0][1]),
        getHemmer()->encrypt2(tensor_c[0][2], tensor_c[0][3]));
    auto ptxt_tensor_d_0 = getHemmer()->encodeDiagonalToRow4(
        tensor_d[0][0], tensor_d[1][0], tensor_d[2][0], tensor_d[3][0]);
    auto ptxt_tensor_d_1 = getHemmer()->encodeDiagonalToRow4(
        tensor_d[0][1], tensor_d[1][1], tensor_d[2][1], tensor_d[3][1]);

    std::vector<HEaaN::Ciphertext> tmp2;
    getHemmer()->matMulPre(ctxt_tensor_c, tmp2);
    auto temp_2 = getHemmer()->matMulReUse(tmp2, ptxt_tensor_d_0);
    auto temp_3 = getHemmer()->matMulReUse(tmp2, ptxt_tensor_d_1);
    ctxt_tensor_c = getHemmer()->repack(temp_2, temp_3);

    tensor_c =
        tensor_c.transpose(1, 2).reshape({1L * ModelArgs::MAX_SEQ_LEN, -1});
    tensor_d =
        tensor_d.transpose(1, 2).reshape({4L * ModelArgs::MAX_SEQ_LEN, -1});
    tensor_c = tensor_c.matmul(tensor_d)
                   .view({ModelArgs::MAX_SEQ_LEN, 2, ModelArgs::HEAD_DIM})
                   .transpose(0, 1);

    std::cout << "PCMM" << std::endl;
    reportCompareTensor3dim(tensor_c, ctxt_tensor_c);


    tensor_a = tensor_a.matmul(tensor_c);
    //getHemmer()->complexPackingInplace(ctxt_tensor_a);
    //getHemmer()->complexPackingInplace(ctxt_tensor_c);
    ctxt_tensor_a = getHemmer()->singleCCMatMul(ctxt_tensor_a, ctxt_tensor_c);
    //ctxt_tensor_a = getHemmer()->repackCC(ctxt_tensor_a);

    std::cout << "CCMM" << std::endl;
    reportCompareTensor3dim(tensor_a, ctxt_tensor_a);


    // upper/lower one
    torch::Tensor upper_one =
torch::zeros({ModelArgs::HEAD_DIM,ModelArgs::HEAD_DIM}); torch::Tensor lower_one
= torch::ones({ModelArgs::HEAD_DIM,ModelArgs::HEAD_DIM}); torch::Tensor
right_mask = torch::zeros({ModelArgs::HEAD_DIM, ModelArgs::HEAD_DIM});
    torch::Tensor left_mask = torch::ones({ModelArgs::HEAD_DIM,
ModelArgs::HEAD_DIM});

    right_mask.slice(1,0,64).fill_(1);
    left_mask.slice(1,0,64).fill_(0);
    upper_one.slice(0,0,64).fill_(1);
    lower_one.slice(0,0,64).fill_(0);

    PtxtTensor upper_one_tensor = getHemmer()->encode2(upper_one, upper_one);
    PtxtTensor lower_one_tensor = getHemmer()->encode2(lower_one, lower_one);

    CtxtTensor mask_right = ctxt_tensor_a;
    CtxtTensor mask_left = ctxt_tensor_a;

    auto tensor_mask_right = tensor_a.mul(torch::cat({right_mask.unsqueeze(0),
right_mask.unsqueeze(0)}, 0)); auto tensor_mask_left =
tensor_a.mul(torch::cat({left_mask.unsqueeze(0), left_mask.unsqueeze(0)}, 0));


    getHemmer()->maskRightLeft(ctxt_tensor_a, mask_right, mask_left);
    mask_right = getHemmer()->singleMatMul(mask_right, upper_one_tensor,5);
    mask_left = getHemmer()->singleMatMul(mask_left, lower_one_tensor,5);

    tensor_mask_right =
tensor_mask_right.matmul(torch::cat({upper_one.unsqueeze(0),
upper_one.unsqueeze(0)}, 0)); tensor_mask_left =
tensor_mask_left.matmul(torch::cat({lower_one.unsqueeze(0),
lower_one.unsqueeze(0)}, 0));


    reportCompareTensor3dim(tensor_mask_right, mask_right);
    reportCompareTensor3dim(tensor_mask_left, mask_left);
} */

/*
TEST_F(HEMMerTest, 1CRotSum) {
    const double prec = std::pow(2.0, -15);

    auto tensor_a =
        torch::rand({1, 4, ModelArgs::MAX_SEQ_LEN, ModelArgs::HEAD_DIM});
    auto tensor_b =
        torch::rand({4, 2, ModelArgs::HEAD_DIM, ModelArgs::HEAD_DIM});

    auto ctxt_tensor_a = getHemmer()->complexPacking(
        getHemmer()->encrypt2(tensor_a[0][0], tensor_a[0][1]),
        getHemmer()->encrypt2(tensor_a[0][2], tensor_a[0][3]));
    auto ptxt_tensor_b_0 = getHemmer()->encodeDiagonalToRow4(
        tensor_b[0][0], tensor_b[1][0], tensor_b[2][0], tensor_b[3][0]);
    auto ptxt_tensor_b_1 = getHemmer()->encodeDiagonalToRow4(
        tensor_b[0][1], tensor_b[1][1], tensor_b[2][1], tensor_b[3][1]);

    std::vector<HEaaN::Ciphertext> tmp;
    getHemmer()->matMulPre(ctxt_tensor_a, tmp);
    auto temp_0 = getHemmer()->matMulReUse(tmp, ptxt_tensor_b_0);
    auto temp_1 = getHemmer()->matMulReUse(tmp, ptxt_tensor_b_1);
    ctxt_tensor_a = getHemmer()->repack(temp_0, temp_1);

    tensor_a =
        tensor_a.transpose(1, 2).reshape({1L * ModelArgs::MAX_SEQ_LEN, -1});
    tensor_b =
        tensor_b.transpose(1, 2).reshape({4L * ModelArgs::MAX_SEQ_LEN, -1});
    tensor_a = tensor_a.matmul(tensor_b)
                   .view({ModelArgs::MAX_SEQ_LEN, 2, ModelArgs::HEAD_DIM})
                   .transpose(0, 1);

    reportCompareTensor3dim(tensor_a, ctxt_tensor_a);

    // upper/lower one
    torch::Tensor upper_one =
torch::zeros({ModelArgs::HEAD_DIM,ModelArgs::HEAD_DIM}); torch::Tensor lower_one
= torch::ones({ModelArgs::HEAD_DIM,ModelArgs::HEAD_DIM}); torch::Tensor
right_mask = torch::zeros({ModelArgs::HEAD_DIM, ModelArgs::HEAD_DIM});
    torch::Tensor left_mask = torch::ones({ModelArgs::HEAD_DIM,
ModelArgs::HEAD_DIM});

    right_mask.slice(1,0,64).fill_(1);
    left_mask.slice(1,0,64).fill_(0);
    upper_one.slice(0,0,64).fill_(1);
    lower_one.slice(0,0,64).fill_(0);

    CtxtTensor mask_right = ctxt_tensor_a;
    CtxtTensor mask_left = ctxt_tensor_a;

    auto tensor_mask_right = tensor_a.mul(torch::cat({right_mask.unsqueeze(0),
right_mask.unsqueeze(0)}, 0)); auto tensor_mask_left =
tensor_a.mul(torch::cat({left_mask.unsqueeze(0), left_mask.unsqueeze(0)}, 0));


    getHemmer()->maskRightLeft(ctxt_tensor_a, mask_right, mask_left);
    //replcae 1CMM to rot&sum
    Ciphertext tmp_rot{getHemmer()->getContext()};
    for (i64 rot = 1 ; rot < ModelArgs::HEAD_DIM/2; rot <<=1 ) {
        getHemmer()->getEval().leftRotate(mask_right.get(),
static_cast<u64>(rot), tmp_rot); getHemmer()->getEval().add(mask_right.get(),
tmp_rot, mask_right.get());

        getHemmer()->getEval().rightRotate(mask_left.get(),
static_cast<u64>(rot), tmp_rot); getHemmer()->getEval().add(mask_left.get(),
tmp_rot, mask_left.get());
    }

    torch::Tensor rms_mask1 =
        torch::cat({torch::full({ModelArgs::HEAD_DIM, 1}, 1.0),
                    torch::zeros({ModelArgs::HEAD_DIM, ModelArgs::HEAD_DIM -
1})}, 1); torch::Tensor rms_mask2 =
        torch::cat({torch::zeros({ModelArgs::HEAD_DIM, ModelArgs::HEAD_DIM -
1}), torch::full({ModelArgs::HEAD_DIM, 1}, 1.0),}, 1);

    auto rms_mask_ = getHemmer()->message2(rms_mask1, rms_mask1);
    auto rms_mask2_ = getHemmer()->message2(rms_mask2, rms_mask2);

    getHemmer()->getEval().mult(mask_right.get(), rms_mask_, mask_right.get());
    getHemmer()->getEval().mult(mask_left.get(), rms_mask2_, mask_left.get());


    for (i64 rot = 1 ; rot < ModelArgs::HEAD_DIM; rot <<=1 ) {
        getHemmer()->getEval().rightRotate(mask_right.get(),
static_cast<u64>(rot), tmp_rot); getHemmer()->getEval().add(mask_right.get(),
tmp_rot, mask_right.get());

        getHemmer()->getEval().leftRotate(mask_left.get(),
static_cast<u64>(rot), tmp_rot); getHemmer()->getEval().add(mask_left.get(),
tmp_rot, mask_left.get());
    }


    tensor_mask_right =
tensor_mask_right.matmul(torch::cat({upper_one.unsqueeze(0),
upper_one.unsqueeze(0)}, 0)); tensor_mask_left =
tensor_mask_left.matmul(torch::cat({lower_one.unsqueeze(0),
lower_one.unsqueeze(0)}, 0));


    reportCompareTensor3dim(tensor_mask_right, mask_right);
    reportCompareTensor3dim(tensor_mask_left, mask_left);
}

 */

/* TEST_F(HEMMerTest, Masking) {
    const double prec = std::pow(2.0, -13);

    //auto tensor_a = torch::ones({2, 128, 128});
    auto tensor_a = torch::ones({2,128,128});

    auto ctxt_tensor_a = getHemmer()->encrypt2(tensor_a[0], tensor_a[1]);

    auto output1 = ctxt_tensor_a;
    auto output2 = ctxt_tensor_a;
    getHemmer()->maskRightLeft(ctxt_tensor_a, output1, output2);

    torch::Tensor out1 = torch::empty({2,128,128});
    torch::Tensor out2 = torch::empty({2,128,128});

    out1 = getHemmer()->decrypt2(output1);
    out2 = getHemmer()->decrypt2(output2);

} */

/* TEST_F(HEMMerTest, dropOut) {
    const double prec = std::pow(2.0, -15);

    auto tensor_a = torch::ones({2, 128, 128});

    auto ctxt_tensor_a = getHemmer()->encrypt2(tensor_a[0], tensor_a[1]);


    getHemmer()->dropoutInplace(ctxt_tensor_a, 0);

    std::cout << "output" << std::endl;
    printingOutput(ctxt_tensor_a);

} */

/* TEST_F(HEMMerTest, tanh) {
    const double prec = std::pow(2.0, -15);
    auto lower = -16.0;
    auto upper = 16.0;

    auto tensor_a = torch::rand({2, 128, 128})*(upper - lower) + lower;
    auto tensor_out = torch::tanh(tensor_a);

    auto ctxt_tensor_a = getHemmer()->encrypt2(tensor_a[0], tensor_a[1]);
    HELLM::Tanh::approxTanh_wide_16(getHemmer()->getEval(),
getHemmer()->getBtp(), ctxt_tensor_a.get(), ctxt_tensor_a.get(), 0);

    reportCompareTensor3dim(tensor_out, ctxt_tensor_a);

} */

/* TEST_F(HEMMerTest, comp) {

    auto upper = 6.0;
    auto lower = -6.0;

    //auto tensor_a = torch::ones({2, 128, 128});
    //tensor_a.index_put_({0,1, torch::indexing::Slice()} , 0.9);

    auto tensor_a = torch::zeros({128,128});
    tensor_a[0][0] = 0.5;
    tensor_a[1][0] = 0.2;

    auto ctxt_tensor_a = getHemmer()->encrypt2(tensor_a, tensor_a);

    std::cout << "input: " << std::endl;
    printing(ctxt_tensor_a);

    //getHemmer()->getEval().add(ctxt_tensor_a.get(), 6, ctxt_tensor_a.get());
    //getHemmer()->getEval().mult(ctxt_tensor_a.get(), 1.0/12,
ctxt_tensor_a.get());

    Ciphertext tmp{ctxt_tensor_a.get()};

    auto res = ctxt_tensor_a;
    auto comp = ctxt_tensor_a;
    auto max_res = ctxt_tensor_a;

    //getHemmer()->getEval().leftRotate(res.get(), 256, tmp);
    //getHemmer()->getEval().add(res.get(), tmp, res.get());

    getHemmer()->getEval().leftRotate(ctxt_tensor_a.get(), 256, tmp);
    getHemmer()->getEval().sub(tmp, ctxt_tensor_a.get(), tmp);
    std::cout << "sign input: " << tmp.getLevel() << std::endl;
    HELLM::Softmax::approxSign(getHemmer()->getEval(), getHemmer()->getBtp(),
        tmp, comp.get(), 1, 1, 0.5);
    getHemmer()->getEval().add(comp.get(), 0.5, comp.get());
    getHemmer()->getEval().mult(tmp, comp.get(), tmp);
    getHemmer()->getEval().add(res.get(), tmp, res.get());

    std::cout << "output level: " << res.get().getLevel() << " max: " <<
std::endl;
    //getHemmer()->getEval().mult(res.get(), 12, res.get());
    //getHemmer()->getEval().sub(res.get(), 6, res.get());
    printing(res);

    HELLM::Softmax::approxMax(getHemmer()->getEval(), getHemmer()->getBtp(),
        max_res.get(), max_res.get(), 2, 256);

    std::cout << "max function output: " << std::endl;
    printing(max_res);

} */

/* TEST_F(HEMMerTest, GK) {

    auto upper = 1.0;
    auto lower = -13.0;

    auto tensor_a = torch::rand({2, 128, 128})*(upper - lower) + lower;
    auto tensor_out = torch::exp(tensor_a);

    auto ctxt_tensor_a = getHemmer()->encrypt2(tensor_a[0], tensor_a[1]);

    //getHemmer()->getEval().levelDown(ctxt_tensor_a.get(), 6,
ctxt_tensor_a.get()); getHemmer()->transposeInplace(ctxt_tensor_a, 4);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0 ; i < 8 ; ++i) {
        ctxt_tensor_a = getHemmer()->encrypt2(tensor_a[0], tensor_a[1]);
        //getHemmer()->getEval().levelDown(ctxt_tensor_a.get(), 6,
ctxt_tensor_a.get()); getHemmer()->transposeInplace(ctxt_tensor_a, 4);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "8 x transpose eval time: " << elapsed.count() << " s" <<
std::endl;

    //Loss::approxExp15_SST2(getHemmer()->getEval(), getHemmer()->getBtp(),
    //    ctxt_tensor_a.get(), ctxt_tensor_a.get());

    //reportCompareTensor3dim(tensor_out, ctxt_tensor_a);
} */

/* TEST_F(HEMMerTest, Inv) {

    auto upper = 3.0;
    auto lower = 0.8;

    auto tensor_a = torch::rand({2, 128, 128})*(upper - lower) + lower;
    auto tensor_out = 1.0/tensor_a;

    auto ctxt_tensor_a = getHemmer()->encrypt2(tensor_a[0], tensor_a[1]);
    Loss::approxInv15_SST2(getHemmer()->getEval(), getHemmer()->getBtp(),
        ctxt_tensor_a.get(), ctxt_tensor_a.get());

    reportCompareTensor3dim(tensor_out, ctxt_tensor_a);
} */

/* TEST_F(HEMMerTest, tanh) {
    const double prec = std::pow(2.0, -15);

    auto tensor_a = torch::rand({2, 128, 128});
    auto tnesor_one = torch::ones({2,128,128});
    auto tensor_out = torch::tanh(tensor_a);

    auto ctxt_tensor_a = getHemmer()->encrypt2(tensor_a[0], tensor_a[1]);
    std::vector<HELLM::CtxtTensor> inputs;
    inputs.reserve(4);
    for (u64 i = 0 ; i < 4 ; ++i) {
        inputs.push_back(ctxt_tensor_a);
    }
    getHemmer()->tanhVectorInplace(inputs, 0);

    compareTensor2(tensor_out, inputs[0], std::pow(2.0,-25));

} */

} // namespace HELLM
