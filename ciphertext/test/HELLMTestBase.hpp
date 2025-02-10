////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2024 Crypto Lab Inc. All rights reserved.               //
//                                                                            //
// This source code is protected under international copyright law.           //
// All rights reserved and protected by the copyright holders.                //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include <ATen/core/TensorBody.h>
#include <gtest/gtest.h>
#include <memory>

#include "HELLM/HEMMer.hpp"
#include "HELLM/ModelArgs.hpp"

#include "HELLM/LoRA.hpp"

namespace HELLM {
class HELLMTestBase : public ::testing::Test {
protected:
    static HEMMer *getHemmer() {
        static auto *hemmer = new HEMMer{HEMMer::genHEMMer()};
        return hemmer;
    };

    static void TearDownTestSuite() { // NOLINT
        delete getHemmer();
        HELLM::cleanUpMatrixTransformer();
    }

    void compareTensor(const torch::Tensor &tensor_a,
                       const torch::Tensor &tensor_b,
                       double max_error = std::pow(2.0, -14)) const {
        ASSERT_EQ(tensor_a.size(0), tensor_b.size(0));
        ASSERT_EQ(tensor_a.size(1), tensor_b.size(1));

        for (i64 i = 0; i < tensor_a.size(0); ++i) {
            for (i64 j = 0; j < tensor_a.size(1); ++j) {
                ASSERT_NEAR(tensor_a.index({i, j}).item<double>(),
                            tensor_b.index({i, j}).item<double>(), max_error)
                    << i << " " << j;
            }
        }
    }

    void printing(const CtxtTensor &ctxt_tensor) const {
        auto dec_tensor = getHemmer()->decrypt2(ctxt_tensor);

        for (i64 k = 0; k < dec_tensor.size(0); ++k) {
            for (i64 i = 0; i < 4; ++i) {
                for (i64 j = 62; j < 62 + 4; ++j) {
                    std::cout << dec_tensor[k].index({i, j}).item<double>()
                              << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    void printingOutput(const CtxtTensor &ctxt_tensor) const {
        auto dec_tensor = getHemmer()->decrypt2(ctxt_tensor);

        for (i64 k = 0; k < dec_tensor.size(0); ++k) {
            for (i64 i = 0; i < 4; ++i) {
                for (i64 j = 0; j < 4; ++j) {
                    std::cout << dec_tensor[k].index({i, j}).item<double>()
                              << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    void compareTensor(const torch::Tensor &tensor,
                       const CtxtTensor &ctxt_tensor,
                       double max_error = std::pow(2.0, -14)) const {
        auto dec_tensor = getHemmer()->decrypt(ctxt_tensor);

        ASSERT_EQ(tensor.size(0), dec_tensor.size(0));
        ASSERT_EQ(tensor.size(1), dec_tensor.size(1));

        for (i64 i = 0; i < tensor.size(0); ++i) {
            for (i64 j = 0; j < tensor.size(1); ++j) {
                ASSERT_NEAR(tensor.index({i, j}).item<double>(),
                            dec_tensor.index({i, j}).item<double>(), max_error)
                    << i << " " << j;
            }
        }
    }

    void compareTensor2(const torch::Tensor &tensor,
                        const CtxtTensor &ctxt_tensor,
                        double max_error = std::pow(2.0, -14)) const {
        auto dec_tensor = getHemmer()->decrypt2(ctxt_tensor);

        ASSERT_EQ(tensor.size(0), dec_tensor.size(0));
        ASSERT_EQ(tensor.size(1), dec_tensor.size(1));
        ASSERT_EQ(tensor.size(2), dec_tensor.size(2));

        for (i64 i = 0; i < tensor.size(0); ++i) {
            for (i64 j = 0; j < tensor.size(1); ++j) {
                for (i64 k = 0; k < tensor.size(2); ++k) {
                    ASSERT_NEAR(tensor.index({i, j, k}).item<double>(),
                                dec_tensor.index({i, j, k}).item<double>(),
                                max_error)
                        << i << " " << j << " " << k;
                }
            }
        }
    }

    void reportCompareTensor3dim(const torch::Tensor &tensor,
                                 const CtxtTensor &ctxt_tensor) const {
        double max_err = 0;
        i64 max_k;
        i64 max_i;
        i64 max_j;

        auto dec_tensor = getHemmer()->decrypt2(ctxt_tensor);

        ASSERT_EQ(tensor.size(0), dec_tensor.size(0));
        ASSERT_EQ(tensor.size(1), dec_tensor.size(1));

        for (i64 k = 0; k < tensor.size(0); ++k) {
            for (i64 i = 0; i < tensor.size(1); ++i) {
                for (i64 j = 0; j < tensor.size(2); ++j) {
                    auto tmp_err =
                        (tensor[k].index({i, j}) - dec_tensor[k].index({i, j}))
                            .abs()
                            .item<double>();
                    if (max_err < tmp_err) {
                        max_err = tmp_err;
                        max_i = i;
                        max_j = j;
                        max_k = k;
                    }
                }
            }
        }

        std::cout << std::endl;
        std::cout << "MAX ERROR = " << std::log2(max_err) << std::endl;
        // std::cout << "tensor(" << max_i << ", " << max_j
        //           << ") = " << tensor[max_k].index({max_i,
        //           max_j}).item<double>()
        //           << std::endl
        //           << "ctxt_tensor(" << max_i << ", " << max_j
        //          << ") = " << dec_tensor[max_k].index({max_i,
        //          max_j}).item<double>()
        //           << std::endl
        //           << std::endl;
    }

    void reportCompareTensor(const torch::Tensor &tensor,
                             const CtxtTensor &ctxt_tensor) const {
        double max_err = 0;
        i64 max_i;
        i64 max_j;

        auto dec_tensor = getHemmer()->decrypt(ctxt_tensor);

        ASSERT_EQ(tensor.size(0), dec_tensor.size(0));
        ASSERT_EQ(tensor.size(1), dec_tensor.size(1));

        for (i64 i = 0; i < tensor.size(0); ++i) {
            for (i64 j = 0; j < tensor.size(1); ++j) {
                auto tmp_err = (tensor.index({i, j}) - dec_tensor.index({i, j}))
                                   .abs()
                                   .item<double>();
                if (max_err < tmp_err) {
                    max_err = tmp_err;
                    max_i = i;
                    max_j = j;
                }
            }
        }

        std::cout << std::endl;
        std::cout << "MAX ERROR = " << max_err << std::endl;
        std::cout << "tensor(" << max_i << ", " << max_j
                  << ") = " << tensor.index({max_i, max_j}).item<double>()
                  << std::endl
                  << "ctxt_tensor(" << max_i << ", " << max_j
                  << ") = " << dec_tensor.index({max_i, max_j}).item<double>()
                  << std::endl
                  << std::endl;
    }

    void reportCompareVectorTensor2(
        const std::vector<torch::Tensor> &tensor_vec,
        const std::vector<CtxtTensor> &ctxt_tensor_vec) const {

        std::vector<torch::Tensor> dec_tensor_vec(ctxt_tensor_vec.size());
        for (i64 idx = 0; idx < ctxt_tensor_vec.size(); ++idx) {
            auto tmp = getHemmer()->decrypt2(ctxt_tensor_vec[idx]);
            dec_tensor_vec[idx] = torch::cat({tmp[0], tmp[1]}, 1);
        }

        auto dec_tensor = torch::cat(dec_tensor_vec, 1);
        auto tensor = torch::cat(tensor_vec, 1);

        auto dec_min = dec_tensor.min();
        auto dec_max = dec_tensor.max();

        auto abs_err = (dec_tensor - tensor).abs();
        auto max_err = abs_err.max();
        auto max_idx = abs_err.argmax().item<int>();

        auto max_i = max_idx / tensor.size(1);
        auto max_j = max_idx % tensor.size(1);

        std::cout << "dec minimum = " << dec_min.item<double>()
                  << ", dec maximum = " << dec_max.item<double>() << std::endl;

        std::cout << std::endl;
        std::cout << "MAX ERROR = " << max_err.item<double>() << std::endl;
        std::cout << "INDEX = (" << max_i << ", " << max_j << ")" << std::endl;
        std::cout << "tensor(INDEX) = "
                  << tensor.index({max_i, max_j}).item<double>() << std::endl;
        std::cout << "dec_tensor(INDEX) = "
                  << dec_tensor.index({max_i, max_j}).item<double>()
                  << std::endl;
    }

    void reportCompareVectorTensor3(
        const std::vector<torch::Tensor> &tensor_vec,
        const std::vector<std::vector<CtxtTensor>> &ctxt_tensor) const {

        std::vector<torch::Tensor> dec_tensor_vec(ctxt_tensor[0].size());
        for (i64 idx = 0; idx < ctxt_tensor[0].size(); ++idx) {
            std::vector<torch::Tensor> dec_tensor_tmp(ctxt_tensor.size());
            for (i64 sb = 0; sb < ctxt_tensor.size(); ++sb) {
                auto tmp = getHemmer()->decrypt2(ctxt_tensor[sb][idx]);
                dec_tensor_tmp[sb] = torch::cat({tmp[0], tmp[1]}, 1);
            }
            dec_tensor_vec[idx] = torch::cat(dec_tensor_tmp, 0);
        }

        auto dec_tensor = torch::cat(dec_tensor_vec, 1);
        auto tensor = torch::cat(tensor_vec, 1);

        auto dec_min = dec_tensor.min();
        auto dec_max = dec_tensor.max();

        auto abs_err = (dec_tensor - tensor).abs();
        auto max_err = abs_err.max();
        auto max_idx = abs_err.argmax().item<int>();

        auto max_i = max_idx / tensor.size(1);
        auto max_j = max_idx % tensor.size(1);

        std::cout << std::endl;
        std::cout << "dec minimum = " << dec_min.item<double>()
                  << ", dec maximum = " << dec_max.item<double>() << std::endl;

        std::cout << std::endl;
        std::cout << "MAX ERROR = " << max_err.item<double>() << std::endl;
        std::cout << "INDEX = (" << max_i << ", " << max_j << ")" << std::endl;
        std::cout << "tensor(INDEX) = "
                  << tensor.index({max_i, max_j}).item<double>() << std::endl;
        std::cout << "dec_tensor(INDEX) = "
                  << dec_tensor.index({max_i, max_j}).item<double>()
                  << std::endl;
    }
};
} // namespace HELLM
