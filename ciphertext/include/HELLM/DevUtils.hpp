////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2024 Crypto Lab Inc. All rights reserved.               //
//                                                                            //
// This source code is protected under international copyright law.           //
// All rights reserved and protected by the copyright holders.                //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "HETensor.hpp"

#include <algorithm>

//#define HELLM_DEBUG
#if defined(HELLM_DEBUG)
#ifndef HDEBUG
#define HDEBUG(...)                                                            \
    do {                                                                       \
        __VA_ARGS__;                                                           \
    } while (0)
#endif
#else
#ifndef HDEBUG
#define HDEBUG(...)
#endif
#endif

namespace HELLM {

inline void printTensorValue(const torch::Tensor &tensor) {
    long height = tensor.size(0);
    long width = tensor.size(1);

    for (long i = 0; i < std::min(height, 4L); ++i) {
        for (long j = 0; j < std::min(width, 4L); ++j) {
            double val = tensor.index({i, j}).item<double>();
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

inline void printTensor(const std::vector<torch::Tensor> &tensor) {
    unsigned long num_tensor = tensor.size();
    long height = tensor[0].size(0);
    long width = tensor[0].size(1);

    double max = 0, min = 0;
    for (unsigned long idx = 0; idx < num_tensor; idx++) {
        for (long i = 0; i < height; ++i) {
            for (long j = 0; j < width; ++j) {
                double val = tensor[idx].index({i, j}).item<double>();
                max = max > val ? max : val;
                min = min < val ? min : val;
            }
        }
    }
    std::cout << "tensor shape = [" << tensor.size() << ", "
              << tensor[0].size(0) << ", " << tensor[0].size(1) << "], ";
    std::cout << "range = (" << min << ", " << max << ")" << std::endl;
    printTensorValue(tensor[0]);
    std::cout << std::endl;
}

inline void printTensor(const torch::Tensor &tensor) {
    long height = tensor.size(0);
    long width = tensor.size(1);

    double max = 0, min = 0;
    for (long i = 0; i < height; ++i) {
        for (long j = 0; j < width; ++j) {
            double val = tensor.index({i, j}).item<double>();
            max = max > val ? max : val;
            min = min < val ? min : val;
        }
    }
    std::cout << "tensor shape = [" << tensor.size(0) << ", " << tensor.size(1)
              << "], ";
    std::cout << "range = (" << min << ", " << max << ")" << std::endl;
    printTensorValue(tensor);
    std::cout << std::endl;
}

inline void compareTensor(const std::vector<torch::Tensor> &tensor_a,
                          const std::vector<torch::Tensor> &tensor_b) {
    unsigned long num_tensor = tensor_a.size();
    long height = tensor_a[0].size(0);
    long width = tensor_a[0].size(1);

    unsigned long max_idx = 0;
    long max_row = 0;
    long max_col = 0;
    double max_error = 0;
    for (unsigned long idx = 0; idx < num_tensor; idx++) {
        for (long i = 0; i < height; ++i) {
            for (long j = 0; j < width; ++j) {
                double error_tmp =
                    std::abs(tensor_a[idx].index({i, j}).item<double>() -
                             tensor_b[idx].index({i, j}).item<double>());
                if (max_error < error_tmp) {
                    max_error = error_tmp;
                    max_idx = idx;
                    max_row = i;
                    max_col = j;
                }
            }
        }
    }

    std::cout << "tensor shape a = [" << tensor_a.size() << ", "
              << tensor_a[0].size(0) << ", " << tensor_a[0].size(1)
              << "], b = [" << tensor_b.size() << ", " << tensor_b[0].size(0)
              << ", " << tensor_b[0].size(1) << "]" << std::endl;
    std::cout << "max_error = " << max_error << std::endl;
    std::cout << "max_idx = " << max_idx << ", max_row = " << max_row
              << ", max_col = " << max_col << std::endl;
    std::cout << "tensor_a = "
              << tensor_a[max_idx].index({max_row, max_col}).item<double>()
              << ", tensor_b = "
              << tensor_b[max_idx].index({max_row, max_col}).item<double>()
              << std::endl
              << std::endl;
}

// Note: The buffer 'buf' will point to the allocated buffer containing the
// serialized data. The caller must free this buffer using delete[].
[[maybe_unused]] inline size_t
serialize(char **buf, const std::vector<HELLM::CtxtTensor> &data) {
    std::stringstream ss{};
    for (const auto &i : data) {
        i.get().save(ss);
    }
    std::string sstr = ss.str();
    size_t buf_size = sstr.size();

    *buf = new char[buf_size + 1]{};
    memcpy(*buf, sstr.c_str(), buf_size + 1);

    return buf_size;
}

[[maybe_unused]] inline void
deserialize(const unsigned int &rank, const char **buf, const size_t &buf_size,
            const HEaaN::Context &context,
            const std::vector<HELLM::CtxtTensor> &origin,
            std::vector<HELLM::CtxtTensor> &data) {
    if (rank > 0) {
        std::stringstream ss{};
        size_t ctxt_size = buf_size / data.capacity();
        char *buf_ptr = const_cast<char *>(*buf);

        for (size_t i = 0; i < data.capacity(); ++i) {
            ss.write(buf_ptr, static_cast<long>(ctxt_size));
            HEaaN::Ciphertext tmp(context);
            tmp.load(ss);
            // FIXME: check HEaaN to synchronize this
            tmp.to(HEaaN::getCurrentCudaDevice());
            data.emplace_back(origin[i].getHeight(), origin[i].getWidth(), tmp,
                              origin[i].getBlockWidth());
            ss.str("");
            buf_ptr += ctxt_size;
        }
    }
}

[[maybe_unused]] inline void printPoly(HEaaN::Ciphertext &ctxt, int size,
                                       int degree) {
    ctxt.to(HEaaN::getDefaultDevice());
    auto *pdata = ctxt.getPolyData(0, 0);
    for (int i = 0; i < size; i++)
        if (i % degree == 0)
            std::cout << pdata[i] << " ";
    std::cout << std::endl;
    ctxt.to(HEaaN::getCurrentCudaDevice());
}

} // namespace HELLM
