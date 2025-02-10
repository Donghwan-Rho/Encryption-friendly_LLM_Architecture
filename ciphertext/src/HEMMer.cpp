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
#include "HELLM/MatrixUtils.hpp"
#include "HELLM/ModelArgs.hpp"
#include "HELLM/Softmax.hpp"

// bert
#include "HELLM/Exp.hpp"
#include "HELLM/LayerNorm.hpp"
#include "HELLM/LoRA.hpp"
#include "HELLM/Loss.hpp"
#include "HELLM/ReLU.hpp"
#include "HELLM/Tanh.hpp"

#include "HELLM/utils/check_macros.hpp"

#include <ATen/core/TensorBody.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/concat.h>
#include <ATen/ops/masked_fill.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/zero.h>
#include <ATen/ops/zeros_like.h>
#include <algorithm>
#include <c10/core/ScalarType.h>
#include <cstdio>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "HEaaN-math/HEaaN-math.hpp"
#include "HEaaN/HEaaN.hpp"
#include "HEaaN/Integers.hpp"
#include "HEaaN/Message.hpp"
#include "HEaaN/ParameterPreset.hpp"
#include "HEaaN/Real.hpp"
#include "HEaaN/device/CudaTools.hpp"
#include "HEaaN/device/Device.hpp"

#include "HELLM/DevUtils.hpp"

#include "torch/script.h"
#include "torch/torch.h"

#include <omp.h>

#ifdef HELLM_MULTIGPU
#include "mpi.h"
#endif

namespace HELLM {

namespace {

MatrixTransformer matrix_transformer;

inline int initMPIandGetRank() {
    // Initialize MPI
    int initialized{};

    MPICHECK(MPI_Initialized(&initialized));
    if (initialized == 0) {
        MPICHECK(MPI_Init(nullptr, nullptr));
    }

    int rank{};
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    return rank;
}

} // namespace

void cleanUpMatrixTransformer() {
    int res = std::atexit([]() { matrix_transformer.cleanUp(); });

    if (res != 0) {
        HELLM_RUNTIME_ERR("std::atexit failed to registration.");
    }
}

HEMMer HEMMer::genHEMMer(int device_id, ParameterPreset preset,
                         const bool is_multi_gpu) {
    int selected_device_id = device_id;

#ifdef HELLM_MULTIGPU
    if (is_multi_gpu)
        selected_device_id = initMPIandGetRank();
#endif

    const std::string key_path = ::getenv("HELLM_KEY_PATH") != nullptr
                                     ? ::getenv("HELLM_KEY_PATH")
                                     : std::string{};

    if (!key_path.empty() &&
        std::filesystem::exists(key_path + "/SecretKey.bin")) {
        return HEMMer{key_path, selected_device_id, preset};
    }

    return HEMMer{selected_device_id, preset};
}

#ifdef HELLM_MULTIGPU
HEMMer HEMMer::genHEMMerMultiGPU(ParameterPreset preset) {
    return HEMMer::genHEMMer(0, preset, true);
}

void HEMMer::ncclDestroy() {
    NCCLCHECK(ncclCommFinalize(HEMMer::comm_));
    NCCLCHECK(ncclCommDestroy(HEMMer::comm_));
}
#endif

HEMMer::HEMMer(int device_id, ParameterPreset preset)
    : device_id_{device_id},

      context_{makeContext(preset, {device_id_})},
      log_slots_{getLogFullSlots(context_)}, sk_{context_}, enc_{context_},
      dec_{context_}, endec_{context_}, pack_{context_}, eval_{context_, pack_},
      btp_{eval_, log_slots_}, mask_matmul_{context_},
      mask_matmul_half_(context_) {
    {
        KeyGenerator keygen{context_, sk_, pack_};

        keygen.genCommonKeys();
        keygen.genRotKeysForBootstrap(log_slots_);
    }

    setCurrentCudaDevice(device_id_);

#ifdef HELLM_MULTIGPU
    // Create unique NCCL ID for each rank (= device_id)
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &max_rank_));

    ncclUniqueId id;
    if (device_id == 0)
        NCCLCHECK(ncclGetUniqueId(&id));
    MPICHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    // Create NCCL communicator
    NCCLCHECK(ncclCommInitRank(&HEMMer::comm_, max_rank_, id, device_id));

    printf("[MPI Rank %d/%d] Initialized.\n", device_id + 1, max_rank_);
#endif

    sk_.to(getCurrentCudaDevice());
    pack_.to(getCurrentCudaDevice());

    genMask();

    generateAxisMasks();
    generateRightLeftMasks();
    generateDropoutMasks();

    std::filesystem::create_directories(he_path_);
    std::filesystem::create_directories(torch_path_);
}

HEMMer::HEMMer(const std::string &key_path, int device_id,
               ParameterPreset preset)
    : device_id_{device_id},

      context_{makeContext(preset, {device_id_})}, log_slots_{getLogFullSlots(
                                                       context_)},

      sk_{context_, key_path + "/SecretKey.bin"}, enc_{context_},
      dec_{context_}, endec_{context_}, pack_{context_, key_path},
      eval_{context_, pack_}, btp_{eval_, log_slots_}, mask_matmul_{context_},
      mask_matmul_half_(context_) {
    setCurrentCudaDevice(device_id_);

#ifdef HELLM_MULTIGPU
    // Create unique NCCL ID for each rank (= device_id)
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &max_rank_));

    ncclUniqueId id;
    if (device_id == 0)
        NCCLCHECK(ncclGetUniqueId(&id));
    MPICHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    // Create NCCL communicator
    NCCLCHECK(ncclCommInitRank(&HEMMer::comm_, max_rank_, id, device_id));

    printf("[MPI Rank %d/%d] Initialized.\n", device_id + 1, max_rank_);
#endif

    sk_.to(getCurrentCudaDevice());
    pack_.to(getCurrentCudaDevice());

    genMask();

    generateAxisMasks();
    generateRightLeftMasks();
    generateDropoutMasks();

    std::filesystem::create_directories(he_path_);
    std::filesystem::create_directories(torch_path_);
}

HEMMer::~HEMMer() {
#ifdef HELLM_MULTIGPU
    // Finalize NCCL and MPI
    HEMMer::ncclDestroy();
    int finalized{};
    MPICHECK(MPI_Finalized(&finalized));
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (finalized == 0) {
        MPICHECK(MPI_Finalize());
    }
#endif
    context_.reset();
}

void HEMMer::save(const std::string &keys_path) {
    sk_.save(keys_path + "/SecretKey.bin");
    pack_.to(getDefaultDevice());
    pack_.save(keys_path);
    pack_.to(Device{DeviceType::GPU, device_id_});
}

PtxtTensor HEMMer::encode(const torch::Tensor &data, u64 level) const {
    i64 height = data.size(0);
    i64 width = data.size(1);

    HEaaN::Message msg{log_slots_, 0};

    auto tmp = data.contiguous().toType(torch::kDouble);
    std::copy_n(tmp.data_ptr<double>(), tmp.numel(), msg.begin());
    msg.to(getCurrentCudaDevice());

    if (level == 0) {
        level = getEncryptionLevel(getContext());
    }
    Plaintext ptxt = endec_.encode(msg, level);
    return PtxtTensor{height, width, ptxt};
}

Message HEMMer::message2(const torch::Tensor &data1,
                         const torch::Tensor &data2) const {
    HEaaN::Message msg{log_slots_, 0};

    // 2 * [height, width] -> [height, 2 * width]
    auto data = torch::cat({data1, data2}, 1);
    data = data.contiguous().toType(torch::kDouble);

    std::copy_n(data.data_ptr<double>(), data.numel(), msg.begin());
    msg.to(getCurrentCudaDevice());

    return msg;
}

PtxtTensor HEMMer::encode2(const torch::Tensor &data1,
                           const torch::Tensor &data2, u64 level) const {
    auto msg = message2(data1, data2);

    i64 height = data1.size(0);
    i64 width = data1.size(1);

    if (level == 0) {
        level = getEncryptionLevel(getContext());
    }
    Plaintext ptxt = endec_.encode(msg, level);
    return PtxtTensor{height, width, ptxt, width * 2};
}

Message HEMMer::msg4096Vector(const torch::Tensor &data) const {
    i64 height = data.size(0);
    i64 width = data.size(1);
    i64 block_width = 256;
    i64 unit_vector = static_cast<i64>(1UL << log_slots_) / (block_width);
    if (height != 1 || width % 32 != 0)
        throw HEaaN::RuntimeException(
            "only 1 X (i*32) size vector is supported");

    HEaaN::Message msg{log_slots_, 0};

    i64 a_vector_len = width / 32;
    i64 margin;
    for (margin = unit_vector; margin < a_vector_len; margin += unit_vector)
        ;

    for (i64 i = 0; i < 16; i++) {
        for (i64 j = margin - a_vector_len; j < margin; j++) {
            msg[static_cast<u64>((8 * i + j / unit_vector) * block_width +
                                 (j % unit_vector))] =
                HEaaN::Complex(
                    data[0][(width / 32) * (2 * i) + j + a_vector_len - margin]
                        .item<double>());
            msg[static_cast<u64>((8 * i + j / unit_vector) * block_width +
                                 (j % unit_vector) + block_width / 2)] =
                HEaaN::Complex(data[0][(width / 32) * (2 * i + 1) + j +
                                       a_vector_len - margin]
                                   .item<double>());
        }
    }

    msg.to(getCurrentCudaDevice());
    return msg;
}

PtxtTensor HEMMer::encode4096Vector(const torch::Tensor &data,
                                    u64 level) const {
    i64 height = data.size(0);
    i64 width = data.size(1);
    i64 block_width = 256;
    Message msg{msg4096Vector(data)};
    if (level == 0) {
        level = getEncryptionLevel(getContext());
    }
    Plaintext ptxt = endec_.encode(msg, level);
    return PtxtTensor{height, width, ptxt, block_width};
}

PtxtTensor HEMMer::encode4096VectorRowwiseRepeat(const torch::Tensor &data,
                                                 u64 level) const {
    i64 height = data.size(0);
    i64 width = data.size(1);
    if (height != 1 || width != 128)
        throw HEaaN::RuntimeException("only 1 X 128 vector is supported");

    HEaaN::Message msg{log_slots_, 0};

    // put identical 128 data on each row
    for (i64 i = 0; i < 16; i++) {
        for (i64 j = 0; j < width; j++) {
            msg[static_cast<u64>(8 * i * 2 * width + j)] =
                HEaaN::Complex(data[0][j].item<double>());
            msg[static_cast<u64>(8 * i * 2 * width + width + j)] =
                HEaaN::Complex(data[0][j].item<double>());
        }
    }

    msg.to(getCurrentCudaDevice());

    if (level == 0) {
        level = getEncryptionLevel(getContext());
    }
    Plaintext ptxt = endec_.encode(msg, level);
    return PtxtTensor{height, width, ptxt};
}

Message HEMMer::msgDiagonalToRow4(const torch::Tensor &data1,
                                  const torch::Tensor &data2,
                                  const torch::Tensor &data3,
                                  const torch::Tensor &data4) const {
    // i64 height = data1.size(0);
    // i64 width = data1.size(1);

    // HEaaN::Message msg{log_slots_, 0};
    // for (i64 i = 0; i < height; ++i) {
    //     for (i64 j = 0; j < width; ++j) {
    //         i64 diagonal = (i - j + height) % height;
    //         msg[static_cast<u64>(2 * diagonal * width + j)] =
    //             HEaaN::Complex(0.5 * data1[i][j].item<double>(),
    //                            -0.5 * data3[i][j].item<double>());
    //         msg[static_cast<u64>((2 * diagonal + 1) * width + j)] =
    //             HEaaN::Complex(0.5 * data2[i][j].item<double>(),
    //                            -0.5 * data4[i][j].item<double>());
    //     }
    // }

    i64 width = data1.size(1);
    // [x], [y] -> [x + iy] -> width * [height]
    auto data_left =
        torch::split(0.5 * torch::complex(data1.toType(torch::kDouble),
                                          -data3.toType(torch::kDouble)),
                     1, 1);
    auto data_right =
        torch::split(0.5 * torch::complex(data2.toType(torch::kDouble),
                                          -data4.toType(torch::kDouble)),
                     1, 1);
    for (i64 j = 1; j < width; ++j) {
        u32 idx = static_cast<u32>(j);
        data_left[idx] = torch::roll(data_left[idx], -j);
        data_right[idx] = torch::roll(data_right[idx], -j);
    }

    auto data =
        torch::cat({torch::cat(data_left, 1), torch::cat(data_right, 1)}, 1);
    data = data.contiguous().toType(torch::kComplexDouble);
    HEaaN::Message msg{log_slots_, 0};

    std::transform(data.data_ptr<c10::complex<double>>(),
                   data.data_ptr<c10::complex<double>>() + data.numel(),
                   msg.begin(), [](auto x) { return HEaaN::Complex(x); });

    return msg;
}

PtxtTensor HEMMer::encodeDiagonalToRow4(const torch::Tensor &data1,
                                        const torch::Tensor &data2,
                                        const torch::Tensor &data3,
                                        const torch::Tensor &data4,
                                        u64 level) const {
    i64 height = data1.size(0);
    i64 width = data1.size(1);

    HEaaN::Message msg = msgDiagonalToRow4(data1, data2, data3, data4);
    msg.to(getCurrentCudaDevice());
    if (level == 0) {
        level = getEncryptionLevel(getContext());
    }
    Plaintext ptxt = endec_.encode(msg, level);
    return PtxtTensor{height, width, ptxt, width * 2};
}

inline int num_change(int i) {
    int tmp1 = i % 128;
    int tmp2 = i / 128;
    int flag = tmp2 % 2;
    return 32 * tmp1 + tmp2 / 2 + 16 * flag;
}

inline u64 slot2ctxnum(int i, int j) {
    int y = num_change(i);
    int x = (num_change(j) + y) % 4096;
    return static_cast<u64>(x / 8);
}

inline u64 slot2slotidx(int i, int j) {
    int y = num_change(i);
    int x = (num_change(j) + y) % 4096;
    return static_cast<u64>(8 * y + x % 8);
}

void HEMMer::encodeWeight(const torch::Tensor &tensor,
                          const std::string &save_path,
                          bool is_generation) const {

    if (is_generation) {
        if (tensor.dim() != 3)
            throw RuntimeException(
                "[encodeWeight] Weight for generation is additionally encoded "
                "only for atn_norm_w and ffn_norm_w");
        // [32, 128, 128] -> [32, 1, 128] -> [1, 4096]
        auto tensor_obj = tensor.slice(1, 0, 1).view({1, ModelArgs::DIM});
        msg4096Vector(tensor_obj).save(save_path + ".msg");
        return;
    }

    if (tensor.dim() == 3) {
        i64 dim_0 = tensor.size(0);
        for (i64 i = 0; i < dim_0 / 2; ++i) {
            auto msg = message2(tensor[i * 2], tensor[i * 2 + 1]);
            msg.save(save_path + std::to_string(i) + ".msg");
        }
    } else {

        i64 height = tensor.size(0);
        i64 width = tensor.size(1);

        for (i64 i = 0; i < height / 4; ++i) {
            for (i64 j = 0; j < width; ++j) {
                msgDiagonalToRow4(tensor[i * 4][j], tensor[i * 4 + 1][j],
                                  tensor[i * 4 + 2][j], tensor[i * 4 + 3][j])
                    .save(save_path + std::to_string(i) + "_" +
                          std::to_string(j) + ".msg");
            }
        }
        if (height % 4 > 0) {
            for (i64 j = 0; j < width; ++j) {
                msgDiagonalToRow4(
                    tensor[height - 2][j], tensor[height - 1][j],
                    torch::zeros({ModelArgs::HEAD_DIM, ModelArgs::HEAD_DIM}),
                    torch::zeros({ModelArgs::HEAD_DIM, ModelArgs::HEAD_DIM}))
                    .save(save_path + std::to_string(height / 4) + "_" +
                          std::to_string(j) + ".msg");
            }
        }
    }
}

// column-wise packing.
void HEMMer::encodeWeight_bert(const torch::Tensor &tensor,
                               const std::string &save_path,
                               bool is_generation) const {

    if (is_generation) {
        if (tensor.dim() != 3)
            throw RuntimeException(
                "[encodeWeight] Weight for generation is additionally encoded "
                "only for atn_norm_w and ffn_norm_w");
        // [32, 128, 128] -> [32, 1, 128] -> [1, 4096]
        auto tensor_obj = tensor.slice(1, 0, 1).view({1, ModelArgs::DIM});
        msg4096Vector(tensor_obj).save(save_path + ".msg");
        return;
    }

    if (tensor.dim() == 3) {
        i64 dim_0 = tensor.size(0);
        for (i64 i = 0; i < dim_0 / 2; ++i) {
            auto msg = message2(tensor[i * 2], tensor[i * 2 + 1]);
            msg.save(save_path + std::to_string(i) + ".msg");
        }
    } else {

        i64 height = tensor.size(0);
        i64 width = tensor.size(1);

        for (i64 i = 0; i < width; ++i) {
            for (i64 j = 0; j < height / 2; ++j) {
                message2(tensor[j * 2][i], tensor[j * 2 + 1][i])
                    .save(save_path + std::to_string(j) + "_" +
                          std::to_string(i) + ".msg");
            }
        }
    }
}

// bert2: row-wise packing.
// for pooler.dense.weight, where we should pack data row-wisly.
void HEMMer::encodeWeight_bert2(const torch::Tensor &tensor,
                                const std::string &save_path,
                                bool is_generation) const {

    if (is_generation) {
        if (tensor.dim() != 3)
            throw RuntimeException(
                "[encodeWeight] Weight for generation is additionally encoded "
                "only for atn_norm_w and ffn_norm_w");
        // [32, 128, 128] -> [32, 1, 128] -> [1, 4096]
        auto tensor_obj = tensor.slice(1, 0, 1).view({1, ModelArgs::DIM});
        msg4096Vector(tensor_obj).save(save_path + ".msg");
        return;
    }

    if (tensor.dim() == 3) {
        i64 dim_0 = tensor.size(0);
        for (i64 i = 0; i < dim_0 / 2; ++i) {
            auto msg = message2(tensor[i * 2], tensor[i * 2 + 1]);
            msg.save(save_path + std::to_string(i) + ".msg");
        }
    } else {

        i64 height = tensor.size(0);
        i64 width = tensor.size(1);

        for (i64 i = 0; i < height; ++i) {
            for (i64 j = 0; j < width / 2; ++j) {
                message2(tensor[i][j * 2], tensor[i][j * 2 + 1])
                    .save(save_path + std::to_string(i) + "_" +
                          std::to_string(j) + ".msg");
            }
        }
    }
}

// bert2: row-wise packing.
// for pooler.dense.weight, where we should pack data row-wisly.
void HEMMer::encodeWeight_pooling(const torch::Tensor &tensor,
                                  const std::string &save_path,
                                  bool is_generation) const {

    if (is_generation) {
        if (tensor.dim() != 3)
            throw RuntimeException(
                "[encodeWeight] Weight for generation is additionally encoded "
                "only for atn_norm_w and ffn_norm_w");
        // [32, 128, 128] -> [32, 1, 128] -> [1, 4096]
        auto tensor_obj = tensor.slice(1, 0, 1).view({1, ModelArgs::DIM});
        msg4096Vector(tensor_obj).save(save_path + ".msg");
        return;
    }

    message2(tensor[0], tensor[1]).save(save_path + ".msg");
}

CtxtTensor HEMMer::encrypt(const torch::Tensor &data) const {

    PtxtTensor tensor = encode(data);
    CtxtTensor ret{context_, tensor.getShape()};
    enc_.encrypt(tensor.get(), sk_, ret.get());
    return ret;
}

CtxtTensor HEMMer::encrypt2(const torch::Tensor &data1,
                            const torch::Tensor &data2) const {

    PtxtTensor tensor = encode2(data1, data2);
    CtxtTensor ret{context_, tensor.getShape()};
    enc_.encrypt(tensor.get(), sk_, ret.get());
    return ret;
}

CtxtTensor HEMMer::encrypt4096Vector(const torch::Tensor &vector) const {
    PtxtTensor tensor = encode4096Vector(vector);
    CtxtTensor ret{context_, tensor.getShape()};
    enc_.encrypt(tensor.get(), sk_, ret.get());
    return ret;
}

torch::Tensor HEMMer::decrypt(const CtxtTensor &tensor) const {
    HEaaN::Message msg{log_slots_};
    dec_.decrypt(tensor.get(), sk_, msg);
    msg.to(getDefaultDevice());

    i64 height = tensor.getHeight();
    i64 width = tensor.getWidth();
    torch::Tensor ret = torch::empty({height, width});

    ret = ret.contiguous().toType(torch::kDouble);
    std::transform(msg.begin(), msg.begin() + ret.numel(),
                   ret.data_ptr<double>(),
                   [](const HEaaN::Complex &x) { return x.real(); });

    return ret;
}

torch::Tensor HEMMer::decrypt2(const CtxtTensor &tensor) const {
    HEaaN::Message msg{log_slots_};
    dec_.decrypt(tensor.get(), sk_, msg);
    msg.to(getDefaultDevice());

    i64 height = tensor.getHeight();
    i64 width = tensor.getWidth();
    torch::Tensor ret = torch::empty({height, 2 * width});

    ret = ret.contiguous().toType(torch::kDouble);
    std::transform(msg.begin(), msg.begin() + ret.numel(),
                   ret.data_ptr<double>(),
                   [](const HEaaN::Complex &x) { return x.real(); });

    // [height, 2 * width] - > 2 * [height, width] -> [2, height, width]
    ret = torch::stack(torch::split(ret, width, 1));

    return ret;
}

torch::Tensor HEMMer::decrypt4096Vector(const CtxtTensor &tensor) const {
    HEaaN::Message msg{log_slots_};
    dec_.decrypt(tensor.get(), sk_, msg);
    msg.to(getDefaultDevice());

    i64 height = tensor.getHeight();
    i64 width = tensor.getWidth();
    i64 block_width = tensor.getBlockWidth();
    i64 unit_vector = static_cast<i64>(1UL << log_slots_) / (block_width);
    if (height != 1 || width % 32 != 0)
        throw HEaaN::RuntimeException(
            "only 1 X (i*32) size vector is supported");

    i64 a_vector_len = width / 32;
    i64 margin;
    for (margin = unit_vector; margin < a_vector_len; margin += unit_vector)
        ;

    torch::Tensor ret = torch::empty({height, width});

    for (i64 i = 0; i < 16; i++) {
        for (i64 j = margin - a_vector_len; j < margin; j++) {

            ret.index_put_(
                {0, (width / 32) * (2 * i) + j + a_vector_len - margin},
                msg[static_cast<u64>((8 * i + j / unit_vector) * block_width +
                                     (j % unit_vector))]
                    .real());
            ret.index_put_(
                {0, (width / 32) * (2 * i + 1) + j + a_vector_len - margin},
                msg[static_cast<u64>((8 * i + j / unit_vector) * block_width +
                                     (j % unit_vector) + 128)]
                    .real());
        }
    }
    return ret;
}

void HEMMer::addInplace(torch::Tensor &tensor_a,
                        const torch::Tensor &tensor_b) const {
    tensor_a += tensor_b;
}
template <class T>
void HEMMer::addInplace(CtxtTensor &tensor_a,
                        const HETensor<T> &tensor_b) const {
    eval_.add(tensor_a.get(), tensor_b.get(), tensor_a.get());
}
template void HEMMer::addInplace(CtxtTensor &tensor_a,
                                 const CtxtTensor &tensor_b) const;
template void HEMMer::addInplace(CtxtTensor &tensor_a,
                                 const PtxtTensor &tensor_b) const;

void HEMMer::hadamardMultInplace(torch::Tensor &tensor_a,
                                 const torch::Tensor &tensor_b) const {
    if (tensor_a.size(0) == tensor_b.size(1))
        tensor_a *= tensor_b;
    else {
        auto temp = tensor_b.slice(0, 0, tensor_a.size(0));
        tensor_a *= temp;
    }
}
template <class T>
void HEMMer::hadamardMultInplace(CtxtTensor &tensor_a,
                                 const HETensor<T> &tensor_b) const {
    eval_.mult(tensor_a.get(), tensor_b.get(), tensor_a.get());
}
template void HEMMer::hadamardMultInplace(CtxtTensor &tensor_a,
                                          const CtxtTensor &tensor_b) const;
template void HEMMer::hadamardMultInplace(CtxtTensor &tensor_a,
                                          const PtxtTensor &tensor_b) const;
void HEMMer::hadamardMultInplace(CtxtTensor &tensor_a,
                                 const Message &tensor_b) const {
    eval_.mult(tensor_a.get(), tensor_b, tensor_a.get());
}

void HEMMer::divInplace(torch::Tensor &tensor, double num) const {
    tensor /= num;
}
void HEMMer::divInplace(CtxtTensor &tensor, double num) const {
    eval_.mult(tensor.get(), 1.0 / num, tensor.get());
}

void HEMMer::divPtxtInplace(PtxtTensor &ptxt, double val) const {
    eval_.mult(ptxt.get(), 1 / val, ptxt.get());
}

void HEMMer::bootstrap(CtxtTensor &tensor) const {
    btp_.bootstrapExtended(tensor.get(), tensor.get());
}

void HEMMer::bootstrap2(CtxtTensor &tensor1, CtxtTensor &tensor2) const {
    eval_.multImagUnit(tensor2.get(), tensor2.get());
    eval_.add(tensor1.get(), tensor2.get(), tensor1.get());
    btp_.bootstrapExtended(tensor1.get(), tensor1.get(), tensor2.get());
}

void HEMMer::bootstrap2_exponly(CtxtTensor &tensor1,
                                CtxtTensor &tensor2) const {
    eval_.multImagUnit(tensor2.get(), tensor2.get());
    eval_.add(tensor1.get(), tensor2.get(), tensor1.get());
    btp_.bootstrap(tensor1.get(), tensor1.get(), tensor2.get());
}

void HEMMer::bootstrapExtendedWithMultRange(u64 range,
                                            const HEaaN::Ciphertext &input,
                                            HEaaN::Ciphertext &res) const {
    Ciphertext error{eval_.getContext()};
    btp_.bootstrap(input, res, true);
    eval_.sub(input, res, error);
    eval_.multInteger(error, static_cast<i64>(range), error);
    btp_.bootstrap(error, error, true);
    eval_.multInteger(res, static_cast<i64>(range), res);
    eval_.add(res, error, res);
}

void HEMMer::bootstrapUnitRange(CtxtTensor &tensor) const {
    btp_.bootstrap(tensor.get(), tensor.get());
}

torch::Tensor HEMMer::matMul(const torch::Tensor &tensor_a,
                             const torch::Tensor &tensor_b) const {
    return tensor_a.mm(tensor_b);
}

void HEMMer::complexPackingInplace(CtxtTensor &tensor) const {
    Ciphertext odd{context_}, even{context_}, fus{context_};
    // [1,0,1,0,...]
    eval_.mult(tensor.get(), col_mask_even_, even);

    // [0,i,0,i,...] <=
    eval_.leftRotate(tensor.get(), 1, odd);
    eval_.multImagUnit(odd, odd);
    eval_.mult(odd, col_mask_even_, odd);

    eval_.add(odd, even, fus);
    eval_.rightRotate(fus, 1, tensor.get());
    eval_.add(tensor.get(), fus, tensor.get());
}

void HEMMer::transposeComplexPackingInplace(CtxtTensor &tensor,
                                            u64 target_level) const {
    Ciphertext odd{context_}, even{context_}, fus{context_};
    // [1,0,1,0,...]
    eval_.mult(tensor.get(), col_mask_even_, even);

    // [0,-i,0,-i,...] <=
    eval_.leftRotate(tensor.get(), 1, odd);
    eval_.multImagUnit(odd, odd);
    eval_.negate(odd, odd);
    eval_.mult(odd, col_mask_even_, odd);

    eval_.add(odd, even, fus);
    eval_.rightRotate(fus, 1, tensor.get());
    eval_.add(tensor.get(), fus, tensor.get());

    if (target_level > 0) {
        eval_.levelDown(tensor.get(), 1 + target_level, tensor.get());
    }
    transposeInplace(tensor);
}

void HEMMer::complexPackingRowInplace(CtxtTensor &tensor) const {
    Ciphertext odd{context_}, even{context_}, fus{context_};
    const u64 block_width = static_cast<u64>(tensor.getBlockWidth());

    // [[1,1,...], [0,0,...], ...]
    eval_.mult(tensor.get(), row_mask_even_, even);

    // [[0,0,...], [-i,-i,...], ...] <=
    eval_.leftRotate(tensor.get(), block_width, odd);
    eval_.multImagUnit(odd, odd);
    eval_.negate(odd, odd);
    eval_.mult(odd, row_mask_even_, odd);

    eval_.add(even, odd, fus);
    eval_.rightRotate(fus, block_width, tensor.get());
    eval_.add(tensor.get(), fus, tensor.get());
}

void HEMMer::splitInTwo(CtxtTensor &tensor, CtxtTensor &tensor1,
                        CtxtTensor &tensor2) const {
    const u64 width = static_cast<u64>(tensor.getWidth());
    Ciphertext tmp{context_};
    eval_.mult(tensor.get(), mask_left_half_, tmp);
    eval_.sub(tensor.get(), tmp, tensor2.get());

    eval_.rightRotate(tmp, width, tensor1.get());
    eval_.add(tensor1.get(), tmp, tensor1.get());

    eval_.leftRotate(tensor2.get(), width, tmp);
    eval_.add(tensor2.get(), tmp, tensor2.get());
}

CtxtTensor HEMMer::packedMatMul(const CtxtTensor &tensor_a,
                                const CtxtTensor &tensor_b,
                                u64 target_level) const {
    return multPackedMatMat(eval_, btp_, tensor_a, tensor_b, target_level,
                            matrix_transformer);
}

CtxtTensor HEMMer::packedMatMulPre(const CtxtTensor &tensor_a,
                                   u64 target_level) const {
    return multPackedMatMatPre(eval_, tensor_a, target_level,
                               matrix_transformer);
}

void HEMMer::packedMatMulPreRot(const CtxtTensor &tensor_a,
                                std::vector<Ciphertext> &tmp,
                                u64 target_level) const {
    multPackedMatMatPreRot(eval_, tensor_a, tmp, target_level,
                           matrix_transformer);
}

CtxtTensor HEMMer::packedMatMulPreRev(const CtxtTensor &tensor_a,
                                      u64 target_level) const {
    return multPackedMatMatPreRev(eval_, tensor_a, target_level,
                                  matrix_transformer);
}

CtxtTensor HEMMer::packedMatMulCCReuse(const std::vector<Ciphertext> &tmp,
                                       const CtxtTensor &tensor_b,
                                       u64 target_level) const {
    return multPackedMatMatCCReuse(eval_, tmp, tensor_b, target_level,
                                   matrix_transformer);
}

CtxtTensor HEMMer::singleMatMul(const CtxtTensor &tensor_a,
                                const PtxtTensor &tensor_b,
                                u64 target_level) const {

    return multMatMat(eval_, btp_, tensor_a, tensor_b, target_level,
                      matrix_transformer);
}

CtxtTensor HEMMer::singleCCMatMul(const CtxtTensor &tensor_a,
                                  const CtxtTensor &tensor_b,
                                  u64 target_level) const {

    return multMatMat(eval_, btp_, tensor_a, tensor_b, target_level,
                      matrix_transformer);
}

void HEMMer::oneMatRotSumInplace(CtxtTensor &tensor_a,
                                 CtxtTensor &tensor_b) const {

    Ciphertext tmp_rot{eval_.getContext()};

    for (i64 rot = 1; rot < ModelArgs::HEAD_DIM / 2; rot <<= 1) {
        eval_.leftRotate(tensor_a.get(), static_cast<u64>(rot), tmp_rot);
        eval_.add(tensor_a.get(), tmp_rot, tensor_a.get());

        eval_.rightRotate(tensor_b.get(), static_cast<u64>(rot), tmp_rot);
        eval_.add(tensor_b.get(), tmp_rot, tensor_b.get());
    }

    torch::Tensor mask1 = torch::cat(
        {torch::full({ModelArgs::HEAD_DIM, 1}, 1.0),
         torch::zeros({ModelArgs::HEAD_DIM, ModelArgs::HEAD_DIM - 1})},
        1);
    torch::Tensor mask2 = torch::cat(
        {
            torch::zeros({ModelArgs::HEAD_DIM, ModelArgs::HEAD_DIM - 1}),
            torch::full({ModelArgs::HEAD_DIM, 1}, 1.0),
        },
        1);

    auto mask1_msg = message2(mask1, mask1);
    auto mask2_msg = message2(mask2, mask2);

    eval_.mult(tensor_a.get(), mask1_msg, tensor_a.get());
    eval_.mult(tensor_b.get(), mask2_msg, tensor_b.get());

    for (i64 rot = 1; rot < ModelArgs::HEAD_DIM; rot <<= 1) {
        eval_.rightRotate(tensor_a.get(), static_cast<u64>(rot), tmp_rot);
        eval_.add(tensor_a.get(), tmp_rot, tensor_a.get());

        eval_.leftRotate(tensor_b.get(), static_cast<u64>(rot), tmp_rot);
        eval_.add(tensor_b.get(), tmp_rot, tensor_b.get());
    }
}

void HEMMer::tr_oneMatRotSumInplace(CtxtTensor &tensor_a, CtxtTensor &tensor_b,
                                    u64 target_level) const {

    if (target_level > 0) {
        eval_.levelDown(tensor_a.get(), target_level + 1, tensor_a.get());
        eval_.levelDown(tensor_b.get(), target_level + 1, tensor_b.get());
    }

    Ciphertext tmp_rot{eval_.getContext()};
    for (i64 rot = 1; rot < ModelArgs::HEAD_DIM; rot <<= 1) {
        eval_.leftRotate(tensor_a.get(), static_cast<u64>(rot), tmp_rot);
        eval_.add(tensor_a.get(), tmp_rot, tensor_a.get());

        eval_.rightRotate(tensor_b.get(), static_cast<u64>(rot), tmp_rot);
        eval_.add(tensor_b.get(), tmp_rot, tensor_b.get());
    }

    torch::Tensor mask1 = torch::cat(
        {torch::full({ModelArgs::HEAD_DIM, 1}, 1.0),
         torch::zeros({ModelArgs::HEAD_DIM, ModelArgs::HEAD_DIM - 1})},
        1);
    torch::Tensor mask2 = torch::cat(
        {
            torch::zeros({ModelArgs::HEAD_DIM, ModelArgs::HEAD_DIM - 1}),
            torch::full({ModelArgs::HEAD_DIM, 1}, 1.0),
        },
        1);

    auto mask1_msg = message2(mask1, mask1);
    auto mask2_msg = message2(mask2, mask2);

    eval_.mult(tensor_a.get(), mask1_msg, tensor_a.get());
    eval_.mult(tensor_b.get(), mask2_msg, tensor_b.get());

    for (i64 rot = 1; rot < ModelArgs::HEAD_DIM / 2; rot <<= 1) {
        eval_.rightRotate(tensor_a.get(), static_cast<u64>(rot), tmp_rot);
        eval_.add(tensor_a.get(), tmp_rot, tensor_a.get());

        eval_.leftRotate(tensor_b.get(), static_cast<u64>(rot), tmp_rot);
        eval_.add(tensor_b.get(), tmp_rot, tensor_b.get());
    }
}

CtxtTensor HEMMer::repackCC(CtxtTensor &tensor_1) const {
    CtxtTensor out{tensor_1};
    eval_.conjugate(tensor_1.get(), out.get());
    eval_.add(tensor_1.get(), out.get(), out.get());
    return out;
}

PtxtTensor HEMMer::complexPacking(const PtxtTensor &tensor_1,
                                  const PtxtTensor &tensor_2) const {
    PtxtTensor res{tensor_1}, tmp{tensor_1};
    eval_.mult(tensor_2.get(), Complex(0, 0.5), res.get());
    eval_.mult(tensor_1.get(), 0.5, tmp.get());
    eval_.sub(tmp.get(), res.get(), res.get());
    return res;
}

CtxtTensor HEMMer::complexPacking(const CtxtTensor &tensor_1,
                                  const CtxtTensor &tensor_2) const {
    CtxtTensor res{tensor_2};
    eval_.multImagUnit(tensor_2.get(), res.get());
    eval_.add(tensor_1.get(), res.get(), res.get());
    return res;
}

CtxtTensor HEMMer::complexPackingRev(const CtxtTensor &tensor_1,
                                     const CtxtTensor &tensor_2) const {
    CtxtTensor res{tensor_2};
    eval_.multImagUnit(tensor_2.get(), res.get());
    eval_.sub(tensor_1.get(), res.get(), res.get());
    return res;
}

CtxtTensor HEMMer::repack(CtxtTensor &tensor_1, CtxtTensor &tensor_2) const {
    CtxtTensor out{tensor_1};
    CtxtTensor tmp{tensor_2};

    eval_.rescale(tensor_1.get()); // 4
    eval_.rescale(tensor_2.get()); // 4

    eval_.conjugate(tensor_1.get(), out.get());
    eval_.add(tensor_1.get(), out.get(), out.get());

    eval_.conjugate(tensor_2.get(), tmp.get());
    eval_.add(tensor_2.get(), tmp.get(), tmp.get());

    eval_.multImagUnit(tmp.get(), tmp.get());

    eval_.add(out.get(), tmp.get(), out.get());

    eval_.leftRotate(out.get(), ModelArgs::HEAD_DIM, tmp.get());
    eval_.add(out.get(), tmp.get(), out.get());
    btp_.bootstrapExtended(out.get(), out.get(), tmp.get());

    eval_.mult(tmp.get(), mask_matmul_, tmp.get());
    eval_.mult(out.get(), mask_matmul_, out.get());

    eval_.rightRotate(tmp.get(), ModelArgs::HEAD_DIM, tmp.get());
    addInplace(out, tmp);
    return out;
}

CtxtTensor HEMMer::repack_notBTS(CtxtTensor &tensor_1,
                                 CtxtTensor &tensor_2) const {
    CtxtTensor out{tensor_1};
    CtxtTensor tmp{tensor_2};

    eval_.rescale(tensor_1.get()); // 4
    eval_.rescale(tensor_2.get()); // 4

    eval_.conjugate(tensor_1.get(), out.get());
    eval_.add(tensor_1.get(), out.get(), out.get());

    eval_.conjugate(tensor_2.get(), tmp.get());
    eval_.add(tensor_2.get(), tmp.get(), tmp.get());

    eval_.multImagUnit(tmp.get(), tmp.get());

    eval_.add(out.get(), tmp.get(), out.get());

    eval_.leftRotate(out.get(), ModelArgs::HEAD_DIM, tmp.get());
    eval_.add(out.get(), tmp.get(), out.get());
    // btp_.bootstrapExtended(out.get(), out.get(), tmp.get());

    eval_.mult(tmp.get(), mask_matmul_, tmp.get());
    eval_.mult(out.get(), mask_matmul_, out.get());

    eval_.rightRotate(tmp.get(), ModelArgs::HEAD_DIM, tmp.get());
    addInplace(out, tmp);
    return out;
}

CtxtTensor HEMMer::repack_loraOpti(CtxtTensor &tensor_1,
                                   CtxtTensor &tensor_2) const {
    CtxtTensor out{tensor_1};
    CtxtTensor tmp{tensor_2};

    eval_.rescale(tensor_1.get()); // 4
    eval_.rescale(tensor_2.get()); // 4
    // std::cout << "after rescale: " << tensor_1.get().getLevel() << std::endl;

    eval_.conjugate(tensor_1.get(), out.get());
    eval_.add(tensor_1.get(), out.get(), out.get());

    eval_.conjugate(tensor_2.get(), tmp.get());
    eval_.add(tensor_2.get(), tmp.get(), tmp.get());

    eval_.multImagUnit(tmp.get(), tmp.get());

    eval_.add(out.get(), tmp.get(), out.get());

    eval_.leftRotate(out.get(), ModelArgs::HEAD_DIM, tmp.get());
    eval_.add(out.get(), tmp.get(), out.get());
    // btp_.bootstrapExtended(out.get(), out.get(), tmp.get());

    CtxtTensor conj_tmp{out};
    eval_.conjugate(out.get(), conj_tmp.get());
    eval_.add(out.get(), conj_tmp.get(), out.get());

    eval_.sub(conj_tmp.get(), tmp.get(), tmp.get());
    eval_.multImagUnit(tmp.get(), tmp.get());

    eval_.mult(tmp.get(), mask_matmul_half_, tmp.get());
    eval_.mult(out.get(), mask_matmul_half_, out.get());

    eval_.rightRotate(tmp.get(), ModelArgs::HEAD_DIM, tmp.get());
    addInplace(out, tmp);
    return out;
}

void HEMMer::repackVector(std::vector<CtxtTensor> &vector) const {
    std::vector<CtxtTensor> output;
    u64 size = vector.size() / 2;
    for (u64 i = 0; i < size; ++i) {
        output.push_back(repack(vector[i * 2], vector[i * 2 + 1]));
    }
    vector = std::move(output);
}

void HEMMer::matMulPre(const CtxtTensor &tensor_a, std::vector<Ciphertext> &tmp,
                       u64 target_level) const {
    multMatMatPre(eval_, tensor_a, tmp, target_level, matrix_transformer);
}

CtxtTensor HEMMer::matMulReUse(const std::vector<Ciphertext> &tmp,
                               const PtxtTensor &tensor_b,
                               u64 target_level) const {
    return multMatMatReUse(eval_, tmp, tensor_b, target_level,
                           matrix_transformer);
}

CtxtTensor HEMMer::matMulPreRev(const CtxtTensor &tensor_a,
                                u64 target_level) const {
    return multMatMatPreRev(eval_, tensor_a, target_level, matrix_transformer);
}

CtxtTensor HEMMer::matMulCCReUse(const std::vector<Ciphertext> &tmp,
                                 const CtxtTensor &tensor_b,
                                 u64 target_level) const {
    return multMatMatCCReUse(eval_, tmp, tensor_b, target_level,
                             matrix_transformer);
}

void HEMMer::multVecPre(const CtxtTensor &vector,
                        std::vector<Ciphertext> &tmp_vectors,
                        u64 target_level) const {
    ::HELLM::multVecPre(eval_, vector, tmp_vectors, target_level);
}

CtxtTensor HEMMer::vec4096MatMul(const Ciphertext &tmp_vector,
                                 const PtxtTensor &weight,
                                 u64 target_level) const {
    return multPVec128Mat(eval_, tmp_vector, weight, target_level,
                          matrix_transformer);
}

CtxtTensor HEMMer::vec4096MatMul(const Ciphertext &tmp_vector,
                                 const Message &weight,
                                 u64 target_level) const {
    return multPVec128Mat(eval_, tmp_vector, weight, target_level,
                          matrix_transformer);
}

CtxtTensor HEMMer::vec4096MatMul(const CtxtTensor &vector,
                                 const std::vector<CtxtTensor> &mat,
                                 u64 target_level) const {
    return multCVec128Mat(eval_, vector, mat, target_level, matrix_transformer);
}

void HEMMer::multVecPost(std::vector<CtxtTensor>::const_iterator begin,
                         std::vector<CtxtTensor>::const_iterator end,
                         CtxtTensor &res) const {
    torch::Tensor mask_tensor = torch::zeros({1, 4096});
    mask_tensor.slice(1, 0, 128).fill_(1);
    // sub 1 level as input ctxts need to be rescaled
    auto mask = encode4096Vector(mask_tensor, begin->getLevel() - 1);
    ::HELLM::multVecPost(eval_, mask, begin, end, res);
}

void HEMMer::transposeInplace(torch::Tensor &tensor) const {
    tensor = tensor.transpose(0, 1);
}

void HEMMer::transposeInplace(CtxtTensor &tensor, u64 target_level) const {
    if (target_level > 0) {
        eval_.levelDown(tensor.get(), 1 + target_level, tensor.get());
    }
    CtxtTensor res = matrix_transformer.transpose(eval_, tensor);
    tensor = std::move(res);
}

std::vector<Plaintext> HEMMer::encodeConcatMasks(u64 col_level,
                                                 u64 row_level) const {
    std::vector<Plaintext> res;
    res.reserve(4);

    HEaaN::Message msg_col_cat_gen_1{log_slots_, 1},
        msg_col_cat_gen_2{log_slots_, 0}, msg_row_cat_gen_1{log_slots_, 1},
        msg_row_cat_gen_2{log_slots_, 0};
    for (u64 i = 1; i <= (ModelArgs::MAX_SEQ_LEN << 1); ++i) {
        msg_col_cat_gen_1[i * ModelArgs::HEAD_DIM - 1] = HEaaN::COMPLEX_ZERO;
        msg_col_cat_gen_2[i * ModelArgs::HEAD_DIM - 1] = HEaaN::Complex(1);
    }
    for (u64 j = 0; j < ModelArgs::HEAD_DIM; ++j) {
        u64 last_row_idx = (static_cast<u64>(ModelArgs::MAX_SEQ_LEN) - 1) * 2 *
                           static_cast<u64>(ModelArgs::HEAD_DIM);
        msg_row_cat_gen_1[last_row_idx + j] = HEaaN::COMPLEX_ZERO;
        msg_row_cat_gen_1[last_row_idx + ModelArgs::HEAD_DIM + j] =
            HEaaN::COMPLEX_ZERO;
        msg_row_cat_gen_2[last_row_idx + j] = HEaaN::Complex(1);
        msg_row_cat_gen_2[last_row_idx + ModelArgs::HEAD_DIM + j] =
            HEaaN::Complex(1);
    }

    msg_col_cat_gen_1.to(getCurrentCudaDevice());
    msg_col_cat_gen_2.to(getCurrentCudaDevice());
    msg_row_cat_gen_1.to(getCurrentCudaDevice());
    msg_row_cat_gen_2.to(getCurrentCudaDevice());

    res.emplace_back(endec_.encode(msg_col_cat_gen_1, col_level));
    res.emplace_back(endec_.encode(msg_col_cat_gen_2, col_level));
    res.emplace_back(endec_.encode(msg_row_cat_gen_1, row_level));
    res.emplace_back(endec_.encode(msg_row_cat_gen_2, row_level));
    return res;
}

void HEMMer::columnConcatInplace(CtxtTensor &tensor, const CtxtTensor &vectors,
                                 const Plaintext &mask_col_cat_gen_1,
                                 const Plaintext &mask_col_cat_gen_2,
                                 const u64 idx) const {
    eval_.leftRotate(tensor.get(), 1, tensor.get());
    eval_.mult(tensor.get(), mask_col_cat_gen_1, tensor.get());

    // column vector
    Ciphertext appendum{context_};
    const u64 rot = static_cast<u64>(tensor.getWidth()) - 8 * idx - 1;
    eval_.rightRotate(vectors.get(), rot, appendum);
    eval_.mult(appendum, mask_col_cat_gen_2, appendum);

    eval_.add(tensor.get(), appendum, tensor.get());
}

void HEMMer::rowConcatInplace(CtxtTensor &tensor, const CtxtTensor &vectors,
                              const Plaintext &mask_row_cat_gen_1,
                              const Plaintext &mask_row_cat_gen_2,
                              const u64 idx) const {
    const u64 left_rot = static_cast<u64>(tensor.getBlockWidth());
    eval_.leftRotate(tensor.get(), left_rot, tensor.get());
    eval_.mult(tensor.get(), mask_row_cat_gen_1, tensor.get());

    // row vector
    Ciphertext appendum{context_};
    const u64 width = static_cast<u64>(tensor.getBlockWidth());
    const u64 height = static_cast<u64>(tensor.getHeight());
    const u64 right_rot = width * (height - 8 * idx - 1);
    eval_.rightRotate(vectors.get(), right_rot, appendum);
    eval_.mult(appendum, mask_row_cat_gen_2, appendum);

    eval_.add(tensor.get(), appendum, tensor.get());
}

// bert
void HEMMer::maskRightLeft(const CtxtTensor &tensor, CtxtTensor &tensor_out1,
                           CtxtTensor &tensor_out2) const {
    eval_.mult(tensor.get(), mask_one_left_, tensor_out1.get());
    eval_.mult(tensor.get(), mask_one_right_, tensor_out2.get());
}

void HEMMer::maskFirstColInplace(CtxtTensor &tensor) const {
    eval_.mult(tensor.get(), mask_first_col_, tensor.get());
}

void HEMMer::maskFirstColOnlyInplace(CtxtTensor &tensor) const {
    eval_.mult(tensor.get(), mask_first_col_only_, tensor.get());
}

void HEMMer::maskFirstRowInplace(CtxtTensor &tensor) const {
    eval_.mult(tensor.get(), mask_first_row_, tensor.get());
}

void HEMMer::maskLeftHalfInplace(CtxtTensor &tensor) const {
    eval_.mult(tensor.get(), mask_left_half_, tensor.get());
}

void HEMMer::maskRightHalfInplace(CtxtTensor &tensor) const {
    eval_.mult(tensor.get(), mask_right_half_, tensor.get());
}

void HEMMer::maskFirsteleInplace(CtxtTensor &tensor1) const {
    eval_.mult(tensor1.get(), mask_first_ele_, tensor1.get());
}

// for loss gradient.
void HEMMer::maskFirstele2Inplace(CtxtTensor &tensor1) const {
    eval_.mult(tensor1.get(), mask_first_ele2_, tensor1.get());
}

// for loss gradient.
void HEMMer::maskFirstele2InplaceSST2(CtxtTensor &tensor1) const {
    eval_.mult(tensor1.get(), mask_first_ele2_sst2_, tensor1.get());
}

// (0,0) component: 0.1 for Cross Entropy approx.
void HEMMer::maskFirsteleOnlyInplace(CtxtTensor &tensor1) const {
    eval_.mult(tensor1.get(), mask_first_ele_only_, tensor1.get());
}

// (0,0) component: 1
void HEMMer::maskFirstele1OnlyInplace(CtxtTensor &tensor1) const {
    eval_.mult(tensor1.get(), mask_first_ele_1_only_, tensor1.get());
}

// (0,0) component: 0.1 for Cross Entropy approx.
void HEMMer::maskFirsteleOnlyInplaceSST2(CtxtTensor &tensor1) const {
    eval_.mult(tensor1.get(), mask_first_ele_only_sst2_, tensor1.get());
}

void HEMMer::maskFirstColPoolingInplace(CtxtTensor &tensor) const {
    eval_.mult(tensor.get(), mask_first_col_pool_, tensor.get());
}

void HEMMer::maskFirstRowPoolingInplace(CtxtTensor &tensor) const {
    eval_.mult(tensor.get(), mask_first_row_pool_, tensor.get());
}

void HEMMer::maskFirstRowsPoolingInplace(CtxtTensor &tensor) const {
    eval_.mult(tensor.get(), mask_first_rows_pool_, tensor.get());
}

void HEMMer::dropoutInplace(CtxtTensor &tensor, const std::string &name,
                            const int layer_n, const u64 idx) const {
    const auto rank = getRank();
    mask_drop_out_[idx].save(
        "./mask/dropout_" + name + "_" + std::to_string(rank) + "_" +
        std::to_string(layer_n) + "_" + std::to_string(idx) + ".msg");
    eval_.mult(tensor.get(), mask_drop_out_[idx], tensor.get());
}
void HEMMer::backwarddropoutInplace(CtxtTensor &tensor, const std::string &name,
                                    const int layer_n, const u64 idx) const {
    const auto rank = getRank();
    Message mask_(tensor.get().getLogSlots());
    mask_.load("./mask/dropout_" + name + "_" + std::to_string(rank) + "_" +
               std::to_string(layer_n) + "_" + std::to_string(idx) + ".msg");
    mask_.to(getCurrentCudaDevice());
    eval_.mult(tensor.get(), mask_drop_out_[idx], tensor.get());
}

void HEMMer::dropoutExpInplace(CtxtTensor &tensor, const std::string &name,
                               const int layer_n, const u64 idx) const {
    const auto rank = getRank();
    mask_drop_out_exp_[idx].save(
        "./mask/dropout_" + name + "_" + std::to_string(rank) + "_" +
        std::to_string(layer_n) + "_" + std::to_string(idx) + ".msg");
    eval_.mult(tensor.get(), mask_drop_out_exp_[idx], tensor.get());
}

void HEMMer::backwarddropoutExpInplace(CtxtTensor &tensor,
                                       const std::string &name,
                                       const int layer_n, const u64 idx) const {
    const auto rank = getRank();
    Message mask_(tensor.get().getLogSlots());
    mask_.load("./mask/dropout_" + name + "_" + std::to_string(rank) + "_" +
               std::to_string(layer_n) + "_" + std::to_string(idx) + ".msg");
    mask_.to(getCurrentCudaDevice());
    eval_.mult(tensor.get(), mask_, tensor.get());
}

void HEMMer::expInplace(torch::Tensor &tensor, const int layer_n,
                        const bool train_mode) const {

    tensor = torch::exp(tensor);

    if (train_mode) {
        torch::save(tensor, torch_path_ + "/exp_" + std::to_string(layer_n) +
                                "_" + ".pth");
    }
}

void HEMMer::lossExpInplace(CtxtTensor &tensor) {

    Loss::approxExp15(eval_, btp_, tensor.get(), tensor.get());
}

void HEMMer::lossExpInplaceSST2(CtxtTensor &tensor) {

    Loss::approxExp15_SST2(eval_, btp_, tensor.get(), tensor.get());
}

void HEMMer::lossInvInplace(CtxtTensor &tensor) {

    Loss::approxInv63(eval_, btp_, tensor.get(), tensor.get());
}

void HEMMer::lossInvInplaceSST2(CtxtTensor &tensor) {

    Loss::approxInv15_SST2(eval_, btp_, tensor.get(), tensor.get());
}

void HEMMer::lossMax(CtxtTensor &tensor) {

    Loss::approxMax(eval_, btp_, tensor.get(), tensor.get());
}

void HEMMer::expVectorInplace(std::vector<CtxtTensor> &tensor_vec,
                              const int layer_n, const bool train_mode) const {

    const auto rank = getRank();
    /* for (u64 i = 0 ; i < tensor_vec.size() ; ++i) {
        Exp::approxExpWide(eval_, btp_, tensor_vec[i].get(),
    tensor_vec[i].get());
    } */

    for (u64 i = 0; i < tensor_vec.size(); ++i) {
        Exp::exp_iter(eval_, btp_, tensor_vec[i].get(), tensor_vec[i].get(),
                      12);
    }
    // TODO: consider bts timing.

    if (train_mode) {
        for (u64 i = 0; i < tensor_vec.size(); i++) {
            tensor_vec[i].get().save(he_path_ + "/exp_" + std::to_string(rank) +
                                     "_" + std::to_string(layer_n) + "_" +
                                     std::to_string(i) + ".bin");
        }
    }
}

void HEMMer::expParallelInplace(std::vector<CtxtTensor> &tensor_vec,
                                const int layer_n,
                                const bool train_mode) const {

    const auto rank = getRank();
    // 11 -> 5
    Exp::exp_iter_Parallel(eval_, btp_, tensor_vec, 14);
    // TODO: consider bts timing.

    for (u64 i = 0; i < 3; ++i) {
        bootstrap2_exponly(tensor_vec[i * 2], tensor_vec[i * 2 + 1]);
    }

    if (train_mode) {

        for (u64 i = 0; i < tensor_vec.size(); i++) {
            tensor_vec[i].get().save(he_path_ + "/exp_" + std::to_string(rank) +
                                     "_" + std::to_string(layer_n) + "_" +
                                     std::to_string(i) + ".bin");
        }
    }
}

void HEMMer::softmaxVectorInplaceHETAL(std::vector<CtxtTensor> &tensor_vec,
                                       const int layer_n, const u64 base_idx,
                                       const bool train_mode,
                                       const Decryptor &dec,
                                       const SecretKey &sk) const {
    auto rank = getRank();

    if (base_idx == 100) {
        std::cout << "Wow" << std::endl;
    }

    Softmax::approxSoftmaxWide_Parallel(eval_, btp_, dec, sk, tensor_vec,
                                        tensor_vec, 128, 1024, layer_n);

    for (u64 i = 0; i < tensor_vec.size() / 2; ++i) {
        bootstrap2_exponly(tensor_vec[2 * i], tensor_vec[2 * i + 1]);
    }

    if (train_mode) {
        for (u64 i = 0; i < tensor_vec.size(); i++) {
            tensor_vec[i].get().save(
                he_path_ + "/softmax_" + std::to_string(rank) + "_" +
                std::to_string(layer_n) + "_" + std::to_string(i) + ".bin");
        }
    }
}

void HEMMer::softmaxVectorInplaceCCS(std::vector<CtxtTensor> &tensor_vec,
                                     const int layer_n,
                                     const bool train_mode) const {
    auto rank = getRank();

    Softmax::Softmax_128_512_Parallel(eval_, btp_, tensor_vec, tensor_vec);

    for (u64 i = 0; i < tensor_vec.size() / 2; ++i) {
        bootstrap2_exponly(tensor_vec[2 * i], tensor_vec[2 * i + 1]);
    }

    if (train_mode) {
        for (u64 i = 0; i < tensor_vec.size(); i++) {
            tensor_vec[i].get().save(
                he_path_ + "/softmax_" + std::to_string(rank) + "_" +
                std::to_string(layer_n) + "_" + std::to_string(i) + ".bin");
        }
    }
}

void HEMMer::reluInplace(torch::Tensor &tensor, const int layer_n,
                         const bool train_mode) const {

    tensor = torch::relu(tensor);

    if (train_mode) {
        // TODO: fix the grad of ReLU.
        auto grad = tensor; /////
        torch::save(grad,
                    torch_path_ + "/relu_" + std::to_string(layer_n) + ".pth");
    }
}

void HEMMer::reluVectorInplace(std::vector<CtxtTensor> &tensor_vec,
                               const int layer_n, const bool train_mode) const {

    const auto rank = getRank();
    // for train_mode.
    std::vector<CtxtTensor> tensor_train;
    tensor_train.reserve(tensor_vec.size());

    for (u64 i = 0; i < tensor_vec.size(); ++i) {
        tensor_train.push_back(tensor_vec[i]);
        eval_.mult(tensor_vec[i].get(), 1.0 / 50, tensor_vec[i].get());
        ReLU::ApproxReLU(eval_, btp_, tensor_vec[i].get(), tensor_vec[i].get(),
                         tensor_train[i].get());
        // should not recover input values by multiplying upper bound!
        eval_.mult(tensor_vec[i].get(), 50, tensor_vec[i].get());
    }

    // TODO: bts timing.
    for (u64 i = 0; i < tensor_vec.size() - 1; i += 2) {
        eval_.multImagUnit(tensor_vec[i + 1].get(), tensor_vec[i + 1].get());
        eval_.add(tensor_vec[i].get(), tensor_vec[i + 1].get(),
                  tensor_vec[i].get());
        btp_.bootstrap(tensor_vec[i].get(), tensor_vec[i].get(),
                       tensor_vec[i + 1].get());
    }

    if (tensor_vec.size() % 2 == 1) {
        const u64 idx = tensor_vec.size() - 1;
        btp_.bootstrap(tensor_vec[idx].get(), tensor_vec[idx].get());
    }

    if (train_mode) {
        PtxtTensor one{context_, tensor_vec[0].getShape()};
        auto height = tensor_vec[0].getHeight();
        auto width = tensor_vec[0].getWidth();
        auto block_width = tensor_vec[0].getBlockWidth();
        torch::Tensor one_tensor =
            torch::ones({height, width, block_width}) * 0.5;
        // one = encode(one_tensor);

        // TODO: find an appropriate BTS position.
        for (u64 i = 0; i < tensor_train.size() - 1; i += 2) {
            eval_.multImagUnit(tensor_train[i + 1].get(),
                               tensor_train[i + 1].get());
            eval_.add(tensor_train[i].get(), tensor_train[i + 1].get(),
                      tensor_train[i].get());
            btp_.bootstrap(tensor_train[i].get(), tensor_train[i].get(),
                           tensor_train[i + 1].get());
        }

        if (tensor_train.size() % 2 == 1) {
            const u64 idx = tensor_train.size() - 1;
            btp_.bootstrap(tensor_train[idx].get(), tensor_train[idx].get());
        }

        for (u64 i = 0; i < tensor_train.size(); i++) {
            Ciphertext grad(context_);
            eval_.add(tensor_train[i].get(), 0.5, grad);
            grad.save(he_path_ + "/relu_backward_" + std::to_string(rank) +
                      "_" + std::to_string(layer_n) + "_" + std::to_string(i) +
                      ".bin");
        }
    }
}

// TODO: replace torch version to HE version tanh.
void HEMMer::tanhVectorInplace(std::vector<CtxtTensor> &tensor_vec,
                               const int layer_n, const bool train_mode) const {

    const auto rank = getRank();

    // 10 -> 3
    for (u64 i = 0; i < tensor_vec.size(); ++i) {
        // auto tensor = decrypt2(tensor_vec[i]);
        Tanh::approxTanh(eval_, btp_, tensor_vec[i].get(), tensor_vec[i].get(),
                         layer_n);
        // Tanh::approxTanh(eval_, btp_, tensor_vec[i].get(),
        // tensor_vec[i].get(), layer_n); tensor = torch::tanh(tensor);

        // tensor_vec[i] = encrypt2(tensor[0], tensor[1]);
    }

    for (u64 i = 0; i < 2; ++i) {
        bootstrap2_exponly(tensor_vec[i * 2], tensor_vec[i * 2 + 1]);
    }

    /* CtxtTensor output{tensor_vec[0]};
    for (u64 i = 1 ; i < 4; ++i) {
        Ciphertext tmp(context_);
        eval_.rightRotate(tensor_vec[i].get(), 256*i, tmp);
        eval_.add(output.get(), tmp, output.get());
    }

    btp_.bootstrap(output.get(), output.get());

    for (u64 i = 0 ; i < 4 ; ++i) {
        CtxtTensor tmp{output};
        eval_.leftRotate(tmp.get(), i*256, tmp.get());
        maskFirstRowInplace(tmp);
        tensor_vec[i] = tmp;
    } */

    if (train_mode) {
        Ciphertext grad(context_);
        for (u64 i = 0; i < tensor_vec.size(); i++) {
            CtxtTensor square{context_, tensor_vec[i].getShape()};
            eval_.square(tensor_vec[i].get(), square.get()); // 11
            // transposeInplace(square, 8); //10
            eval_.mult(square.get(), -1.0, square.get());
            eval_.add(square.get(), 1.0, grad);
            eval_.mult(grad, mask_first_col_, grad); // 9
            grad.save(he_path_ + "/tanh_" + std::to_string(rank) + "_" +
                      std::to_string(layer_n) + "_" + std::to_string(i) +
                      ".bin");
        }
    }
}

void HEMMer::tanhInplace(CtxtTensor &tensor, const int layer_n,
                         const bool train_mode) const {

    const auto rank = getRank();

    // 10 -> 3
    Tanh::approxTanh(eval_, btp_, tensor.get(), tensor.get(), layer_n);

    /* auto tensor_tmp = decrypt2(tensor);
    tensor_tmp = torch::tanh(tensor_tmp);
    tensor = encrypt2(tensor_tmp[0], tensor_tmp[1]); */

    btp_.bootstrap(tensor.get(), tensor.get());

    if (train_mode) {
        Ciphertext grad(context_);
        CtxtTensor square{tensor};
        eval_.square(tensor.get(), square.get()); // 11
        // transposeInplace(square, 8); //10
        eval_.mult(square.get(), -1.0, square.get());
        eval_.add(square.get(), 1.0, grad);
        eval_.mult(grad, mask_first_rows_pool_, grad); // 9
        grad.save(he_path_ + "/tanh_" + std::to_string(rank) + "_" +
                  std::to_string(layer_n) + ".bin");
    }
}

void HEMMer::tanhInplace_SST2(CtxtTensor &tensor, const int layer_n,
                              const bool train_mode) const {

    const auto rank = getRank();

    // 10 -> 5
    Tanh::approxTanh_wide_16(eval_, btp_, tensor.get(), tensor.get(), layer_n);

    /* auto tensor_tmp = decrypt2(tensor);
    tensor_tmp = torch::tanh(tensor_tmp);
    tensor = encrypt2(tensor_tmp[0], tensor_tmp[1]); */

    btp_.bootstrap(tensor.get(), tensor.get());

    if (train_mode) {
        Ciphertext grad(context_);
        CtxtTensor square{tensor};
        eval_.square(tensor.get(), square.get()); // 11
        // transposeInplace(square, 8); //10
        eval_.mult(square.get(), -1.0, square.get());
        eval_.add(square.get(), 1.0, grad);
        eval_.mult(grad, mask_first_rows_pool_, grad); // 9
        grad.save(he_path_ + "/tanh_" + std::to_string(rank) + "_" +
                  std::to_string(layer_n) + ".bin");
    }
}

// bert
std::vector<torch::Tensor>
HEMMer::LayerNorm(const std::vector<torch::Tensor> &tensor_vec,
                  const std::string &module_name, const int layer_n,
                  const bool train_mode) const {
    auto var = torch::zeros_like(tensor_vec[0]);
    auto mean = torch::zeros_like(tensor_vec[0]);
    for (const auto &iter : tensor_vec) {
        var += iter.pow(2);
        mean += iter;
    }
    mean = mean.sum(-1, true) / ModelArgs::DIM;
    var = var.sum(-1, true) / ModelArgs::DIM;

    auto LN = torch::rsqrt(var - mean.pow(2) + ModelArgs::EPS);

    std::vector<torch::Tensor> ret_vec;
    ret_vec.reserve(tensor_vec.size());
    for (const auto &tensor : tensor_vec) {
        ret_vec.push_back(tensor * LN);
    }

    if (train_mode) {
        torch::save(LN, torch_path_ + "/layernorm_" + module_name + "_" +
                            std::to_string(layer_n) + "_invsqrt.pth");
        for (u64 i = 0; i < tensor_vec.size(); ++i) {
            torch::save(ret_vec[i], torch_path_ + "/layernorm_" + module_name +
                                        "_" + std::to_string(layer_n) + "_" +
                                        std::to_string(i) + ".pth");
        }
    }

    return ret_vec;
}

std::vector<CtxtTensor>
HEMMer::LayerNorm(const std::vector<CtxtTensor> &tensor_vec,
                  const std::string &module_name, const int layer_n,
                  const bool train_mode) const {

    Ciphertext ctxt_tmp{context_};
    Ciphertext ctxt_mean_tmp{context_};
    CtxtTensor inv_sqrt{context_, tensor_vec[0].getShape()};
    CtxtTensor inv_mean_sqrt{context_, tensor_vec[0].getShape()};
    const auto rank = getRank();

    inv_mean_sqrt.get() = tensor_vec[0].get();
    eval_.tensor(tensor_vec[0].get(), tensor_vec[0].get(), inv_sqrt.get());

    // squaring

    for (u64 i = 1; i < tensor_vec.size(); i++) {
        eval_.tensor(tensor_vec[i].get(), tensor_vec[i].get(), ctxt_tmp);
        eval_.add(inv_sqrt.get(), ctxt_tmp, inv_sqrt.get());

        eval_.add(inv_mean_sqrt.get(), tensor_vec[i].get(),
                  inv_mean_sqrt.get());
    }

    eval_.relinearize(inv_sqrt.get(), inv_sqrt.get());
    eval_.rescale(inv_sqrt.get());

    // summation
    Ciphertext tmp{context_};
    Ciphertext tmp_mean{context_};
    for (i64 rot = 1; rot < inv_sqrt.getBlockWidth(); rot <<= 1) {
        eval_.leftRotate(inv_sqrt.get(), static_cast<u64>(rot), tmp);
        eval_.add(inv_sqrt.get(), tmp, inv_sqrt.get());

        eval_.leftRotate(inv_mean_sqrt.get(), static_cast<u64>(rot), tmp_mean);
        eval_.add(inv_mean_sqrt.get(), tmp_mean, inv_mean_sqrt.get());
    }

    // alse can use rms_mask_ , which contains dividing with the number of input
    // tokens, in LN.
    eval_.mult(inv_sqrt.get(), rms_mask_, inv_sqrt.get());
    eval_.mult(inv_mean_sqrt.get(), rms_mask_, inv_mean_sqrt.get());

    // distribute summation valeu to each slots.
    for (i64 rot = 1; rot < inv_sqrt.getBlockWidth(); rot <<= 1) {
        eval_.rightRotate(inv_sqrt.get(), static_cast<u64>(rot), tmp);
        eval_.add(inv_sqrt.get(), tmp, inv_sqrt.get());

        eval_.rightRotate(inv_mean_sqrt.get(), static_cast<u64>(rot), tmp_mean);
        eval_.add(inv_mean_sqrt.get(), tmp_mean, inv_mean_sqrt.get());
    }

    /* if (max_rank_ > 1) {
        allReduceWrapper(inv_sqrt);
        eval_.modReduct(inv_sqrt.get());

        allReduceWrapper(inv_mean_sqrt);
        eval_.modReduct(inv_mean_sqrt.get());
    } */

    CtxtTensor mean_square{context_, inv_mean_sqrt.getShape()};
    eval_.square(inv_mean_sqrt.get(), mean_square.get());

    eval_.sub(inv_sqrt.get(), mean_square.get(), inv_sqrt.get());
    eval_.add(inv_sqrt.get(), ModelArgs::EPS, inv_sqrt.get());

    if (train_mode) {
        inverseSqrtLNFT(inv_sqrt, module_name, layer_n);
    } else {
        inverseSqrtLN(inv_sqrt, module_name, layer_n);
    }

    // Here, we need to find the right target Level.
    // Q. why do we need 1 + 4 more depth?
    if (inv_sqrt.getLevel() < 3 + btp_.getMinLevelForBootstrap()) {
        btp_.bootstrapExtended(inv_sqrt.get(), inv_sqrt.get());
    }

    std::vector<CtxtTensor> ret;
    for (u64 i = 0; i < tensor_vec.size(); ++i) {

        ret.emplace_back(context_, tensor_vec[i].getShape());
        eval_.sub(tensor_vec[i].get(), inv_mean_sqrt.get(), ret[i].get());
        eval_.mult(ret[i].get(), inv_sqrt.get(), ret[i].get());
    }

    if (train_mode) {
        // save for backward process
        inv_sqrt.get().save(he_path_ + "/layernorm_" + module_name + "_" +
                            std::to_string(rank) + "_" +
                            std::to_string(layer_n) + "_invsqrt.bin");
        for (u64 i = 0; i < tensor_vec.size(); ++i) {
            ret[i].get().save(he_path_ + "/layernorm_" + module_name + "_" +
                              std::to_string(rank) + "_" +
                              std::to_string(layer_n) + "_" +
                              std::to_string(i) + ".bin");
        }
    }

    return ret;
}

std::vector<CtxtTensor>
HEMMer::LayerNorm_multi(const std::vector<CtxtTensor> &tensor_vec,
                        const std::string &module_name, const int layer_n,
                        const bool train_mode) const {

    Ciphertext ctxt_tmp{context_};
    Ciphertext ctxt_mean_tmp{context_};
    CtxtTensor inv_sqrt{context_, tensor_vec[0].getShape()};
    CtxtTensor inv_mean_sqrt{context_, tensor_vec[0].getShape()};

    inv_mean_sqrt.get() = tensor_vec[0].get();
    eval_.tensor(tensor_vec[0].get(), tensor_vec[0].get(), inv_sqrt.get());

    // squaring
    if (getRank() == 0) {
        for (u64 i = 1; i < tensor_vec.size(); i++) {
            eval_.tensor(tensor_vec[i].get(), tensor_vec[i].get(), ctxt_tmp);
            eval_.add(inv_sqrt.get(), ctxt_tmp, inv_sqrt.get());

            eval_.add(inv_mean_sqrt.get(), tensor_vec[i].get(),
                      inv_mean_sqrt.get());
        }
    }

    eval_.relinearize(inv_sqrt.get(), inv_sqrt.get());
    eval_.rescale(inv_sqrt.get());

    // summation
    Ciphertext tmp{context_};
    Ciphertext tmp_mean{context_};
    for (i64 rot = 1; rot < inv_sqrt.getBlockWidth(); rot <<= 1) {
        eval_.leftRotate(inv_sqrt.get(), static_cast<u64>(rot), tmp);
        eval_.add(inv_sqrt.get(), tmp, inv_sqrt.get());

        eval_.leftRotate(inv_mean_sqrt.get(), static_cast<u64>(rot), tmp_mean);
        eval_.add(inv_mean_sqrt.get(), tmp_mean, inv_mean_sqrt.get());
    }

    // alse can use rms_mask_ , which contains dividing with the number of input
    // tokens, in LN.
    eval_.mult(inv_sqrt.get(), rms_mask_, inv_sqrt.get());
    eval_.mult(inv_mean_sqrt.get(), rms_mask_, inv_mean_sqrt.get());

    // distribute summation valeu to each slots.
    for (i64 rot = 1; rot < inv_sqrt.getBlockWidth(); rot <<= 1) {
        eval_.rightRotate(inv_sqrt.get(), static_cast<u64>(rot), tmp);
        eval_.add(inv_sqrt.get(), tmp, inv_sqrt.get());

        eval_.rightRotate(inv_mean_sqrt.get(), static_cast<u64>(rot), tmp_mean);
        eval_.add(inv_mean_sqrt.get(), tmp_mean, inv_mean_sqrt.get());
    }

    if (max_rank_ > 1) {
        allReduceWrapper(inv_sqrt);
        eval_.modReduct(inv_sqrt.get());

        allReduceWrapper(inv_mean_sqrt);
        eval_.modReduct(inv_mean_sqrt.get());
    }

    CtxtTensor mean_square{context_, inv_mean_sqrt.getShape()};
    eval_.square(inv_mean_sqrt.get(), mean_square.get());

    eval_.sub(inv_sqrt.get(), mean_square.get(), inv_sqrt.get());
    eval_.add(inv_sqrt.get(), ModelArgs::EPS, inv_sqrt.get());

    if (train_mode) {
        inverseSqrtLNFT(inv_sqrt, module_name, layer_n);
    } else {
        inverseSqrtLN(inv_sqrt, module_name, layer_n);
    }

    // Here, we need to find the right target Level.
    // Q. why do we need 1 + 4 more depth?
    if (inv_sqrt.getLevel() < 5 + btp_.getMinLevelForBootstrap()) {
        btp_.bootstrapExtended(inv_sqrt.get(), inv_sqrt.get());
    }

    std::vector<CtxtTensor> ret;
    for (u64 i = 0; i < tensor_vec.size(); ++i) {

        ret.emplace_back(context_, tensor_vec[i].getShape());
        eval_.sub(tensor_vec[i].get(), inv_mean_sqrt.get(), ret[i].get());
        eval_.mult(ret[i].get(), inv_sqrt.get(), ret[i].get());
    }

    if (train_mode) {
        // save for backward process
        if (getRank() == 0)
            inv_sqrt.get().save(he_path_ + "/layernorm_" + module_name + "_" +
                                std::to_string(layer_n) + "_invsqrt.bin");
        for (u64 i = 0; i < tensor_vec.size(); ++i) {
            ret[i].get().save(he_path_ + "/layernorm_" + module_name + "_" +
                              std::to_string(layer_n) + "_" +
                              std::to_string(i) + ".bin");
        }
    }

    return ret;
}

// backward BERT
std::vector<CtxtTensor>
HEMMer::backwardLayerNorm(const std::vector<CtxtTensor> &tensor_vec,
                          const std::string &module_name,
                          const int layer_n) const {

    const u64 ctxt_num = tensor_vec.size();
    const i64 block_width = tensor_vec[0].getBlockWidth();
    auto device = tensor_vec[0].get().getDevice();
    const auto rank = getRank();

    std::vector<Ciphertext> ctxt_y;
    ctxt_y.reserve(ctxt_num);
    // 6
    for (u64 i = 0; i < ctxt_num; ++i) {
        ctxt_y.emplace_back(context_);
        ctxt_y[i].load(he_path_ + "/layernorm_" + module_name + "_" +
                       std::to_string(rank) + "_" + std::to_string(layer_n) +
                       "_" + std::to_string(i) + ".bin");
        ctxt_y[i].to(device);
    }

    if (ctxt_y[0].getLevel() < 6 && ctxt_num == 3) {
        eval_.multImagUnit(ctxt_y[1], ctxt_y[1]);
        eval_.add(ctxt_y[0], ctxt_y[1], ctxt_y[0]);
        btp_.bootstrapExtended(ctxt_y[0], ctxt_y[0], ctxt_y[1]);

        btp_.bootstrapExtended(ctxt_y[2], ctxt_y[2]);
    }

    // inner product
    CtxtTensor prod_sum{context_, tensor_vec[0].getShape()};
    CtxtTensor grad_sum{context_, tensor_vec[0].getShape()};
    for (u64 i = 0; i < tensor_vec.size(); ++i) {
        const u64 level1 = ctxt_y[i].getLevel();     // 6 | 5
        const u64 level2 = tensor_vec[i].getLevel(); // 11 | 11
        Ciphertext prod(context_), tmp(context_);

        if (level1 == level2) {
            eval_.tensor(ctxt_y[i], tensor_vec[i].get(), prod);
        } else if (level1 > level2) {
            eval_.levelDown(ctxt_y[i], level2, tmp);
            eval_.tensor(tmp, tensor_vec[i].get(), prod);
        } else {
            eval_.levelDown(tensor_vec[i].get(), level1, tmp);
            eval_.tensor(ctxt_y[i], tmp, prod);
        }

        if (i == 0) {
            prod_sum.get() = std::move(prod);
            grad_sum.get() = std::move(tensor_vec[i].get());
        } else {
            eval_.add(prod_sum.get(), prod, prod_sum.get());
            eval_.add(grad_sum.get(), tensor_vec[i].get(),
                      grad_sum.get()); // 12
        }
    }

    eval_.relinearize(prod_sum.get(), prod_sum.get());
    eval_.rescale(prod_sum.get()); // 5 | 4

    // TODO: clean up layer-wise level.
    /* if (prod_sum.get().getLevel() == 4) {
        btp_.bootstrapExtended(prod_sum.get(), prod_sum.get());
    } */

    Ciphertext tmp(context_);
    for (i64 rot = 1; rot < block_width; rot <<= 1) {
        eval_.leftRotate(prod_sum.get(), static_cast<u64>(rot), tmp);
        eval_.add(prod_sum.get(), tmp, prod_sum.get());

        eval_.leftRotate(grad_sum.get(), static_cast<u64>(rot), tmp);
        eval_.add(grad_sum.get(), tmp, grad_sum.get());
    }
    eval_.mult(prod_sum.get(), rms_mask_, prod_sum.get()); // 4
    eval_.mult(grad_sum.get(), rms_mask_, grad_sum.get()); // 10
    for (i64 rot = 1; rot < block_width; rot <<= 1) {
        eval_.rightRotate(prod_sum.get(), static_cast<u64>(rot), tmp);
        eval_.add(prod_sum.get(), tmp, prod_sum.get());

        eval_.rightRotate(grad_sum.get(), static_cast<u64>(rot), tmp);
        eval_.add(grad_sum.get(), tmp, grad_sum.get());
    }

    if (prod_sum.get().getLevel() == 4)
        btp_.bootstrapExtended(prod_sum.get(), prod_sum.get());

    // Handling multi-GPU
    /* if (max_rank_ > 1) {
        allReduceWrapper(prod_sum);
        eval_.modReduct(prod_sum.get());
    } */

    // 7
    Ciphertext inv_sqrt(context_);
    inv_sqrt.load(he_path_ + "/layernorm_" + module_name + "_" +
                  std::to_string(rank) + "_" + std::to_string(layer_n) +
                  "_invsqrt.bin");
    inv_sqrt.to(device);

    std::vector<CtxtTensor> ret;
    ret.reserve(ctxt_num);
    for (u64 i = 0; i < ctxt_num; ++i) {
        ret.emplace_back(context_, tensor_vec[i].getShape());
        auto &ctxt_ret = ret[i].get();
        eval_.mult(ctxt_y[i], prod_sum.get(), ctxt_ret); // 5 (prod_sum btp) | 4
        eval_.sub(tensor_vec[i].get(), ctxt_ret, ctxt_ret); // 5
        eval_.sub(ctxt_ret, grad_sum.get(), ctxt_ret);      // 5
        eval_.mult(ctxt_ret, inv_sqrt, ctxt_ret);           // 4 | 3
    }
    return ret;
}

void HEMMer::backwardexpVectorInplace(std::vector<CtxtTensor> &tensor_vec,
                                      const int layer_n) const {

    const auto rank = getRank();
    auto device = tensor_vec[0].get().getDevice();
    Ciphertext grad(context_);
    for (u64 i = 0; i < tensor_vec.size(); ++i) {
        grad.load(he_path_ + "/exp_" + std::to_string(rank) + "_" +
                  std::to_string(layer_n) + "_" + std::to_string(i) + ".bin");
        grad.to(device);
        eval_.mult(tensor_vec[i].get(), grad, tensor_vec[i].get());
    }
}

void HEMMer::backwardsoftmaxVectorInplaceHETAL(std::vector<CtxtTensor> &grad_y,
                                               const int layer_n) const {

    auto rank = getRank();

    const auto device = grad_y[0].get().getDevice();
    const u64 n_ctxt = grad_y.size();

    // generate mask
    auto log_slots = getLogFullSlots(context_);
    HEaaN::Message msg_mask(log_slots, 0.0);
    for (u64 i = 0; i < 128; i++) {
        for (u64 j = 0; j < 2; ++j) {
            msg_mask[i * 256 + i + 128 * j] = 1.0;
        }
    }
    msg_mask.to(device);

    // each vector element has two matrices
    Ciphertext sum(context_), tmp(context_);
    std::vector<Ciphertext> ctxt_y;
    ctxt_y.reserve(n_ctxt);
    for (u64 sb = 0; sb < n_ctxt; ++sb) {
        ctxt_y.emplace_back(context_);
    }

    for (u64 i = 0; i < n_ctxt; i++) {
        Ciphertext tmp2(context_); // (pmj): Because tensor
        ctxt_y[i].load(he_path_ + "/softmax_" + std::to_string(rank) + "_" +
                       std::to_string(layer_n) + "_" + std::to_string(i) +
                       ".bin");
        ctxt_y[i].to(device);

        eval_.sub(ctxt_y[i], msg_mask, tmp);
        eval_.mult(ctxt_y[i], tmp, ctxt_y[i]);

        eval_.mult(grad_y[i].get(), ctxt_y[i], grad_y[i].get());
    }
}

void HEMMer::backwardreluVectorInplace(std::vector<CtxtTensor> &tensor_vec,
                                       const int layer_n) const {

    const auto rank = getRank();
    auto device = tensor_vec[0].get().getDevice();
    Ciphertext grad(context_);
    for (u64 i = 0; i < tensor_vec.size(); ++i) {
        grad.load(he_path_ + "/relu_backward_" + std::to_string(rank) + "_" +
                  std::to_string(layer_n) + "_" + std::to_string(i) + ".bin");
        grad.to(device);
        eval_.mult(tensor_vec[i].get(), grad, tensor_vec[i].get());
    }
}

void HEMMer::backwardtanhVectorInplace(std::vector<CtxtTensor> &tensor_vec,
                                       const int layer_n) const {
    const auto rank = getRank();
    auto device = tensor_vec[0].get().getDevice();
    Ciphertext grad(context_);
    for (u64 i = 0; i < tensor_vec.size(); ++i) {
        grad.load(he_path_ + "/tanh_" + std::to_string(rank) + "_" +
                  std::to_string(layer_n) + "_" + std::to_string(i) +
                  ".bin"); // 9
        grad.to(device);
        eval_.mult(tensor_vec[i].get(), grad, tensor_vec[i].get());
    }
}

void HEMMer::backwardtanhInplace(CtxtTensor &tensor, const int layer_n) const {
    const auto rank = getRank();
    auto device = tensor.get().getDevice();
    Ciphertext grad(context_);
    grad.load(he_path_ + "/tanh_" + std::to_string(rank) + "_" +
              std::to_string(layer_n) + ".bin"); // 9
    grad.to(device);
    eval_.mult(tensor.get(), grad, tensor.get());
}

// lora
CtxtTensor HEMMer::matMulHighLow(const CtxtTensor &tensor_a,
                                 const CtxtTensor &tensor_b, u64 in_col_block,
                                 u64 target_level) const {
    return multMatMatHighLow(eval_, tensor_a, tensor_b, in_col_block,
                             target_level, matrix_transformer);
}

CtxtTensor HEMMer::matMulLowHigh(const CtxtTensor &tensor_a,
                                 const CtxtTensor &tensor_b, u64 in_row_block,
                                 u64 target_level) const {
    return multMatMatLowHigh(eval_, tensor_a, tensor_b, in_row_block,
                             target_level, matrix_transformer);
}

CtxtTensor HEMMer::matMulLowLow(const CtxtTensor &tensor_a,
                                const CtxtTensor &tensor_b, u64 in_col_block,
                                u64 in_row_block, u64 target_level) const {
    return multMatMatLowLow(eval_, tensor_a, tensor_b, in_col_block,
                            in_row_block, target_level, matrix_transformer);
}

// bert version.
CtxtTensor HEMMer::repackToOneCol(const CtxtTensor &tensor,
                                  u64 out_col_block) const {
    const u64 low_dim = ModelArgs::LOW_DIM;
    const u64 num_rows = static_cast<u64>(tensor.getHeight());
    const u64 num_cols = static_cast<u64>(tensor.getWidth());
    const u64 block_width = static_cast<u64>(tensor.getBlockWidth());

    // hard coding
    Ciphertext tmp(context_);
    CtxtTensor ret{tensor};
    u64 col_pos = out_col_block;
    for (u64 i = low_dim; i < block_width; i <<= 1) {
        if (col_pos % 2 == 0) {
            eval_.leftRotate(ret.get(), i, tmp);
        } else {
            eval_.rightRotate(ret.get(), i, tmp);
        }
        eval_.add(ret.get(), tmp, ret.get());
        col_pos /= 2;
    }

    // eval_.leftRotate(ret.get(), 128, tmp);
    // eval_.add(ret.get(), tmp, ret.get());

    // hard coding
    Message col_block_mask(ret.getLogSlots(), 0);
    for (u64 i = 0; i < num_rows; i++) {
        for (u64 j = out_col_block * low_dim; j < (1 + out_col_block) * low_dim;
             j++) {
            col_block_mask[i * block_width + j] = 1;
        }
    }
    col_block_mask.to(ret.get().getDevice());
    eval_.mult(ret.get(), col_block_mask, ret.get());

    // copy to muliple block
    for (u64 i = num_cols; i < block_width; i <<= 1) {
        eval_.rightRotate(ret.get(), num_cols, tmp);
        eval_.add(ret.get(), tmp, ret.get());
    }

    return ret;
}

CtxtTensor HEMMer::repackToMultiCol(const CtxtTensor &tensor,
                                    u64 out_col_block) const {
    const u64 low_dim = ModelArgs::LOW_DIM;
    // const u64 low_dim = 2;
    const u64 num_rows = static_cast<u64>(tensor.getHeight());
    const u64 num_cols = static_cast<u64>(tensor.getWidth());
    const u64 block_width = static_cast<u64>(tensor.getBlockWidth());

    // hard coding
    Ciphertext tmp(context_);
    CtxtTensor ret{tensor};
    u64 col_pos = out_col_block;
    for (u64 i = low_dim; i < block_width / 2; i <<= 1) {
        if (col_pos % 2 == 0) {
            eval_.leftRotate(ret.get(), i, tmp);
        } else {
            eval_.rightRotate(ret.get(), i, tmp);
        }
        eval_.add(ret.get(), tmp, ret.get());
        col_pos /= 2;
    }

    // hard coding
    Message col_block_mask(ret.getLogSlots(), 0);
    for (u64 i = 0; i < num_rows; i++) {
        for (u64 k = 0; k < block_width; k += num_cols) {
            for (u64 j = out_col_block * low_dim;
                 j < (1 + out_col_block) * low_dim; j++) {
                col_block_mask[i * block_width + k + j] = 1;
            }
        }
    }
    col_block_mask.to(ret.get().getDevice());
    eval_.mult(ret.get(), col_block_mask, ret.get());

    return ret;
}

CtxtTensor HEMMer::repackToOneRow(const CtxtTensor &tensor,
                                  u64 out_row_block) const {
    const u64 low_dim = ModelArgs::LOW_DIM;
    const u64 num_rows = static_cast<u64>(tensor.getHeight());
    const u64 num_cols = static_cast<u64>(tensor.getWidth());
    const u64 block_width = static_cast<u64>(tensor.getBlockWidth());

    // hard coding
    Ciphertext tmp(context_);
    CtxtTensor ret{tensor};
    u64 row_pos = out_row_block;
    for (u64 i = low_dim; i < num_rows; i <<= 1) {
        if (row_pos % 2 == 0) {
            eval_.leftRotate(ret.get(), i * block_width, tmp);
        } else {
            eval_.rightRotate(ret.get(), i * block_width, tmp);
        }
        eval_.add(ret.get(), tmp, ret.get());
        row_pos /= 2;
    }

    for (u64 i = num_cols; i < block_width; i <<= 1) {
        eval_.leftRotate(ret.get(), num_cols, tmp);
        eval_.add(ret.get(), tmp, ret.get());
    }

    // hard coding
    Message row_block_mask(ret.getLogSlots(), 0);
    for (u64 i = out_row_block * low_dim; i < (1 + out_row_block) * low_dim;
         ++i) {
        for (u64 j = 0; j < num_cols; j++) {
            row_block_mask[i * block_width + j] = 1;
        }
    }
    row_block_mask.to(ret.get().getDevice());
    eval_.mult(ret.get(), row_block_mask, ret.get());

    // copy to muliple block
    for (u64 i = num_cols; i < block_width; i += num_cols) {
        eval_.rightRotate(ret.get(), num_cols, tmp);
        eval_.add(ret.get(), tmp, ret.get());
    }

    return ret;
}

CtxtTensor HEMMer::repackToMultiRow(const CtxtTensor &tensor,
                                    u64 out_row_block) const {
    const u64 low_dim = ModelArgs::LOW_DIM;
    const u64 num_rows = static_cast<u64>(tensor.getHeight());
    const u64 block_width = static_cast<u64>(tensor.getBlockWidth());

    // hard coding
    Ciphertext tmp(context_);
    CtxtTensor ret{tensor};
    u64 row_pos = out_row_block;
    for (u64 i = low_dim; i < num_rows; i <<= 1) {
        if (row_pos % 2 == 0) {
            eval_.leftRotate(ret.get(), i * block_width, tmp);
        } else {
            eval_.rightRotate(ret.get(), i * block_width, tmp);
        }
        eval_.add(ret.get(), tmp, ret.get());
        row_pos /= 2;
    }

    // hard coding
    Message row_block_mask(ret.getLogSlots(), 0);
    for (u64 i = out_row_block * low_dim; i < (1 + out_row_block) * low_dim;
         ++i) {
        for (u64 j = 0; j < block_width; j++) {
            row_block_mask[i * block_width + j] = 1;
        }
    }
    row_block_mask.to(ret.get().getDevice());
    eval_.mult(ret.get(), row_block_mask, ret.get());

    return ret;
}

CtxtTensor HEMMer::getLowColBlock(const CtxtTensor &tensor,
                                  u64 col_block) const {
    const u64 block_width = static_cast<u64>(tensor.getBlockWidth());
    const u64 num_rows = static_cast<u64>(tensor.getHeight());
    const u64 num_cols = static_cast<u64>(tensor.getWidth());
    Message msg_mask(tensor.getLogSlots(), 0);
    for (u64 i = 0; i < num_rows; i++) {
        for (u64 k = 0; k < block_width; k += num_cols) {
            for (u64 j = col_block * ModelArgs::LOW_DIM;
                 j < (1 + col_block) * ModelArgs::LOW_DIM; j++) {
                msg_mask[i * block_width + k + j] = 1;
            }
        }
    }
    msg_mask.to(tensor.get().getDevice());
    CtxtTensor res{tensor};
    eval_.mult(tensor.get(), msg_mask, res.get());

    return res;
}

CtxtTensor HEMMer::getLowRowBlock(const CtxtTensor &tensor,
                                  u64 row_block) const {
    const u64 block_width = static_cast<u64>(tensor.getBlockWidth());
    const u64 num_cols = static_cast<u64>(tensor.getWidth());
    Message msg_mask(tensor.getLogSlots(), 0);
    for (u64 i = row_block * ModelArgs::LOW_DIM;
         i < (1 + row_block) * ModelArgs::LOW_DIM; i++) {
        for (u64 k = 0; k < block_width; k += num_cols) {
            for (u64 j = 0; j < num_cols; j++) {
                msg_mask[i * block_width + k + j] = 1;
            }
        }
    }
    msg_mask.to(tensor.get().getDevice());
    CtxtTensor res{tensor};
    eval_.mult(tensor.get(), msg_mask, res.get());

    return res;
}

// private

void HEMMer::genMask() {
    auto one = torch::full({ModelArgs::HEAD_DIM, ModelArgs::HEAD_DIM}, 1);
    auto zero = torch::full({ModelArgs::HEAD_DIM, ModelArgs::HEAD_DIM}, 0);
    mask_matmul_ = encode2(one, zero).get();

    auto half = torch::full({ModelArgs::HEAD_DIM, ModelArgs::HEAD_DIM}, 0.5);
    mask_matmul_half_ = encode2(half, zero).get();

    CudaTools::cudaDeviceSynchronize();
}

void HEMMer::generateAxisMasks() {
    const int dim = ModelArgs::HEAD_DIM;
    torch::Tensor even_mask = (1 - torch::arange(0, 2).view({1, 2}));
    even_mask = even_mask.repeat_interleave(dim * dim / 2, 0).view({dim, dim});
    torch::Tensor tr_even_mask = even_mask.transpose(0, 1);

    col_mask_even_ = message2(even_mask, even_mask);
    row_mask_even_ = message2(tr_even_mask, tr_even_mask);

    torch::Tensor rms_mask1 =
        torch::cat({torch::full({dim, 1}, 1.0 / ModelArgs::DIM),
                    torch::zeros({dim, dim - 1})},
                   1);
    torch::Tensor rms_mask2 = torch::zeros({dim, dim});

    rms_mask_ = message2(rms_mask1, rms_mask2);
}

// bert
void HEMMer::generateRightLeftMasks() {
    const int dim = ModelArgs::HEAD_DIM; // 128

    torch::Tensor right_mask = torch::zeros({dim, dim});
    torch::Tensor left_mask = torch::ones({dim, dim});

    right_mask.slice(1, 0, 64).fill_(1);
    left_mask.slice(1, 0, 64).fill_(0);

    mask_one_left_ = message2(right_mask, right_mask);
    mask_one_right_ = message2(left_mask, left_mask);

    torch::Tensor first_col_mask1 = torch::cat(
        {torch::full({dim, 1}, 1.0), torch::zeros({dim, dim - 1})}, 1);
    torch::Tensor first_col_mask2 = torch::zeros({dim, dim});

    // mask_first_col_ = message2(first_col_mask1, first_col_mask2);
    mask_first_col_ = message2(first_col_mask1, first_col_mask1);

    torch::Tensor first_row_mask = torch::cat(
        {torch::full({1, dim}, 1.0), torch::zeros({dim - 1, dim})}, 0);

    mask_first_row_ = message2(first_row_mask, first_row_mask);

    torch::Tensor ones = torch::ones({dim, dim});
    torch::Tensor zeros = torch::zeros({dim, dim});

    mask_left_half_ = message2(ones, zeros);
    mask_right_half_ = message2(zeros, ones);

    torch::Tensor upper_mask = right_mask.transpose(0, 1);
    torch::Tensor lower_mask = left_mask.transpose(0, 1);

    mask_upper_half_ = message2(upper_mask, upper_mask);
    mask_lower_half_ = message2(lower_mask, lower_mask);

    torch::Tensor first_ele = torch::zeros({dim, dim});
    first_ele[0][0] = 1.0;
    first_ele[0][1] = 1.0;

    mask_first_ele_ = message2(first_ele, zeros);

    torch::Tensor first_ele2 = torch::zeros({dim, dim});
    first_ele2[0][0] = 1.0;
    first_ele2[1][0] = 1.0;
    mask_first_ele2_ = message2(first_ele2, zeros);

    torch::Tensor first_ele2_sst2 = torch::zeros({dim, dim});
    first_ele2_sst2[0][0] = 1.0 / (6 * 2);
    first_ele2_sst2[1][0] = 1.0 / (6 * 2);
    mask_first_ele2_sst2_ = message2(first_ele2_sst2, zeros);

    mask_first_col_only_ = message2(first_col_mask1, zeros);

    torch::Tensor first_ele_only = torch::zeros({dim, dim});
    first_ele_only[0][0] = 0.1;
    mask_first_ele_only_ = message2(first_ele_only, zeros);

    torch::Tensor first_ele_1_only = torch::zeros({dim, dim});
    first_ele_1_only[0][0] = 1.0;
    mask_first_ele_1_only_ = message2(first_ele_1_only, zeros);

    torch::Tensor first_ele_only_sst2 = torch::zeros({dim, dim});
    first_ele_only_sst2[0][0] = 12;
    mask_first_ele_only_sst2_ = message2(first_ele_only_sst2, zeros);

    torch::Tensor first_col_pool = torch::zeros({dim, dim});
    for (int i = 0; i < ModelArgs::LOW_DIM_HEAD; ++i) {
        first_col_pool[i][0] = 1.0;
    }
    mask_first_col_pool_ = message2(first_col_pool, zeros);

    torch::Tensor first_row_pool = torch::zeros({dim, dim});
    for (int i = 0; i < ModelArgs::LOW_DIM; ++i) {
        first_row_pool[0][i] = 1.0;
    }
    mask_first_row_pool_ = message2(first_row_pool, zeros);

    torch::Tensor first_rows_pool = torch::zeros({dim, dim});
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < dim; ++j) {
            first_rows_pool[i * ModelArgs::LOW_DIM_HEAD][j] = 1.0;
        }
    }

    mask_first_rows_pool_ = message2(first_rows_pool, first_rows_pool);
}

void HEMMer::generateDropoutMasks() {
    torch::Tensor dropout_mask = torch::empty({128, 768});

    for (int row = 0; row < dropout_mask.size(0); ++row) {
        for (int col = 0; col < dropout_mask.size(1); ++col) {
            if (torch::rand({}).item<double>() < 0.1) {
                dropout_mask[row][col] = 0.0;
            } else {
                dropout_mask[row][col] = 10.0 / 9.0;
            }
        }
    }

    mask_drop_out_.clear();
    mask_drop_out_.reserve(3);

    for (u64 i = 0; i < 3; i++) {
        torch::Tensor tmp1 =
            dropout_mask.slice(1, (2 * i) * 128, (2 * i + 1) * 128);
        torch::Tensor tmp2 =
            dropout_mask.slice(1, (2 * i + 1) * 128, (2 * i + 2) * 128);

        mask_drop_out_.push_back(message2(tmp1, tmp2));
    }

    torch::Tensor dropout_mask_exp = torch::empty({128, 128});

    for (int row = 0; row < dropout_mask_exp.size(0); ++row) {
        for (int col = 0; col < dropout_mask_exp.size(1); ++col) {
            if (torch::rand({}).item<double>() < 0.1) {
                dropout_mask_exp[row][col] = 0.0;
            } else {
                dropout_mask_exp[row][col] = 10.0 / 9.0;
            }
        }
    }

    mask_drop_out_exp_.clear();
    mask_drop_out_exp_.reserve(6);

    for (u64 i = 0; i < 6; ++i)
        mask_drop_out_exp_.push_back(
            message2(dropout_mask_exp, dropout_mask_exp));
}

void HEMMer::squareSum(const std::vector<CtxtTensor> &tensor_vec,
                       CtxtTensor &tensor) const {
    Ciphertext tmp(context_);

    eval_.tensor(tensor_vec[0].get(), tensor_vec[0].get(), tensor.get());
    for (u64 i = 1; i < tensor_vec.size(); ++i) {
        eval_.tensor(tensor_vec[i].get(), tensor_vec[i].get(), tmp);
        eval_.add(tensor.get(), tmp, tensor.get());
    }
    eval_.relinearize(tensor.get(), tensor.get());
    eval_.rescale(tensor.get());
}

void HEMMer::reduceRowSumWithScaledMask(CtxtTensor &tensor,
                                        const HEaaN::Message &mask) const {
    const u64 block_width = static_cast<u64>(tensor.getBlockWidth());

    Ciphertext tmp(context_);
    for (u64 i = 1; i < block_width; i <<= 1) {
        eval_.leftRotate(tensor.get(), i, tmp);
        eval_.add(tensor.get(), tmp, tensor.get());
    }
    eval_.mult(tensor.get(), mask, tensor.get());
    for (u64 i = 1; i < block_width; i <<= 1) {
        eval_.rightRotate(tensor.get(), i, tmp);
        eval_.add(tensor.get(), tmp, tensor.get());
    }
    if (max_rank_ > 1) {
        allReduceWrapper(tensor);
        eval_.modReduct(tensor.get());
    }
}

// bert
void HEMMer::inverseSqrtLN(CtxtTensor &tensor, const std::string &module_name,
                           const int layer_n) const {
    if (module_name == "atn") {
        // Caution: should know the uppder bound of whole inputs.
        // We also can make case-wise LN evaluation by adjusting upper bound.
        if (layer_n == 0) {
            eval_.mult(tensor.get(), 1.0 / 300, tensor.get());
            LayerNorm::approxInverseSqrtLN(eval_, btp_, tensor.get(),
                                           tensor.get(), layer_n, 3);
            eval_.mult(tensor.get(), 1.0 / std::sqrt(300), tensor.get());
        } else if (layer_n == 1) {
            eval_.mult(tensor.get(), 1.0 / 300, tensor.get());
            LayerNorm::approxInverseSqrtLN(eval_, btp_, tensor.get(),
                                           tensor.get(), layer_n, 3);
            eval_.mult(tensor.get(), 1.0 / std::sqrt(300), tensor.get());
        } else {
            eval_.mult(tensor.get(), 1.0 / 280, tensor.get());
            LayerNorm::approxInverseSqrtLN(eval_, btp_, tensor.get(),
                                           tensor.get(), layer_n, 3);
            eval_.mult(tensor.get(), 1.0 / std::sqrt(280), tensor.get());
        }
    } else if (module_name == "ffn") {
        // Caution: should know the uppder bound of whole inputs.
        // We also can make case-wise LN evaluation by adjusting upper bound.
        if (layer_n == 0) {
            eval_.mult(tensor.get(), 1.0 / 300, tensor.get());
            LayerNorm::approxInverseSqrtLN(eval_, btp_, tensor.get(),
                                           tensor.get(), layer_n, 3);
            eval_.mult(tensor.get(), 1.0 / std::sqrt(300), tensor.get());
        } else {
            eval_.mult(tensor.get(), 1.0 / 300, tensor.get());
            LayerNorm::approxInverseSqrtLN(eval_, btp_, tensor.get(),
                                           tensor.get(), layer_n, 3);
            eval_.mult(tensor.get(), 1.0 / std::sqrt(300), tensor.get());
        }
    } else {
        eval_.mult(tensor.get(), 1.0 / 300, tensor.get());
        LayerNorm::approxInverseSqrtLN(eval_, btp_, tensor.get(), tensor.get(),
                                       layer_n, 3);
        eval_.mult(tensor.get(), 1.0 / std::sqrt(300), tensor.get());
    }
}

// bert
void HEMMer::inverseSqrtLNFT(CtxtTensor &tensor, const std::string &module_name,
                             const int layer_n) const {
    if (module_name == "atn") {
        // Caution: should know the uppder bound of whole inputs.
        // We also can make case-wise LN evaluation by adjusting upper bound.
        if (layer_n == 0) {
            eval_.mult(tensor.get(), 1.0 / 20, tensor.get());
            LayerNorm::approxInverseSqrtLN(eval_, btp_, tensor.get(),
                                           tensor.get(), layer_n, 3);
            eval_.mult(tensor.get(), 1.0 / std::sqrt(20), tensor.get());
        } else if (layer_n == 1) {
            eval_.mult(tensor.get(), 1.0 / 300, tensor.get());
            LayerNorm::approxInverseSqrtLN(eval_, btp_, tensor.get(),
                                           tensor.get(), layer_n, 3);
            eval_.mult(tensor.get(), 1.0 / std::sqrt(300), tensor.get());
        } else {
            eval_.mult(tensor.get(), 1.0 / 280, tensor.get());
            LayerNorm::approxInverseSqrtLN(eval_, btp_, tensor.get(),
                                           tensor.get(), layer_n, 3);
            eval_.mult(tensor.get(), 1.0 / std::sqrt(280), tensor.get());
        }
    } else if (module_name == "ffn") {
        // Caution: should know the uppder bound of whole inputs.
        // We also can make case-wise LN evaluation by adjusting upper bound.
        if (layer_n == 0) {
            eval_.mult(tensor.get(), 1.0 / 150, tensor.get());
            LayerNorm::approxInverseSqrtLN(eval_, btp_, tensor.get(),
                                           tensor.get(), layer_n, 3);
            eval_.mult(tensor.get(), 1.0 / std::sqrt(150), tensor.get());
        } else {
            eval_.mult(tensor.get(), 1.0 / 300, tensor.get());
            LayerNorm::approxInverseSqrtLN(eval_, btp_, tensor.get(),
                                           tensor.get(), layer_n, 3);
            eval_.mult(tensor.get(), 1.0 / std::sqrt(300), tensor.get());
        }
    } else {
        eval_.mult(tensor.get(), 1.0 / 350, tensor.get());
        LayerNorm::approxInverseSqrtLN(eval_, btp_, tensor.get(), tensor.get(),
                                       layer_n, 3);
        eval_.mult(tensor.get(), 1.0 / std::sqrt(350), tensor.get());
    }
}

#ifdef HELLM_MULTIGPU
void HEMMer::allReduceWrapper(CtxtTensor &ctxt_tensor) const {
    ncclAllReduceWrapper(ctxt_tensor);
}

void HEMMer::ncclAllReduceWrapper(const CtxtTensor &ctxt_tensor) const {
    const auto degree = getDegree(context_);
    NCCLCHECK(ncclGroupStart());
    for (u64 i = 0; i < 2; ++i) {
        NCCLCHECK(ncclAllReduce(ctxt_tensor.get().getPolyData(i, 0),
                                ctxt_tensor.get().getPolyData(i, 0),
                                (ctxt_tensor.getLevel() + 1) * degree,
                                ncclUint64, ncclSum, comm_, nullptr));
    }
    NCCLCHECK(ncclGroupEnd());
}
#endif
} // namespace HELLM
