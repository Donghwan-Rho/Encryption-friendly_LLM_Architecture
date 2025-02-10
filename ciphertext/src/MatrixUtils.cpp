////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2024 Crypto Lab Inc. All rights reserved.               //
//                                                                            //
// This source code is protected under international copyright law.           //
// All rights reserved and protected by the copyright holders.                //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "HELLM/MatrixUtils.hpp"
#include "HELLM/HETensor.hpp"
#include "HELLM/ModelArgs.hpp"
#include "HEaaN/Ciphertext.hpp"

#include "HEaaN/HEaaN.hpp"

#include <fstream>
#include <iostream>

namespace {
using namespace HEaaN;
using namespace HELLM;

template <class T1, class T2>
i64 getCommonInsideLength(const HETensor<T1> &op1, const HETensor<T2> &op2) {
    if (op1.getWidth() != op2.getHeight())
        throw std::invalid_argument(
            "[getCommonInsideLength] different inside length.");

    return op1.getWidth();
}

template <class T1, class T2>
i64 getCommonBlockWidth(const HETensor<T1> &op1, const HETensor<T2> &op2) {
    if (op1.getBlockWidth() != op2.getBlockWidth())
        throw std::invalid_argument(
            "[getCommonBlockWidth] different block width");

    return op1.getBlockWidth();
}

template i64 getCommonBlockWidth(const CtxtTensor &op1, const CtxtTensor &op2);
template i64 getCommonBlockWidth(const CtxtTensor &op1, const PtxtTensor &op2);

template <class T>
[[maybe_unused]] void copyInplace(const HomEvaluator &eval, HETensor<T> &op,
                                  const u64 idx_to, const u64 axis) {
    const u64 idx_from =
        static_cast<u64>((axis == 0) ? op.getHeight() : op.getWidth());
    const u64 base = static_cast<u64>((axis == 0) ? op.getBlockWidth() : 1);

    const u64 log_idx_start =
        static_cast<u64>(std::ceil(std::log2(idx_from * base)));
    const u64 log_idx_end =
        static_cast<u64>(std::ceil(std::log2(idx_to * base)));

    T tmp(eval.getContext());
    for (u64 i = log_idx_start; i < log_idx_end; i++) {
        eval.rightRotate(op.get(), 1UL << i, tmp);
        eval.add(op.get(), tmp, op.get());
    }
}

#if defined(__clang__)
template void copyInplace(const HomEvaluator &eval, CtxtTensor &op,
                          const u64 idx_to, const u64 axis);
template void copyInplace(const HomEvaluator &eval, PtxtTensor &op,
                          const u64 idx_to, const u64 axis);
#elif defined(__GNUC__)
template [[maybe_unused]] void copyInplace(const HomEvaluator &eval,
                                           CtxtTensor &op, const u64 idx_to,
                                           const u64 axis);
template [[maybe_unused]] void copyInplace(const HomEvaluator &eval,
                                           PtxtTensor &op, const u64 idx_to,
                                           const u64 axis);
#endif

} // namespace

namespace HELLM {

using namespace HEaaN;

constexpr u64 LOG_SLOTS = 15;

/////////////////////////////
/////// TransformMask ///////
/////////////////////////////

void TransformMask::generateDiagoanlMask(const HomEvaluator &eval,
                                         const TileTensorShape matrix_shape) {
    const i64 num_rows = matrix_shape.getNumRows();
    const i64 num_cols = matrix_shape.getNumCols();
    const i64 width = matrix_shape.getBlockWidth();

    const u64 num_mask = static_cast<u64>(num_rows + num_cols - 1);

    // bs * (gs - 1) < num_mask <= bs * gs
    const i64 bs = static_cast<i64>(std::ceil(std::sqrt(num_mask)));
    const i64 gs = static_cast<i64>(
        std::ceil(static_cast<Real>(num_mask) / static_cast<Real>(bs)));

    // EnDecoder endec(eval.getContext());
    for (i64 i = 0; i < gs; i++) {
        const i64 rot_num = i * bs * (width - 1);

        for (i64 j = 0; j < bs; j++) {
            const i64 idx = 1 - num_rows + i * bs + j;
            if (idx >= num_cols)
                break;

            Message msg(LOG_SLOTS, 0);
            const i64 start_row = (idx < 0) ? -idx : 0;
            const i64 start_col = (idx < 0) ? 0 : idx;
            for (i64 row = start_row, col = start_col;
                 row < num_rows && col < num_cols; row++, col++) {
                msg[static_cast<u64>(row * width + col)] = 1;
                if (width != num_cols)
                    msg[static_cast<u64>(row * width + num_rows + col)] = 1;
            }
            // Prerotation for baby-step giant-step
            eval.rightRotate(msg, static_cast<u64>(rot_num), msg);
            msg.to(getCurrentCudaDevice());
            diagonal_mask_.emplace(idx, msg);
        }
    }
}

void TransformMask::generateHorizontalMask(const HomEvaluator &eval,
                                           const TileTensorShape matrix_shape) {
    const i64 num_rows = matrix_shape.getNumRows();
    const i64 num_cols = matrix_shape.getNumCols();
    const i64 width = matrix_shape.getBlockWidth();
    const i64 height = (1 << LOG_SLOTS) / width;

    const i64 sub_dim = (num_rows > num_cols) ? num_cols : num_rows;

    // bs * (gs - 1) < sub_dim + 1 <= bs * gs
    const i64 bs = static_cast<i64>(std::ceil(std::sqrt(2 * sub_dim - 1)));
    const i64 gs = static_cast<i64>(
        std::ceil(static_cast<Real>(sub_dim - 1) / static_cast<Real>(bs)));

    // EnDecoder endec(eval.getContext());
    for (i64 i = 0; i < gs; i++) {
        if (i == 0) {
            Message msg(LOG_SLOTS, 0);
            for (i64 row = 0; row < num_rows; row += sub_dim) {
                for (i64 col = 0; col < num_cols; col++) {
                    // msg[static_cast<u64>(row * width + col)] = 1;
                    // multiple block
                    for (i64 row_block = 0; row_block < height;
                         row_block += num_rows) {
                        for (i64 col_block = 0; col_block < width;
                             col_block += num_cols) {
                            msg[static_cast<u64>((row_block + row) * width +
                                                 col_block + col)] = 1;
                        }
                    }
                }
            }
            msg.to(getCurrentCudaDevice());
            horizontal_mask_.emplace(0, msg);
        }

        const i64 rot_num = i * bs;

        for (i64 j = 0; j < bs; j++) {
            const i64 idx = 1 + i * bs + j - num_cols;
            if (idx >= sub_dim - num_cols)
                break;

            Message msg(LOG_SLOTS, 0);
            for (i64 row = num_cols + idx; row < num_rows; row += sub_dim) {
                for (i64 col = -idx; col < num_cols; col++) {
                    // msg[static_cast<u64>(row * width + col)] = 1;
                    // multiple block
                    for (i64 row_block = 0; row_block < height;
                         row_block += num_rows) {
                        for (i64 col_block = 0; col_block < width;
                             col_block += num_cols) {
                            msg[static_cast<u64>((row_block + row) * width +
                                                 col_block + col)] = 1;
                        }
                    }
                }
            }
            eval.rightRotate(msg, static_cast<u64>(rot_num - num_cols), msg);
            msg.to(getCurrentCudaDevice());
            horizontal_mask_.emplace(idx, msg);
        }

        for (i64 j = 0; j < bs; j++) {
            const i64 idx = 1 + i * bs + j;
            if (idx >= sub_dim)
                break;

            Message msg(LOG_SLOTS, 0);
            for (i64 row = idx; row < num_rows; row += sub_dim) {
                for (i64 col = 0; col < num_cols - idx; col++) {
                    // msg[static_cast<u64>(row * width + col)] = 1;
                    // multiple block
                    for (i64 row_block = 0; row_block < height;
                         row_block += num_rows) {
                        for (i64 col_block = 0; col_block < width;
                             col_block += num_cols) {
                            msg[static_cast<u64>((row_block + row) * width +
                                                 col_block + col)] = 1;
                        }
                    }
                }
            }
            eval.rightRotate(msg, static_cast<u64>(rot_num), msg);
            msg.to(getCurrentCudaDevice());
            horizontal_mask_.emplace(idx, msg);
        }
    }
}

void TransformMask::generatePackedHorizontalMask(
    const HomEvaluator &eval, const TileTensorShape matrix_shape) {
    const i64 num_rows = matrix_shape.getNumRows() / 2;
    const i64 num_cols = matrix_shape.getNumCols() / 2;
    const i64 width = matrix_shape.getBlockWidth();

    const i64 sub_dim = (num_rows > num_cols) ? num_cols : num_rows;

    // bs * (gs - 1) < sub_dim + 1 <= bs * gs
    const i64 bs = static_cast<i64>(std::ceil(std::sqrt(2 * sub_dim - 1)));
    const i64 gs = static_cast<i64>(
        std::ceil(static_cast<Real>(sub_dim - 1) / static_cast<Real>(bs)));

    // EnDecoder endec(eval.getContext());
    for (i64 i = 0; i < gs; i++) {
        if (i == 0) {
            Message msg(LOG_SLOTS, 0);
            for (i64 row = 0; row < num_rows; row += sub_dim)
                for (i64 col = 0; col < num_cols; col++) {
                    msg[static_cast<u64>((2 * row) * width + (2 * col))] = 0.5;
                    msg[static_cast<u64>((2 * row + 1) * width + (2 * col))] =
                        0.5;
                    msg[static_cast<u64>((2 * row) * width + (2 * col + 1))] =
                        0.5;
                    msg[static_cast<u64>((2 * row + 1) * width +
                                         (2 * col + 1))] = 0.5;
                    if (width != num_cols) {
                        msg[static_cast<u64>((2 * row) * width + 2 * num_cols +
                                             (2 * col))] = 0.5;
                        msg[static_cast<u64>((2 * row + 1) * width +
                                             2 * num_cols + (2 * col))] = 0.5;
                        msg[static_cast<u64>((2 * row) * width + 2 * num_cols +
                                             (2 * col + 1))] = 0.5;
                        msg[static_cast<u64>((2 * row + 1) * width +
                                             2 * num_cols + (2 * col + 1))] =
                            0.5;
                    }
                }
            msg.to(getCurrentCudaDevice());
            packed_horizontal_mask_.emplace(0, msg);
        }

        const i64 rot_num = 2 * i * bs;

        for (i64 j = 0; j < bs; j++) {
            const i64 idx = 1 + i * bs + j - num_cols;
            if (idx >= sub_dim - num_cols)
                break;

            Message msg(LOG_SLOTS, 0);
            for (i64 row = num_cols + idx; row < num_rows; row += sub_dim)
                for (i64 col = -idx; col < num_cols; col++) {
                    msg[static_cast<u64>((2 * row) * width + (2 * col))] = 0.5;
                    msg[static_cast<u64>((2 * row + 1) * width + (2 * col))] =
                        0.5;
                    msg[static_cast<u64>((2 * row) * width + (2 * col + 1))] =
                        0.5;
                    msg[static_cast<u64>((2 * row + 1) * width +
                                         (2 * col + 1))] = 0.5;
                    if (width != num_cols) {
                        msg[static_cast<u64>((2 * row) * width + 2 * num_cols +
                                             (2 * col))] = 0.5;
                        msg[static_cast<u64>((2 * row + 1) * width +
                                             2 * num_cols + (2 * col))] = 0.5;
                        msg[static_cast<u64>((2 * row) * width + 2 * num_cols +
                                             (2 * col + 1))] = 0.5;
                        msg[static_cast<u64>((2 * row + 1) * width +
                                             2 * num_cols + (2 * col + 1))] =
                            0.5;
                    }
                }
            eval.rightRotate(msg, static_cast<u64>(rot_num - 2 * num_cols),
                             msg);
            msg.to(getCurrentCudaDevice());
            packed_horizontal_mask_.emplace(idx, msg);
        }

        for (i64 j = 0; j < bs; j++) {
            const i64 idx = 1 + i * bs + j;
            if (idx >= sub_dim)
                break;

            Message msg(LOG_SLOTS, 0);
            for (i64 row = idx; row < num_rows; row += sub_dim)
                for (i64 col = 0; col < num_cols - idx; col++) {
                    msg[static_cast<u64>((2 * row) * width + (2 * col))] = 0.5;
                    msg[static_cast<u64>((2 * row + 1) * width + (2 * col))] =
                        0.5;
                    msg[static_cast<u64>((2 * row) * width + (2 * col + 1))] =
                        0.5;
                    msg[static_cast<u64>((2 * row + 1) * width +
                                         (2 * col + 1))] = 0.5;
                    if (width != num_cols) {
                        msg[static_cast<u64>((2 * row) * width + 2 * num_cols +
                                             (2 * col))] = 0.5;
                        msg[static_cast<u64>((2 * row + 1) * width +
                                             2 * num_cols + (2 * col))] = 0.5;
                        msg[static_cast<u64>((2 * row) * width + 2 * num_cols +
                                             (2 * col + 1))] = 0.5;
                        msg[static_cast<u64>((2 * row + 1) * width +
                                             2 * num_cols + (2 * col + 1))] =
                            0.5;
                    }
                }
            eval.rightRotate(msg, static_cast<u64>(rot_num), msg);
            msg.to(getCurrentCudaDevice());
            packed_horizontal_mask_.emplace(idx, msg);
        }
    }
}

void TransformMask::generateVerticalMask(const HomEvaluator &eval,
                                         const TileTensorShape matrix_shape) {
    const i64 num_rows = matrix_shape.getNumRows();
    const i64 num_cols = matrix_shape.getNumCols();
    const i64 width = matrix_shape.getBlockWidth();
    const i64 height = (1 << LOG_SLOTS) / width;

    const i64 sub_dim = (num_cols > num_rows) ? num_rows : num_cols;

    // bs * (gs - 1) < sub_dim + 1 <= bs * gs
    const i64 bs = static_cast<i64>(std::ceil(std::sqrt(2 * sub_dim - 1)));
    const i64 gs = static_cast<i64>(
        std::ceil(static_cast<Real>(sub_dim - 1) / static_cast<Real>(bs)));

    // EnDecoder endec(eval.getContext());
    for (i64 i = 0; i < gs; i++) {
        if (i == 0) {
            Message msg(LOG_SLOTS, 0);
            for (i64 col = 0; col < num_cols; col += sub_dim) {
                for (i64 row = 0; row < num_rows; row++) {
                    // msg[static_cast<u64>(row * width + col)] = 1;
                    // multiple block
                    for (i64 col_block = 0; col_block < width;
                         col_block += num_cols) {
                        for (i64 row_block = 0; row_block < height;
                             row_block += num_rows) {
                            msg[static_cast<u64>((row_block + row) * width +
                                                 col_block + col)] = 1;
                        }
                    }
                }
            }
            msg.to(getCurrentCudaDevice());
            vertical_mask_.emplace(0, msg);
        }

        const i64 rot_num = i * bs * width;

        for (i64 j = 0; j < bs; j++) {
            const i64 idx = 1 + i * bs + j - num_rows;
            if (idx >= sub_dim - num_rows)
                break;

            Message msg(LOG_SLOTS, 0);
            for (i64 col = num_rows + idx; col < num_cols; col += sub_dim) {
                for (i64 row = -idx; row < num_rows; row++) {
                    // msg[static_cast<u64>(row * width + col)] = 1;
                    // multiple block
                    for (i64 col_block = 0; col_block < width;
                         col_block += num_cols) {
                        for (i64 row_block = 0; row_block < height;
                             row_block += num_rows) {
                            msg[static_cast<u64>((row_block + row) * width +
                                                 col_block + col)] = 1;
                        }
                    }
                }
            }
            eval.leftRotate(msg, static_cast<u64>(num_rows * width - rot_num),
                            msg);
            msg.to(getCurrentCudaDevice());
            vertical_mask_.emplace(idx, msg);
        }

        for (i64 j = 0; j < bs; j++) {
            const i64 idx = 1 + i * bs + j;
            if (idx >= sub_dim)
                break;

            Message msg(LOG_SLOTS, 0);
            for (i64 col = idx; col < num_cols; col += sub_dim) {
                for (i64 row = 0; row < num_rows - idx; row++) {
                    // msg[static_cast<u64>(row * width + col)] = 1;
                    // multiple block
                    for (i64 col_block = 0; col_block < width;
                         col_block += num_cols) {
                        for (i64 row_block = 0; row_block < height;
                             row_block += num_rows) {
                            msg[static_cast<u64>((row_block + row) * width +
                                                 col_block + col)] = 1;
                        }
                    }
                }
            }
            eval.rightRotate(msg, static_cast<u64>(rot_num), msg);
            msg.to(getCurrentCudaDevice());
            vertical_mask_.emplace(idx, msg);
        }
    }
}

void TransformMask::generatePackedVerticalMask(
    const HomEvaluator &eval, const TileTensorShape matrix_shape) {
    const i64 num_rows = matrix_shape.getNumRows() / 2;
    const i64 num_cols = matrix_shape.getNumCols() / 2;
    const i64 width = matrix_shape.getBlockWidth();

    const i64 sub_dim = (num_cols > num_rows) ? num_rows : num_cols;

    // bs * (gs - 1) < sub_dim + 1 <= bs * gs
    const i64 bs = static_cast<i64>(std::ceil(std::sqrt(2 * sub_dim - 1)));
    const i64 gs = static_cast<i64>(
        std::ceil(static_cast<Real>(sub_dim - 1) / static_cast<Real>(bs)));

    // EnDecoder endec(eval.getContext());
    for (i64 i = 0; i < gs; i++) {
        if (i == 0) {
            Message msg(LOG_SLOTS, 0);
            for (i64 col = 0; col < num_cols; col += sub_dim)
                for (i64 row = 0; row < num_rows; row++) {
                    msg[static_cast<u64>((2 * row) * width + (2 * col))] = 1;
                    msg[static_cast<u64>((2 * row + 1) * width + (2 * col))] =
                        1;
                    msg[static_cast<u64>((2 * row) * width + (2 * col + 1))] =
                        1;
                    msg[static_cast<u64>((2 * row + 1) * width +
                                         (2 * col + 1))] = 1;
                    if (width != num_cols) {
                        msg[static_cast<u64>((2 * row) * width + 2 * num_cols +
                                             (2 * col))] = 1;
                        msg[static_cast<u64>((2 * row + 1) * width +
                                             2 * num_cols + (2 * col))] = 1;
                        msg[static_cast<u64>((2 * row) * width + 2 * num_cols +
                                             (2 * col + 1))] = 1;
                        msg[static_cast<u64>((2 * row + 1) * width +
                                             2 * num_cols + (2 * col + 1))] = 1;
                    }
                }
            msg.to(getCurrentCudaDevice());
            packed_vertical_mask_.emplace(0, msg);
        }

        const i64 rot_num = 2 * i * bs * width;

        for (i64 j = 0; j < bs; j++) {
            const i64 idx = 1 + i * bs + j - num_rows;
            if (idx >= sub_dim - num_rows)
                break;

            Message msg(LOG_SLOTS, 0);
            for (i64 col = num_rows + idx; col < num_cols; col += sub_dim)
                for (i64 row = -idx; row < num_rows; row++) {
                    msg[static_cast<u64>((2 * row) * width + (2 * col))] = 1;
                    msg[static_cast<u64>((2 * row + 1) * width + (2 * col))] =
                        1;
                    msg[static_cast<u64>((2 * row) * width + (2 * col + 1))] =
                        1;
                    msg[static_cast<u64>((2 * row + 1) * width +
                                         (2 * col + 1))] = 1;
                    if (width != num_cols) {
                        msg[static_cast<u64>((2 * row) * width + 2 * num_cols +
                                             (2 * col))] = 1;
                        msg[static_cast<u64>((2 * row + 1) * width +
                                             2 * num_cols + (2 * col))] = 1;
                        msg[static_cast<u64>((2 * row) * width + 2 * num_cols +
                                             (2 * col + 1))] = 1;
                        msg[static_cast<u64>((2 * row + 1) * width +
                                             2 * num_cols + (2 * col + 1))] = 1;
                    }
                }
            eval.leftRotate(
                msg, static_cast<u64>(2 * num_rows * width - rot_num), msg);
            msg.to(getCurrentCudaDevice());
            packed_vertical_mask_.emplace(idx, msg);
        }

        for (i64 j = 0; j < bs; j++) {
            const i64 idx = 1 + i * bs + j;
            if (idx >= sub_dim)
                break;

            Message msg(LOG_SLOTS, 0);
            for (i64 col = idx; col < num_cols; col += sub_dim)
                for (i64 row = 0; row < num_rows - idx; row++) {
                    msg[static_cast<u64>((2 * row) * width + (2 * col))] = 1;
                    msg[static_cast<u64>((2 * row + 1) * width + (2 * col))] =
                        1;
                    msg[static_cast<u64>((2 * row) * width + (2 * col + 1))] =
                        1;
                    msg[static_cast<u64>((2 * row + 1) * width +
                                         (2 * col + 1))] = 1;
                    if (width != num_cols) {
                        msg[static_cast<u64>((2 * row) * width + 2 * num_cols +
                                             (2 * col))] = 1;
                        msg[static_cast<u64>((2 * row + 1) * width +
                                             2 * num_cols + (2 * col))] = 1;
                        msg[static_cast<u64>((2 * row) * width + 2 * num_cols +
                                             (2 * col + 1))] = 1;
                        msg[static_cast<u64>((2 * row + 1) * width +
                                             2 * num_cols + (2 * col + 1))] = 1;
                    }
                }
            eval.rightRotate(msg, static_cast<u64>(rot_num), msg);
            msg.to(getCurrentCudaDevice());
            packed_vertical_mask_.emplace(idx, msg);
        }
    }
}

void TransformMask::generateRowMask(const HomEvaluator &eval,
                                    const TileTensorShape matrix_shape,
                                    u64 target_level) {
    const i64 num_rows = matrix_shape.getNumRows();
    const i64 width = matrix_shape.getBlockWidth();
    EnDecoder endec(eval.getContext());
    // bs * (gs - 1) < num_mask <= bs * gs
    for (int i = 0; i < num_rows; i++) {
        Message msg(getLogFullSlots(eval.getContext()), 1);
        Message msg_rev(getLogFullSlots(eval.getContext()), 0);
        for (int j = 0; j < int(msg.getSize()); j++) {
            if ((j / width) % num_rows < num_rows - i) {
                msg[u64(j)] = 0;
                msg_rev[u64(j)] = 1;
            }
        }
        msg.to(getCurrentCudaDevice());
        msg_rev.to(getCurrentCudaDevice());
        auto ptxt = endec.encode(msg, target_level);
        row_mask_.emplace_back(ptxt);
        ptxt = endec.encode(msg_rev, target_level);
        row_mask_.emplace_back(ptxt);
    }
}

void TransformMask::generateColMask(const HomEvaluator &eval,
                                    const TileTensorShape matrix_shape,
                                    u64 target_level) {
    // const i64 num_rows = matrix_shape.getNumRows();
    const i64 num_cols = matrix_shape.getNumCols();
    EnDecoder endec(eval.getContext());
    // bs * (gs - 1) < num_mask <= bs * gs
    for (int i = 0; i < num_cols; i++) {
        Message msg(getLogFullSlots(eval.getContext()), 1);
        Message msg_rev(getLogFullSlots(eval.getContext()), 0);
        for (int j = 0; j < int(msg.getSize()); j++) {
            if (j % num_cols < num_cols - i) {
                msg[u64(j)] = 0;
                msg_rev[u64(j)] = 1;
            }
        }
        msg.to(getCurrentCudaDevice());
        msg_rev.to(getCurrentCudaDevice());

        auto ptxt = endec.encode(msg, target_level);
        col_mask_.emplace_back(ptxt);
        ptxt = endec.encode(msg_rev, target_level);
        col_mask_.emplace_back(ptxt);
    }
}

/////////////////////////////
///// MatrixTransformer /////
/////////////////////////////

TransformMask &
MatrixTransformer::getTransformMask(u64 targe_level,
                                    const TileTensorShape &matrix_shape) {
    std::pair<int, TileTensorShape> search(targe_level, matrix_shape);
    auto iterator = masks_.find(search);
    if (iterator == masks_.end()) {
        TransformMask transform_mask;
        masks_.emplace(search, transform_mask);
    }
    return masks_[search];
}

std::map<i64, Message> &
MatrixTransformer::getDiagonalMask(const HomEvaluator &eval,
                                   const TileTensorShape &matrix_shape) {
    TransformMask &transform_mask{getTransformMask(0, matrix_shape)};
    std::map<i64, Message> &diagonal_mask{transform_mask.getDiagoanlMask()};
    if (diagonal_mask.empty())
        transform_mask.generateDiagoanlMask(eval, matrix_shape);
    return diagonal_mask;
}

std::map<i64, Message> &
MatrixTransformer::getHorizontalMask(const HomEvaluator &eval,
                                     const TileTensorShape &matrix_shape) {
    TransformMask &transform_mask{getTransformMask(0, matrix_shape)};
    std::map<i64, Message> &horizontal_mask{transform_mask.getHorizontalMask()};
    if (horizontal_mask.empty())
        transform_mask.generateHorizontalMask(eval, matrix_shape);
    return horizontal_mask;
}

std::map<i64, Message> &MatrixTransformer::getPackedHorizontalMask(
    const HomEvaluator &eval, const TileTensorShape &matrix_shape) {
    TransformMask &transform_mask{getTransformMask(0, matrix_shape)};
    std::map<i64, Message> &packed_horizontal_mask{
        transform_mask.getPackedHorizontalMask()};
    if (packed_horizontal_mask.empty())
        transform_mask.generatePackedHorizontalMask(eval, matrix_shape);
    return packed_horizontal_mask;
}

std::map<i64, Message> &
MatrixTransformer::getVerticalMask(const HomEvaluator &eval,
                                   const TileTensorShape &matrix_shape) {
    TransformMask &transform_mask{getTransformMask(0, matrix_shape)};
    std::map<i64, Message> &vertical_mask{transform_mask.getVerticalMask()};
    if (vertical_mask.empty())
        transform_mask.generateVerticalMask(eval, matrix_shape);
    return vertical_mask;
}

std::map<i64, Message> &
MatrixTransformer::getPackedVerticalMask(const HomEvaluator &eval,
                                         const TileTensorShape &matrix_shape) {
    TransformMask &transform_mask{getTransformMask(0, matrix_shape)};
    std::map<i64, Message> &packed_vertical_mask{
        transform_mask.getPackedVerticalMask()};
    if (packed_vertical_mask.empty())
        transform_mask.generatePackedVerticalMask(eval, matrix_shape);
    return packed_vertical_mask;
}

std::vector<Plaintext> &
MatrixTransformer::getRowMask(const HomEvaluator &eval,
                              const TileTensorShape &matrix_shape,
                              const u64 target_level) {
    TransformMask &transform_mask{getTransformMask(target_level, matrix_shape)};
    std::vector<Plaintext> &row_mask{transform_mask.getRowMask()};
    if (row_mask.empty()) {
        transform_mask.generateRowMask(eval, matrix_shape, target_level);
    }
    return row_mask;
}

std::vector<Plaintext> &
MatrixTransformer::getColMask(const HomEvaluator &eval,
                              const TileTensorShape &matrix_shape,
                              const u64 target_level) {
    TransformMask &transform_mask{getTransformMask(target_level, matrix_shape)};
    std::vector<Plaintext> &col_mask{transform_mask.getColMask()};
    if (col_mask.empty()) {
        transform_mask.generateColMask(eval, matrix_shape, target_level);
    }
    return col_mask;
}

template <>
HETensor<Ciphertext>
MatrixTransformer::transpose<Ciphertext>(const HomEvaluator &eval,
                                         const HETensor<Ciphertext> &op) {
    const i64 num_cols = op.getHeight(); // num_cols after transpose
    const i64 num_rows = op.getWidth();  // num_rows after transpose
    const i64 width = op.getBlockWidth();

    if (num_cols > width)
        throw std::invalid_argument("[transpose] num_cols > width.");

    TileTensorShape matrix_shape(num_rows, num_cols, width);
    std::map<i64, Message> &diagonal_mask{getDiagonalMask(eval, matrix_shape)};

    // bs * (gs - 1) < num_rows + num_cols - 1 <= bs * gs
    const i64 bs =
        static_cast<i64>(std::ceil(std::sqrt(num_rows + num_cols - 1)));
    const i64 gs = static_cast<i64>(std::ceil(
        static_cast<Real>(num_rows + num_cols - 1) / static_cast<Real>(bs)));
    const i64 gap = width - 1;

    HETensor<Ciphertext> res(eval.getContext(), matrix_shape);
    Ciphertext tmp(eval.getContext()), gs_tmp(eval.getContext());
    std::vector<Ciphertext> bs_rot;
    bs_rot.reserve(static_cast<size_t>(bs));
    for (i64 j = 0; j < bs; j++)
        bs_rot.emplace_back(Ciphertext(eval.getContext()));
    eval.rightRotate(op.get(), static_cast<u64>((num_rows - 1) * gap),
                     bs_rot[0]);
    for (i64 j = 1; j < bs; j++) {
        eval.leftRotate(bs_rot[static_cast<u64>(j - 1)], static_cast<u64>(gap),
                        bs_rot[static_cast<u64>(j)]);
    }

    for (i64 i = 0; i < gs; i++) {
        if (i == 0) {
            i64 idx = 1 - num_rows;
            eval.mult(bs_rot[0], diagonal_mask[idx], res.get());

            for (i64 j = 1; j < bs; j++) {
                if (++idx >= num_cols)
                    break;

                eval.mult(bs_rot[static_cast<size_t>(j)], diagonal_mask[idx],
                          tmp);
                eval.add(res.get(), tmp, res.get());
            }
        } else {
            // baby-step
            i64 idx = 1 - num_rows + i * bs;
            eval.mult(bs_rot[0], diagonal_mask[idx], gs_tmp);

            for (i64 j = 1; j < bs; j++) {
                if (++idx >= num_cols)
                    break;

                eval.mult(bs_rot[static_cast<size_t>(j)], diagonal_mask[idx],
                          tmp);
                eval.add(gs_tmp, tmp, gs_tmp);
            }

            // giant-step
            eval.leftRotate(gs_tmp, static_cast<u64>((i * bs) * gap), gs_tmp);
            eval.add(res.get(), gs_tmp, res.get());
        }
    }

    return res;
}

template <>
HETensor<Plaintext>
MatrixTransformer::transpose<Plaintext>(const HomEvaluator &eval,
                                        const HETensor<Plaintext> &op) {
    const i64 num_cols = op.getHeight(); // num_cols after transpose
    const i64 num_rows = op.getWidth();  // num_rows after transpose
    const i64 width = op.getBlockWidth();

    if (num_cols > width)
        throw std::invalid_argument("[transpose] num_cols > width.");

    TileTensorShape matrix_shape(num_rows, num_cols, width);

    HETensor<Plaintext> res(eval.getContext(), matrix_shape);

    EnDecoder endec(eval.getContext());
    Message msg = endec.decode(op.get());
    msg.to(getDefaultDevice());
    Message msg2(msg);

    for (i64 i = 0; i < num_cols; i++) {
        for (i64 j = 0; j < num_rows; j++) {
            msg2[static_cast<u64>(i * width + j)] =
                msg[static_cast<u64>(j * width + i)];
            msg2[static_cast<u64>(i * width + j + num_cols)] =
                msg[static_cast<u64>(j * width + i + num_cols)];
        }
    }

    msg2.to(getCurrentCudaDevice());
    res.get() = endec.encode(msg2, 1);

    return res;
}

template <class T>
HETensor<T> MatrixTransformer::diagonalToColumn(const HomEvaluator &eval,
                                                const HETensor<T> &op) {
    const i64 num_rows = op.getHeight();
    const i64 num_cols = op.getWidth();
    const i64 width = op.getBlockWidth();

    TileTensorShape matrix_shape(num_rows, num_cols, width);
    std::map<i64, Message> &horizontal_mask{
        getHorizontalMask(eval, matrix_shape)};

    const i64 sub_dim = (num_rows > num_cols) ? num_cols : num_rows;

    // bs * (gs - 1) < sub_dim - 1 <= bs * gs
    const i64 bs = static_cast<i64>(std::ceil(std::sqrt(2 * sub_dim - 1)));
    const i64 gs = static_cast<i64>(
        std::ceil(static_cast<Real>(sub_dim - 1) / static_cast<Real>(bs)));
    const i64 gap = 1;

    HETensor<T> res(eval.getContext(), matrix_shape);
    T tmp(eval.getContext()), gs_tmp(eval.getContext());
    std::vector<T> bs_rot;
    bs_rot.reserve(static_cast<size_t>(bs));
    bs_rot.emplace_back(T(eval.getContext()));
    eval.leftRotate(op.get(), gap, bs_rot[0]);
    for (i64 j = 1; j < bs; j++) {
        bs_rot.emplace_back(T(eval.getContext()));
        eval.leftRotate(bs_rot[static_cast<u64>(j - 1)], gap,
                        bs_rot[static_cast<u64>(j)]);
    }

    eval.mult(op.get(), horizontal_mask[0], res.get());

    for (i64 i = 0; i < gs; i++) {
        const i64 rot_num = i * bs * gap;

        // negative idx
        // baby-step
        i64 idx = 1 + i * bs - num_cols;
        eval.mult(bs_rot[0], horizontal_mask[idx], gs_tmp);

        for (i64 j = 1; j < bs; j++) {
            if (++idx >= sub_dim - num_cols)
                break;

            eval.mult(bs_rot[static_cast<u64>(j)], horizontal_mask[idx], tmp);
            eval.add(gs_tmp, tmp, gs_tmp);
        }

        // giant-step
        eval.rightRotate(gs_tmp, static_cast<u64>(num_cols - rot_num), gs_tmp);
        eval.add(res.get(), gs_tmp, res.get());

        // positive idx
        // baby-step
        idx = 1 + i * bs;
        eval.mult(bs_rot[0], horizontal_mask[idx], gs_tmp);

        for (i64 j = 1; j < bs; j++) {
            if (++idx >= sub_dim)
                break;

            eval.mult(bs_rot[static_cast<u64>(j)], horizontal_mask[idx], tmp);
            eval.add(gs_tmp, tmp, gs_tmp);
        }

        // giant-step
        eval.leftRotate(gs_tmp, static_cast<u64>(rot_num), gs_tmp);
        eval.add(res.get(), gs_tmp, res.get());
    }

    return res;
}

CtxtTensor MatrixTransformer::packedDiagonalToColumn(const HomEvaluator &eval,
                                                     const CtxtTensor &op) {
    const i64 num_rows = op.getHeight() / 2;
    const i64 num_cols = op.getWidth() / 2;
    const i64 width = op.getBlockWidth();

    TileTensorShape matrix_shape(2 * num_rows, 2 * num_cols, width);
    std::map<i64, Message> &horizontal_mask{
        getPackedHorizontalMask(eval, matrix_shape)};

    const i64 sub_dim = (num_rows > num_cols) ? num_cols : num_rows;

    // bs * (gs - 1) < sub_dim - 1 <= bs * gs
    const i64 bs = static_cast<i64>(std::ceil(std::sqrt(2 * sub_dim - 1)));
    const i64 gs = static_cast<i64>(
        std::ceil(static_cast<Real>(sub_dim - 1) / static_cast<Real>(bs)));
    const i64 gap = 2;

    CtxtTensor res(eval.getContext(), matrix_shape);
    Ciphertext tmp(eval.getContext()), gs_tmp(eval.getContext());
    std::vector<Ciphertext> bs_rot;
    bs_rot.reserve(static_cast<size_t>(bs));
    bs_rot.emplace_back(Ciphertext(eval.getContext()));
    eval.leftRotate(op.get(), gap, bs_rot[0]);
    for (i64 j = 1; j < bs; j++) {
        bs_rot.emplace_back(Ciphertext(eval.getContext()));
        eval.leftRotate(bs_rot[static_cast<u64>(j - 1)], gap,
                        bs_rot[static_cast<u64>(j)]);
    }

    eval.mult(op.get(), horizontal_mask[0], res.get());

    for (i64 i = 0; i < gs; i++) {
        const i64 rot_num = i * bs * gap;

        // negative idx
        // baby-step
        i64 idx = 1 + i * bs - num_cols;
        eval.mult(bs_rot[0], horizontal_mask[idx], gs_tmp);

        for (i64 j = 1; j < bs; j++) {
            if (++idx >= sub_dim - num_cols)
                break;

            eval.mult(bs_rot[static_cast<u64>(j)], horizontal_mask[idx], tmp);
            eval.add(gs_tmp, tmp, gs_tmp);
        }

        // giant-step
        eval.rightRotate(gs_tmp, static_cast<u64>(gap * num_cols - rot_num),
                         gs_tmp);
        eval.add(res.get(), gs_tmp, res.get());

        // positive idx
        // baby-step
        idx = 1 + i * bs;
        eval.mult(bs_rot[0], horizontal_mask[idx], gs_tmp);

        for (i64 j = 1; j < bs; j++) {
            if (++idx >= sub_dim)
                break;

            eval.mult(bs_rot[static_cast<u64>(j)], horizontal_mask[idx], tmp);
            eval.add(gs_tmp, tmp, gs_tmp);
        }

        // giant-step
        eval.leftRotate(gs_tmp, static_cast<u64>(rot_num), gs_tmp);
        eval.add(res.get(), gs_tmp, res.get());
    }

    return res;
}

template <class T>
HETensor<T> MatrixTransformer::diagonalToRow(const HomEvaluator &eval,
                                             const HETensor<T> &op) {
    const i64 num_rows = op.getHeight();
    const i64 num_cols = op.getWidth();
    const i64 width = op.getBlockWidth();

    TileTensorShape matrix_shape(num_rows, num_cols, width);
    std::map<i64, Message> &vertical_mask{getVerticalMask(eval, matrix_shape)};

    const i64 sub_dim = (num_cols > num_rows) ? num_rows : num_cols;

    // bs * (gs - 1) < sub_dim - 1 <= bs * gs
    const i64 bs = static_cast<i64>(std::ceil(std::sqrt(2 * sub_dim - 1)));
    const i64 gs = static_cast<i64>(
        std::ceil(static_cast<Real>(sub_dim - 1) / static_cast<Real>(bs)));
    const i64 gap = width;

    HETensor<T> res(eval.getContext(), matrix_shape);
    T tmp(eval.getContext()), gs_tmp(eval.getContext());
    std::vector<T> bs_rot;
    bs_rot.reserve(static_cast<size_t>(bs));
    bs_rot.emplace_back(T(eval.getContext()));
    eval.leftRotate(op.get(), static_cast<u64>(gap), bs_rot[0]);
    for (i64 j = 1; j < bs; j++) {
        bs_rot.emplace_back(T(eval.getContext()));
        eval.leftRotate(bs_rot[static_cast<size_t>(j - 1)],
                        static_cast<u64>(gap), bs_rot[static_cast<size_t>(j)]);
    }

    eval.mult(op.get(), vertical_mask[0], res.get());

    for (i64 i = 0; i < gs; i++) {
        const i64 rot_num = i * bs * gap;

        // negative idx
        // baby-step
        i64 idx = 1 + i * bs - num_rows;
        eval.mult(bs_rot[0], vertical_mask[idx], gs_tmp);

        for (i64 j = 1; j < bs; j++) {
            if (++idx >= sub_dim - num_rows)
                break;

            eval.mult(bs_rot[static_cast<size_t>(j)], vertical_mask[idx], tmp);
            eval.add(gs_tmp, tmp, gs_tmp);
        }

        // giant-step
        eval.rightRotate(gs_tmp, static_cast<u64>(num_rows * gap - rot_num),
                         gs_tmp);
        eval.add(res.get(), gs_tmp, res.get());

        // positive idx
        // baby-step
        idx = 1 + i * bs;
        eval.mult(bs_rot[0], vertical_mask[idx], gs_tmp);

        for (i64 j = 1; j < bs; j++) {
            if (++idx >= sub_dim)
                break;

            eval.mult(bs_rot[static_cast<size_t>(j)], vertical_mask[idx], tmp);
            eval.add(gs_tmp, tmp, gs_tmp);
        }

        // giant-step
        eval.leftRotate(gs_tmp, static_cast<u64>(rot_num), gs_tmp);
        eval.add(res.get(), gs_tmp, res.get());
    }

    return res;
}

CtxtTensor MatrixTransformer::packedDiagonalToRow(const HomEvaluator &eval,
                                                  const CtxtTensor &op) {
    const i64 num_rows = op.getHeight() / 2;
    const i64 num_cols = op.getWidth() / 2;
    const i64 width = op.getBlockWidth();

    TileTensorShape matrix_shape(2 * num_rows, 2 * num_cols, width);
    std::map<i64, Message> &vertical_mask{
        getPackedVerticalMask(eval, matrix_shape)};

    const i64 sub_dim = (num_cols > num_rows) ? num_rows : num_cols;

    // bs * (gs - 1) < sub_dim - 1 <= bs * gs
    const i64 bs = static_cast<i64>(std::ceil(std::sqrt(2 * sub_dim - 1)));
    const i64 gs = static_cast<i64>(
        std::ceil(static_cast<Real>(sub_dim - 1) / static_cast<Real>(bs)));
    const i64 gap = 2 * width;

    CtxtTensor res(eval.getContext(), matrix_shape);
    Ciphertext tmp(eval.getContext()), gs_tmp(eval.getContext());
    std::vector<Ciphertext> bs_rot;
    bs_rot.reserve(static_cast<size_t>(bs));
    bs_rot.emplace_back(Ciphertext(eval.getContext()));
    eval.leftRotate(op.get(), static_cast<u64>(gap), bs_rot[0]);
    for (i64 j = 1; j < bs; j++) {
        bs_rot.emplace_back(Ciphertext(eval.getContext()));
        eval.leftRotate(bs_rot[static_cast<size_t>(j - 1)],
                        static_cast<u64>(gap), bs_rot[static_cast<size_t>(j)]);
    }

    eval.mult(op.get(), vertical_mask[0], res.get());

    for (i64 i = 0; i < gs; i++) {
        const i64 rot_num = i * bs * gap;

        // negative idx
        // baby-step
        i64 idx = 1 + i * bs - num_rows;
        eval.mult(bs_rot[0], vertical_mask[idx], gs_tmp);

        for (i64 j = 1; j < bs; j++) {
            if (++idx >= sub_dim - num_rows)
                break;

            eval.mult(bs_rot[static_cast<size_t>(j)], vertical_mask[idx], tmp);
            eval.add(gs_tmp, tmp, gs_tmp);
        }

        // giant-step
        eval.rightRotate(gs_tmp, static_cast<u64>(num_rows * gap - rot_num),
                         gs_tmp);
        eval.add(res.get(), gs_tmp, res.get());

        // positive idx
        // baby-step
        idx = 1 + i * bs;
        eval.mult(bs_rot[0], vertical_mask[idx], gs_tmp);

        for (i64 j = 1; j < bs; j++) {
            if (++idx >= sub_dim)
                break;

            eval.mult(bs_rot[static_cast<size_t>(j)], vertical_mask[idx], tmp);
            eval.add(gs_tmp, tmp, gs_tmp);
        }

        // giant-step
        eval.leftRotate(gs_tmp, static_cast<u64>(rot_num), gs_tmp);
        eval.add(res.get(), gs_tmp, res.get());
    }

    return res;
}

template <>
HETensor<Plaintext>
MatrixTransformer::diagonalToRow<Plaintext>(const HomEvaluator &eval,
                                            const HETensor<Plaintext> &op) {

    const i64 num_rows = op.getHeight();
    const i64 num_cols = op.getWidth();
    const i64 width = op.getBlockWidth();

    HETensor<Plaintext> res{op};
    EnDecoder endec(eval.getContext());
    Message msg = endec.decode(op.get());
    msg.to(getDefaultDevice());
    Message msg2(op.getLogSlots(), 0);
    for (i64 i = 0; i < num_rows; i++)
        for (i64 j = 0; j < num_cols; j++) {
            msg2[static_cast<u64>(((i - j + num_rows) % num_rows) * width +
                                  j)] = msg[static_cast<u64>(i * width + j)];
            if (width != num_cols)
                msg2[static_cast<u64>(((i - j + num_rows) % num_rows) * width +
                                      num_cols + j)] =
                    msg[static_cast<u64>(i * width + j + num_cols)];
        }
    msg2.to(getCurrentCudaDevice());
    res.get() = endec.encode(msg2);
    return res;
}

CtxtTensor multPackedMatMat(const HomEvaluator &eval, const Bootstrapper &btp,
                            const CtxtTensor &op1, const CtxtTensor &op2,
                            u64 target_level,
                            MatrixTransformer &matrix_transformer) {
    // const u64 log_slots = op1.getLogSlots();
    const u64 num_rows = static_cast<u64>(op1.getHeight());
    const u64 num_cols = static_cast<u64>(op2.getWidth());
    u64 num_k = static_cast<u64>(getCommonInsideLength(op1, op2));
    const u64 width = static_cast<u64>(getCommonBlockWidth(op1, op2));

    // level check
    if (std::min(op1.getLevel(), op2.getLevel()) <
        1 + btp.getMinLevelForBootstrap())
        throw std::invalid_argument(
            "[multMatMat] level >= " +
            std::to_string(1 + btp.getMinLevelForBootstrap()) + " is needed.");

    // TODO: algorithm using num_k which is not a power of 2.
    num_k = static_cast<u64>(std::pow(2, std::ceil(std::log2(num_k))));

    CtxtTensor op1_d_to_c{op1}, op2_d_to_r{op2};
    op1_d_to_c.setShape(static_cast<i64>(num_rows), static_cast<i64>(num_k));
    op2_d_to_r.setShape(static_cast<i64>(num_k), static_cast<i64>(num_cols));
    std::vector<Plaintext> &col_mask{matrix_transformer.getColMask(
        eval, op1_d_to_c.getShape(), target_level + 2)};

    // diagonal transforms

    eval.levelDown(op1_d_to_c.get(), target_level + 3, op1_d_to_c.get());
    eval.levelDown(op2_d_to_r.get(), target_level + 2, op2_d_to_r.get());

    op1_d_to_c = matrix_transformer.packedDiagonalToColumn(eval, op1_d_to_c);
    op2_d_to_r = matrix_transformer.packedDiagonalToRow(eval, op2_d_to_r);

    // copy inplace
    // copyInplace(eval, op1_d_to_c, num_cols, 1);
    // copyInplace(eval, op2_d_to_r, num_rows, 0);

    // make same level

    // compute matrix multiplication
    const u64 num_rows_tmp =
        (num_k > num_rows)
            ? num_k
            : 1 << static_cast<u64>(std::ceil(std::log2(num_rows)));
    const u64 num_cols_tmp =
        (num_k > num_cols)
            ? num_k
            : 1 << static_cast<u64>(std::ceil(std::log2(num_cols)));
    const TileTensorShape matrix_shape_tmp(static_cast<i64>(num_rows_tmp),
                                           static_cast<i64>(num_cols_tmp),
                                           static_cast<i64>(width));

    CtxtTensor res(eval.getContext(), static_cast<i64>(num_rows),
                   static_cast<i64>(num_cols), static_cast<i64>(width));
    eval.mult(op1_d_to_c.get(), op2_d_to_r.get(), res.get());

    eval.leftRotate(op1_d_to_c.get(), 2, op1_d_to_c.get());
    eval.leftRotate(op2_d_to_r.get(), 2 * width, op2_d_to_r.get());

    Ciphertext op1_d_to_c_rot(eval.getContext()),
        op2_d_to_r_rot(eval.getContext());
    eval.rightRotate(op1_d_to_c.get(), num_cols_tmp, op1_d_to_c_rot);

    /*
    msg_cols                    msg_rows
        = [1, 1, ..., 1, 1]         = [1, 1, ..., 1, 1]
          [1, 1, ..., 1, 1]           [1, 1, ..., 1, 1]
           .                           .
           .                           .
           .                           .
          [1, 1, ..., 1, 1]           [1, 1, ..., 1, 1]
          [1, 1, ..., 1, 1]           [1, 1, ..., 1, 1]

    msg_col_reverse             msg_rows_reverse
        = [0, 0, ..., 0, 0]          = [0, 0, ..., 0, 0]
          [0, 0, ..., 0, 0]            [0, 0, ..., 0, 0]
           .                            .
           .                            .
           .                            .
          [0, 0, ..., 0, 0]            [0, 0, ..., 0, 0]
          [0, 0, ..., 0, 0]            [0, 0, ..., 0, 0]

    */

    Ciphertext tmp1(eval.getContext()), tmp2(eval.getContext()),
        tmp3(eval.getContext());
    Ciphertext itxt_tmp(eval.getContext()), itxt_res(eval.getContext());

    for (u64 i = 1; i < num_k / 2; i++) {
        // rotate
        if (i > 1) {
            eval.leftRotate(op1_d_to_c.get(), 2, op1_d_to_c.get());
            eval.leftRotate(op1_d_to_c_rot, 2, op1_d_to_c_rot);

            eval.leftRotate(op2_d_to_r.get(), 2 * width, op2_d_to_r.get());
        }

        // assemble
        eval.multWithoutRescale(op1_d_to_c.get(), col_mask[4 * i + 1], tmp1);
        eval.multWithoutRescale(op1_d_to_c_rot, col_mask[4 * i], tmp3);
        eval.add(tmp1, tmp3, tmp1);
        eval.rescale(tmp1);

        if (i == 1) {
            eval.tensor(tmp1, op2_d_to_r.get(), itxt_res);
        } else {
            eval.tensor(tmp1, op2_d_to_r.get(), itxt_tmp);
            eval.add(itxt_res, itxt_tmp, itxt_res);
        }
    }
    eval.relinearize(itxt_res, tmp1);
    eval.rescale(tmp1);
    eval.add(res.get(), tmp1, res.get());

    return res;
}

CtxtTensor multPackedMatMatPre(const HomEvaluator &eval,
                               const CtxtTensor &tensor_a, u64 target_level,
                               MatrixTransformer &matrix_transformer) {
    CtxtTensor op1_d_to_c(tensor_a);
    eval.levelDown(op1_d_to_c.get(), target_level + 3, op1_d_to_c.get());
    op1_d_to_c = matrix_transformer.packedDiagonalToColumn(eval, op1_d_to_c);

    return op1_d_to_c;
}

void multPackedMatMatPreRot(const HomEvaluator &eval,
                            const CtxtTensor &tensor_a,
                            std::vector<Ciphertext> &tmp, u64 target_level,
                            MatrixTransformer &matrix_transformer) {
    tmp.clear();
    const u64 num_rows = static_cast<u64>(tensor_a.getHeight());
    CtxtTensor op1_d_to_c(tensor_a), tmp_c(tensor_a);

    eval.levelDownOne(op1_d_to_c.get(), tmp_c.get());
    tmp.push_back(tmp_c.get());

    eval.leftRotate(op1_d_to_c.get(), 2, op1_d_to_c.get());

    Ciphertext op1_d_to_c_rot(eval.getContext());

    eval.rightRotate(op1_d_to_c.get(), num_rows, op1_d_to_c_rot);

    Ciphertext tmp1(eval.getContext()), tmp2(eval.getContext());

    std::vector<Plaintext> &col_mask{matrix_transformer.getColMask(
        eval, op1_d_to_c.getShape(), target_level + 2)};

    for (u64 i = 1; i < num_rows / 2; i++) {
        // rotate
        if (i > 1) {
            eval.leftRotate(op1_d_to_c.get(), 2, op1_d_to_c.get());
            eval.leftRotate(op1_d_to_c_rot, 2, op1_d_to_c_rot);
        }

        // assemble
        eval.multWithoutRescale(op1_d_to_c.get(), col_mask[4 * i + 1], tmp1);
        eval.multWithoutRescale(op1_d_to_c_rot, col_mask[4 * i], tmp2);
        eval.add(tmp1, tmp2, tmp1);
        eval.rescale(tmp1);
        tmp.push_back(tmp1);
    }
}

CtxtTensor multPackedMatMatPreRev(const HomEvaluator &eval,
                                  const CtxtTensor &tensor_a, u64 target_level,
                                  MatrixTransformer &matrix_transformer) {
    CtxtTensor op2_d_to_r(tensor_a);
    eval.levelDown(op2_d_to_r.get(), target_level + 2, op2_d_to_r.get());
    op2_d_to_r = matrix_transformer.packedDiagonalToRow(eval, op2_d_to_r);

    return op2_d_to_r;
}

CtxtTensor multPackedMatMatCCReuse(
    const HomEvaluator &eval, const std::vector<Ciphertext> &tmp,
    const CtxtTensor &tensor_b, [[maybe_unused]] u64 target_level,
    [[maybe_unused]] MatrixTransformer &matrix_transformer) {
    const u64 num_rows = static_cast<u64>(tensor_b.getHeight());
    const u64 num_cols = static_cast<u64>(tensor_b.getWidth());
    u64 num_k = static_cast<u64>(tensor_b.getHeight());
    const u64 width = static_cast<u64>(tensor_b.getBlockWidth());

    // TODO: algorithm using num_k which is not a power of 2.
    num_k = static_cast<u64>(std::pow(2, std::ceil(std::log2(num_k))));

    CtxtTensor op2_d_to_r(tensor_b);
    op2_d_to_r.setShape(static_cast<i64>(num_k), static_cast<i64>(num_cols));

    CtxtTensor res(eval.getContext(), static_cast<i64>(num_rows),
                   static_cast<i64>(num_cols), static_cast<i64>(width));

    Ciphertext sum{tmp[0]}, mul{tmp[0]};
    eval.tensor(tmp[0], op2_d_to_r.get(), sum);
    for (u64 i = 1; i < tmp.size(); ++i) {
        eval.leftRotate(op2_d_to_r.get(), 2 * width, op2_d_to_r.get());
        eval.tensor(tmp[i], op2_d_to_r.get(), mul);
        eval.add(sum, mul, sum);
    }
    eval.relinearize(sum, res.get());
    eval.rescale(res.get());

    return res;
}

template <>
CtxtTensor multMatMat<Ciphertext>(const HomEvaluator &eval,
                                  const Bootstrapper &btp,
                                  const CtxtTensor &op1, const CtxtTensor &op2,
                                  u64 target_level,
                                  MatrixTransformer &matrix_transformer) {
    // const u64 log_slots = op1.getLogSlots();
    const u64 num_rows = static_cast<u64>(op1.getHeight());
    const u64 num_cols = static_cast<u64>(op2.getWidth());
    u64 num_k = static_cast<u64>(getCommonInsideLength(op1, op2));
    const u64 width = static_cast<u64>(getCommonBlockWidth(op1, op2));

    // level check
    if (std::min(op1.getLevel(), op2.getLevel()) <
        1 + btp.getMinLevelForBootstrap())
        throw std::invalid_argument(
            "[multMatMat] level >= " +
            std::to_string(1 + btp.getMinLevelForBootstrap()) + " is needed.");

    // TODO: algorithm using num_k which is not a power of 2.
    num_k = static_cast<u64>(std::pow(2, std::ceil(std::log2(num_k))));

    CtxtTensor op1_d_to_c{op1}, op2_d_to_r{op2};
    op1_d_to_c.setShape(static_cast<i64>(num_rows), static_cast<i64>(num_k));
    op2_d_to_r.setShape(static_cast<i64>(num_k), static_cast<i64>(num_cols));
    std::vector<Plaintext> &col_mask{matrix_transformer.getColMask(
        eval, op1_d_to_c.getShape(), target_level + 2)};
    std::vector<Plaintext> &row_mask{matrix_transformer.getRowMask(
        eval, op2_d_to_r.getShape(), target_level + 2)};

    if (std::min(op1_d_to_c.getLevel(), op2_d_to_r.getLevel()) <
        4 + btp.getMinLevelForBootstrap()) {
        eval.multImagUnit(op2_d_to_r.get(), op2_d_to_r.get());
        eval.add(op1_d_to_c.get(), op2_d_to_r.get(), op1_d_to_c.get());
        btp.bootstrapExtended(op1_d_to_c.get(), op1_d_to_c.get(),
                              op2_d_to_r.get());
    }

    // diagonal transforms
    op1_d_to_c = matrix_transformer.diagonalToColumn(eval, op1_d_to_c);
    op2_d_to_r = matrix_transformer.diagonalToRow(eval, op2_d_to_r);

    eval.levelDown(op1_d_to_c.get(), target_level + 2, op1_d_to_c.get());
    eval.levelDown(op2_d_to_r.get(), target_level + 2, op2_d_to_r.get());
    // copy inplace
    // copyInplace(eval, op1_d_to_c, num_cols, 1);
    // copyInplace(eval, op2_d_to_r, num_rows, 0);

    // make same level
    if (op1_d_to_c.getLevel() > op2_d_to_r.getLevel())
        eval.levelDown(op1_d_to_c.get(), op2_d_to_r.getLevel(),
                       op1_d_to_c.get());
    if (op2_d_to_r.getLevel() > op1_d_to_c.getLevel())
        eval.levelDown(op2_d_to_r.get(), op1_d_to_c.getLevel(),
                       op2_d_to_r.get());

    // compute matrix multiplication
    const u64 num_rows_tmp =
        (num_k > num_rows)
            ? num_k
            : 1 << static_cast<u64>(std::ceil(std::log2(num_rows)));
    const u64 num_cols_tmp =
        (num_k > num_cols)
            ? num_k
            : 1 << static_cast<u64>(std::ceil(std::log2(num_cols)));
    const TileTensorShape matrix_shape_tmp(static_cast<i64>(num_rows_tmp),
                                           static_cast<i64>(num_cols_tmp),
                                           static_cast<i64>(width));

    CtxtTensor res(eval.getContext(), static_cast<i64>(num_rows),
                   static_cast<i64>(num_cols), static_cast<i64>(width));
    eval.mult(op1_d_to_c.get(), op2_d_to_r.get(), res.get());

    eval.leftRotate(op1_d_to_c.get(), 1, op1_d_to_c.get());
    eval.leftRotate(op2_d_to_r.get(), width, op2_d_to_r.get());

    Ciphertext op1_d_to_c_rot(eval.getContext()),
        op2_d_to_r_rot(eval.getContext());
    eval.rightRotate(op1_d_to_c.get(), num_cols_tmp, op1_d_to_c_rot);
    eval.rightRotate(op2_d_to_r.get(), num_rows_tmp * width, op2_d_to_r_rot);

    /*
    msg_cols                    msg_rows
        = [1, 1, ..., 1, 1]         = [1, 1, ..., 1, 1]
          [1, 1, ..., 1, 1]           [1, 1, ..., 1, 1]
           .                           .
           .                           .
           .                           .
          [1, 1, ..., 1, 1]           [1, 1, ..., 1, 1]
          [1, 1, ..., 1, 1]           [1, 1, ..., 1, 1]

    msg_col_reverse             msg_rows_reverse
        = [0, 0, ..., 0, 0]          = [0, 0, ..., 0, 0]
          [0, 0, ..., 0, 0]            [0, 0, ..., 0, 0]
           .                            .
           .                            .
           .                            .
          [0, 0, ..., 0, 0]            [0, 0, ..., 0, 0]
          [0, 0, ..., 0, 0]            [0, 0, ..., 0, 0]

    */

    Ciphertext tmp1(eval.getContext()), tmp2(eval.getContext()),
        tmp3(eval.getContext());
    Ciphertext itxt_tmp(eval.getContext()), itxt_res(eval.getContext());

    for (u64 i = 1; i < num_k; i++) {
        // rotate
        if (i > 1) {
            eval.leftRotate(op1_d_to_c.get(), 1, op1_d_to_c.get());
            eval.leftRotate(op1_d_to_c_rot, 1, op1_d_to_c_rot);

            eval.leftRotate(op2_d_to_r.get(), width, op2_d_to_r.get());
            eval.leftRotate(op2_d_to_r_rot, width, op2_d_to_r_rot);
        }

        // assemble
        eval.multWithoutRescale(op1_d_to_c.get(), col_mask[2 * i + 1], tmp1);
        eval.multWithoutRescale(op1_d_to_c_rot, col_mask[2 * i], tmp3);
        eval.add(tmp1, tmp3, tmp1);
        eval.rescale(tmp1);

        eval.multWithoutRescale(op2_d_to_r.get(), row_mask[2 * i + 1], tmp2);
        eval.multWithoutRescale(op2_d_to_r_rot, row_mask[2 * i], tmp3);
        eval.add(tmp2, tmp3, tmp2);
        eval.rescale(tmp2);

        if (i == 1) {
            eval.tensor(tmp1, tmp2, itxt_res);
        } else {
            eval.tensor(tmp1, tmp2, itxt_tmp);
            eval.add(itxt_res, itxt_tmp, itxt_res);
        }
    }
    eval.relinearize(itxt_res, tmp1);
    eval.rescale(tmp1);
    eval.add(res.get(), tmp1, res.get());

    return res;
}

template <>
CtxtTensor multMatMat<Plaintext>(const HomEvaluator &eval,
                                 const Bootstrapper &btp, const CtxtTensor &op1,
                                 const PtxtTensor &op2, u64 target_level,
                                 MatrixTransformer &matrix_transformer) {
    // const u64 log_slots = op1.getLogSlots();
    const u64 num_rows = static_cast<u64>(op1.getHeight());
    const u64 num_cols = static_cast<u64>(op2.getWidth());
    u64 num_k = static_cast<u64>(getCommonInsideLength(op1, op2));
    const u64 width = static_cast<u64>(getCommonBlockWidth(op1, op2));
    // level check
    if (std::min(op1.getLevel(), op2.getLevel()) <
        1 + btp.getMinLevelForBootstrap())
        throw std::invalid_argument(
            "[multMatMat] level >= " +
            std::to_string(1 + btp.getMinLevelForBootstrap()) + " is needed.");

    // TODO: algorithm using num_k which is not a power of 2.
    num_k = static_cast<u64>(std::pow(2, std::ceil(std::log2(num_k))));

    CtxtTensor op1_d_to_c{op1};
    PtxtTensor op2_d_to_r{op2};
    op1_d_to_c.setShape(static_cast<i64>(num_rows), static_cast<i64>(num_k));
    op2_d_to_r.setShape(static_cast<i64>(num_k), static_cast<i64>(num_cols));
    if (op1_d_to_c.getLevel() < 4 + btp.getMinLevelForBootstrap()) {
        btp.bootstrapExtended(op1_d_to_c.get(), op1_d_to_c.get());
    }

    // diagonal transforms
    op1_d_to_c = matrix_transformer.diagonalToColumn(eval, op1_d_to_c);
    op2_d_to_r = matrix_transformer.diagonalToRow(eval, op2_d_to_r);

    eval.levelDown(op1_d_to_c.get(), target_level + 2, op1_d_to_c.get());
    EnDecoder endec(eval.getContext());
    op2_d_to_r.get() =
        (endec.encode(endec.decode(op2_d_to_r.get()), target_level + 2));

    // compute matrix multiplication
    const u64 num_rows_tmp =
        (num_k > num_rows)
            ? num_k
            : 1 << static_cast<u64>(std::ceil(std::log2(num_rows)));
    const u64 num_cols_tmp =
        (num_k > num_cols)
            ? num_k
            : 1 << static_cast<u64>(std::ceil(std::log2(num_cols)));
    const TileTensorShape matrix_shape_tmp(static_cast<i64>(num_rows_tmp),
                                           static_cast<i64>(num_cols_tmp),
                                           static_cast<i64>(width));

    CtxtTensor res(eval.getContext(), static_cast<i64>(num_rows),
                   static_cast<i64>(num_cols), static_cast<i64>(width));
    eval.mult(op1_d_to_c.get(), op2_d_to_r.get(), res.get());

    eval.leftRotate(op1_d_to_c.get(), 1, op1_d_to_c.get());
    eval.leftRotate(op2_d_to_r.get(), width, op2_d_to_r.get());

    Ciphertext op1_d_to_c_rot(eval.getContext());
    Plaintext op2_d_to_r_rot(eval.getContext());
    eval.rightRotate(op1_d_to_c.get(), num_cols_tmp, op1_d_to_c_rot);
    eval.rightRotate(op2_d_to_r.get(), num_rows_tmp * width, op2_d_to_r_rot);

    /*
    msg_cols                    msg_rows
        = [1, 1, ..., 1, 1]         = [1, 1, ..., 1, 1]
          [1, 1, ..., 1, 1]           [1, 1, ..., 1, 1]
           .                           .
           .                           .
           .                           .
          [1, 1, ..., 1, 1]           [1, 1, ..., 1, 1]
          [1, 1, ..., 1, 1]           [1, 1, ..., 1, 1]

    msg_col_reverse             msg_rows_reverse
        = [0, 0, ..., 0, 0]          = [0, 0, ..., 0, 0]
          [0, 0, ..., 0, 0]            [0, 0, ..., 0, 0]
           .                            .
           .                            .
           .                            .
          [0, 0, ..., 0, 0]            [0, 0, ..., 0, 0]
          [0, 0, ..., 0, 0]            [0, 0, ..., 0, 0]

    */

    Ciphertext tmp1(eval.getContext()), tmp2(eval.getContext());
    Plaintext tmp3(eval.getContext()), tmp4(eval.getContext());
    Ciphertext itxt_tmp(eval.getContext()), itxt_res(eval.getContext());

    std::vector<Plaintext> &col_mask{matrix_transformer.getColMask(
        eval, op1_d_to_c.getShape(), target_level + 2)};
    std::vector<Plaintext> &row_mask{matrix_transformer.getRowMask(
        eval, op2_d_to_r.getShape(), target_level + 2)};

    for (u64 i = 1; i < num_k; i++) {
        // rotate
        if (i > 1) {
            eval.leftRotate(op1_d_to_c.get(), 1, op1_d_to_c.get());
            eval.leftRotate(op1_d_to_c_rot, 1, op1_d_to_c_rot);

            eval.leftRotate(op2_d_to_r.get(), width, op2_d_to_r.get());
            eval.leftRotate(op2_d_to_r_rot, width, op2_d_to_r_rot);
        }

        // assemble
        eval.multWithoutRescale(op1_d_to_c.get(), col_mask[2 * i + 1], tmp1);
        eval.multWithoutRescale(op1_d_to_c_rot, col_mask[2 * i], tmp2);
        eval.add(tmp1, tmp2, tmp1);
        eval.rescale(tmp1);

        eval.mult(op2_d_to_r.get(), row_mask[2 * i + 1], tmp3);
        eval.mult(op2_d_to_r_rot, row_mask[2 * i], tmp4);
        eval.add(tmp3, tmp4, tmp3);

        if (i == 1) {
            eval.mult(tmp1, tmp3, itxt_res);
        } else {
            eval.mult(tmp1, tmp3, itxt_tmp);
            eval.add(itxt_res, itxt_tmp, itxt_res);
        }
    }
    eval.add(res.get(), itxt_res, res.get());

    return res;
}

void multMatMatPre(const HomEvaluator &eval, const CtxtTensor &tensor_a,
                   std::vector<Ciphertext> &tmp, u64 target_level,
                   MatrixTransformer &matrix_transformer) {
    tmp.clear();
    const u64 num_rows = static_cast<u64>(tensor_a.getHeight());
    CtxtTensor op1_d_to_c(tensor_a), tmp_c(tensor_a);

    eval.levelDown(op1_d_to_c.get(), target_level + 3, op1_d_to_c.get());
    op1_d_to_c = matrix_transformer.diagonalToColumn(eval, op1_d_to_c);

    eval.levelDownOne(op1_d_to_c.get(), tmp_c.get());
    tmp.push_back(tmp_c.get());

    eval.leftRotate(op1_d_to_c.get(), 1, op1_d_to_c.get());

    Ciphertext op1_d_to_c_rot(eval.getContext());

    eval.rightRotate(op1_d_to_c.get(), num_rows, op1_d_to_c_rot);

    Ciphertext tmp1(eval.getContext()), tmp2(eval.getContext());

    std::vector<Plaintext> &col_mask{matrix_transformer.getColMask(
        eval, op1_d_to_c.getShape(), target_level + 2)};

    for (u64 i = 1; i < num_rows; i++) {
        // rotate
        if (i > 1) {
            eval.leftRotate(op1_d_to_c.get(), 1, op1_d_to_c.get());
            eval.leftRotate(op1_d_to_c_rot, 1, op1_d_to_c_rot);
        }

        // assemble
        eval.multWithoutRescale(op1_d_to_c.get(), col_mask[2 * i + 1], tmp1);
        eval.multWithoutRescale(op1_d_to_c_rot, col_mask[2 * i], tmp2);
        eval.add(tmp1, tmp2, tmp1);
        eval.rescale(tmp1);
        tmp.push_back(tmp1);
    }
}

CtxtTensor
multMatMatReUse(const HomEvaluator &eval, const std::vector<Ciphertext> &tmp,
                const PtxtTensor &tensor_b, u64 target_level,
                [[maybe_unused]] MatrixTransformer &matrix_transformer) {
    const u64 num_rows = static_cast<u64>(tensor_b.getHeight());
    const u64 num_cols = static_cast<u64>(tensor_b.getWidth());
    u64 num_k = static_cast<u64>(tensor_b.getHeight());
    const u64 width = static_cast<u64>(tensor_b.getBlockWidth());

    // TODO: algorithm using num_k which is not a power of 2.
    num_k = static_cast<u64>(std::pow(2, std::ceil(std::log2(num_k))));

    PtxtTensor op2_d_to_r(tensor_b);
    op2_d_to_r.setShape(static_cast<i64>(num_k), static_cast<i64>(num_cols));

    if (op2_d_to_r.getLevel() != target_level + 1) {
        EnDecoder endec(eval.getContext());
        Message tmp_m = endec.decode(op2_d_to_r.get());
        op2_d_to_r.get() = endec.encode(tmp_m, target_level + 1);
    }

    CtxtTensor res(eval.getContext(), static_cast<i64>(num_rows),
                   static_cast<i64>(num_cols), static_cast<i64>(width));

    eval.matmul(tmp, op2_d_to_r.get(), res.get());

    return res;
}

CtxtTensor multMatMatPreRev(const HomEvaluator &eval,
                            const CtxtTensor &tensor_a, u64 target_level,
                            MatrixTransformer &matrix_transformer) {
    CtxtTensor op2_d_to_r(tensor_a);
    eval.levelDown(op2_d_to_r.get(), target_level + 2, op2_d_to_r.get());
    op2_d_to_r = matrix_transformer.diagonalToRow(eval, op2_d_to_r);

    return op2_d_to_r;
}

CtxtTensor
multMatMatCCReUse(const HomEvaluator &eval, const std::vector<Ciphertext> &tmp,
                  const CtxtTensor &tensor_b, [[maybe_unused]] u64 target_level,
                  [[maybe_unused]] MatrixTransformer &matrix_transformer) {
    const u64 num_rows = static_cast<u64>(tensor_b.getHeight());
    const u64 num_cols = static_cast<u64>(tensor_b.getWidth());
    u64 num_k = static_cast<u64>(tensor_b.getHeight());
    const u64 width = static_cast<u64>(tensor_b.getBlockWidth());

    // TODO: algorithm using num_k which is not a power of 2.
    num_k = static_cast<u64>(std::pow(2, std::ceil(std::log2(num_k))));

    CtxtTensor op2_d_to_r(tensor_b);
    op2_d_to_r.setShape(static_cast<i64>(num_k), static_cast<i64>(num_cols));

    CtxtTensor res(eval.getContext(), static_cast<i64>(num_rows),
                   static_cast<i64>(num_cols), static_cast<i64>(width));

    Ciphertext sum{tmp[0]}, mul{tmp[0]};
    eval.tensor(tmp[0], op2_d_to_r.get(), sum);
    for (u64 i = 1; i < tmp.size(); ++i) {
        eval.leftRotate(op2_d_to_r.get(), width, op2_d_to_r.get());
        eval.tensor(tmp[i], op2_d_to_r.get(), mul);
        eval.add(sum, mul, sum);
    }
    eval.relinearize(sum, res.get());

    return res;
}

CtxtTensor multMatMatHighLow(const HomEvaluator &eval, const CtxtTensor &op1,
                             const CtxtTensor &op2, const u64 in_col_block,
                             u64 target_level,
                             MatrixTransformer &matrix_transformer) {
    const u64 low_dim = ModelArgs::LOW_DIM;
    const u64 num_rows = static_cast<u64>(op1.getHeight());
    const u64 block_width = static_cast<u64>(op2.getBlockWidth());
    CtxtTensor op1_d_to_c{op1}, op2_d_to_r{op2};
    op2_d_to_r.setShape(op2.getHeight(), low_dim);

    // diagonal transforms
    eval.levelDown(op1_d_to_c.get(), target_level + 3, op1_d_to_c.get());
    eval.levelDown(op2_d_to_r.get(), target_level + 3, op2_d_to_r.get());

    op1_d_to_c = matrix_transformer.diagonalToColumn(eval, op1_d_to_c);
    op2_d_to_r = matrix_transformer.diagonalToRow(eval, op2_d_to_r);

    // hard coding
    u64 col_pos = in_col_block;
    Ciphertext tmp(eval.getContext());
    for (u64 i = low_dim; i < block_width / 2; i <<= 1) {
        if (col_pos % 2 == 0) {
            eval.leftRotate(op2_d_to_r.get(), i * block_width, tmp);
            eval.rightRotate(tmp, i, tmp);
        } else {
            eval.leftRotate(op2_d_to_r.get(), i, tmp);
            eval.leftRotate(op2_d_to_r.get(), i * block_width,
                            op2_d_to_r.get());
        }
        eval.add(op2_d_to_r.get(), tmp, op2_d_to_r.get());
        col_pos /= 2;
    }

    // compute matrix multiplication
    CtxtTensor res(eval.getContext(), op1.getHeight(), op2.getWidth(),
                   op2.getBlockWidth());
    eval.mult(op1_d_to_c.get(), op2_d_to_r.get(), res.get());

    eval.leftRotate(op1_d_to_c.get(), 1, op1_d_to_c.get());
    eval.leftRotate(op2_d_to_r.get(), block_width, op2_d_to_r.get());

    Ciphertext op1_d_to_c_rot(eval.getContext()),
        op2_d_to_r_rot(eval.getContext());
    eval.rightRotate(op1_d_to_c.get(), block_width / 2, op1_d_to_c_rot);
    eval.rightRotate(op2_d_to_r.get(), num_rows * block_width, op2_d_to_r_rot);

    std::vector<Plaintext> &col_mask{
        matrix_transformer.getColMask(eval, op1.getShape(), target_level + 2)};
    std::vector<Plaintext> &row_mask{
        matrix_transformer.getRowMask(eval, op2.getShape(), target_level + 2)};

    Ciphertext tmp1(eval.getContext()), tmp2(eval.getContext()),
        tmp3(eval.getContext());
    Ciphertext itxt_tmp(eval.getContext()), itxt_res(eval.getContext());

    for (u64 i = 1; i < low_dim; i++) {
        // rotate
        if (i > 1) {
            eval.leftRotate(op1_d_to_c.get(), 1, op1_d_to_c.get());
            eval.leftRotate(op1_d_to_c_rot, 1, op1_d_to_c_rot);

            eval.leftRotate(op2_d_to_r.get(), block_width, op2_d_to_r.get());
            eval.leftRotate(op2_d_to_r_rot, block_width, op2_d_to_r_rot);
        }

        // assemble
        eval.multWithoutRescale(op1_d_to_c.get(), col_mask[2 * i + 1], tmp1);
        eval.multWithoutRescale(op1_d_to_c_rot, col_mask[2 * i], tmp3);
        eval.add(tmp1, tmp3, tmp1);
        eval.rescale(tmp1);

        eval.multWithoutRescale(op2_d_to_r.get(), row_mask[2 * i + 1], tmp2);
        eval.multWithoutRescale(op2_d_to_r_rot, row_mask[2 * i], tmp3);
        eval.add(tmp2, tmp3, tmp2);
        eval.rescale(tmp2);

        if (i == 1) {
            eval.tensor(tmp1, tmp2, itxt_res);
        } else {
            eval.tensor(tmp1, tmp2, itxt_tmp);
            eval.add(itxt_res, itxt_tmp, itxt_res);
        }
    }
    eval.relinearize(itxt_res, tmp1);
    eval.rescale(tmp1);
    eval.add(res.get(), tmp1, res.get());

    return res;
}

CtxtTensor multMatMatLowHigh(const HomEvaluator &eval, const CtxtTensor &op1,
                             const CtxtTensor &op2, const u64 in_row_block,
                             u64 target_level,
                             MatrixTransformer &matrix_transformer) {
    //
    const u64 low_dim = ModelArgs::LOW_DIM;
    const u64 num_rows = static_cast<u64>(op1.getHeight());
    const u64 block_width = static_cast<u64>(getCommonBlockWidth(op1, op2));

    CtxtTensor op1_d_to_c{op1}, op2_d_to_r{op2};

    eval.levelDown(op1_d_to_c.get(), target_level + 3, op1_d_to_c.get());
    eval.levelDown(op2_d_to_r.get(), target_level + 3, op2_d_to_r.get());

    // hard coding
    u64 row_pos = in_row_block;
    Ciphertext tmp(eval.getContext());
    for (u64 i = low_dim; i < block_width / 2; i <<= 1) {
        if (row_pos % 2 == 0) {
            eval.rightRotate(op1_d_to_c.get(), i * block_width, tmp);
        } else {
            eval.leftRotate(op1_d_to_c.get(), i * block_width, tmp);
        }
        eval.add(op1_d_to_c.get(), tmp, op1_d_to_c.get());
        row_pos /= 2;
    }

    // diagonal transforms
    op1_d_to_c = matrix_transformer.diagonalToColumn(eval, op1_d_to_c);
    op2_d_to_r = matrix_transformer.diagonalToRow(eval, op2_d_to_r);

    std::vector<Plaintext> &col_mask{
        matrix_transformer.getColMask(eval, op1.getShape(), target_level + 2)};
    std::vector<Plaintext> &row_mask{
        matrix_transformer.getRowMask(eval, op2.getShape(), target_level + 2)};

    // compute matrix multiplication
    CtxtTensor res(eval.getContext(), op1.getHeight(), op2.getWidth(),
                   op2.getBlockWidth());
    eval.mult(op1_d_to_c.get(), op2_d_to_r.get(), res.get());

    eval.leftRotate(op1_d_to_c.get(), 1, op1_d_to_c.get());
    eval.leftRotate(op2_d_to_r.get(), block_width, op2_d_to_r.get());

    Ciphertext op1_d_to_c_rot(eval.getContext()),
        op2_d_to_r_rot(eval.getContext());
    eval.rightRotate(op1_d_to_c.get(), block_width / 2, op1_d_to_c_rot);
    eval.rightRotate(op2_d_to_r.get(), num_rows * block_width, op2_d_to_r_rot);

    /*
    msg_cols                    msg_rows
        = [1, 1, ..., 1, 1]         = [1, 1, ..., 1, 1]
          [1, 1, ..., 1, 1]           [1, 1, ..., 1, 1]
           .                           .
           .                           .
           .                           .
          [1, 1, ..., 1, 1]           [1, 1, ..., 1, 1]
          [1, 1, ..., 1, 1]           [1, 1, ..., 1, 1]

    msg_col_reverse             msg_rows_reverse
        = [0, 0, ..., 0, 0]          = [0, 0, ..., 0, 0]
          [0, 0, ..., 0, 0]            [0, 0, ..., 0, 0]
           .                            .
           .                            .
           .                            .
          [0, 0, ..., 0, 0]            [0, 0, ..., 0, 0]
          [0, 0, ..., 0, 0]            [0, 0, ..., 0, 0]

    */

    Ciphertext tmp1(eval.getContext()), tmp2(eval.getContext()),
        tmp3(eval.getContext());
    Ciphertext itxt_tmp(eval.getContext()), itxt_res(eval.getContext());

    for (u64 i = 1; i < low_dim; i++) {
        // rotate
        if (i > 1) {
            eval.leftRotate(op1_d_to_c.get(), 1, op1_d_to_c.get());
            eval.leftRotate(op1_d_to_c_rot, 1, op1_d_to_c_rot);

            eval.leftRotate(op2_d_to_r.get(), block_width, op2_d_to_r.get());
            eval.leftRotate(op2_d_to_r_rot, block_width, op2_d_to_r_rot);
        }

        // assemble
        eval.multWithoutRescale(op1_d_to_c.get(), col_mask[2 * i + 1], tmp1);
        eval.multWithoutRescale(op1_d_to_c_rot, col_mask[2 * i], tmp3);
        eval.add(tmp1, tmp3, tmp1);
        eval.rescale(tmp1);

        eval.multWithoutRescale(op2_d_to_r.get(), row_mask[2 * i + 1], tmp2);
        eval.multWithoutRescale(op2_d_to_r_rot, row_mask[2 * i], tmp3);
        eval.add(tmp2, tmp3, tmp2);
        eval.rescale(tmp2);

        if (i == 1) {
            eval.tensor(tmp1, tmp2, itxt_res);
        } else {
            eval.tensor(tmp1, tmp2, itxt_tmp);
            eval.add(itxt_res, itxt_tmp, itxt_res);
        }
    }
    eval.relinearize(itxt_res, tmp1);
    eval.rescale(tmp1);
    eval.add(res.get(), tmp1, res.get());

    return res;
}

CtxtTensor multMatMatLowLow(const HomEvaluator &eval, const CtxtTensor &op1,
                            const CtxtTensor &op2, const u64 in_col_block,
                            const u64 in_row_block, u64 target_level,
                            MatrixTransformer &matrix_transformer) {
    const u64 low_dim = ModelArgs::LOW_DIM;
    const u64 num_rows = static_cast<u64>(op1.getHeight());
    const u64 block_width = static_cast<u64>(getCommonBlockWidth(op1, op2));

    CtxtTensor op1_d_to_c{op1}, op2_d_to_r{op2};
    op1_d_to_c.setShape(op1.getHeight(), low_dim);
    op2_d_to_r.setShape(low_dim, op2.getWidth());

    eval.levelDown(op1_d_to_c.get(), target_level + 3, op1_d_to_c.get());
    eval.levelDown(op2_d_to_r.get(), target_level + 3, op2_d_to_r.get());

    // diagonal transforms
    op1_d_to_c = matrix_transformer.diagonalToColumn(eval, op1_d_to_c);
    op2_d_to_r = matrix_transformer.diagonalToRow(eval, op2_d_to_r);

    // hard coding
    u64 col_pos = in_col_block;
    u64 row_pos = in_row_block;
    Ciphertext tmp(eval.getContext());
    for (u64 i = low_dim; i < block_width / 2; i <<= 1) {
        if (col_pos % 2 == 0) {
            eval.rightRotate(op1_d_to_c.get(), i, tmp);
        } else {
            eval.leftRotate(op1_d_to_c.get(), i, tmp);
        }
        eval.add(op1_d_to_c.get(), tmp, op1_d_to_c.get());
        col_pos /= 2;
        if (row_pos % 2 == 0) {
            eval.rightRotate(op2_d_to_r.get(), i * block_width, tmp);
        } else {
            eval.leftRotate(op2_d_to_r.get(), i * block_width, tmp);
        }
        eval.add(op2_d_to_r.get(), tmp, op2_d_to_r.get());
        row_pos /= 2;
    }

    std::vector<Plaintext> &col_mask{
        matrix_transformer.getColMask(eval, op1.getShape(), target_level + 2)};
    std::vector<Plaintext> &row_mask{
        matrix_transformer.getRowMask(eval, op2.getShape(), target_level + 2)};

    // compute matrix multiplication
    CtxtTensor res(eval.getContext(), op1.getHeight(), op2.getWidth(),
                   op2.getBlockWidth());
    eval.mult(op1_d_to_c.get(), op2_d_to_r.get(), res.get());

    eval.leftRotate(op1_d_to_c.get(), 1, op1_d_to_c.get());
    eval.leftRotate(op2_d_to_r.get(), block_width, op2_d_to_r.get());

    Ciphertext op1_d_to_c_rot(eval.getContext()),
        op2_d_to_r_rot(eval.getContext());
    eval.rightRotate(op1_d_to_c.get(), block_width / 2, op1_d_to_c_rot);
    eval.rightRotate(op2_d_to_r.get(), num_rows * block_width, op2_d_to_r_rot);

    /*
    msg_cols                    msg_rows
        = [1, 1, ..., 1, 1]         = [1, 1, ..., 1, 1]
          [1, 1, ..., 1, 1]           [1, 1, ..., 1, 1]
           .                           .
           .                           .
           .                           .
          [1, 1, ..., 1, 1]           [1, 1, ..., 1, 1]
          [1, 1, ..., 1, 1]           [1, 1, ..., 1, 1]

    msg_col_reverse             msg_rows_reverse
        = [0, 0, ..., 0, 0]          = [0, 0, ..., 0, 0]
          [0, 0, ..., 0, 0]            [0, 0, ..., 0, 0]
           .                            .
           .                            .
           .                            .
          [0, 0, ..., 0, 0]            [0, 0, ..., 0, 0]
          [0, 0, ..., 0, 0]            [0, 0, ..., 0, 0]

    */

    Ciphertext tmp1(eval.getContext()), tmp2(eval.getContext()),
        tmp3(eval.getContext());
    Ciphertext itxt_tmp(eval.getContext()), itxt_res(eval.getContext());

    for (u64 i = 1; i < low_dim; i++) {
        // rotate
        if (i > 1) {
            eval.leftRotate(op1_d_to_c.get(), 1, op1_d_to_c.get());
            eval.leftRotate(op1_d_to_c_rot, 1, op1_d_to_c_rot);

            eval.leftRotate(op2_d_to_r.get(), block_width, op2_d_to_r.get());
            eval.leftRotate(op2_d_to_r_rot, block_width, op2_d_to_r_rot);
        }

        // assemble
        eval.multWithoutRescale(op1_d_to_c.get(), col_mask[2 * i + 1], tmp1);
        eval.multWithoutRescale(op1_d_to_c_rot, col_mask[2 * i], tmp3);
        eval.add(tmp1, tmp3, tmp1);
        eval.rescale(tmp1);

        eval.multWithoutRescale(op2_d_to_r.get(), row_mask[2 * i + 1], tmp2);
        eval.multWithoutRescale(op2_d_to_r_rot, row_mask[2 * i], tmp3);
        eval.add(tmp2, tmp3, tmp2);
        eval.rescale(tmp2);

        if (i == 1) {
            eval.tensor(tmp1, tmp2, itxt_res);
        } else {
            eval.tensor(tmp1, tmp2, itxt_tmp);
            eval.add(itxt_res, itxt_tmp, itxt_res);
        }
    }
    eval.relinearize(itxt_res, tmp1);
    eval.rescale(tmp1);
    eval.add(res.get(), tmp1, res.get());

    return res;
}

Message genRowMask(const u64 log_slot) {
    Message res(log_slot, 0);
    for (u64 i = 0; i < 256; i++) {
        res[i] = 1;
    }
    return res;
}

Message genFirstHalfRowMask(const u64 log_slot) {
    Message res(log_slot, 0);
    for (u64 i = 0; i < 128; i++) {
        res[i] = 0.5;
    }
    return res;
}

Message genSecondHalfRowMask(const u64 log_slot) {
    Message res(log_slot, 0);
    for (u64 i = 128; i < 256; i++) {
        res[i] = 0.5;
    }
    return res;
}

Message genFirstMask(const u64 log_slot) {
    Message res(log_slot, 0);
    for (u64 i = 0; i < 128; i++)
        for (u64 j = 0; j < 128; j++)
            res[i * 256 + j] = 1.0;

    return res;
}

Message genSecondMask(const u64 log_slot) {
    Message res(log_slot, 0);
    for (u64 i = 0; i < 128; i++)
        for (u64 j = 128; j < 256; j++)
            res[i * 256 + j] = 1.0;

    return res;
}

void split8(const HomEvaluator &eval, const CtxtTensor &vector,
            std::vector<Ciphertext> &packed_vector) {
    static Message row_mask1 = genFirstHalfRowMask(vector.getLogSlots());
    static Message row_mask2 = genSecondHalfRowMask(vector.getLogSlots());
    static Message first_mask = genFirstMask(vector.getLogSlots());
    static Message second_mask = genSecondMask(vector.getLogSlots());
    row_mask1.to(getCurrentCudaDevice());
    row_mask2.to(getCurrentCudaDevice());
    first_mask.to(getCurrentCudaDevice());
    second_mask.to(getCurrentCudaDevice());

    Ciphertext tmp(vector.get()), tmp2(vector.get());

    // vector.getBlockWidth();
    u64 block_width = 256;

    // vector.getWidth();
    u64 width = 128;
    // vector.getHeight()
    u64 height = 128;

    // complex_pack
    eval.leftRotate(vector.get(), block_width * 8, tmp);
    eval.multImagUnit(tmp, tmp);
    eval.add(vector.get(), tmp, tmp);

    Ciphertext origin(tmp);
    for (u64 i = 0; i < 8; i++) {
        packed_vector.push_back(origin);
        if (i != 0)
            eval.leftRotate(origin, i * block_width * 16, packed_vector[i]);

        // fill [0,blowidth]
        eval.mult(packed_vector[i], row_mask1, tmp);
        eval.mult(packed_vector[i], row_mask2, tmp2);

        eval.rightRotate(tmp, width, packed_vector[i]);
        eval.add(tmp, packed_vector[i], tmp);

        eval.rightRotate(tmp2, width, packed_vector[i]);
        eval.add(tmp2, packed_vector[i], tmp2);

        // eval.rightRotateReduce(tmp,(block_width-1),height,tmp);
        // eval.rightRotateReduce(tmp2,(block_width-1),height,tmp2);
        // below is faster than..
        for (u64 j = height / 2; j >= 1; j = j / 2) {
            eval.rightRotate(tmp, j * (block_width - 1), packed_vector[i]);
            eval.add(tmp, packed_vector[i], tmp);
        }
        for (u64 j = height / 2; j >= 1; j = j / 2) {
            eval.rightRotate(tmp2, j * (block_width - 1), packed_vector[i]);
            eval.add(tmp2, packed_vector[i], tmp2);
        }

        eval.mult(tmp, first_mask, tmp);
        eval.mult(tmp2, second_mask, tmp2);
        eval.add(tmp, tmp2, packed_vector[i]);
    }
}

Ciphertext repack16(const HomEvaluator &eval,
                    const std::vector<Ciphertext> &packed_vector) {
    Ciphertext res(packed_vector[0]);

    static Message mask = genRowMask(packed_vector[0].getLogSlots());

    // vector.getBlockWidth();
    u64 block_width = 256;
    // vector.getWidth();
    // u64 width = 128;
    // vector.getHegiht();
    //  u64 hegiht = 128;

    mask.to(getCurrentCudaDevice());
    for (u64 i = 0; i < packed_vector.size(); i++) {
        Ciphertext tmp(packed_vector[i]);
        eval.leftRotateReduce(packed_vector[i], 256, 128, tmp);
        if (i == 0)
            eval.mult(tmp, mask, res);
        else {
            eval.mult(tmp, mask, tmp);
            eval.rightRotate(tmp, 8 * i * block_width, tmp);
            eval.add(res, tmp, res);
        }
    }
    return res;
}

CtxtTensor
multCVec128Mat(const HomEvaluator &eval, const CtxtTensor &vector,
               const std::vector<CtxtTensor> &packed_mat,
               [[maybe_unused]] u64 target_level,
               [[maybe_unused]] MatrixTransformer &matrix_transformer) {
    CtxtTensor res(vector);
    std::vector<Ciphertext> tmp_vec;
    std::vector<Ciphertext> packed_vector(16, Ciphertext(eval.getContext()));
    eval.levelDown(res.get(), target_level + 4, res.get());

    split8(eval, res, tmp_vec);
    for (u64 i = 0; i < 8; i++) {
        eval.conjugate(tmp_vec[i], packed_vector[2 * i + 1]);
        eval.add(tmp_vec[i], packed_vector[2 * i + 1], packed_vector[2 * i]);
        eval.multInteger(packed_vector[2 * i + 1], 2, packed_vector[2 * i + 1]);
        eval.sub(packed_vector[2 * i + 1], packed_vector[2 * i],
                 packed_vector[2 * i + 1]);
        eval.multImagUnit(packed_vector[2 * i + 1], packed_vector[2 * i + 1]);
    }

    for (u64 i = 0; i < packed_mat.size(); i++) {
        CtxtTensor tmp(packed_mat[i]);
        eval.levelDown(tmp.get(), target_level + 3, tmp.get());
        tmp = matrix_transformer.diagonalToRow(eval, tmp);
        eval.mult(packed_vector[i], tmp.get(), packed_vector[i]);
    }

    res.get() = repack16(eval, packed_vector);

    return res;
}

void multVecPre(const HomEvaluator &eval, const CtxtTensor &vector,
                std::vector<Ciphertext> &tmp_vectors, u64 target_level) {
    CtxtTensor tmp(vector);

    eval.levelDown(vector.get(), target_level + 4, tmp.get());
    split8(eval, tmp, tmp_vectors);
    for (u64 i = 0; i < 8; i++) {
        eval.multInteger(tmp_vectors[i], 2, tmp_vectors[i]);
    }
}

void multVecPost(const HomEvaluator &eval, const PtxtTensor &mask,
                 std::vector<CtxtTensor>::const_iterator begin,
                 std::vector<CtxtTensor>::const_iterator end, CtxtTensor &res) {
    if (begin->get().getLevel() == 0)
        throw RuntimeException(
            "[multVecPost] The level of input ctxts should be >= 1.");

    const auto &context = eval.getContext();

    const size_t size = static_cast<size_t>(std::distance(begin, end));
    std::vector<Ciphertext> ctxts;
    ctxts.reserve(size);
    for (auto it = begin; it != end; ++it) {
        Ciphertext tmp(context);
        eval.leftRotateReduce(it->get(), 128, 256, tmp);
        // input ctxts have r_counter 1
        // rescale here as multWithoutRescale requires r_counter to be 0
        eval.rescale(tmp);
        eval.multWithoutRescale(tmp, mask.get(), tmp);
        ctxts.push_back(std::move(tmp));
    }

    // rot idx of ctxt[2 * i] : 256 * 8 * i (first half of 8 * i row)
    // rot idx of ctxt[2 * i + 1] : 256 * 8 * i + 128 (second half of 8 * i row)
    // NOTE : this impl accepts arbitrary size (!= 32).
    // TODO : optimize with rotSum, by adding more keys
    res.get() = ctxts[size - 2];
    for (int i = static_cast<int>(size) - 4; i >= 0; i -= 2) {
        eval.rightRotate(res.get(), 256 * 8, res.get());
        eval.add(res.get(), ctxts[static_cast<size_t>(i)], res.get());
    }

    auto res_odd = ctxts[size - 1];
    for (int i = static_cast<int>(size) - 3; i >= 0; i -= 2) {
        eval.rightRotate(res_odd, 256 * 8, res_odd);
        eval.add(res_odd, ctxts[static_cast<size_t>(i)], res_odd);
    }
    eval.rightRotate(res_odd, 128, res_odd);

    eval.add(res.get(), res_odd, res.get());

    // Finish hoisted part on multPVec128Mat
    Ciphertext res_conj(eval.getContext());
    eval.conjugate(res.get(), res_conj);
    eval.add(res.get(), res_conj, res.get());
    eval.rescale(res.get());
}

CtxtTensor
multPVec128Mat(const HomEvaluator &eval, const Ciphertext &vector,
               const PtxtTensor &weight, [[maybe_unused]] u64 target_level,
               [[maybe_unused]] MatrixTransformer &matrix_transformer) {
    Ciphertext res(eval.getContext());
    // 4096/16 -> 256  = 16 * 16
    eval.multWithoutRescale(vector, weight.get(), res);
    return CtxtTensor{1, 4096, res, 256};
}

CtxtTensor
multPVec128Mat(const HomEvaluator &eval, const Ciphertext &vector,
               const Message &weight, [[maybe_unused]] u64 target_level,
               [[maybe_unused]] MatrixTransformer &matrix_transformer) {
    Ciphertext res(eval.getContext());
    // 4096/16 -> 256  = 16 * 16
    EnDecoder endec(eval.getContext());
    auto ptxt = endec.encode(weight, vector.getLevel());
    eval.multWithoutRescale(vector, ptxt, res);
    return CtxtTensor{1, 4096, res, 256};
}

} // namespace HELLM
