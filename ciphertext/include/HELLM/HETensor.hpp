////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2024 Crypto Lab Inc. All rights reserved.               //
//                                                                            //
// This source code is protected under international copyright law.           //
// All rights reserved and protected by the copyright holders.                //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "HEaaN/HEaaN.hpp"

namespace HELLM {

class TileTensorShape {
public:
    TileTensorShape(HEaaN::i64 num_rows, HEaaN::i64 num_cols,
                    HEaaN::i64 block_width)
        : num_rows_(num_rows), num_cols_(num_cols), block_width_(block_width) {}

    TileTensorShape(HEaaN::i64 num_rows, HEaaN::i64 num_cols)
        : num_rows_(num_rows), num_cols_(num_cols) {
        int i = 1;
        while (num_cols > i)
            i = i << 1;
        block_width_ = i;
    }

    inline HEaaN::i64 getNumRows() const { return num_rows_; }
    inline HEaaN::i64 getNumCols() const { return num_cols_; }
    inline HEaaN::i64 getBlockWidth() const { return block_width_; }
    void shapeTranspose() { setShape(num_cols_, num_rows_); }
    void setShape(HEaaN::i64 num_rows, HEaaN::i64 num_cols) {
        num_rows_ = num_rows;
        num_cols_ = num_cols;
    }
    void setBlockWidth(HEaaN::i64 width) { block_width_ = width; }

private:
    HEaaN::i64 num_rows_;
    HEaaN::i64 num_cols_;
    HEaaN::i64 block_width_;
};

template <typename T> class HETensor {
public:
    HETensor(const HEaaN::Context &context, HEaaN::i64 height, HEaaN::i64 width)
        : shape_(height, width), data_{context} {}

    HETensor(const HEaaN::Context &context, HEaaN::i64 height, HEaaN::i64 width,
             HEaaN::i64 block_width)
        : shape_(height, width, block_width), data_{context} {}

    HETensor(const HEaaN::Context &context, TileTensorShape shape)
        : shape_(shape), data_{context} {}

    HETensor(HEaaN::i64 height, HEaaN::i64 width, T &data)
        : shape_(height, width), data_{data} {}

    HETensor(HEaaN::i64 height, HEaaN::i64 width, T &data,
             HEaaN::i64 block_width)
        : shape_(height, width, block_width), data_{data} {}

    HEaaN::i64 getHeight() const { return shape_.getNumRows(); }
    HEaaN::i64 getWidth() const { return shape_.getNumCols(); }
    HEaaN::i64 getBlockWidth() const { return shape_.getBlockWidth(); }
    HEaaN::u64 getLogSlots() const { return data_.getLogSlots(); }
    HEaaN::u64 getLevel() const { return data_.getLevel(); }
    void setShape(HEaaN::i64 h, HEaaN::i64 w) { shape_.setShape(h, w); }
    TileTensorShape getShape() const { return shape_; }
    void setBlockWidth(HEaaN::i64 width) { shape_.setBlockWidth(width); }

    T &get() { return data_; }
    const T &get() const { return data_; }

private:
    TileTensorShape shape_;
    T data_;
};

template class HETensor<HEaaN::Plaintext>;
template class HETensor<HEaaN::Ciphertext>;

using CtxtTensor = HETensor<HEaaN::Ciphertext>;
using PtxtTensor = HETensor<HEaaN::Plaintext>;
} // namespace HELLM
