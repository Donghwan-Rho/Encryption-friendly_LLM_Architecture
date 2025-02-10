////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2024 Crypto Lab Inc. All rights reserved.               //
//                                                                            //
// This source code is protected under international copyright law.           //
// All rights reserved and protected by the copyright holders.                //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <map>

#include "HELLM/HETensor.hpp"
#include "HEaaN/HomEvaluator.hpp"

namespace HELLM {

using namespace HEaaN;
// This class has three Message vector which are used for matrix transforms
// transpose, diagonal to column and diagonal to row.
class TransformMask {
public:
    std::map<i64, Message> &getDiagoanlMask() { return diagonal_mask_; }
    std::map<i64, Message> &getHorizontalMask() { return horizontal_mask_; }
    std::map<i64, Message> &getPackedHorizontalMask() {
        return packed_horizontal_mask_;
    }
    std::map<i64, Message> &getVerticalMask() { return vertical_mask_; }
    std::map<i64, Message> &getPackedVerticalMask() {
        return packed_vertical_mask_;
    }
    std::vector<Plaintext> &getRowMask() { return row_mask_; }
    std::vector<Plaintext> &getColMask() { return col_mask_; }

    void generateDiagoanlMask(const HomEvaluator &eval,
                              const TileTensorShape matrix_shape);
    void generateHorizontalMask(const HomEvaluator &eval,
                                const TileTensorShape matrix_shape);
    void generatePackedHorizontalMask(const HomEvaluator &eval,
                                      const TileTensorShape matrix_shape);
    void generateVerticalMask(const HomEvaluator &eval,
                              const TileTensorShape matrix_shape);
    void generatePackedVerticalMask(const HomEvaluator &eval,
                                    const TileTensorShape matrix_shape);
    void generateRowMask(const HomEvaluator &eval,
                         const TileTensorShape matrix_shape, u64 target_level);
    void generateColMask(const HomEvaluator &eval,
                         const TileTensorShape matrix_shape, u64 target_level);

private:
    std::map<i64, Message> diagonal_mask_;
    std::map<i64, Message> horizontal_mask_;
    std::map<i64, Message> vertical_mask_;
    std::map<i64, Message> packed_horizontal_mask_;
    std::map<i64, Message> packed_vertical_mask_;
    std::vector<Plaintext> row_mask_;
    std::vector<Plaintext> col_mask_;
};

struct TileTensorShapeCompare {
    bool operator()(const std::pair<int, TileTensorShape> &left,
                    const std::pair<int, TileTensorShape> &right) const {
        if (left.first != right.first)
            return left.first < right.first;
        if (left.second.getNumRows() != right.second.getNumRows())
            return left.second.getNumRows() < right.second.getNumRows();
        if (left.second.getNumCols() != right.second.getNumCols())
            return left.second.getNumCols() < right.second.getNumCols();
        return left.second.getBlockWidth() < right.second.getBlockWidth();
    }
};

// MatrixTransformer has a family of TransformMask.
// We compute the Message vector for transforms only once using this class.
class MatrixTransformer {
public:
    MatrixTransformer() = default;

    std::map<i64, Message> &
    getDiagonalMask(const HomEvaluator &eval,
                    const TileTensorShape &matrix_shape);
    std::map<i64, Message> &
    getHorizontalMask(const HomEvaluator &eval,
                      const TileTensorShape &matrix_shape);
    std::map<i64, Message> &
    getPackedHorizontalMask(const HomEvaluator &eval,
                            const TileTensorShape &matrix_shape);

    std::map<i64, Message> &
    getVerticalMask(const HomEvaluator &eval,
                    const TileTensorShape &matrix_shape);

    std::map<i64, Message> &
    getPackedVerticalMask(const HomEvaluator &eval,
                          const TileTensorShape &matrix_shape);

    std::vector<Plaintext> &getRowMask(const HomEvaluator &eval,
                                       const TileTensorShape &matrix_shape,
                                       u64 target_level);
    std::vector<Plaintext> &getColMask(const HomEvaluator &eval,
                                       const TileTensorShape &matrix_shape,
                                       u64 target_level);

    template <class T>
    HETensor<T> transpose(const HomEvaluator &eval, const HETensor<T> &op);
    template <class T>
    HETensor<T> diagonalToColumn(const HomEvaluator &eval,
                                 const HETensor<T> &op);

    CtxtTensor packedDiagonalToColumn(const HomEvaluator &eval,
                                      const CtxtTensor &op);
    template <class T>
    HETensor<T> diagonalToRow(const HomEvaluator &eval, const HETensor<T> &op);

    CtxtTensor packedDiagonalToRow(const HomEvaluator &eval,
                                   const CtxtTensor &op);

    void cleanUp() { masks_.clear(); }

private:
    TransformMask &getTransformMask(u64 target_level,
                                    const TileTensorShape &matrix_shape);

    std::map<std::pair<int, TileTensorShape>, TransformMask,
             TileTensorShapeCompare>
        masks_;
};

template <class T>
CtxtTensor multMatMat(const HomEvaluator &eval, const Bootstrapper &btp,
                      const CtxtTensor &op1, const HETensor<T> &op2,
                      u64 target_level, MatrixTransformer &matrix_transformer);

CtxtTensor multPackedMatMat(const HomEvaluator &eval, const Bootstrapper &btp,
                            const CtxtTensor &op1, const CtxtTensor &op2,
                            u64 target_level,
                            MatrixTransformer &matrix_transformer);

CtxtTensor multPackedMatMatPre(const HomEvaluator &eval,
                               const CtxtTensor &tensor_a, u64 target_level,
                               MatrixTransformer &matrix_transformer);

void multPackedMatMatPreRot(const HomEvaluator &eval,
                            const CtxtTensor &tensor_a,
                            std::vector<Ciphertext> &tmp, u64 target_level,
                            MatrixTransformer &matrix_transformer);

CtxtTensor multPackedMatMatPreRev(const HomEvaluator &eval,
                                  const CtxtTensor &tensor_a, u64 target_level,
                                  MatrixTransformer &matrix_transformer);

CtxtTensor multPackedMatMatCCReuse(const HomEvaluator &eval,
                                   const std::vector<Ciphertext> &tmp,
                                   const CtxtTensor &tensor_b, u64 target_level,
                                   MatrixTransformer &matrix_transformer);

PtxtTensor multMatMat(const HomEvaluator &eval, const Bootstrapper &btp,
                      const PtxtTensor &op1, const PtxtTensor &op2,
                      u64 target_level, MatrixTransformer &matrix_transformer);

void multMatMatPre(const HomEvaluator &eval, const CtxtTensor &tensor_a,
                   std::vector<Ciphertext> &tmp, u64 target_level,
                   MatrixTransformer &matrix_transformer);

CtxtTensor multMatMatReUse(const HomEvaluator &eval,
                           const std::vector<Ciphertext> &tmp,
                           const PtxtTensor &tensor_b, u64 target_level,
                           MatrixTransformer &matrix_transformer);

CtxtTensor multMatMatPreRev(const HomEvaluator &eval,
                            const CtxtTensor &tensor_a, u64 target_level,
                            MatrixTransformer &matrix_transformer);

CtxtTensor multMatMatCCReUse(const HomEvaluator &eval,
                             const std::vector<Ciphertext> &tmp,
                             const CtxtTensor &tensor_b, u64 target_level,
                             MatrixTransformer &matrix_transformer);

CtxtTensor multMatMatHighLow(const HomEvaluator &eval, const CtxtTensor &op1,
                             const CtxtTensor &op2, const u64 in_col_block,
                             u64 target_level,
                             MatrixTransformer &matrix_transformer);

CtxtTensor multMatMatLowHigh(const HomEvaluator &eval, const CtxtTensor &op1,
                             const CtxtTensor &op2, const u64 in_row_block,
                             u64 target_level,
                             MatrixTransformer &matrix_transformer);

CtxtTensor multMatMatLowLow(const HomEvaluator &eval, const CtxtTensor &op1,
                            const CtxtTensor &op2, const u64 in_col_block,
                            const u64 in_row_block, u64 target_level,
                            MatrixTransformer &matrix_transformer);

void multVecPre(const HomEvaluator &eval, const CtxtTensor &vector,
                std::vector<Ciphertext> &tmp_vectors, u64 target_level);

CtxtTensor multPVec128Mat(const HomEvaluator &eval,
                          const Ciphertext &tmp_vector,
                          const PtxtTensor &weight, u64 target_level,
                          MatrixTransformer &matrix_transformer);

CtxtTensor multPVec128Mat(const HomEvaluator &eval,
                          const Ciphertext &tmp_vector, const Message &weight,
                          u64 target_level,
                          MatrixTransformer &matrix_transformer);

CtxtTensor multCVec128Mat(const HomEvaluator &eval, const CtxtTensor &vector,
                          const std::vector<CtxtTensor> &mat, u64 target_level,
                          MatrixTransformer &matrix_transformer);

// Assume the input ctxts have r_counter 1
void multVecPost(const HomEvaluator &eval, const PtxtTensor &mask,
                 std::vector<CtxtTensor>::const_iterator begin,
                 std::vector<CtxtTensor>::const_iterator end, CtxtTensor &res);

} // namespace HELLM
