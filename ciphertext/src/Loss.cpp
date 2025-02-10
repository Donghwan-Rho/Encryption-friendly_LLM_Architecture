////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2024 Crypto Lab Inc. All rights reserved.               //
//                                                                            //
// This source code is protected under international copyright law.           //
// All rights reserved and protected by the copyright holders.                //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "HELLM/Loss.hpp"
#include "HELLM/HEMMer.hpp"
#include "HELLM/LossRemezCoeff.hpp"
#include "HELLM/Softmax.hpp"

#include "HEaaN-math/tools/PolynomialEvaluator.hpp"

#include "torch/torch.h"
#include <ATen/core/TensorBody.h>
#include <stdexcept>
#include <vector>

#include <iostream>
#include <string>

namespace HELLM::Loss {

using namespace HEaaN;
using namespace Math;

namespace {

void bootstrapIfNecessary(const Bootstrapper &btp, Ciphertext &ctxt,
                          const u64 depth) {
    if (ctxt.getLevel() < depth + btp.getMinLevelForBootstrap()) {
        btp.bootstrap(ctxt, ctxt);
    }
}
} // namespace

// input [-2.0, 2.0]
void approxExp15(const HomEvaluator &eval, const Bootstrapper &btp,
                 const Ciphertext &op, Ciphertext &res) {

    const ChebyshevCoefficients &cheby_coeffs = EXP_COEFFS_15;

    Ciphertext ctxt_x =
        linearTransform(eval, btp, op, InputInterval(-2.0, 2.0));

    bootstrapIfNecessary(btp, ctxt_x, cheby_coeffs.level_cost);

    // TODO: put layer_n valeu into function.
    Ciphertext ctxt_y(eval.getContext());

    ctxt_y = evaluateChebyshevExpansion(eval, btp, ctxt_x, cheby_coeffs, 1.0);

    res = ctxt_y;
}

// input [-13.0, 1.0]
void approxExp15_SST2(const HomEvaluator &eval, const Bootstrapper &btp,
                      const Ciphertext &op, Ciphertext &res) {

    const ChebyshevCoefficients &cheby_coeffs = EXP_COEFFS_15_SST2;

    Ciphertext ctxt_x =
        linearTransform(eval, btp, op, InputInterval(-13.0, 1.0));

    bootstrapIfNecessary(btp, ctxt_x, cheby_coeffs.level_cost);

    // TODO: put layer_n valeu into function.
    Ciphertext ctxt_y(eval.getContext());

    ctxt_y = evaluateChebyshevExpansion(eval, btp, ctxt_x, cheby_coeffs, 1.0);

    res = ctxt_y;
}

// input [0.04, 1.0]
void approxInv63(const HomEvaluator &eval, const Bootstrapper &btp,
                 const Ciphertext &op, Ciphertext &res) {

    const ChebyshevCoefficients &cheby_coeffs = INV_COEFFS_63;

    Ciphertext ctxt_x =
        linearTransform(eval, btp, op, InputInterval(0.04, 1.0));

    bootstrapIfNecessary(btp, ctxt_x, cheby_coeffs.level_cost);

    // TODO: put layer_n valeu into function.
    Ciphertext ctxt_y(eval.getContext());

    ctxt_y = evaluateChebyshevExpansion(eval, btp, ctxt_x, cheby_coeffs, 1.0);

    res = ctxt_y;
}

// input [0.8, 3.0]
void approxInv15_SST2(const HomEvaluator &eval, const Bootstrapper &btp,
                      const Ciphertext &op, Ciphertext &res) {

    const ChebyshevCoefficients &cheby_coeffs = INV_COEFFS_15_SST2;

    Ciphertext ctxt_x = linearTransform(eval, btp, op, InputInterval(0.8, 3.0));

    bootstrapIfNecessary(btp, ctxt_x, cheby_coeffs.level_cost);

    // TODO: put layer_n valeu into function.
    Ciphertext ctxt_y(eval.getContext());

    ctxt_y = evaluateChebyshevExpansion(eval, btp, ctxt_x, cheby_coeffs, 1.0);

    res = ctxt_y;
}

// input [-6, 6]
void approxMax(const HomEvaluator &eval, const Bootstrapper &btp,
               const Ciphertext &op, Ciphertext &res) {
    Ciphertext tmp{op}, comp{op};

    res = op;
    auto bound = 6.0 / 12;

    eval.add(res, bound, res);

    eval.leftRotate(res, 256, tmp);
    eval.sub(tmp, res, tmp);
    HELLM::Softmax::approxSign(eval, btp, tmp, comp, 1, 1, 0.5); // 3
    eval.add(comp, 0.5, comp);
    btp.bootstrap(comp, comp); // 3 -> 12
    eval.mult(tmp, comp, tmp); // 10 -> 9
    eval.add(res, tmp, res);   // 9

    eval.sub(res, bound, res); // 9
}

} // namespace HELLM::Loss
