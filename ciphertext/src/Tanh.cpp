////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2024 Crypto Lab Inc. All rights reserved.               //
//                                                                            //
// This source code is protected under international copyright law.           //
// All rights reserved and protected by the copyright holders.                //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "HELLM/Tanh.hpp"
#include "HELLM/Softmax.hpp"
#include "HELLM/TanhRemezCoeff.hpp"

#include "HEaaN-math/tools/PolynomialEvaluator.hpp"

#include <ATen/core/TensorBody.h>
#include <stdexcept>
#include <vector>

#include <iostream>
#include <string>

namespace HELLM::Tanh {

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

// input [-5.0, 5.0]
void approxTanh(const HomEvaluator &eval, const Bootstrapper &btp,
                const Ciphertext &op, Ciphertext &res, const int layer_n) {

    const ChebyshevCoefficients &cheby_coeffs = TANH_COEFFS_63;

    Ciphertext ctxt_x =
        linearTransform(eval, btp, op, InputInterval(-5.0, 5.0));

    bootstrapIfNecessary(btp, ctxt_x, cheby_coeffs.level_cost);

    // TODO: put layer_n valeu into function.
    Ciphertext ctxt_y(eval.getContext());

    if (layer_n == 0) {
        ctxt_y =
            evaluateChebyshevExpansion(eval, btp, ctxt_x, cheby_coeffs, 1.0);
    } else {
        ctxt_y =
            evaluateChebyshevExpansion(eval, btp, ctxt_x, cheby_coeffs, 1.0);
    }

    res = ctxt_y;
}

// input [-12.0, 12.0]
void approxTanh_wide_12(const HomEvaluator &eval, const Bootstrapper &btp,
                        const Ciphertext &op, Ciphertext &res,
                        const int layer_n) {

    const ChebyshevCoefficients &cheby_coeffs = TANH_COEFFS12_127;

    Ciphertext ctxt_x =
        linearTransform(eval, btp, op, InputInterval(-12.0, 12.0));

    bootstrapIfNecessary(btp, ctxt_x, cheby_coeffs.level_cost);

    // TODO: put layer_n valeu into function.
    Ciphertext ctxt_y(eval.getContext());

    if (layer_n == 0) {
        ctxt_y =
            evaluateChebyshevExpansion(eval, btp, ctxt_x, cheby_coeffs, 1.0);
    } else {
        ctxt_y =
            evaluateChebyshevExpansion(eval, btp, ctxt_x, cheby_coeffs, 1.0);
    }

    res = ctxt_y;
}

// input [-16.0, 16.0]
void approxTanh_wide_16(const HomEvaluator &eval, const Bootstrapper &btp,
                        const Ciphertext &op, Ciphertext &res,
                        const int layer_n) {

    const ChebyshevCoefficients &cheby_coeffs = TANH_COEFFS16_127;

    Ciphertext ctxt_x =
        linearTransform(eval, btp, op, InputInterval(-16.0, 16.0));

    bootstrapIfNecessary(btp, ctxt_x, cheby_coeffs.level_cost);

    // TODO: put layer_n valeu into function.
    Ciphertext ctxt_y(eval.getContext());

    if (layer_n == 0) {
        ctxt_y =
            evaluateChebyshevExpansion(eval, btp, ctxt_x, cheby_coeffs, 1.0);
    } else {
        ctxt_y =
            evaluateChebyshevExpansion(eval, btp, ctxt_x, cheby_coeffs, 1.0);
    }

    res = ctxt_y;
}

// input [-12.0, 12.0]
void approxTanh_wide(const HomEvaluator &eval, const Bootstrapper &btp,
                     const Ciphertext &op, Ciphertext &res, const int layer_n) {

    const ChebyshevCoefficients &cheby_coeffs = TANH_COEFFS_63_WIDE;

    Ciphertext ctxt_x =
        linearTransform(eval, btp, op, InputInterval(-12.0, 12.0));

    bootstrapIfNecessary(btp, ctxt_x, cheby_coeffs.level_cost);

    // TODO: put layer_n valeu into function.
    Ciphertext ctxt_y(eval.getContext());

    if (layer_n == 0) {
        ctxt_y =
            evaluateChebyshevExpansion(eval, btp, ctxt_x, cheby_coeffs, 1.0);
    } else {
        ctxt_y =
            evaluateChebyshevExpansion(eval, btp, ctxt_x, cheby_coeffs, 1.0);
    }

    res = ctxt_y;
}

} // namespace HELLM::Tanh
