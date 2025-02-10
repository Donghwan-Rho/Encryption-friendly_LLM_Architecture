////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2024 Crypto Lab Inc. All rights reserved.               //
//                                                                            //
// This source code is protected under international copyright law.           //
// All rights reserved and protected by the copyright holders.                //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "HELLM/LayerNorm.hpp"
#include "HELLM/LNRemezCoeff.hpp"

#include "HEaaN-math/tools/PolynomialEvaluator.hpp"

#include <ATen/core/TensorBody.h>
#include <stdexcept>
#include <vector>

#include <iostream>
#include <string>

namespace HELLM::LayerNorm {

using namespace HEaaN;
using namespace Math;

namespace {

void bootstrapIfNecessary(const Bootstrapper &btp, Ciphertext &ctxt,
                          const u64 depth) {
    if (ctxt.getLevel() < depth + btp.getMinLevelForBootstrap()) {
        btp.bootstrap(ctxt, ctxt);
    }
}

/* void bootstrapExtendedIfNecessary(const Bootstrapper &btp, Ciphertext &ctxt,
                                  const u64 depth) {
    if (ctxt.getLevel() < depth + btp.getMinLevelForBootstrap()) {
        btp.bootstrapExtended(ctxt, ctxt);
    }
} */

/* void approxInverseSqrtNewton(const HomEvaluator &eval, const Bootstrapper
&btp, const Ciphertext &ctxt, Ciphertext &ctxt_out, const Real initial, u64
num_iter) { Ciphertext ctxt_x(eval.getContext()); Ciphertext
tmp1(eval.getContext()); Ciphertext tmp2(eval.getContext());

    eval.mult(ctxt, -0.5, ctxt_x);

    for (u64 i = 0; i < num_iter; i++) {
        if (i == 0) {
            eval.mult(ctxt, -initial * initial, tmp1);
            eval.add(tmp1, 3, tmp1);
            eval.mult(tmp1, 0.5 * initial, ctxt_out);
        } else {
            bootstrapIfNecessary(btp, ctxt_out, 2);

            eval.mult(ctxt_out, 1.5, tmp1);
            eval.mult(ctxt_out, ctxt_x, tmp2);
            eval.square(ctxt_out, ctxt_out);
            eval.mult(ctxt_out, tmp2, ctxt_out);
            eval.add(ctxt_out, tmp1, ctxt_out);
        }
    }
}
 */
void approxInverseSqrtNewton_TS(const HomEvaluator &eval,
                                const Bootstrapper &btp, const Ciphertext &ctxt,
                                Ciphertext &ctxt_out,
                                const Ciphertext &ctxt_init, u64 num_iter) {
    Ciphertext ctxt_x(eval.getContext());
    Ciphertext tmp1(eval.getContext());
    Ciphertext tmp2(eval.getContext());

    eval.mult(ctxt, -0.5, ctxt_x); // 3

    ctxt_out = ctxt_init; // 4

    for (u64 i = 0; i < num_iter; i++) {

        if (ctxt_out.getLevel() < btp.getMinLevelForBootstrap() + 2)
            btp.bootstrap(ctxt_out, ctxt_out); // 12 12
        // shodul consider precision problem. extBTS -> BTS
        if (ctxt_x.getLevel() < ctxt_out.getLevel())
            btp.bootstrap(ctxt_x, ctxt_x); // 12 12

        eval.mult(ctxt_out, 1.5, tmp1);      // 11 8
        eval.mult(ctxt_out, ctxt_x, tmp2);   // 11 8
        eval.square(ctxt_out, ctxt_out);     // 11 8
        eval.mult(ctxt_out, tmp2, ctxt_out); // 10 7
        eval.add(ctxt_out, tmp1, ctxt_out);  // 10 7
    }
}

/* void approxInverseSqrtNewton(const HomEvaluator &eval, const Bootstrapper
&btp, const Ciphertext &ctxt, Ciphertext &ctxt_out, const Ciphertext &ctxt_init,
u64 num_iter) { Ciphertext ctxt_x(eval.getContext()); Ciphertext
tmp1(eval.getContext()); Ciphertext tmp2(eval.getContext());

    eval.mult(ctxt, -0.5, ctxt_x);

    ctxt_out = ctxt_init;
    for (u64 i = 0; i < num_iter; i++) {
        bootstrapExtendedIfNecessary(btp, ctxt_out, 3);

        eval.mult(ctxt_out, 1.5, tmp1);
        eval.mult(ctxt_out, ctxt_x, tmp2);
        eval.square(ctxt_out, ctxt_out);
        eval.mult(ctxt_out, tmp2, ctxt_out);
        eval.add(ctxt_out, tmp1, ctxt_out);
    }
}
 */
} // namespace

void approxInverseSqrtLN(const HomEvaluator &eval, const Bootstrapper &btp,
                         const Ciphertext &op, Ciphertext &res,
                         const int layer_n, const u64 num_iter) {

    const ChebyshevCoefficients &cheby_coeffs = LAYER_INVERSE_SQRT_127;

    Ciphertext ctxt_x = linearTransform(eval, btp, op, InputInterval(0, 1));

    // bootstrapIfNecessary(btp, ctxt_x, 1 + cheby_coeffs.level_cost);
    // Q. Why do we need one more depth?
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

    if (num_iter > 0)
        approxInverseSqrtNewton_TS(eval, btp, op, ctxt_y, ctxt_y, num_iter);

    res = ctxt_y;
}

} // namespace HELLM::LayerNorm
