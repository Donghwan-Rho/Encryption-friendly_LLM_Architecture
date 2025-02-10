////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2024 Crypto Lab Inc. All rights reserved.               //
//                                                                            //
// This source code is protected under international copyright law.           //
// All rights reserved and protected by the copyright holders.                //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "HELLM/Exp.hpp"
#include "HELLM/HETensor.hpp"

#include <ATen/core/TensorBody.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace HELLM::Exp {

using namespace HEaaN;

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
}
 */
} // namespace

void exp_iter(const HomEvaluator &eval, const Bootstrapper &btp,
              const Ciphertext &op, Ciphertext &res, const int num_iter) {

    eval.mult(op, 1.0 / std::pow(2, num_iter), res);
    eval.add(res, 1, res);

    for (int i = 0; i < num_iter; i++) {
        if (res.getLevel() == btp.getMinLevelForBootstrap())
            btp.bootstrap(res, res);
        eval.square(res, res);
    }
}

void exp_iter_Parallel(const HomEvaluator &eval, const Bootstrapper &btp,
                       std::vector<HELLM::CtxtTensor> &op, const int num_iter) {

    for (u64 i = 0; i < op.size(); ++i) {
        eval.mult(op[i].get(), 1.0 / std::pow(2, num_iter), op[i].get());
        eval.add(op[i].get(), 1, op[i].get());
    }

    for (int i = 0; i < num_iter; i++) {
        for (u64 j = 0; j < op.size() / 2; ++j) {
            if (op[2 * j].getLevel() == btp.getMinLevelForBootstrap()) {
                eval.multImagUnit(op[2 * j + 1].get(), op[2 * j + 1].get());
                eval.add(op[2 * j].get(), op[2 * j + 1].get(), op[2 * j].get());
                btp.bootstrap(op[2 * j].get(), op[2 * j].get(),
                              op[2 * j + 1].get());
            }
            eval.square(op[2 * j].get(), op[2 * j].get());
            eval.square(op[2 * j + 1].get(), op[2 * j + 1].get());
        }
    }
}

Ciphertext approxDomainExtension(const HomEvaluator &eval,
                                 const Bootstrapper &btp,
                                 const Ciphertext &ctxt, const Real base_range,
                                 const Real extended_range,
                                 const Real domain_extension_rate) {
    const u64 target_level = (base_range > 1.0)
                                 ? 1 + btp.getMinLevelForBootstrap()
                                 : btp.getMinLevelForBootstrap();
    // n: domain extension.
    // L: domain extension rate.
    const u64 domain_extension_order =
        static_cast<u64>(std::ceil(std::log2(extended_range / base_range) /
                                   std::log2(domain_extension_rate)));
    i64 bound = static_cast<i64>(std::ceil(extended_range));
    Real div = 1.0 / (static_cast<Real>(bound) * static_cast<Real>(bound));

    const u64 level_cost = 2;

    Ciphertext ctxt_x{ctxt};
    Ciphertext x_stable(ctxt);
    Ciphertext tmp(eval.getContext());
    // input = 10
    for (u64 i = 0; i < domain_extension_order; i++) {
        if (ctxt_x.getLevel() < level_cost + target_level + 1) {
            eval.multWithoutRescale(x_stable, 0.5 / static_cast<Real>(bound),
                                    ctxt_x);
            eval.rescale(ctxt_x);
            btp.bootstrap(ctxt_x, ctxt_x);
            eval.multInteger(ctxt_x, bound, ctxt_x);
            eval.conjugate(ctxt_x, tmp);
            eval.add(ctxt_x, tmp, ctxt_x);
        }

        div *= domain_extension_rate * domain_extension_rate;

        Ciphertext x_copy(eval.getContext()), pow2(eval.getContext()),
            x_tmp(eval.getContext());

        x_copy = ctxt_x;
        x_tmp = x_copy;
        Real div_tmp = div;

        Real coeff = -4.0 * div_tmp / 27.0;

        eval.square(x_copy, pow2);
        eval.multWithoutRescale(x_tmp, coeff, tmp);
        eval.rescale(tmp);
        eval.mult(pow2, tmp, tmp);
        eval.add(ctxt_x, tmp, ctxt_x);
        eval.add(x_stable, tmp, x_stable);

        bound = static_cast<i64>(
            std::ceil(static_cast<Real>(bound) / domain_extension_rate));
    }

    return x_stable;
}

Ciphertext approxDomainExtensionInverse(const HomEvaluator &eval,
                                        const Ciphertext &ctxt,
                                        const u64 domain_extension_order,
                                        const Real domain_extension_rate) {
    Real coeff = (4.0 / 27.0) *
                 (1.0 - 1.0 / std::pow(domain_extension_rate,
                                       2 * domain_extension_order)) /
                 (1.0 - 1.0 / std::pow(domain_extension_rate, 2));
    Real coeff2 = coeff * (4.0 / 9.0) *
                  (1.0 - 1.0 / std::pow(domain_extension_rate,
                                        2 * (domain_extension_order + 1))) /
                  (1.0 - 1.0 / std::pow(domain_extension_rate, 4));
    Real eps = 0.1;

    Ciphertext ctxt_out(eval.getContext()), ctxt_tmp(eval.getContext()),
        pow2(eval.getContext());

    eval.square(ctxt, pow2);
    eval.square(pow2, ctxt_out);
    eval.negate(ctxt_out, ctxt_out);
    eval.mult(pow2, coeff2 / (coeff + coeff2 + eps), ctxt_tmp);
    eval.add(ctxt_out, ctxt_tmp, ctxt_out);
    eval.add(ctxt_out, coeff / (coeff + coeff2 + eps), ctxt_out);

    eval.mult(ctxt, coeff + coeff2 + eps, ctxt_tmp);
    eval.mult(ctxt_tmp, pow2, ctxt_tmp);
    eval.mult(ctxt_out, ctxt_tmp, ctxt_out);
    eval.add(ctxt_out, ctxt, ctxt_out);

    return ctxt_out;
}

void approxExpWide(const HomEvaluator &eval, const Bootstrapper &btp,
                   const Ciphertext &ctxt, Ciphertext &ctxt_out) {
    const u64 log_base_range = 5;
    const Real base_range = 1 << log_base_range;
    const Real domain_extension_rate = 2;
    const Real extended_range = 512;
    const u64 domain_extension_order =
        static_cast<u64>(std::ceil(std::log2(extended_range / base_range)));

    Ciphertext ctxt_x = ctxt;
    eval.mult(ctxt_x, 1.0 / base_range, ctxt_x); // requires 1

    ctxt_x =
        approxDomainExtension(eval, btp, ctxt_x, 1, extended_range / base_range,
                              domain_extension_rate); // requires 6

    bootstrapIfNecessary(btp, ctxt_x, 3);
    ctxt_x = approxDomainExtensionInverse(eval, ctxt_x, domain_extension_order,
                                          domain_extension_rate); // requires 3

    std::vector<Real> coeffs = {
        1.000000001945585,     0.9999999388932228,     0.499999859398392,
        0.16666798775227673,   0.04166711228500117,    0.008328336039797088,
        0.0013881473292497602, 0.00020492629147503022, 2.551385701569812e-05};

    Ciphertext pow2(eval.getContext()), pow4(eval.getContext()),
        tmp(eval.getContext());

    bootstrapIfNecessary(btp, ctxt_x, 3);
    eval.mult(ctxt_x, ctxt_x, pow2);
    eval.mult(pow2, pow2, pow4);
    eval.mult(pow4, coeffs[8], ctxt_out);

    eval.mult(ctxt_x, coeffs[7], tmp);

    eval.add(tmp, coeffs[6], tmp);
    eval.mult(tmp, pow2, tmp);
    eval.add(ctxt_out, tmp, ctxt_out);

    eval.mult(ctxt_x, coeffs[5], tmp);
    eval.add(tmp, coeffs[4], tmp);
    eval.add(ctxt_out, tmp, ctxt_out);
    eval.mult(ctxt_out, pow4, ctxt_out);

    eval.mult(ctxt_x, coeffs[3], tmp);
    eval.add(tmp, coeffs[2], tmp);
    eval.mult(tmp, pow2, tmp);
    eval.add(ctxt_out, tmp, ctxt_out);

    eval.mult(ctxt_x, coeffs[1], tmp);
    eval.add(ctxt_out, tmp, ctxt_out); // 5

    eval.add(ctxt_out, coeffs[0], ctxt_out); // 5

    bootstrapIfNecessary(btp, ctxt_out, log_base_range);

    for (u64 i = 0; i < log_base_range; i++) {
        eval.square(ctxt_out, ctxt_out);
    }
}

} // namespace HELLM::Exp
