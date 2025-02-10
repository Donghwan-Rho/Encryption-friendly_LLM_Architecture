////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2024 Crypto Lab Inc. All rights reserved.               //
//                                                                            //
// This source code is protected under international copyright law.           //
// All rights reserved and protected by the copyright holders.                //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "HELLM/Softmax.hpp"

#include "HELLM/HETensor.hpp"
#include "HELLM/SoftmaxRemezCoeff.hpp"
#include "HEaaN-math/tools/PolynomialEvaluator.hpp"
#include "HEaaN/Ciphertext.hpp"

#include <ATen/core/TensorBody.h>
#include <stdexcept>
#include <vector>

#include <iostream>
#include <string>

#define UNUSED(x) (void)(x)

namespace HELLM::Softmax {

using namespace HEaaN;
using namespace Math;

namespace {

void bootstrapIfNecessary(const Bootstrapper &btp, Ciphertext &op,
                          const u64 depth) {
    if (op.getLevel() < depth + btp.getMinLevelForBootstrap()) {
        btp.bootstrap(op, op);
    }
}

} // anonymous namespace

////////////////////
////// HETAL ///////
////////////////////

// for BERT.

Real degreeThreePolySolver(const std::vector<Real> &coeffs) {
    const Real coeff_a = coeffs[2] / coeffs[3], coeff_b = coeffs[1] / coeffs[3],
               coeff_c = coeffs[0] / coeffs[3];
    Real mid_p = coeff_b - coeff_a * coeff_a / 3,
         mid_q = coeff_c + coeff_a * (2 * coeff_a * coeff_a - 9 * coeff_b) / 27;
    Real determine = mid_q * mid_q / 4 + mid_p * mid_p * mid_p / 27;
    Real sol;
    if (determine > 0) {
        Real root_d = std::sqrt(determine);
        Real y_1 = root_d - mid_q / 2, y_2 = -root_d - mid_q / 2;
        if (y_1 > 0)
            y_1 = std::pow(y_1, 1.0 / 3);
        else
            y_1 = -std::pow(-y_1, 1.0 / 3);
        if (y_2 > 0)
            y_2 = std::pow(y_2, 1.0 / 3);
        else
            y_2 = -std::pow(-y_2, 1.0 / 3);
        sol = y_1 + y_2;
    } else {
        Real scale = std::sqrt(-determine + mid_q * mid_q / 4);
        Real cos_value = std::cos((1.0 / 3) * std::acos(-mid_q / (2 * scale)));
        sol = 2 * std::pow(scale, 1.0 / 3) * cos_value;
    }
    sol -= coeff_a / 3;
    return sol;
}

void computeDegreeSevenOddPolyWithSol(const HomEvaluator &eval,
                                      const std::vector<Real> &coeff,
                                      const Real sol, const Ciphertext &ctx,
                                      Ciphertext &res, const Real scale) {
    Ciphertext tmp1(eval.getContext()), tmp2(eval.getContext());

    eval.square(ctx, tmp1);
    eval.add(tmp1, (coeff[2] + sol) / 2, tmp2);
    eval.square(tmp2, tmp2);
    eval.add(tmp2, -coeff[0] / sol - (coeff[2] + sol) * (coeff[2] + sol) / 4,
             tmp2);
    eval.add(tmp1, -sol, tmp1);
    eval.mult(ctx, coeff[3] * scale, res);
    eval.mult(res, tmp1, res);
    eval.mult(res, tmp2, res);
}

Ciphertext approxDomainExtension(const HomEvaluator &eval,
                                 const Bootstrapper &btp,
                                 const Ciphertext &ctxt, const Real base_range,
                                 const Real extended_range,
                                 const Real domain_extension_rate) {
    const u64 target_level = (base_range > 1.0)
                                 ? 1 + btp.getMinLevelForBootstrap()
                                 : btp.getMinLevelForBootstrap();
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

// generate (scale, scale , ... , scale , 0 , ... ,0)
// the number of non-zero elemensts is num_class
// will be updated to multInteger case.
Message genColumnMask(const u64 log_slots, const u64 num_class,
                      const Real scale) {
    Message mask(log_slots, 0);
    /* for (u64 j = 0 ; j < 256 ; j = j + 128){
        for (u64 i = 0; i < num_class; i++) {
            mask[i*256 + j] = scale;
        }
    } */
    for (u64 j = 0; j < static_cast<u64>(std::pow(2, log_slots));
         j = j + num_class) {
        mask[j] = scale;
    }

    return mask;
}

void approxSign(const HomEvaluator &eval, const Bootstrapper &btp,
                const Ciphertext &ctxt, Ciphertext &ctxt_out,
                const u64 num_iter_g, const u64 num_iter_f, const Real scale) {
    static const std::vector<Real> COEFFS_F = {-7.0, 7.0, -4.2, 1.0};
    static const std::vector<Real> COEFFS_G = {-4589.0 / 12860, 16577.0 / 12860,
                                               -25614.0 / 12860, 1.0};

    static const u64 ONE_ITER_COST = 3; // level consumed in each iteration
    static const Real SOL_F = degreeThreePolySolver(COEFFS_F),
                      SCALE_F = -5.0 / 16;
    static const Real SOL_G = degreeThreePolySolver(COEFFS_G),
                      SCALE_G = -12860.0 / 1024;
    Ciphertext sign{ctxt};
    Ciphertext tmp(eval.getContext());

    for (u64 i = 0; i < num_iter_g; i++) {
        if (sign.getLevel() < ONE_ITER_COST + btp.getMinLevelForBootstrap()) {
            btp.bootstrap(sign, sign);
        }
        computeDegreeSevenOddPolyWithSol(eval, COEFFS_G, SOL_G, sign, sign,
                                         SCALE_G);
    }

    for (u64 i = 0; i < num_iter_f; i++) {
        if (sign.getLevel() < ONE_ITER_COST + btp.getMinLevelForBootstrap()) {
            btp.bootstrap(sign, sign);
        }
        if (i == num_iter_f - 1) {
            computeDegreeSevenOddPolyWithSol(eval, COEFFS_F, SOL_F, sign,
                                             ctxt_out, SCALE_F * scale);
        } else {
            computeDegreeSevenOddPolyWithSol(eval, COEFFS_F, SOL_F, sign, sign,
                                             SCALE_F);
        }
    }
}

void approxMax(const HomEvaluator &eval, const Bootstrapper &btp,
               const Ciphertext &ctxt, Ciphertext &ctxt_out,
               const int num_data_par, const int num_group) {
    Ciphertext tmp(eval.getContext()), comp(eval.getContext());
    ctxt_out = ctxt;
    for (u64 i = 1; i < static_cast<u64>(num_data_par); i <<= 1) {
        eval.leftRotate(ctxt_out, i * static_cast<u64>(num_group), tmp);
        eval.sub(tmp, ctxt_out, tmp);
        // 2048: 3,2
        // 1024: 2,2
        // 512: 2,2
        // 256: 3,1
        // 128: 2,2
        approxSign(eval, btp, tmp, comp, 2, 2, 0.5);
        bootstrapIfNecessary(btp, comp, 1);
        eval.add(comp, 0.5, comp);
        eval.mult(tmp, comp, tmp);
        eval.add(ctxt_out, tmp, ctxt_out);
        bootstrapIfNecessary(btp, ctxt_out, 1);
    }
}

void approxMax_Parallel(const HomEvaluator &eval, const Bootstrapper &btp,
                        const std::vector<Ciphertext> &ctxt,
                        Ciphertext &ctxt_out, const int num_data_par,
                        const int num_group) {
    if (num_data_par == -5 && num_group == -1) {
        std::cout << "Bye" << std::endl;
    }
    const u64 num_ctxt = ctxt.size();
    Ciphertext tmp(eval.getContext()), comp(eval.getContext());
    Ciphertext tmp_res(eval.getContext());
    for (u64 i = 1; i < num_ctxt; i++) {
        if (i == 1) {
            eval.sub(ctxt[0], ctxt[1], tmp);
            approxSign(eval, btp, tmp, comp, 2, 2, 0.5);
            bootstrapIfNecessary(btp, comp, 1);
            eval.add(comp, 0.5, comp);
            eval.mult(tmp, comp, tmp);
            eval.add(ctxt[1], tmp, tmp_res);
            bootstrapIfNecessary(btp, tmp_res, 1);
        } else {
            eval.sub(tmp_res, ctxt[i], tmp);
            approxSign(eval, btp, tmp, comp, 2, 2, 0.5);
            bootstrapIfNecessary(btp, comp, 1);
            eval.add(comp, 0.5, comp);
            eval.mult(tmp, comp, tmp);
            eval.add(ctxt[i], tmp, tmp_res);
            bootstrapIfNecessary(btp, tmp_res, 1);
        }
    }
    ctxt_out = tmp_res;
}

void approxExpWide(const HomEvaluator &eval, const Bootstrapper &btp,
                   const Ciphertext &ctxt, Ciphertext &ctxt_out) {
    const u64 log_base_range = 5;
    const Real base_range = 1 << log_base_range;
    const Real domain_extension_rate = 2;
    const Real extended_range = 1024;
    const u64 domain_extension_order =
        static_cast<u64>(std::ceil(std::log2(extended_range / base_range)));

    Ciphertext ctxt_x = ctxt;
    eval.mult(ctxt_x, 1.0 / base_range, ctxt_x);

    ctxt_x =
        approxDomainExtension(eval, btp, ctxt_x, 1, extended_range / base_range,
                              domain_extension_rate);

    bootstrapIfNecessary(btp, ctxt_x, 3);
    ctxt_x = approxDomainExtensionInverse(eval, ctxt_x, domain_extension_order,
                                          domain_extension_rate);

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
    eval.add(ctxt_out, tmp, ctxt_out);

    btp.bootstrap(ctxt_out, ctxt_out);
    eval.add(ctxt_out, coeffs[0], ctxt_out);

    for (u64 i = 0; i < log_base_range; i++) {
        eval.square(ctxt_out, ctxt_out);
    }
}

// update to the version of decreasing of log_slots
void approxInv(const HomEvaluator &eval, const Bootstrapper &btp,
               const Ciphertext &ctxt, Ciphertext &ctxt_out, const Real initial,
               const u64 num_iter, const u64 num_log_slots) {
    const u64 one_iter_cost{1};
    const u64 num_slots = ctxt.getLogSlots();
    Ciphertext ctxt_x(eval.getContext()), ctxt_y(eval.getContext());
    Ciphertext ctxt_z(eval.getContext()), ctxt_tmp(eval.getContext());

    if (ctxt.getLevel() < btp.getLevelAfterFullSlotBootstrap()) {
        ctxt_x = ctxt;
        ctxt_x.setLogSlots(num_log_slots);
        btp.bootstrapExtended(ctxt_x, ctxt_x);
        ctxt_x.setLogSlots(num_slots);
    } else {
        ctxt_x = ctxt;
    }

    eval.mult(ctxt_x, initial, ctxt_z);
    eval.negate(ctxt_z, ctxt_tmp);
    eval.add(ctxt_tmp, 2.0, ctxt_tmp);
    eval.mult(ctxt_tmp, initial, ctxt_y);

    for (u64 iter = 1; iter < num_iter; iter++) {
        if (ctxt_y.getLevel() < one_iter_cost + btp.getMinLevelForBootstrap()) {
            ctxt_y.setLogSlots(num_log_slots);
            btp.bootstrap(ctxt_y, ctxt_y);
            ctxt_y.setLogSlots(num_slots);
            eval.mult(ctxt_x, ctxt_y, ctxt_z);
        } else {
            eval.mult(ctxt_z, ctxt_tmp, ctxt_z);
        }

        eval.negate(ctxt_z, ctxt_tmp); // tmp = 2 - z_n
        eval.add(ctxt_tmp, 2.0, ctxt_tmp);
        eval.mult(ctxt_y, ctxt_tmp, ctxt_y);
    }

    ctxt_out = ctxt_y;
}

void approxSoftmaxWide_Parallel(const HomEvaluator &eval,
                                const Bootstrapper &btp, const Decryptor &dec,
                                const SecretKey &sk,
                                std::vector<CtxtTensor> &ctxt,
                                std::vector<CtxtTensor> &ctxt_out,
                                const u64 num_data, const u64 M,
                                const int layer_n) {
    UNUSED(dec);
    UNUSED(sk);
    // constexpr Real MAXR = 128;
    const u64 log_slots = 15;
    const Device device = ctxt[0].get().getDevice();
    const u64 num_ctxt = ctxt.size();
    // Assumption: all inputs are packed into one ciphertext.
    // const int num_data_par = static_cast<int>(std::pow(2,log_slots) /
    // num_data); const int num_data_par = static_cast<int>(num_data /
    // num_ctxt); // num_data_par = the number of partial data
    // in one ciphertext
    // const u64 num_group = static_cast<u64>(std::pow(2,log_slots) /
    // num_data_par); // num_group = the number of
    //  data packed in one ciphertext.
    int log_num_slots = 15;

    // intialize
    std::vector<Ciphertext> ctxt_max;
    ctxt_max.reserve(num_ctxt);
    for (u64 i = 0; i < num_ctxt; i++) {
        Ciphertext tmp(eval.getContext());
        ctxt_max.emplace_back(tmp);
    }

    // 1. subtract approx max
    for (u64 i = 0; i < num_ctxt; i++)
        eval.mult(ctxt[i].get(), 1.0 / static_cast<double>(M),
                  ctxt_max[i]); // 11

    std::vector<Ciphertext> max;
    max.reserve(num_ctxt);
    // initialization
    for (u64 i = 0; i < num_ctxt; ++i) {
        max.emplace_back(ctxt_max[i]); // 11
    }

    // 11 -> 5
    for (u64 i = 0; i < num_ctxt; ++i) {
        approxMax(eval, btp, ctxt_max[i], max[i], static_cast<int>(num_data),
                  1);
    }

    // timer.start("getting max ");
    // timer.end();
    // bootstrapIfNecessary(btp, max, 2);

    for (u64 i = 0; i < num_ctxt; ++i) {
        // eval.multImagUnit(max[2*i+1], max[2*i+1]);
        // eval.add(max[2*i], max[2*i+1], max[2*i]);
        btp.bootstrap(max[i], max[i]);
    }

    Message max_mask =
        genColumnMask(log_slots, num_data, static_cast<double>(M));
    max_mask.to(device);
    for (u64 i = 0; i < num_ctxt; ++i) {
        eval.mult(max[i], max_mask, max[i]); // (max_1, 0 , ... , max_2 ,0 , ...
                                             // , max_(group) , 0 ,... 0) //11
    }

    /* Message dmsg(log_slots, 0);
    dec.decrypt(max[0], sk, dmsg);  */

    // std::cout << "after max" << std::endl;
    /* Message dmsg(log_slots, 0);
    dec.decrypt(max[0], sk, dmsg);
    dmsg.to(DeviceType::CPU);
    for (u64 i = 0 ; i < 4 ; ++i) {
        std::cout << dmsg[256*i].real() << ", ";
    }
    std::cout << std::endl;
    dmsg.to(device);
 */
    Ciphertext tmp(eval.getContext());
    for (u64 j = 0; j < num_ctxt; ++j) {
        for (u64 i = 1; i < num_data; i <<= 1) {
            eval.rightRotate(max[j], i, tmp);
            eval.add(max[j], tmp, max[j]);
        }

        eval.sub(ctxt[j].get(), max[j], ctxt_out[j].get());
        // 11 -> 7
        approxExpWide(eval, btp, ctxt_out[j].get(), ctxt_out[j].get());
    }

    /* std::cout << "after exp" << std::endl;
    dec.decrypt(ctxt_out[0].get(), sk, dmsg);
    dmsg.to(DeviceType::CPU);
    for (u64 i = 0 ; i < 8 ; ++i) {
        std::cout << dmsg[i*256].real() << ", ";
    }
    std::cout << std::endl;
    dmsg.to(device);
 */

    // 3. inverse
    Message inv_mask = genColumnMask(
        log_slots, num_data, 1.0); // generate (1, 0,..., 1, 0,... , 1,0,.. 0)
    inv_mask.to(device);
    std::vector<Ciphertext> ctxt_inv;
    ctxt_inv.reserve(num_ctxt);
    // init
    for (u64 i = 0; i < num_ctxt; ++i) {
        ctxt_inv.emplace_back(ctxt_out[i].get());

        for (u64 j = 1; j < num_data; j <<= 1) {
            eval.leftRotate(ctxt_inv[i], j, tmp);
            eval.add(ctxt_inv[i], tmp, ctxt_inv[i]);
        }

        eval.mult(ctxt_inv[i], inv_mask, ctxt_inv[i]);

        for (u64 j = 1; j < num_data; j <<= 1) {
            eval.rightRotate(ctxt_inv[i], j, tmp);
            eval.add(ctxt_inv[i], tmp, ctxt_inv[i]);
        }
    }

    /*  std::cout << "inv input " << std::endl;
     dec.decrypt(ctxt_inv[0], sk, dmsg);
     dmsg.to(DeviceType::CPU);
     for (u64 i = 0 ; i < 8 ; ++i) {
         std::cout << dmsg[256*i].real() << ", ";
     }
     std::cout << std::endl; */

    for (u64 i = 0; i < num_ctxt; ++i) {

        // approxInv(eval, btp, ctxt_inv[i], ctxt_inv[i],
        //         1/static_cast<double>(num_data) , 12,
        //         static_cast<u64>(log_num_slots));

        if (layer_n == 0) {
            // output: 5
            // 128: 12
            // 256~ :14
            approxInv(eval, btp, ctxt_inv[i], ctxt_inv[i],
                      1 / static_cast<double>(num_data), 12,
                      static_cast<u64>(log_num_slots));

        } else {
            // eval.mult(ctxt_inv[i], 100, ctxt_inv[i]);
            approxInv(eval, btp, ctxt_inv[i], ctxt_inv[i],
                      1 / static_cast<double>(num_data), 13,
                      static_cast<u64>(log_num_slots));
            // eval.mult(ctxt_inv[i], 100, ctxt_inv[i]);
        }

        /* if (i == 0) {
            std::cout << "inv output " << std::endl;
            dec.decrypt(ctxt_inv[0], sk, dmsg);
            dmsg.to(DeviceType::CPU);
            for (u64 k = 0 ; k < 8 ; ++k) {
                std::cout << dmsg[256*k].real() << ", ";
            }
            std::cout << std::endl;
        } */

        eval.mult(ctxt_out[i].get(), ctxt_inv[i], ctxt_out[i].get());
    }
}

//////////////////
////// BERT //////
//////////////////

Ciphertext linearTransform_exp(const HomEvaluator &eval, const Ciphertext &ctxt,
                               const int k, const int N) {
    const Real left_end = -8.0;
    const Real right_end = 0.0;

    Ciphertext res{ctxt};

    eval.sub(res, pow(2, k - 1) * log(N), res);
    eval.sub(res, (right_end + left_end) * pow(2, k - 1), res);
    eval.mult(res, 1.0 / (pow(2, k - 1) * (right_end - left_end)), res);

    return res;
}

Message generateMask_V2(const u64 log_slots, const u64 num_data) {

    Message mask(log_slots, 0.0);

    for (u64 i = 0; i < num_data; i++) {
        mask[i * 256] = 1.0;
        mask[i * 256 + 128] = 1.0;
    }

    return mask;
}

Message generateMask_garbage(const u64 log_slots, const u64 num_data) {

    Message mask(log_slots, 0.0);

    for (u64 i = 0; i < num_data; i++) {
        for (u64 j = 96; j < 128; ++j) {
            mask[i * 256 + j] = 1.0;
            mask[i * 256 + 128 + j] = 1.0;
        }
    }

    return mask;
}

void bootstrap_02(const HomEvaluator &eval, const Bootstrapper &btp,
                  Ciphertext &op) {

    eval.add(op, -1, op);
    btp.bootstrap(op, op);
    eval.add(op, 1, op);
}

void bootstrap2_02(const HomEvaluator &eval, const Bootstrapper &btp,
                   Ciphertext &op1, Ciphertext &op2) {

    eval.add(op1, -1, op1);
    eval.add(op2, -1, op2);

    eval.multImagUnit(op2, op2);
    eval.add(op1, op2, op1);
    btp.bootstrap(op1, op1, op2);

    eval.add(op1, 1, op1);
    eval.add(op2, 1, op2);
}

void rotateSum_masking(const HomEvaluator &eval, const Ciphertext &op,
                       Ciphertext &res, const Message &mask,
                       const u64 num_data) {
    Ciphertext tmp(eval.getContext());

    res = op;
    for (u64 i = 1; i < num_data; i <<= 1) {
        eval.leftRotate(res, i, tmp);
        eval.add(res, tmp, res);
    }

    eval.mult(res, mask, res);

    for (u64 i = 1; i < num_data; i <<= 1) {
        eval.rightRotate(res, i, tmp);
        eval.add(res, tmp, res);
    }
}

void rotateSum_masking_first(const HomEvaluator &eval, const Ciphertext &op,
                             Ciphertext &res, const Message &mask,
                             const u64 num_data) {
    Ciphertext tmp(eval.getContext());

    res = op;
    for (u64 i = 1; i < num_data; i <<= 1) {
        eval.leftRotate(res, i, tmp);
        eval.add(res, tmp, res);
    }
    eval.mult(res, mask, res);
}

void rotateSum_masking_after(const HomEvaluator &eval, const Ciphertext &op,
                             Ciphertext &res, const Message &mask,
                             const u64 num_data) {
    Ciphertext tmp(eval.getContext());

    res = op;
    eval.mult(res, mask, res);
    for (u64 i = 1; i < num_data; i <<= 1) {
        eval.rightRotate(res, i, tmp);
        eval.add(res, tmp, res);
    }
}

void Softmax_UNI_approxExp_BSGS(const HomEvaluator &eval,
                                const Bootstrapper &btp, const Ciphertext &op,
                                Ciphertext &res, const int k, const int N) {

    const ChebyshevCoefficients &cheby_coeffs = EXPONENTIAL_UNI_15_BSGS;

    Ciphertext ctxt_x = linearTransform_exp(eval, op, k, N);

    res = evaluateChebyshevExpansion(eval, btp, ctxt_x, cheby_coeffs, 1.0);
}

void Softmax_UNI_approxInverseSqrt_63_BSGS(const HomEvaluator &eval,
                                           const Bootstrapper &btp,
                                           const Ciphertext &op,
                                           Ciphertext &res) {
    const ChebyshevCoefficients &cheby_coeffs = INVERSE_SQRT_UNI_FIRST_63_BSGS;

    Ciphertext ctxt_x = linearTransform(eval, btp, op, InputInterval(0, 1));

    bootstrapIfNecessary(btp, ctxt_x, 1 + cheby_coeffs.level_cost);

    res = evaluateChebyshevExpansion(eval, btp, ctxt_x, cheby_coeffs, 1.0);
}

void Softmax_UNI128_Scaled_approxInverseSqrt_15_BSGS_GH(
    const HomEvaluator &eval, const Bootstrapper &btp, double scale,
    const Ciphertext &op, Ciphertext &res) {
    const ChebyshevCoefficients &cheby_coeffs = INVERSE_SQRT_UNI128_15_BSGS_GH;

    Ciphertext ctxt_x = linearTransform(eval, btp, op, InputInterval(0, 1.0));

    bootstrapIfNecessary(btp, ctxt_x, 1 + cheby_coeffs.level_cost);

    res = evaluateChebyshevExpansion(eval, btp, ctxt_x, cheby_coeffs, scale);
}

void Softmax_UNI128_LAST_approxInverseSqrt_31_BSGS(const HomEvaluator &eval,
                                                   const Bootstrapper &btp,
                                                   const Ciphertext &op,
                                                   Ciphertext &res) {
    const ChebyshevCoefficients &cheby_coeffs =
        INVERSE_SQRT_UNI128_LAST_31_BSGS;

    Ciphertext ctxt_x = linearTransform(eval, btp, op, InputInterval(0, 1.238));

    bootstrapIfNecessary(btp, ctxt_x, 1 + cheby_coeffs.level_cost);

    res = evaluateChebyshevExpansion(eval, btp, ctxt_x, cheby_coeffs, 1.0);
}

// N = 128, range= [-256,256]
void Softmax_128_512(const HomEvaluator &eval, const Bootstrapper &btp,
                     const Ciphertext &op, Ciphertext &res) {

    const int num_data = 128;
    const int k = 7;
    const int N = 128;
    const int M = 512;
    const Device device = op.getDevice();
    // const u64 log_slots = op.getLogSlots();
    const int scale_for_bts = 2 * static_cast<int>(std::sqrt(128.0));

    Message mask = generateMask_V2(op.getLogSlots(), num_data);
    mask.to(device);

    Ciphertext ctxt_inv_sqrt(eval.getContext());
    Ciphertext ctxt_inv(eval.getContext());
    Ciphertext ctxt_2(eval.getContext());

    eval.sub(op, M, res);
    Softmax_UNI_approxExp_BSGS(eval, btp, res, res, k, N); // 7

    // step0
    eval.square(res, ctxt_inv_sqrt);                                       // 6
    rotateSum_masking(eval, ctxt_inv_sqrt, ctxt_inv_sqrt, mask, num_data); // 5
    bootstrap_02(eval, btp, ctxt_inv_sqrt);                                // 12

    Softmax_UNI_approxInverseSqrt_63_BSGS(eval, btp, ctxt_inv_sqrt,
                                          ctxt_inv_sqrt); // 6

    eval.mult(res, ctxt_inv_sqrt, res); // 5
    eval.square(res, res);              // 4
    bootstrap_02(eval, btp, res);       // 12

    // step1
    eval.square(res, ctxt_inv_sqrt);                                       // 11
    rotateSum_masking(eval, ctxt_inv_sqrt, ctxt_inv_sqrt, mask, num_data); // 10
    Softmax_UNI128_Scaled_approxInverseSqrt_15_BSGS_GH(
        eval, btp, 1.0, ctxt_inv_sqrt, ctxt_inv_sqrt); // 6

    eval.mult(ctxt_inv_sqrt, 1.0 / scale_for_bts, ctxt_inv_sqrt); // 5
    bootstrap_02(eval, btp, ctxt_inv_sqrt);                       // 12

    eval.multInteger(ctxt_inv_sqrt, scale_for_bts, ctxt_inv_sqrt); // 12

    eval.mult(res, ctxt_inv_sqrt, res); // 11
    eval.square(res, res);              // 10

    // step2
    eval.square(res, ctxt_inv_sqrt);                                       // 9
    rotateSum_masking(eval, ctxt_inv_sqrt, ctxt_inv_sqrt, mask, num_data); // 8
    Softmax_UNI128_Scaled_approxInverseSqrt_15_BSGS_GH(
        eval, btp, 1.0, ctxt_inv_sqrt, ctxt_inv_sqrt); // 4

    eval.mult(ctxt_inv_sqrt, 1.0 / scale_for_bts, ctxt_inv_sqrt);  // 3
    bootstrap_02(eval, btp, ctxt_inv_sqrt);                        // 12
    eval.multInteger(ctxt_inv_sqrt, scale_for_bts, ctxt_inv_sqrt); // 12

    eval.mult(res, ctxt_inv_sqrt, res); // 9
    eval.square(res, res);              // 8

    // step3
    eval.square(res, ctxt_inv_sqrt);                                       // 7
    rotateSum_masking(eval, ctxt_inv_sqrt, ctxt_inv_sqrt, mask, num_data); // 6
    bootstrap_02(eval, btp, ctxt_inv_sqrt);                                // 12
    Softmax_UNI128_Scaled_approxInverseSqrt_15_BSGS_GH(
        eval, btp, 1.0, ctxt_inv_sqrt, ctxt_inv_sqrt); // 8

    eval.mult(res, ctxt_inv_sqrt, res); // 7
    eval.square(res, res);              // 6

    // step4
    eval.square(res, ctxt_inv_sqrt);                                       // 5
    rotateSum_masking(eval, ctxt_inv_sqrt, ctxt_inv_sqrt, mask, num_data); // 4
    bootstrap_02(eval, btp, ctxt_inv_sqrt);                                // 12

    Softmax_UNI128_Scaled_approxInverseSqrt_15_BSGS_GH(
        eval, btp, 1.0, ctxt_inv_sqrt, ctxt_inv_sqrt); // 8

    eval.mult(res, ctxt_inv_sqrt, res); // 5
    eval.square(res, res);              // 4
    bootstrap_02(eval, btp, res);       // 12

    // step5
    eval.square(res, ctxt_inv_sqrt);                                       // 11
    rotateSum_masking(eval, ctxt_inv_sqrt, ctxt_inv_sqrt, mask, num_data); // 10

    Softmax_UNI128_Scaled_approxInverseSqrt_15_BSGS_GH(
        eval, btp, 1.0, ctxt_inv_sqrt, ctxt_inv_sqrt);             // 6
    eval.mult(ctxt_inv_sqrt, 1.0 / scale_for_bts, ctxt_inv_sqrt);  // 5
    bootstrap_02(eval, btp, ctxt_inv_sqrt);                        // 12
    eval.multInteger(ctxt_inv_sqrt, scale_for_bts, ctxt_inv_sqrt); // 12

    eval.mult(res, ctxt_inv_sqrt, res); // 11
    eval.square(res, res);              // 10

    // step6
    eval.square(res, ctxt_inv_sqrt);                                       // 9
    rotateSum_masking(eval, ctxt_inv_sqrt, ctxt_inv_sqrt, mask, num_data); // 8
    Softmax_UNI128_LAST_approxInverseSqrt_31_BSGS(eval, btp, ctxt_inv_sqrt,
                                                  ctxt_inv_sqrt); // 7
    bootstrap_02(eval, btp, ctxt_inv_sqrt);                       // 12

    eval.mult(res, ctxt_inv_sqrt, res); // 9
    eval.square(res, res);              // 8

    rotateSum_masking(eval, res, ctxt_inv_sqrt, mask, num_data); // 7
    eval.negate(ctxt_inv_sqrt, ctxt_inv_sqrt);
    eval.add(ctxt_inv_sqrt, 2.0, ctxt_inv_sqrt);
    eval.mult(res, ctxt_inv_sqrt, res); // 6

    rotateSum_masking(eval, res, ctxt_inv_sqrt, mask, num_data); // 5
    eval.negate(ctxt_inv_sqrt, ctxt_inv_sqrt);
    eval.add(ctxt_inv_sqrt, 2.0, ctxt_inv_sqrt);
    eval.mult(res, ctxt_inv_sqrt, res); // 4
}

// N = 128, range= [-256,256]
void Softmax_128_512_Parallel(const HomEvaluator &eval, const Bootstrapper &btp,
                              const std::vector<CtxtTensor> &op,
                              std::vector<CtxtTensor> &res) {

    const u64 num_ctxt = op.size();
    const int num_data = 128;
    const int k = 8;
    const int N = 128;
    const int M = 1024;
    const Device device = op[0].get().getDevice();
    // const u64 log_slots = op.getLogSlots();
    const int scale_for_bts = 2 * static_cast<int>(std::sqrt(128.0));

    Message mask = generateMask_V2(op[0].get().getLogSlots(), num_data);
    mask.to(device);
    Message mask_gar =
        generateMask_garbage(op[0].get().getLogSlots(), num_data);
    mask_gar.to(device);

    std::vector<Ciphertext> ctxt_inv_sqrt;
    // std::vector<Ciphertext> ctxt_inv;
    // std::vector<Ciphertext> ctxt_2;
    // initialization
    for (u64 i = 0; i < num_ctxt; ++i) {
        ctxt_inv_sqrt.emplace_back(op[i].get());
    }

    for (u64 i = 0; i < num_ctxt; ++i) {
        eval.sub(op[i].get(), M, res[i].get());
        Softmax_UNI_approxExp_BSGS(eval, btp, res[i].get(), res[i].get(), k,
                                   N); // 7
    }

    // step0
    eval.square(res[0].get(), ctxt_inv_sqrt[0]); // 6
    rotateSum_masking_first(eval, ctxt_inv_sqrt[0], ctxt_inv_sqrt[0], mask,
                            num_data); // 5
    for (u64 i = 1; i < num_ctxt; ++i) {
        eval.square(res[i].get(), ctxt_inv_sqrt[i]); // 6
        rotateSum_masking_first(eval, ctxt_inv_sqrt[i], ctxt_inv_sqrt[i], mask,
                                num_data); // 5
        eval.rightRotate(ctxt_inv_sqrt[i], i, ctxt_inv_sqrt[i]);
        eval.add(ctxt_inv_sqrt[0], ctxt_inv_sqrt[i], ctxt_inv_sqrt[0]);
    }

    Ciphertext tmp(eval.getContext());
    for (u64 i = 6; i < 64; i <<= 1) {
        eval.rightRotate(ctxt_inv_sqrt[0], i, tmp);
        eval.add(ctxt_inv_sqrt[0], tmp, ctxt_inv_sqrt[0]);
    }
    eval.add(ctxt_inv_sqrt[0], mask_gar, ctxt_inv_sqrt[0]);

    bootstrap_02(eval, btp, ctxt_inv_sqrt[0]); // 12
    Softmax_UNI_approxInverseSqrt_63_BSGS(eval, btp, ctxt_inv_sqrt[0],
                                          tmp); // 6

    rotateSum_masking_after(eval, tmp, ctxt_inv_sqrt[0], mask, num_data); // 5
    for (u64 i = 1; i < num_ctxt; ++i) {
        eval.leftRotate(tmp, i, ctxt_inv_sqrt[i]);
        rotateSum_masking_after(eval, ctxt_inv_sqrt[i], ctxt_inv_sqrt[i], mask,
                                num_data); // 5
    }

    for (u64 i = 0; i < num_ctxt; ++i) {
        eval.mult(res[i].get(), ctxt_inv_sqrt[i], res[i].get()); // 4
        eval.square(res[i].get(), res[i].get());                 // 3
    }

    for (u64 i = 0; i < num_ctxt / 2; ++i) {
        bootstrap2_02(eval, btp, res[2 * i].get(), res[2 * i + 1].get()); // 12
    }

    // step1
    eval.square(res[0].get(), ctxt_inv_sqrt[0]); // 11
    rotateSum_masking_first(eval, ctxt_inv_sqrt[0], ctxt_inv_sqrt[0], mask,
                            num_data); // 10
    for (u64 i = 1; i < num_ctxt; ++i) {
        eval.square(res[i].get(), ctxt_inv_sqrt[i]); // 11
        rotateSum_masking_first(eval, ctxt_inv_sqrt[i], ctxt_inv_sqrt[i], mask,
                                num_data); // 10
        eval.rightRotate(ctxt_inv_sqrt[i], i, ctxt_inv_sqrt[i]);
        eval.add(ctxt_inv_sqrt[0], ctxt_inv_sqrt[i], ctxt_inv_sqrt[0]);
    }

    for (u64 i = 6; i < 64; i <<= 1) {
        eval.rightRotate(ctxt_inv_sqrt[0], i, tmp);
        eval.add(ctxt_inv_sqrt[0], tmp, ctxt_inv_sqrt[0]);
    }
    eval.add(ctxt_inv_sqrt[0], mask_gar, ctxt_inv_sqrt[0]);
    Softmax_UNI128_Scaled_approxInverseSqrt_15_BSGS_GH(
        eval, btp, 1.0, ctxt_inv_sqrt[0], tmp); // 6

    rotateSum_masking_after(eval, tmp, ctxt_inv_sqrt[0], mask, num_data); // 5
    for (u64 i = 1; i < num_ctxt; ++i) {
        eval.leftRotate(tmp, i, ctxt_inv_sqrt[i]);
        rotateSum_masking_after(eval, ctxt_inv_sqrt[i], ctxt_inv_sqrt[i], mask,
                                num_data); // 5
    }

    for (u64 i = 0; i < num_ctxt / 2; ++i) {
        eval.mult(ctxt_inv_sqrt[i], 1.0 / scale_for_bts, ctxt_inv_sqrt[i]); // 4
        eval.mult(ctxt_inv_sqrt[2 * i + 1], 1.0 / scale_for_bts,
                  ctxt_inv_sqrt[2 * i + 1]); // 4
        bootstrap2_02(eval, btp, ctxt_inv_sqrt[2 * i],
                      ctxt_inv_sqrt[2 * i + 1]); // 12
        // bootstrap_02(eval, btp, ctxt_inv_sqrt[i]);
        eval.multInteger(ctxt_inv_sqrt[i], scale_for_bts,
                         ctxt_inv_sqrt[i]); // 12
        eval.multInteger(ctxt_inv_sqrt[2 * i + 1], scale_for_bts,
                         ctxt_inv_sqrt[2 * i + 1]); // 12
    }

    for (u64 i = 0; i < num_ctxt; ++i) {
        eval.mult(res[i].get(), ctxt_inv_sqrt[i], res[i].get()); // 11
        eval.square(res[i].get(), res[i].get());                 // 10
    }

    // step2
    eval.square(res[0].get(), ctxt_inv_sqrt[0]); // 9
    rotateSum_masking_first(eval, ctxt_inv_sqrt[0], ctxt_inv_sqrt[0], mask,
                            num_data); // 8
    for (u64 i = 1; i < num_ctxt; ++i) {
        eval.square(res[i].get(), ctxt_inv_sqrt[i]); // 9
        rotateSum_masking_first(eval, ctxt_inv_sqrt[i], ctxt_inv_sqrt[i], mask,
                                num_data); // 8
        eval.rightRotate(ctxt_inv_sqrt[i], i, ctxt_inv_sqrt[i]);
        eval.add(ctxt_inv_sqrt[0], ctxt_inv_sqrt[i], ctxt_inv_sqrt[0]);
    }

    for (u64 i = 6; i < 64; i <<= 1) {
        eval.rightRotate(ctxt_inv_sqrt[0], i, tmp);
        eval.add(ctxt_inv_sqrt[0], tmp, ctxt_inv_sqrt[0]);
    }
    eval.add(ctxt_inv_sqrt[0], mask_gar, ctxt_inv_sqrt[0]);

    Softmax_UNI128_Scaled_approxInverseSqrt_15_BSGS_GH(
        eval, btp, 1.0, ctxt_inv_sqrt[0], tmp); // 4
    eval.mult(tmp, 1.0 / scale_for_bts, tmp);   // 3
    bootstrap_02(eval, btp, tmp);               // 12
    eval.multInteger(tmp, scale_for_bts, tmp);  // 12

    rotateSum_masking_after(eval, tmp, ctxt_inv_sqrt[0], mask, num_data); // 11
    for (u64 i = 1; i < num_ctxt; ++i) {
        eval.leftRotate(tmp, i, ctxt_inv_sqrt[i]);
        rotateSum_masking_after(eval, ctxt_inv_sqrt[i], ctxt_inv_sqrt[i], mask,
                                num_data); // 11
    }

    for (u64 i = 0; i < num_ctxt; ++i) {
        eval.mult(res[i].get(), ctxt_inv_sqrt[i], res[i].get()); // 9
        eval.square(res[i].get(), res[i].get());                 // 8
    }

    // step3
    eval.square(res[0].get(), ctxt_inv_sqrt[0]); // 7
    rotateSum_masking_first(eval, ctxt_inv_sqrt[0], ctxt_inv_sqrt[0], mask,
                            num_data); // 6
    for (u64 i = 1; i < num_ctxt; ++i) {
        eval.square(res[i].get(), ctxt_inv_sqrt[i]); // 7
        rotateSum_masking_first(eval, ctxt_inv_sqrt[i], ctxt_inv_sqrt[i], mask,
                                num_data); // 6
        eval.rightRotate(ctxt_inv_sqrt[i], i, ctxt_inv_sqrt[i]);
        eval.add(ctxt_inv_sqrt[0], ctxt_inv_sqrt[i], ctxt_inv_sqrt[0]);
    }

    for (u64 i = 6; i < 64; i <<= 1) {
        eval.rightRotate(ctxt_inv_sqrt[0], i, tmp);
        eval.add(ctxt_inv_sqrt[0], tmp, ctxt_inv_sqrt[0]);
    }
    eval.add(ctxt_inv_sqrt[0], mask_gar, ctxt_inv_sqrt[0]);

    Softmax_UNI128_Scaled_approxInverseSqrt_15_BSGS_GH(
        eval, btp, 1.0, ctxt_inv_sqrt[0], tmp); // 8

    rotateSum_masking_after(eval, tmp, ctxt_inv_sqrt[0], mask, num_data); // 7
    for (u64 i = 1; i < num_ctxt; ++i) {
        eval.leftRotate(tmp, i, ctxt_inv_sqrt[i]);
        rotateSum_masking_after(eval, ctxt_inv_sqrt[i], ctxt_inv_sqrt[i], mask,
                                num_data); // 7
    }

    for (u64 i = 0; i < num_ctxt; ++i) {
        eval.mult(res[i].get(), ctxt_inv_sqrt[i], res[i].get()); // 6
        eval.square(res[i].get(), res[i].get());                 // 5
    }

    for (u64 i = 0; i < num_ctxt / 2; ++i) {
        bootstrap2_02(eval, btp, res[2 * i].get(), res[2 * i + 1].get()); // 12
    }

    // step4
    eval.square(res[0].get(), ctxt_inv_sqrt[0]); // 11
    rotateSum_masking_first(eval, ctxt_inv_sqrt[0], ctxt_inv_sqrt[0], mask,
                            num_data); // 10
    for (u64 i = 1; i < num_ctxt; ++i) {
        eval.square(res[i].get(), ctxt_inv_sqrt[i]); // 11
        rotateSum_masking_first(eval, ctxt_inv_sqrt[i], ctxt_inv_sqrt[i], mask,
                                num_data); // 10
        eval.rightRotate(ctxt_inv_sqrt[i], i, ctxt_inv_sqrt[i]);
        eval.add(ctxt_inv_sqrt[0], ctxt_inv_sqrt[i], ctxt_inv_sqrt[0]);
    }

    for (u64 i = 6; i < 64; i <<= 1) {
        eval.rightRotate(ctxt_inv_sqrt[0], i, tmp);
        eval.add(ctxt_inv_sqrt[0], tmp, ctxt_inv_sqrt[0]);
    }
    eval.add(ctxt_inv_sqrt[0], mask_gar, ctxt_inv_sqrt[0]);

    Softmax_UNI128_Scaled_approxInverseSqrt_15_BSGS_GH(
        eval, btp, 1.0, ctxt_inv_sqrt[0], tmp); // 6
    eval.mult(tmp, 1.0 / scale_for_bts, tmp);   // 5
    bootstrap_02(eval, btp, tmp);               // 12
    eval.multInteger(tmp, scale_for_bts, tmp);  // 12

    rotateSum_masking_after(eval, tmp, ctxt_inv_sqrt[0], mask, num_data); // 11
    for (u64 i = 1; i < num_ctxt; ++i) {
        eval.leftRotate(tmp, i, ctxt_inv_sqrt[i]);
        rotateSum_masking_after(eval, ctxt_inv_sqrt[i], ctxt_inv_sqrt[i], mask,
                                num_data); // 11
    }

    for (u64 i = 0; i < num_ctxt; ++i) {
        eval.mult(res[i].get(), ctxt_inv_sqrt[i], res[i].get()); // 10
        eval.square(res[i].get(), res[i].get());                 // 9
    }

    // step5
    eval.square(res[0].get(), ctxt_inv_sqrt[0]); // 8
    rotateSum_masking_first(eval, ctxt_inv_sqrt[0], ctxt_inv_sqrt[0], mask,
                            num_data); // 7
    for (u64 i = 1; i < num_ctxt; ++i) {
        eval.square(res[i].get(), ctxt_inv_sqrt[i]); // 8
        rotateSum_masking_first(eval, ctxt_inv_sqrt[i], ctxt_inv_sqrt[i], mask,
                                num_data); // 7
        eval.rightRotate(ctxt_inv_sqrt[i], i, ctxt_inv_sqrt[i]);
        eval.add(ctxt_inv_sqrt[0], ctxt_inv_sqrt[i], ctxt_inv_sqrt[0]);
    }

    for (u64 i = 6; i < 64; i <<= 1) {
        eval.rightRotate(ctxt_inv_sqrt[0], i, tmp);
        eval.add(ctxt_inv_sqrt[0], tmp, ctxt_inv_sqrt[0]);
    }
    eval.add(ctxt_inv_sqrt[0], mask_gar, ctxt_inv_sqrt[0]); // 7

    Softmax_UNI128_Scaled_approxInverseSqrt_15_BSGS_GH(
        eval, btp, 1.0, ctxt_inv_sqrt[0], tmp); // 8
    eval.mult(tmp, 1.0 / scale_for_bts, tmp);   // 7
    bootstrap_02(eval, btp, tmp);               // 12
    eval.multInteger(tmp, scale_for_bts, tmp);  // 12

    rotateSum_masking_after(eval, tmp, ctxt_inv_sqrt[0], mask, num_data); // 11
    for (u64 i = 1; i < num_ctxt; ++i) {
        eval.leftRotate(tmp, i, ctxt_inv_sqrt[i]);
        rotateSum_masking_after(eval, ctxt_inv_sqrt[i], ctxt_inv_sqrt[i], mask,
                                num_data); // 11
    }

    for (u64 i = 0; i < num_ctxt; ++i) {
        eval.mult(res[i].get(), ctxt_inv_sqrt[i], res[i].get()); // 8
        eval.square(res[i].get(), res[i].get());                 // 7
    }

    // step6
    eval.square(res[0].get(), ctxt_inv_sqrt[0]); // 6
    rotateSum_masking_first(eval, ctxt_inv_sqrt[0], ctxt_inv_sqrt[0], mask,
                            num_data); // 5
    for (u64 i = 1; i < num_ctxt; ++i) {
        eval.square(res[i].get(), ctxt_inv_sqrt[i]); // 6
        rotateSum_masking_first(eval, ctxt_inv_sqrt[i], ctxt_inv_sqrt[i], mask,
                                num_data); // 5
        eval.rightRotate(ctxt_inv_sqrt[i], i, ctxt_inv_sqrt[i]);
        eval.add(ctxt_inv_sqrt[0], ctxt_inv_sqrt[i], ctxt_inv_sqrt[0]);
    }

    for (u64 i = 6; i < 64; i <<= 1) {
        eval.rightRotate(ctxt_inv_sqrt[0], i, tmp);
        eval.add(ctxt_inv_sqrt[0], tmp, ctxt_inv_sqrt[0]);
    }
    eval.add(ctxt_inv_sqrt[0], mask_gar, ctxt_inv_sqrt[0]); // 5

    Softmax_UNI128_Scaled_approxInverseSqrt_15_BSGS_GH(
        eval, btp, 1.0, ctxt_inv_sqrt[0], tmp); // 8

    rotateSum_masking_after(eval, tmp, ctxt_inv_sqrt[0], mask, num_data); // 7
    for (u64 i = 1; i < num_ctxt; ++i) {
        eval.leftRotate(tmp, i, ctxt_inv_sqrt[i]);
        rotateSum_masking_after(eval, ctxt_inv_sqrt[i], ctxt_inv_sqrt[i], mask,
                                num_data); // 7
    }

    for (u64 i = 0; i < num_ctxt; ++i) {
        eval.mult(res[i].get(), ctxt_inv_sqrt[i], res[i].get()); // 6
        eval.square(res[i].get(), res[i].get());                 // 5
    }

    for (u64 i = 0; i < num_ctxt / 2; ++i) {
        bootstrap2_02(eval, btp, res[2 * i].get(), res[2 * i + 1].get()); // 12
    }

    // step7
    eval.square(res[0].get(), ctxt_inv_sqrt[0]); // 11
    rotateSum_masking_first(eval, ctxt_inv_sqrt[0], ctxt_inv_sqrt[0], mask,
                            num_data); // 10
    for (u64 i = 1; i < num_ctxt; ++i) {
        eval.square(res[i].get(), ctxt_inv_sqrt[i]); // 11
        rotateSum_masking_first(eval, ctxt_inv_sqrt[i], ctxt_inv_sqrt[i], mask,
                                num_data); // 10
        eval.rightRotate(ctxt_inv_sqrt[i], i, ctxt_inv_sqrt[i]);
        eval.add(ctxt_inv_sqrt[0], ctxt_inv_sqrt[i], ctxt_inv_sqrt[0]);
    }

    for (u64 i = 6; i < 64; i <<= 1) {
        eval.rightRotate(ctxt_inv_sqrt[0], i, tmp);
        eval.add(ctxt_inv_sqrt[0], tmp, ctxt_inv_sqrt[0]);
    }
    eval.add(ctxt_inv_sqrt[0], mask_gar, ctxt_inv_sqrt[0]); // 10

    Softmax_UNI128_LAST_approxInverseSqrt_31_BSGS(eval, btp, ctxt_inv_sqrt[0],
                                                  tmp); // 4

    eval.mult(tmp, 1.0 / scale_for_bts, tmp);
    bootstrap_02(eval, btp, tmp);              // 12
    eval.multInteger(tmp, scale_for_bts, tmp); // 12

    rotateSum_masking_after(eval, tmp, ctxt_inv_sqrt[0], mask, num_data); // 11
    for (u64 i = 1; i < num_ctxt; ++i) {
        eval.leftRotate(tmp, i, ctxt_inv_sqrt[i]);
        rotateSum_masking_after(eval, ctxt_inv_sqrt[i], ctxt_inv_sqrt[i], mask,
                                num_data); // 11
    }

    for (u64 i = 0; i < num_ctxt; ++i) {
        eval.mult(res[i].get(), ctxt_inv_sqrt[i], res[i].get()); // 10
        eval.square(res[i].get(), res[i].get());                 // 9
    }

    // out: 5
    for (u64 i = 0; i < num_ctxt; ++i) {
        for (u64 j = 0; j < 2; ++j) {
            rotateSum_masking(eval, res[i].get(), ctxt_inv_sqrt[i], mask,
                              num_data);
            eval.negate(ctxt_inv_sqrt[i], ctxt_inv_sqrt[i]);
            eval.add(ctxt_inv_sqrt[i], 2.0, ctxt_inv_sqrt[i]);
            eval.mult(res[i].get(), ctxt_inv_sqrt[i], res[i].get());
        }
    }
}

} // namespace HELLM::Softmax
