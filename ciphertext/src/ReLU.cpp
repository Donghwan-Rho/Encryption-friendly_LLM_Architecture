////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2024 Crypto Lab Inc. All rights reserved.               //
//                                                                            //
// This source code is protected under international copyright law.           //
// All rights reserved and protected by the copyright holders.                //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "HELLM/ReLU.hpp"

#include "HEaaN-math/tools/PolynomialEvaluator.hpp"

#include <ATen/core/TensorBody.h>
#include <stdexcept>
#include <vector>

#include <iostream>
#include <string>

namespace HELLM::ReLU {

using namespace HEaaN;
using namespace Math;

void multWithoutRelin(const HomEvaluator &eval, const Ciphertext &ctxt1,
                      const Ciphertext &ctxt2, Ciphertext &res) {

    const u64 level1 = ctxt1.getLevel();
    const u64 level2 = ctxt2.getLevel();

    if (level1 == level2) {
        eval.tensor(ctxt1, ctxt2, res);
        eval.rescale(res);
        return;
    }

    Ciphertext ctxt_tmp(eval.getContext());

    if (level1 > level2) {
        eval.levelDown(ctxt1, level2, ctxt_tmp);
        eval.tensor(ctxt_tmp, ctxt2, res);
        eval.rescale(res);
    } else { // level1 < level2
        eval.levelDown(ctxt2, level1, ctxt_tmp);
        eval.tensor(ctxt1, ctxt_tmp, res);
        eval.rescale(res);
    }
}

// Construct BabyStep basis and GiantStep basis.
void oddSetUp(const HomEvaluator &eval, Ciphertext &ctxt,
              std::vector<Ciphertext> &oddBS_basis,
              std::vector<Ciphertext> &evenBS_basis,
              std::vector<Ciphertext> &GS_basis, const int k, const int l) {

    oddBS_basis.push_back(ctxt);
    evenBS_basis.push_back(ctxt);

    HEaaN::Ciphertext ctxt_temp(eval.getContext());

    for (u64 i = 1; i <= static_cast<u64>(std::log2(k)); ++i) {
        eval.mult(evenBS_basis[i - 1], evenBS_basis[i - 1], ctxt_temp);
        evenBS_basis.push_back(ctxt_temp);
    }

    for (u64 i = 1; i < static_cast<u64>(k / 2); ++i) {

        u64 alpha = static_cast<u64>(std::floor(std::log2(2 * i + 1)));
        u64 ind = static_cast<u64>(std::pow(2, alpha - 1));

        eval.mult(evenBS_basis[alpha], oddBS_basis[i - ind], ctxt_temp);
        oddBS_basis.push_back(ctxt_temp);
    }

    // first element of GS_basis is x^k
    GS_basis.push_back(evenBS_basis[static_cast<u64>(std::log2(k))]);

    for (u64 j = 1; j < static_cast<u64>(l); ++j) {
        eval.mult(GS_basis[j - 1], GS_basis[j - 1], ctxt_temp);
        GS_basis.push_back(ctxt_temp);
    }
}

// BabyStep algo in Han-Ki.
void oddBabyStep(const HomEvaluator &eval,
                 const std::vector<HEaaN::Ciphertext> &oddBS_basis,
                 const std::vector<double> &polynomial, Ciphertext &ctxt_result,
                 const int k) {

    eval.mult(oddBS_basis[0], polynomial[1], ctxt_result);

    Ciphertext ctxt_temp(eval.getContext());

    if (k > 2) {
        for (u64 i = 1; i < static_cast<u64>(k / 2); ++i) {
            eval.mult(oddBS_basis[i], polynomial[2 * i + 1], ctxt_temp);
            eval.add(ctxt_result, ctxt_temp, ctxt_result);
        }
    }
}

// For vector slicing. slice vector from a_index to b_index
std::vector<double> vectorSlice(const std::vector<double> &input, int a,
                                int b) {
    auto first = input.begin() + a;
    auto last = input.begin() + b;
    return std::vector<double>(first, last);
}

// GiantStep algorithm in Han-Ki
void oddGiantStep(const HomEvaluator &eval,
                  const std::vector<Ciphertext> &oddBS_basis,
                  const std::vector<Ciphertext> &GS_basis,
                  const std::vector<double> &polynomial,
                  Ciphertext &ctxt_result, int k, int l) {

    // integer a s.t. 2^a <= n/k < 2^(a+1)
    int degree = static_cast<int>(polynomial.size()) - 1;

    i64 a = static_cast<i64>(
        std::floor(std::log2(degree / static_cast<double>(k))));

    if (a < 0) {
        oddBabyStep(eval, oddBS_basis, polynomial, ctxt_result, k);
        /*
        std::cout << "Current Ciphertext evaluating one-time odd Baby Step  -
        level "
            << ctxt_result.getLevel() << std::endl
            << std::endl;
            */
        return;

    } else {
        int deg_div = static_cast<int>(k * std::pow(2, a));

        std::vector<double> quotient = vectorSlice(
            polynomial, deg_div, static_cast<int>(polynomial.size()));
        std::vector<double> remainder = vectorSlice(polynomial, 0, deg_div);

        Ciphertext ctxt_quotient_relin(eval.getContext());
        Ciphertext ctxt_remainder(eval.getContext());

        if (static_cast<int>(quotient.size()) <= k) {
            oddGiantStep(eval, oddBS_basis, GS_basis, quotient,
                         ctxt_quotient_relin, k, l);
        }

        else {
            Ciphertext ctxt_quotient(eval.getContext());
            oddGiantStep(eval, oddBS_basis, GS_basis, quotient, ctxt_quotient,
                         k, l);
            eval.relinearize(ctxt_quotient, ctxt_quotient_relin);
        }

        oddGiantStep(eval, oddBS_basis, GS_basis, remainder, ctxt_remainder, k,
                     l);
        multWithoutRelin(eval, GS_basis[static_cast<u64>(a)],
                         ctxt_quotient_relin, ctxt_result);
        eval.add(ctxt_result, ctxt_remainder, ctxt_result);
    }
}

void evalOddPolynomial(const HomEvaluator &eval, Ciphertext &ctxt,
                       Ciphertext &ctxt_poly,
                       const std::vector<double> &polynomial, int k, int l) {

    std::vector<Ciphertext> oddBS_basis;
    std::vector<Ciphertext> evenBS_basis;
    std::vector<Ciphertext> GS_basis;

    oddSetUp(eval, ctxt, oddBS_basis, evenBS_basis, GS_basis, k, l);

    Ciphertext ctxt_temp(eval.getContext());
    oddGiantStep(eval, oddBS_basis, GS_basis, polynomial, ctxt_temp, k, l);

    oddBS_basis.clear();
    evenBS_basis.shrink_to_fit();

    eval.relinearize(ctxt_temp, ctxt_poly);
}

// Aproximated ReLU function.
void ApproxReLU(const HomEvaluator &eval, const Bootstrapper &btp,
                Ciphertext &ctxt, Ciphertext &ctxt_relu,
                Ciphertext &ctxt_train) {

    Ciphertext ctxt_temp(eval.getContext());
    eval.conjugate(ctxt, ctxt_temp);
    eval.add(ctxt_temp, ctxt, ctxt_temp);
    eval.mult(ctxt_temp, 0.5, ctxt_temp);

    Ciphertext ctxt_real_BTS(eval.getContext());

    btp.bootstrap(ctxt_temp, ctxt_real_BTS);

    std::vector<double> polynomial_1 = {
        1.34595769293910e-33,  2.45589415425004e1,    4.85095667238242e-32,
        -6.69660449716894e2,   -2.44541235853840e-30, 6.67299848301339e3,
        1.86874811944640e-29,  -3.06036656163898e4,   -5.76227817577242e-29,
        7.31884032987787e4,    8.53680673009259e-29,  -9.44433217050084e4,
        -6.02701474694667e-29, 6.23254094212546e4,    1.62342843661940e-29,
        -1.64946744117805e4};

    std::vector<double> polynomial_2 = {
        1.53261588585630e-47,  9.35625636035439,      -3.68972123048249e-46,
        -5.91638963933626e1,   1.74254399703303e-45,  1.48860930626448e2,
        -3.20672110002213e-45, -1.75812874878582e2,   2.79115738948645e-45,
        1.09111299685955e2,    -1.22590309306100e-45, -3.66768839978755e1,
        2.62189142557962e-46,  6.31846290311294,      -2.16662326421275e-47,
        -4.37113415082177e-01};

    std::vector<double> polynomial_3 = {
        6.43551938319983e-48,  5.07813569758861,      8.12601038855762e-46,
        -3.07329918137186e1,   -1.60198474678427e-44, 1.44109746812809e2,
        1.07463154460511e-43,  -4.59661688826142e2,   -3.63448723044512e-43,
        1.02152064470459e3,    7.25207125369784e-43,  -1.62056256708877e3,
        -9.27306397853655e-43, 1.86467646416570e3,    7.95843097354065e-43,
        -1.56749300877143e3,   -4.69190103147527e-43, 9.60970309093422e2,
        1.90863349654016e-43,  -4.24326161871646e2,   -5.27439678020696e-44,
        1.31278509256003e2,    9.47044937974786e-45,  -2.69812576626115e1,
        -9.98181561763750e-46, 3.30651387315565,      4.69390466192199e-47,
        -1.82742944627533e-1};

    // for optimization
    for (u64 i = 0; i < 28; ++i) {
        polynomial_3[i] = polynomial_3[i] * 0.5;
    }

    evalOddPolynomial(eval, ctxt_real_BTS, ctxt_temp, polynomial_1, 2, 3);

    // btp.bootstrap(ctxt_temp, ctxt_temp);

    Ciphertext ctxt_temp1(eval.getContext());
    evalOddPolynomial(eval, ctxt_temp, ctxt_temp1, polynomial_2, 2, 3);

    btp.bootstrap(ctxt_temp1, ctxt_temp);

    evalOddPolynomial(eval, ctxt_temp, ctxt_temp1, polynomial_3, 4, 3);

    // for train_mode
    ctxt_train = ctxt_temp1;

    eval.mult(ctxt_real_BTS, 0.5, ctxt_temp);
    eval.mult(ctxt_real_BTS, ctxt_temp1, ctxt_relu);
    eval.add(ctxt_temp, ctxt_relu, ctxt_relu);
}

} // namespace HELLM::ReLU
