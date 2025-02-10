////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2024 Crypto Lab Inc.                                    //
//                                                                            //
// - This file is part of HEaaN homomorphic encryption library.               //
// - HEaaN cannot be copied and/or distributed without the express permission //
//  of Crypto Lab Inc.                                                        //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "HEaaN/HEaaN.hpp"
#include "HEaaN-math/tools/Intervals.hpp"
#include <vector>

namespace HEaaN::Math {

enum PolynomialBasis { basic, odd };

struct ChebyshevCoefficients {
    ChebyshevCoefficients(std::vector<Real> coefficient, u64 num_baby_step,
                          PolynomialBasis base = PolynomialBasis::basic)
        : coeffs{std::move(coefficient)}, basis{base} {
        num_bs = (num_baby_step > 0) ? num_baby_step : coeffs.size();
        num_gs = static_cast<u64>(std::ceil(static_cast<Real>(coeffs.size()) /
                                            static_cast<Real>(num_bs)));
        log_bs = static_cast<u64>(std::ceil(std::log2(num_bs)));
        log_gs = static_cast<u64>(std::ceil(std::log2(num_gs)));
        level_cost = log_bs + log_gs;
    }

    std::vector<Real> coeffs;
    u64 num_bs;
    u64 num_gs;
    u64 log_bs;
    u64 log_gs;
    u64 level_cost;
    PolynomialBasis basis;
};

// Linear transform from [left_boundary, right_boundary] to [-1, 1].
// res = (2 / (right_boundary - left_boundary))
//       * (ctxt - (right_boundary + left_boundary) / 2).
Ciphertext linearTransform(const HomEvaluator &eval, const Bootstrapper &btp,
                           const Ciphertext &ctxt,
                           const InputInterval &input_interval);

// Evaluate Chebyshev expansion with baby-step giant-step.
// res = Σ ( Σ d_{i * bs + j} * T_j[ctxt] ) * ( Π T_{2^k * bs}[ctxt]^{i &
// 2^k} )
//     = Σ d_{j} * T_j[ctxt]
//       + (Σ d_{bs + j} * T_j[ctxt]) * T_{bs}[ctxt]
//       + (Σ d_{2 * bs + j} * T_j[ctxt]) * T_{2 * bs}[ctxt]
//       + (Σ d_{3 * bs + j} * T_j[ctxt]) * T_{bs}[ctxt] * T_{2 * bs}[ctxt]
//       + ...
// where T's are Chebyshev polynomials and d_i's are the multiplied
// coefficients defined by d_{i} = multiplier * coeffs_[i].
Ciphertext evaluateChebyshevExpansion(const HomEvaluator &eval,
                                      const Bootstrapper &btp,
                                      const Ciphertext &ctxt,
                                      const ChebyshevCoefficients &cheby_coeffs,
                                      const Real multiplier);

} // namespace HEaaN::Math
