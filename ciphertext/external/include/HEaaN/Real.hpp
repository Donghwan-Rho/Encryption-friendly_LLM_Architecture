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

#include "HEaaN/HEaaNExport.hpp"
#include <complex>

namespace HEaaN {

///@brief Default type for representing a real number
using Real = double;

///@brief Default type for representing a complex number
using Complex = std::complex<Real>;

// Constants (complex numbers included)

///@brief Real number 0
constexpr Real REAL_ZERO = 0.0;

///@brief Real number 1
constexpr Real REAL_ONE = 1.0;

///@brief Real number pi
constexpr Real REAL_PI = 3.14159265358979323846;

///@brief Real number 2 * pi
constexpr Real REAL_TWO_PI = 6.283185307179586476925286766559;

///@brief Complex number 0
constexpr Complex COMPLEX_ZERO = Complex(REAL_ZERO, REAL_ZERO);

///@brief Complex number i
constexpr Complex COMPLEX_IMAG_UNIT = Complex(REAL_ZERO, REAL_ONE);
} // namespace HEaaN
