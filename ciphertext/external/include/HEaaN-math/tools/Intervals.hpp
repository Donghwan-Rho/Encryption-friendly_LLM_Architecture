////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2024 CryptoLab, Inc.                                    //
//                                                                            //
// - This file is part of HEaaN homomorphic encryption library.               //
// - HEaaN cannot be copied and/or distributed without the express permission //
//  of CryptoLab, Inc.                                                        //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "HEaaN/HEaaN.hpp"

namespace HEaaN::Math {

struct Interval {
    Interval(const Real left, const Real right)
        : left_end{left}, right_end{right} {}

    Real left_end;
    Real right_end;
};

using InputInterval = Interval;

using DiffInterval = Interval;

} // namespace HEaaN::Math
