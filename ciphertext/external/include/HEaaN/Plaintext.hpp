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

#include "HEaaN/Context.hpp"
#include "HEaaN/HEaaNExport.hpp"
#include "HEaaN/Integers.hpp"
#include "HEaaN/Pointer.hpp"
#include "HEaaN/device/Device.hpp"

namespace HEaaN {
class PlaintextImpl;
class Polynomial;

///
///@brief A class of plaintexts each of which is simply a polynomial.
///
class HEAAN_API Plaintext {
public:
    explicit Plaintext(const Context &context);

    ///@brief Set log(number of slots) of a plaintext.
    ///@param[in] log_slots
    void setLogSlots(u64 log_slots);

    ///@brief Get log(number of slots) of a plaintext.
    ///@returns log(number of slots)
    u64 getLogSlots() const;

    ///@brief Get number of slots of a plaintext.
    ///@returns number of slots
    u64 getNumberOfSlots() const;

    ///@brief Get polynomial representing plaintext.
    Polynomial &getMx();

    ///@brief Get const polynomial representing plaintext.
    const Polynomial &getMx() const;

    ///@brief Get i-th data of a plaintext.
    ///@details i-th data is a modulo q_i information of the polynomial
    u64 *getMxData(u64 i) const;

    ///@brief Save a plaintext to a file
    void save(const std::string &path) const;

    ///@brief Save a plaintext
    void save(std::ostream &stream) const;

    ///@brief Load a plaintext from a file
    void load(const std::string &path);

    ///@brief Load a plaintext
    void load(std::istream &stream);

    ///@brief Rescaling flag
    ///@returns The amount of extra deltas multiplied.
    int getRescaleCounter() const;

    ///@brief set rescale counter
    void setRescaleCounter(int r_counter);

    ///@brief Get level of a plaintext.
    ///@returns The current level of the plaintext.
    u64 getLevel() const;

    ///@brief Set level of a plaintext.
    ///@param[in] level
    void setLevel(u64 level);

    ///@brief Get device which a plaintext reside in.
    ///@returns The device in which the plaintext resides
    const Device &getDevice() const;

    ///@brief Send a plaintext to given device.
    ///@param[in] device
    void to(const Device &device);

    ///@brief Allocate memory for a plaintext at given device.
    ///@param[in] device
    void allocate(const Device &device);

    ///@brief Check whether a plaintext has the zero polynomial
    bool isZero() const;

private:
    Pointer<PlaintextImpl> impl_;
};

} // namespace HEaaN
