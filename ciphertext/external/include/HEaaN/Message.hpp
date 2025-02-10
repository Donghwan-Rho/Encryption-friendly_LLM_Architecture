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

#include <iterator>
#include <vector>

#include "HEaaN/HEaaNExport.hpp"
#include "HEaaN/Integers.hpp"
#include "HEaaN/Pointer.hpp"
#include "HEaaN/Real.hpp"
#include "HEaaN/device/Device.hpp"

namespace HEaaN {
///
///@brief A class consists of complex messages which correspond to a
/// slot-encoded plaintexts.
///@details Each slot value, which is a complex number, should have a real and
/// imaginary number whose absolute values are less than 2^64. Otherwise, it is
/// undefined behavior.
///

class MessageImpl;
class HEAAN_API Message {
public:
    using MessageIterator = Complex *;
    using ConstMessageIterator = const Complex *;

    Message();

    ///@brief Create an uninitialized message.
    ///@param[in] log_slots The number of log(slots).
    ///@details A message which has two to @p log_slots slots is constructed.
    /// Because each slot, which is a complex number, is not initialized, you
    /// have to fill them by yourself.
    explicit Message(u64 log_slots);

    ///@brief Create a message filled with a given value.
    ///@param[in] log_slots The number of log(slots).
    ///@param[in] initial The value of each slot.
    ///@details A message which has two to @p log_slots slots whose values are
    /// @p initial is constructed..
    explicit Message(u64 log_slots, Complex initial);

    /// @brief Get reference to an element at given index
    /// @param[in] idx
    Complex &operator[](u64 idx);

    /// @brief Get const reference to an element at given index
    /// @param[in] idx
    const Complex &operator[](u64 idx) const;

    ///@brief Determine whether the message is empty or not.
    ///@returns true if the message is empty, false otherwise
    bool isEmpty() const;

    ///@brief Get log(number of slots) of a message
    ///@returns log(number of slots)
    u64 getLogSlots() const;

    /// @brief Get size of a message
    u64 getSize() const;

    /// @brief Resize message to given size
    /// @param[in] size
    /// @details The first @p size elements of the original message is
    /// preserved, and the rest of the elements are undefined.
    void resize(u64 size);

    /// @brief An iterator pointing the initial element of the message.
    MessageIterator begin() noexcept;

    /// @brief A const iterator pointing the initial element of the message.
    ConstMessageIterator begin() const noexcept;

    /// @brief An iterator pointing the final element of the message.
    MessageIterator end() noexcept;

    /// @brief A const iterator pointing the final element of the message.
    ConstMessageIterator end() const noexcept;

    /// @brief A reverse iterator pointing the final element of the message.
    auto rbegin() { return std::reverse_iterator(end()); }

    /// @brief A const reverse iterator pointing the final element of the
    /// message.
    auto rbegin() const { return std::reverse_iterator(end()); }

    /// @brief A reverse iterator pointing the initial element of the message.
    auto rend() { return std::reverse_iterator(begin()); }

    /// @brief A const reverse iterator pointing the initial element of the
    /// message.
    auto rend() const { return std::reverse_iterator(begin()); }

    template <class Archive> void serialize(Archive &ar);

    ///@brief Send a message to given device.
    ///@param[in] device
    void to(const Device &device);

    ///@brief Allocate memory for a message at given device.
    ///@param[in] device
    void allocate(const Device &device);

    ///@brief Get device which a message reside in.
    const Device &getDevice() const;

    /// @brief Save a message to a file
    void save(const std::string &path) const;

    /// @brief Save a message to given stream
    void save(std::ostream &stream) const;

    /// @brief Load a message from a file
    void load(const std::string &path);

    /// @brief Load a message from given stream
    void load(std::istream &stream);

private:
    Pointer<MessageImpl> impl_;
};

/// @brief A type alias to represent the message for coefficient-encoded
/// ciphertexts
/// @details Coefficient-encoded ciphertexts is encrypted from and decrypted to
/// real vector, with length equal to the number of coefficients.
using CoeffMessage = std::vector<Real>;

} // namespace HEaaN
