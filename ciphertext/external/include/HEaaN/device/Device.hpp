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

#include <initializer_list>
#include <string>
#include <vector>

#include "HEaaN/Exception.hpp"
#include "HEaaN/HEaaNExport.hpp"
#include "HEaaN/Integers.hpp"
#include "HEaaN/device/CudaTools.hpp"
#include <set>
#include <tuple>

namespace HEaaN {

enum class DeviceType {
    CPU,
    GPU,
    UNDEFINED,
    CPU_PINNED,
};

class HEAAN_API Device {
public:
    // If a type is given only, we infer the id.
    constexpr Device(const DeviceType type)
        : Device{type,
                 type == DeviceType::GPU ? CudaTools::cudaGetDevice() : 0} {}

    constexpr explicit Device(const DeviceType type, int device_id)
        : type_{type}, device_id_{device_id} {}
    constexpr DeviceType type() const { return type_; }
    constexpr int id() const { return device_id_; }
    constexpr bool operator==(const Device &other) const {
        return type() == other.type() && device_id_ == other.device_id_;
    }
    constexpr bool operator!=(const Device &other) const {
        return !(*this == other);
    }

    constexpr bool operator<(const Device &other) const {
        return std::tie(device_id_, type_) <
               std::tie(other.device_id_, other.type_);
    }

private:
    DeviceType type_;
    // cuda device id. Zero when type_ is CPU.
    int device_id_;
};

HEAAN_API constexpr Device getDefaultDevice() {
    return Device{DeviceType::CPU};
} // TODO(wk): remove?

using DeviceSet = std::set<Device>;

// A set of CUDA device IDs.
using CudaDeviceIds = std::set<int>;

HEAAN_API Device getCurrentCudaDevice();
// Set the current CUDA device id. The default id is set to 0.
HEAAN_API void setCurrentCudaDevice(int device_id);

class ScopedCudaDeviceSelector {
public:
    ScopedCudaDeviceSelector(Device device)
        : select_{device.type() == DeviceType::GPU},
          before_{select_ ? getCurrentCudaDevice() : getDefaultDevice()} {
        if (select_)
            setCurrentCudaDevice(device.id());
    }

    ~ScopedCudaDeviceSelector() {
        if (select_)
            setCurrentCudaDevice(before_.id());
    }

    ScopedCudaDeviceSelector(const ScopedCudaDeviceSelector &) = delete;
    ScopedCudaDeviceSelector(ScopedCudaDeviceSelector &&) = delete;
    ScopedCudaDeviceSelector &
    operator=(const ScopedCudaDeviceSelector &) = delete;
    ScopedCudaDeviceSelector &operator=(ScopedCudaDeviceSelector &&) = delete;

private:
    bool select_;
    Device before_;
};

} // namespace HEaaN
