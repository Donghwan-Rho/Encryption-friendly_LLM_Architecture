////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2024 Crypto Lab Inc. All rights reserved.               //
//                                                                            //
// This source code is protected under international copyright law.           //
// All rights reserved and protected by the copyright holders.                //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <sstream>
#include <stdexcept>

// MPI and NCCL validation macros
#ifdef HELLM_MULTIGPU

#define MPICHECK(cmd)                                                          \
    do {                                                                       \
        int e = cmd;                                                           \
        if (e != MPI_SUCCESS) {                                                \
            printf("[HELLM ERROR] MPI error '%d'. (%s:%d)\n", e, __FILE__,     \
                   __LINE__);                                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define NCCLCHECK(cmd)                                                         \
    do {                                                                       \
        ncclResult_t r = cmd;                                                  \
        if (r != ncclSuccess) {                                                \
            printf("[HELLM ERROR] NCCL error '%s'. (%s:%d)\n",                 \
                   ncclGetErrorString(r), __FILE__, __LINE__);                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#endif

namespace {

// extract the filename from a file path
[[maybe_unused]] const char *getFilename(const char *path) {
    const char *filename = strrchr(path, '/');
    return filename ? filename + 1 : path;
}

} // namespace

#define HELLM_RUNTIME_ERR(msg)                                                 \
    do {                                                                       \
        std::ostringstream oss;                                                \
        oss << "[HELLM ERROR] " << msg << ". (" << getFilename(__FILE__)       \
            << ":" << __LINE__ << ":" << __func__ << ")";                      \
        throw std::runtime_error(oss.str());                                   \
    } while (0)
