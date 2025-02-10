////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2024 Crypto Lab Inc. All rights reserved.               //
//                                                                            //
// This source code is protected under international copyright law.           //
// All rights reserved and protected by the copyright holders.                //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "HELLM/HEMMer.hpp"

#include "HEaaN/HEaaN.hpp"

#include <cstdlib>
#include <filesystem>

int main() {
    std::string key_path{std::getenv("HELLM_KEY_PATH")};
    std::filesystem::create_directory(key_path);
    std::filesystem::create_directory(key_path + "/PK");

    HELLM::HEMMer hemmer{HELLM::HEMMer::genHEMMer()};
    hemmer.save(key_path);
}
