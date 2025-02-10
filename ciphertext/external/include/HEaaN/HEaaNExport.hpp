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

#ifndef HEAAN_API
#ifdef HEaaN_EXPORTS
/* We are building this library */
#ifdef _WIN32
#define HEAAN_API __declspec(dllexport)
#else
#define HEAAN_API __attribute__((visibility("default")))
#endif
#else // HEaaN_EXPORTS
/* We are using this library */
#ifdef _WIN32
#define HEAAN_API __declspec(dllimport)
#else
#define HEAAN_API __attribute__((visibility("default")))
#endif
#endif // HEaaN_EXPORTS
#endif // HEAAN_API
