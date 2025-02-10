add_executable(keygen keygen.cpp)
target_link_libraries(keygen PUBLIC HELLM)

add_executable(convert convert2.cpp)
target_link_libraries(convert PUBLIC HELLM)

add_executable(eval bert-test.cpp)
target_link_libraries(eval PUBLIC HELLM)

add_executable(train backward-bert-multi.cpp)
target_link_libraries(train PUBLIC HELLM)

add_executable(time backward-time.cpp)
target_link_libraries(time PUBLIC HELLM)

add_executable(bert-matmul bert-matmul.cpp)
target_link_libraries(bert-matmul PUBLIC HELLM)

add_executable(bert-matmul-backward bert-matmul-backward.cpp)
target_link_libraries(bert-matmul-backward PUBLIC HELLM)
