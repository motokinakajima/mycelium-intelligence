set(NN_SOURCES
        node_nn/nn.cpp
        node_nn/utils/io.cpp
)

add_library(node_nn STATIC ${NN_SOURCES})

target_include_directories(node_nn
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        node_nn/utils
)