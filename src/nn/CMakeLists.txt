add_library(node_nn STATIC
        nn.cpp
)

target_include_directories(node_nn
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)