set(SIM_SOURCES
        config.cpp
        graph.cpp
        maze.cpp
        export.cpp
)

add_library(node_sim STATIC ${SIM_SOURCES})

# Allow #include "node_nn/nn.h" inside sim sources
target_include_directories(node_sim
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}          # allows #include "graph.h" etc.
        ${CMAKE_SOURCE_DIR}/src/nn           # allows #include "node_nn/nn.h"
)

target_link_libraries(node_sim
        PUBLIC
        node_nn
)
