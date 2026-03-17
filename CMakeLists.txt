cmake_minimum_required(VERSION 3.16)
project(mycelium_intelligence LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(src/nn)
add_subdirectory(src/sim)

add_executable(mycelium
        src/main.cpp
)

add_executable(train_nn
        src/train_nn.cpp
)

add_executable(eval_seed_trials
        src/eval_seed_trials.cpp
)

target_link_libraries(mycelium
        PRIVATE
        node_sim   # node_nn is transitively linked via node_sim PUBLIC
)

target_link_libraries(train_nn
        PRIVATE
        node_nn
)

target_link_libraries(eval_seed_trials
        PRIVATE
        node_sim
)
