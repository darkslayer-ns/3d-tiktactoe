cmake_minimum_required(VERSION 3.16)
project(tfm CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# NOTE: no -ffast-math / -Ofast anywhere. Numerical parity with PyTorch
# requires IEEE-754 semantics (no reassociation, no FMA contraction changes).
add_compile_options(-O2 -fPIC)

add_library(tfm_core OBJECT
  src/ops.cpp
  src/layers.cpp
  src/model.cpp
  src/weights.cpp
  src/search.cpp)
target_include_directories(tfm_core PUBLIC include)
set_target_properties(tfm_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(tfm STATIC $<TARGET_OBJECTS:tfm_core>)
target_include_directories(tfm PUBLIC include)

# Apple: link Accelerate (BLAS) — used by ops.cpp under __APPLE__. Elsewhere the
# scalar fallback is used (or Eigen when TFM_USE_EIGEN is defined externally).
if(APPLE)
  target_link_libraries(tfm PUBLIC "-framework Accelerate")
endif()

add_library(model SHARED $<TARGET_OBJECTS:tfm_core> src/c_api.cpp)
target_include_directories(model PUBLIC include)
set_target_properties(model PROPERTIES OUTPUT_NAME model)

add_executable(tfm-cli src/main.cpp)
target_link_libraries(tfm-cli tfm)
target_link_libraries(tfm-cli m)

enable_testing()
add_executable(parity_test tests/parity.cpp)
target_link_libraries(parity_test tfm m)
add_test(NAME parity COMMAND parity_test
  ${CMAKE_SOURCE_DIR}/model.bin ${CMAKE_SOURCE_DIR}/fixtures)

add_executable(search_parity_test tests/search_parity.cpp)
target_link_libraries(search_parity_test tfm m)
add_test(NAME search_parity COMMAND search_parity_test
  ${CMAKE_SOURCE_DIR}/model.bin ${CMAKE_SOURCE_DIR}/fixtures)