include_directories(${DDK_PATH})
include_directories($(CCSRC_DIR)/plugin/device/cpu/kernel)
include_directories(${CMAKE_SOURCE_DIR}/../../../../../../foundation/ai/neural_network_runtime/)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE NNRT_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
)
file(GLOB convert_source checker/*.cc)

add_library(nnrt_mid OBJECT ${NNRT_SRC} ${convert_source})
target_include_directories(nnrt_mid PUBLIC ${CMAKE_SOURCE_DIR}/../../../../../../foundation/ai/neural_network_runtime/)