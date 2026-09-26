add_subdirectory(fmt)
add_subdirectory(volk)
add_subdirectory(vk-bootstrap)
add_subdirectory(SDL EXCLUDE_FROM_ALL)
add_subdirectory(VulkanMemoryAllocator)
add_subdirectory(fastgltf)
add_subdirectory(glm)
add_subdirectory(meshoptimizer)

add_library(basis_universal)

target_sources(basis_universal PRIVATE
  basis_universal/transcoder/basisu_transcoder.cpp
  basis_universal/zstd/zstd.c
)

target_include_directories(engine PRIVATE basis_universal/transcoder)
target_include_directories(engine PRIVATE basis_universal/zstd)

target_link_libraries(engine PRIVATE
  fmt::fmt
  volk
  vk-bootstrap::vk-bootstrap
  SDL3::SDL3
  GPUOpen::VulkanMemoryAllocator
  fastgltf::fastgltf
  glm::glm
  meshoptimizer
  basis_universal
)

find_package(TBB)
if (TBB_FOUND)
  target_link_libraries(engine PRIVATE TBB::tbb)
endif()
