add_subdirectory(fmt)
add_subdirectory(volk)
add_subdirectory(vk-bootstrap)
add_subdirectory(SDL EXCLUDE_FROM_ALL)
add_subdirectory(VulkanMemoryAllocator)
add_subdirectory(fastgltf)
add_subdirectory(glm)

target_link_libraries(engine PRIVATE
  fmt::fmt
  volk
  vk-bootstrap::vk-bootstrap
  SDL3::SDL3
  GPUOpen::VulkanMemoryAllocator
  fastgltf::fastgltf
  glm::glm
)
