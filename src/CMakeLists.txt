target_sources(engine PRIVATE
  engine.cpp
  swapchain.cpp
  resources.cpp
  descriptors.cpp
)

set_target_properties(engine PROPERTIES CXX_STANDARD 20)
