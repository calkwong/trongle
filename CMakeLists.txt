cmake_minimum_required(VERSION 3.14)
project(engine)

find_package(Vulkan REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(engine)
add_subdirectory(src)
add_subdirectory(external)

find_program(SLANG_COMPILER slangc HINTS /usr/bin /usr/local/bin $ENV{VULKAN_SDK}/Bin/ $ENV{VULKAN_SDK}/Bin32/)

if(SLANG_COMPILER)
  execute_process(
    COMMAND ${SLANG_COMPILER} -v
    COMMAND_ECHO STDOUT
    )
endif()

file(GLOB SLANG_SOURCE_FILES
    "${PROJECT_SOURCE_DIR}/shaders/*.slang"
    )

file(GLOB SLANG_MODULES
    "${PROJECT_SOURCE_DIR}/shaders/modules/*.slang"
    )

foreach(SLANG ${SLANG_MODULES})
  message(STATUS "BUILDING MODULES")
  get_filename_component(FILE_NAME ${SLANG} NAME)
  set(SLANG_MODULE "${PROJECT_SOURCE_DIR}/shaders/compiled/${FILE_NAME}.slang-module")
  set(SLANG_MODULE_DEP "${SLANG_MODULE}.d")
  message(STATUS ${SLANG})
  add_custom_command(
    OUTPUT ${SLANG_MODULE}
    COMMAND ${SLANG_COMPILER} ${SLANG} -matrix-layout-column-major -fvk-use-scalar-layout -depfile ${SLANG_MODULE_DEP} -o ${SLANG_MODULE}
    DEPFILE ${SLANG_MODULE_DEP})
  list(APPEND SPIRV_BINARY_FILES ${SLANG_MODULE})
endforeach(SLANG)

foreach(SLANG ${SLANG_SOURCE_FILES})
  message(STATUS "BUILDING SHADER")
  get_filename_component(FILE_NAME ${SLANG} NAME)
  set(SPIRV "${PROJECT_SOURCE_DIR}/shaders/compiled/${FILE_NAME}.spv")
  set(SPIRV_DEP "${SPIRV}.d")
  message(STATUS ${SLANG})
  add_custom_command(
    OUTPUT ${SPIRV}
    COMMAND ${SLANG_COMPILER} ${SLANG} -target spirv -matrix-layout-column-major -fvk-use-entrypoint-name -fvk-use-scalar-layout
      -I "${PROJECT_SOURCE_DIR}/shaders/compiled/" -I "${PROJECT_SOURCE_DIR}/shaders/modules/"
      -Wno-39001 -depfile ${SPIRV_DEP} -o ${SPIRV}
    DEPFILE ${SPIRV_DEP})
  list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(SLANG)

add_custom_target(
    Shaders
    DEPENDS ${SPIRV_BINARY_FILES}
    )

add_dependencies(engine Shaders)
