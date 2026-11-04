#pragma once

#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/ext/vector_float3.hpp>

#include <cstdint>

void decompose_transform(const glm::mat4& m, glm::vec3& t, glm::vec3& s, glm::vec4& rotation);
uint32_t get_group_count(uint32_t size, uint32_t threads);
uint32_t nearest_pow2(uint32_t extent);
uint32_t next_pow2(uint32_t extent);
