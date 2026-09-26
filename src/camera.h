#pragma once

#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/quaternion_float.hpp>

union SDL_Event;

class Camera
{
public:
    glm::vec3 position;
    float pitch;
    glm::vec3 velocity;
    float yaw;

    glm::mat4 perspective;

    float near = 0.01f;
    float far = 100.0f;
    float fov = 70.0f;
    float speed = 7.0f;
    float sensitivity = 0.005f;

    glm::mat4 get_view_matrix() const;
    glm::quat get_rotation_matrix() const;
    glm::mat4 set_perspective_matrix(float fovy, float aspect, float znear);
    void process_sdl_event(const SDL_Event& e, bool relative_mouse_mode);
    void update(float deltatime);
};
