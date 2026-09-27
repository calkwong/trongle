
#include "camera.h"

#include <glm/common.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/trigonometric.hpp>
#include <glm/ext/quaternion_common.hpp>
#include <glm/ext/quaternion_float.hpp>
#include <glm/ext/quaternion_trigonometric.hpp>
#include <glm/geometric.hpp>
#include <SDL3/SDL_keyboard.h>
#include <SDL3/SDL_keycode.h>
#include <SDL3/SDL_mouse.h>
#include <SDL3/SDL_scancode.h>
#include <SDL3/SDL_video.h>
#include <SDL3/SDL_events.h>

#include <cstddef>

namespace
{
constexpr auto MIN_PITCH = glm::radians(-89.0f);
constexpr auto MAX_PITCH = glm::radians(89.0f);

glm::vec3 rotate_quat(glm::vec3 v, glm::quat quat)
{
    glm::vec3 q = glm::vec3(quat.x, quat.y, quat.z);
    return v + glm::vec3(2.0) * cross(q, cross(q, v) + quat.w * v);
}
} // namespace

glm::mat4 Camera::get_view_matrix() const
{
    glm::quat inv_rot = glm::conjugate(get_rotation_matrix());
    glm::mat4 view = glm::mat4_cast(inv_rot);

    view[3] = glm::vec4(rotate_quat(-position, inv_rot), 1.0);

    return view;
}

glm::quat Camera::get_rotation_matrix() const
{
    const glm::quat pitch_rotation = glm::angleAxis(pitch, glm::vec3(1.0f, 0.0f, 0.0f));
    const glm::quat yaw_rotation = glm::angleAxis(yaw, glm::vec3(0.0f, -1.0f, 0.0f));

    return yaw_rotation * pitch_rotation;
}

void Camera::process_sdl_event(const SDL_Event& e, bool relative_mouse_mode)
{
    if (!relative_mouse_mode)
    {
        velocity.x = 0.0;
        velocity.z = 0.0;
        return;
    }

    const bool* state = SDL_GetKeyboardState(NULL);
    velocity.x = static_cast<float>(state[SDL_SCANCODE_D]) - static_cast<float>(state[SDL_SCANCODE_A]);
    velocity.z = static_cast<float>(state[SDL_SCANCODE_S]) - static_cast<float>(state[SDL_SCANCODE_W]);

    if (e.type == SDL_EVENT_MOUSE_MOTION)
    {
        yaw += static_cast<float>(e.motion.xrel) * sensitivity;
        pitch -= static_cast<float>(e.motion.yrel) * sensitivity;
        pitch = glm::clamp(pitch, MIN_PITCH, MAX_PITCH);
    }
}

void Camera::update(float deltatime)
{
    glm::quat camera_rotation = get_rotation_matrix();
    position += rotate_quat(glm::vec3(velocity * speed * deltatime), camera_rotation);
}

glm::mat4 Camera::set_perspective_matrix(float fovy, float aspect, float znear)
{
    const float f = 1.0f / glm::tan(fovy / 2.0f);

    // clang-format off
	perspective = glm::mat4(
	    f / aspect, 0.0f, 0.0f, 0.0f,
	    0.0f, f, 0.0f, 0.0f,
	    0.0f, 0.0f, 0.0f, -1.0f,
	    0.0f, 0.0f, znear, 0.0f
	);
    // clang-format on

    return perspective;
}
