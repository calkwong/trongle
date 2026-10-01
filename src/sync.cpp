#include "sync.h"

#include <volk.h>

#include <cassert>

AccessInfo get_access_info(AccessType access_type)
{
    switch (access_type)
    {
    case AccessType::Nothing:
        return {
            VK_PIPELINE_STAGE_2_NONE,
            VK_ACCESS_2_NONE
        };

    case AccessType::CommandBufferReadNVX:
        return {
            VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_NV,
            VK_ACCESS_2_COMMAND_PREPROCESS_READ_BIT_NV
        };

    case AccessType::IndirectBuffer:
        return {
            VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
            VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT
        };

    case AccessType::IndexBuffer:
        return {
            VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT,
            VK_ACCESS_2_INDEX_READ_BIT
        };

    case AccessType::VertexBuffer:
        return {
            VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT,
            VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT
        };

    case AccessType::VertexShaderReadUniformBuffer:
        return {
            VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
            VK_ACCESS_2_SHADER_READ_BIT
        };

    case AccessType::VertexShaderReadSampledImageOrUniformTexelBuffer:
        return {
            VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
            VK_ACCESS_2_SHADER_READ_BIT
        };

    case AccessType::VertexShaderReadOther:
        return {
            VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
            VK_ACCESS_2_SHADER_READ_BIT
        };

    case AccessType::MeshShaderReadUniformBuffer:
        return {
            VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
            VK_ACCESS_2_SHADER_READ_BIT
        };

    case AccessType::MeshShaderReadSampledImageOrUniformTexelBuffer:
        return {
            VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
            VK_ACCESS_2_SHADER_READ_BIT
        };

    case AccessType::MeshShaderReadOther:
        return {
            VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
            VK_ACCESS_2_SHADER_READ_BIT
        };

    case AccessType::TaskShaderReadUniformBuffer:
        return {
            VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT,
            VK_ACCESS_2_SHADER_READ_BIT
        };

    case AccessType::TaskShaderReadSampledImageOrUniformTexelBuffer:
        return {
            VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT,
            VK_ACCESS_2_SHADER_READ_BIT
        };

    case AccessType::TaskShaderReadOther:
        return {
            VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT,
            VK_ACCESS_2_SHADER_READ_BIT
        };

    case AccessType::TessellationControlShaderReadUniformBuffer:
        return {
            VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT,
            VK_ACCESS_2_UNIFORM_READ_BIT
        };

    case AccessType::TessellationControlShaderReadSampledImageOrUniformTexelBuffer:
        return {
            VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT,
            VK_ACCESS_2_SHADER_READ_BIT
        };

    case AccessType::TessellationControlShaderReadOther:
        return {
            VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT,
            VK_ACCESS_2_SHADER_READ_BIT
        };

    case AccessType::TessellationEvaluationShaderReadUniformBuffer:
        return {
            VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT,
            VK_ACCESS_2_UNIFORM_READ_BIT
        };

    case AccessType::TessellationEvaluationShaderReadSampledImageOrUniformTexelBuffer:
        return {
            VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT,
            VK_ACCESS_2_SHADER_READ_BIT
        };

    case AccessType::TessellationEvaluationShaderReadOther:
        return {
            VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT,
            VK_ACCESS_2_SHADER_READ_BIT
        };

    case AccessType::GeometryShaderReadUniformBuffer:
        return {
            VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT,
            VK_ACCESS_2_UNIFORM_READ_BIT
        };

    case AccessType::GeometryShaderReadSampledImageOrUniformTexelBuffer:
        return {
            VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT,
            VK_ACCESS_2_SHADER_READ_BIT
        };

    case AccessType::GeometryShaderReadOther:
        return {
            VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT,
            VK_ACCESS_2_SHADER_READ_BIT
        };

    case AccessType::FragmentShaderReadUniformBuffer:
        return {
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_UNIFORM_READ_BIT
        };

    case AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer:
        return {
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_READ_BIT
        };

    case AccessType::FragmentShaderReadColorInputAttachment:
        return {
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT
        };

    case AccessType::FragmentShaderReadDepthStencilInputAttachment:
        return {
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT
        };

    case AccessType::FragmentShaderReadOther:
        return {
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_READ_BIT
        };

    case AccessType::ColorAttachmentRead:
        return {
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT
        };

    case AccessType::DepthStencilAttachmentRead:
        return {
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
        };

    case AccessType::DepthStencilAttachmentReadWrite:
        return {
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
        };

    case AccessType::ComputeShaderReadUniformBuffer:
        return {
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_UNIFORM_READ_BIT
        };

    case AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer:
        return {
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_READ_BIT
        };

    case AccessType::ComputeShaderReadOther:
        return {
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_READ_BIT
        };

    case AccessType::AnyShaderReadUniformBuffer:
        return {
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_ACCESS_2_UNIFORM_READ_BIT
        };

    case AccessType::AnyShaderReadUniformBufferOrVertexBuffer:
        return {
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT
        };

    case AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer:
        return {
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_ACCESS_2_SHADER_READ_BIT
        };

    case AccessType::AnyShaderReadOther:
        return {
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_ACCESS_2_SHADER_READ_BIT
        };

    case AccessType::TransferRead:
        return {
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT
        };

    case AccessType::HostRead:
        return {
            VK_PIPELINE_STAGE_2_HOST_BIT,
            VK_ACCESS_2_HOST_READ_BIT
        };

    case AccessType::Present:
        return {
            VK_PIPELINE_STAGE_2_NONE,
            VK_ACCESS_2_NONE
        };

    case AccessType::CommandBufferWriteNVX:
        return {
            VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_NV,
            VK_ACCESS_2_COMMAND_PREPROCESS_WRITE_BIT_NV
        };

    case AccessType::VertexShaderWrite:
        return {
            VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
            VK_ACCESS_2_SHADER_WRITE_BIT
        };

    case AccessType::MeshShaderWrite:
        return {
            VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
            VK_ACCESS_2_SHADER_WRITE_BIT
        };

    case AccessType::TaskShaderWrite:
        return {
            VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT,
            VK_ACCESS_2_SHADER_WRITE_BIT
        };

    case AccessType::TessellationControlShaderWrite:
        return {
            VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT,
            VK_ACCESS_2_SHADER_WRITE_BIT
        };

    case AccessType::TessellationEvaluationShaderWrite:
        return {
            VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT,
            VK_ACCESS_2_SHADER_WRITE_BIT
        };

    case AccessType::GeometryShaderWrite:
        return {
            VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT,
            VK_ACCESS_2_SHADER_WRITE_BIT
        };

    case AccessType::FragmentShaderWrite:
        return {
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_WRITE_BIT
        };

    case AccessType::ColorAttachmentWrite:
        return {
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
        };

    case AccessType::DepthStencilAttachmentWrite:
        return {
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
        };

    case AccessType::DepthAttachmentWriteStencilReadOnly:
        return {
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
        };

    case AccessType::StencilAttachmentWriteDepthReadOnly:
        return {
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
        };

    case AccessType::ComputeShaderWrite:
        return {
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_WRITE_BIT
        };

    case AccessType::ComputeShaderReadWrite:
        return {
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT
        };

    case AccessType::AnyShaderWrite:
        return {
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_ACCESS_2_SHADER_WRITE_BIT
        };

    case AccessType::TransferWrite:
        return {
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT
        };

    case AccessType::HostWrite:
        return {
            VK_PIPELINE_STAGE_2_HOST_BIT,
            VK_ACCESS_2_HOST_WRITE_BIT
        };

    case AccessType::ColorAttachmentReadWrite:
        return {
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
        };

    case AccessType::General:
        return {
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT
        };

    case AccessType::RayTracingShaderReadSampledImageOrUniformTexelBuffer:
        return {
            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            VK_ACCESS_2_SHADER_READ_BIT
        };

    case AccessType::RayTracingShaderReadColorInputAttachment:
        return {
            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT
        };

    case AccessType::RayTracingShaderReadDepthStencilInputAttachment:
        return {
            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT
        };

    case AccessType::RayTracingShaderReadAccelerationStructure:
        return {
            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR
        };

    case AccessType::RayTracingShaderReadOther:
        return {
            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            VK_ACCESS_2_SHADER_READ_BIT
        };

    case AccessType::AccelerationStructureBuildWrite:
        return {
            VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR
        };

    case AccessType::AccelerationStructureBuildRead:
        return {
            VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR
        };

    case AccessType::AccelerationStructureBufferWrite:
        return {
            VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_ACCESS_2_TRANSFER_WRITE_BIT
        };
    }

    assert(0 && "Unknown AccessType");
}

bool is_write_access(AccessType access_type)
{
    switch (access_type)
    {
    case AccessType::CommandBufferWriteNVX:
    case AccessType::VertexShaderWrite:
    case AccessType::MeshShaderWrite:
    case AccessType::TaskShaderWrite:
    case AccessType::TessellationControlShaderWrite:
    case AccessType::TessellationEvaluationShaderWrite:
    case AccessType::GeometryShaderWrite:
    case AccessType::FragmentShaderWrite:
    case AccessType::ColorAttachmentWrite:
    case AccessType::DepthStencilAttachmentWrite:
    case AccessType::DepthAttachmentWriteStencilReadOnly:
    case AccessType::DepthStencilAttachmentReadWrite:
    case AccessType::StencilAttachmentWriteDepthReadOnly:
    case AccessType::ComputeShaderWrite:
    case AccessType::ComputeShaderReadWrite:
    case AccessType::AnyShaderWrite:
    case AccessType::TransferWrite:
    case AccessType::HostWrite:
    case AccessType::ColorAttachmentReadWrite:
    case AccessType::General:
        return true;
    default:
        return false;
    }
}
