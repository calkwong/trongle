module bindings;

public static const uint INVALID_ID = 0xFFFFFFFFu;

[vk::binding(0, 0)]
__DynamicResource<__DynamicResourceKind.General> storage_buffers[];

[vk::binding(0, 1)]
__DynamicResource<__DynamicResourceKind.General> storage_images[];

[vk::binding(1, 1)]
__DynamicResource<__DynamicResourceKind.General> sampled_images[];

[vk::binding(0, 2)]
__DynamicResource<__DynamicResourceKind.Sampler> samplers[];

public T get_storage_image<T>(uint handle)
    where T : IOpaqueDescriptor
{
    return storage_images[handle].asOpaqueDescriptor<T>();
}

public T get_nonuniform_sampled_image<T>(uint handle)
    where T : IOpaqueDescriptor
{
    return sampled_images[NonUniformResourceIndex(handle)].asOpaqueDescriptor<T>();
}

public T get_sampled_image<T>(uint handle)
    where T : IOpaqueDescriptor
{
    return sampled_images[handle].asOpaqueDescriptor<T>();
}

public static const uint LINEAR_SAMPLER = 0;

public T get_sampler<T>(uint handle)
    where T : IOpaqueDescriptor
{
    return samplers[handle].asOpaqueDescriptor<T>();
}
