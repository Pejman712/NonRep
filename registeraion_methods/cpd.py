def apply_cpd(source_cloud, target_cloud, downsampling_resolution=0.25):
    """
    Register two point clouds using Coherent Point Drift (CPD).
    
    Args:
        source_cloud: Source point cloud (either o3d.geometry.PointCloud or PCDWithIntensity).
        target_cloud: Target point cloud (either o3d.geometry.PointCloud or PCDWithIntensity).
        downsampling_resolution (float): Voxel size for downsampling point clouds.
        
    Returns:
        numpy.ndarray: 4x4 transformation matrix from source to target frame.
    """
    import numpy as np
    from probreg import cpd
    import open3d as o3d
    
    # Extract o3d.geometry.PointCloud from inputs if they are PCDWithIntensity objects
    source_o3d = source_cloud.pcd if hasattr(source_cloud, 'pcd') else source_cloud
    target_o3d = target_cloud.pcd if hasattr(target_cloud, 'pcd') else target_cloud
    
    # Ensure we're working with proper Open3D point clouds
    if not isinstance(source_o3d, o3d.geometry.PointCloud) or not isinstance(target_o3d, o3d.geometry.PointCloud):
        raise TypeError("Source and target must be either o3d.geometry.PointCloud or PCDWithIntensity objects")
    
    # Verify that point clouds have points
    if len(source_o3d.points) == 0:
        raise ValueError("Source point cloud has no points")
    if len(target_o3d.points) == 0:
        raise ValueError("Target point cloud has no points")
    
    # Print debug info
    print(f"Source cloud has {len(source_o3d.points)} points")
    print(f"Target cloud has {len(target_o3d.points)} points")
    
    # Downsample the point clouds
    source_down = source_o3d.voxel_down_sample(voxel_size=downsampling_resolution)
    target_down = target_o3d.voxel_down_sample(voxel_size=downsampling_resolution)
    
    # Verify that downsampled point clouds have points
    if len(source_down.points) == 0:
        raise ValueError(f"Downsampled source cloud has no points. Try a larger voxel size (current: {downsampling_resolution})")
    if len(target_down.points) == 0:
        raise ValueError(f"Downsampled target cloud has no points. Try a larger voxel size (current: {downsampling_resolution})")
    
    print(f"Downsampled source cloud has {len(source_down.points)} points")
    print(f"Downsampled target cloud has {len(target_down.points)} points")
    
    try:
        # Perform CPD registration
        tf_param, _, _ = cpd.registration_cpd(source_o3d, target_o3d)
        
        # Construct the 4x4 transformation matrix
        transformation_matrix = np.vstack((
            np.hstack((tf_param.rot, tf_param.t.reshape(3, 1))),
            np.array([[0, 0, 0, 1]])
        ))
        
        return transformation_matrix
    except Exception as e:
        print(f"CPD registration failed: {str(e)}")
        print("Returning identity matrix as fallback")
        return np.eye(4)