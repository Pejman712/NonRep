def apply_ndt(source_cloud, target_cloud, voxel_size=0.5, max_iter=1000, 
              tol=1e-8, init_transform=None):
    """
    Register two point clouds using NDT from the point_cloud_registration library.
    
    Args:
        source_cloud: Source point cloud (either o3d.geometry.PointCloud or PCDWithIntensity).
        target_cloud: Target point cloud (either o3d.geometry.PointCloud or PCDWithIntensity).
        voxel_size (float): Voxel size for the NDT algorithm.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.
        init_transform (numpy.ndarray, optional): Initial 4x4 transformation. If None, identity is used.
        
    Returns:
        numpy.ndarray: 4x4 transformation matrix from source to target frame.
    """
    import numpy as np
    import open3d as o3d
    from point_cloud_registration import NDT
    from numpy.linalg import inv
    
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
    
    # Convert Open3D point clouds to numpy arrays for NDT
    source_points = np.asarray(source_o3d.points)
    target_points = np.asarray(target_o3d.points)
    
    print(f"Source cloud has {len(source_points)} points")
    print(f"Target cloud has {len(target_points)} points")
    
    # Initialize NDT without the min_step_size parameter
    print(f"Initializing NDT with voxel_size={voxel_size}, max_iter={max_iter}, tol={tol}")
    ndt_reg = NDT(voxel_size=voxel_size, max_iter=max_iter, tol=tol)
    
    # Set the target point cloud
    print("Setting target point cloud")
    ndt_reg.set_target(target_points)
    
    # Set initial transformation
    if init_transform is None:
        init_transform = np.eye(4)
    
    # Align source to target
    print("Running NDT alignment")
    try:
        final_transform = ndt_reg.align(source_points, init_T=init_transform)
        print("NDT registration completed successfully")
        print(f"Final transformation:\n{final_transform}")
        return final_transform
    except Exception as e:
        print(f"Error during NDT registration: {str(e)}")
        print("Returning initial transformation as fallback")
        return init_transform