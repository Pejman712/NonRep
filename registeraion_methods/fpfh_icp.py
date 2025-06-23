def apply_fpfh_icp(source_cloud, target_cloud, voxel_size=0.01, distance_threshold=0.05, 
                   ransac_n=5, icp_max_iteration=200, icp_threshold=0.01):
    """
    Register two point clouds using FPFH features for global registration followed by ICP refinement.
    
    Args:
        source_cloud: Source point cloud (either o3d.geometry.PointCloud or PCDWithIntensity).
        target_cloud: Target point cloud (either o3d.geometry.PointCloud or PCDWithIntensity).
        voxel_size (float): Voxel size for downsampling.
        distance_threshold (float): Maximum correspondence distance for RANSAC.
        ransac_n (int): Number of points to use for RANSAC.
        icp_max_iteration (int): Maximum number of ICP iterations.
        icp_threshold (float): Distance threshold for ICP convergence.
        
    Returns:
        numpy.ndarray: 4x4 transformation matrix from source to target frame.
    """
    import numpy as np
    import open3d as o3d
    import copy
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
    
    print(f"Source cloud has {len(source_o3d.points)} points")
    print(f"Target cloud has {len(target_o3d.points)} points")
    
    # Copy to avoid modifying original point clouds
    source = copy.deepcopy(source_o3d)
    target = copy.deepcopy(target_o3d)
    
    # Compute normals for the full point clouds (needed for point-to-plane ICP)
    print("Computing normals for full point clouds")
    source.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    
    # Downsample point clouds
    source_down = source.voxel_down_sample(voxel_size=voxel_size)
    target_down = target.voxel_down_sample(voxel_size=voxel_size)
    
    print(f"Downsampled source cloud has {len(source_down.points)} points")
    print(f"Downsampled target cloud has {len(target_down.points)} points")
    
    # The downsampled point clouds inherit normals from the original cloud,
    # but we recompute them to be safe
    print("Computing normals for downsampled clouds")
    source_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    target_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    
    # Compute FPFH features
    print("Computing FPFH features")
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, 
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
    
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, 
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
    
    # Global registration with RANSAC
    print("Performing global registration with RANSAC")
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    
    if result_ransac.transformation.trace() == 4.0:
        print("Global registration failed. Using identity as initial transformation.")
        initial_transform = np.eye(4)
    else:
        initial_transform = result_ransac.transformation
        print(f"Global registration succeeded with fitness: {result_ransac.fitness}")
        print(f"RANSAC Transformation:\n{initial_transform}")
    
    # Refine with ICP
    print("Refining with point-to-plane ICP")
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, icp_threshold, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_max_iteration))
    
    final_transform = result_icp.transformation
    print(f"ICP registration fitness: {result_icp.fitness}")
    print(f"Final transformation:\n{final_transform}")
    
    return final_transform