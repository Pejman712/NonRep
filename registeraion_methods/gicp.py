import numpy as np
import small_gicp
from numpy.linalg import inv
def apply_gicp(source_cloud, target_cloud, voxel_size=0.05):
    """
    Register source cloud to target cloud using small_gicp
    
    Args:
        source_cloud: PCDWithIntensity object
        target_cloud: PCDWithIntensity object
        voxel_size: voxel size for downsampling
        
    Returns:
        4x4 transformation matrix
    """
    # Extract raw point data
    target_raw_numpy = np.asarray(target_cloud.pcd.points, dtype=np.float64)
    source_raw_numpy = np.asarray(source_cloud.pcd.points, dtype=np.float64)

    # Perform alignment
    result = small_gicp.align(source_raw_numpy, target_raw_numpy)

    return result.T_target_source