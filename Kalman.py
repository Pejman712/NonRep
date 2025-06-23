import numpy as np
import open3d as o3d
import os
import glob
from typing import List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
@dataclass
class KalmanState:
    """Kalman filter state for 6DOF transformation (translation + rotation)"""
    # State vector: [x, y, z, roll, pitch, yaw, dx, dy, dz, droll, dpitch, dyaw]
    state: np.ndarray # 12x1 state vector
    covariance: np.ndarray # 12x12 covariance matrix

class PointCloudKalmanFilter:
    def __init__(self, dt: float = 5.0, process_noise: float = 0.5, measurement_noise: float = 0.01):
        """
        Initialize Kalman filter for point cloud tracking using GICP for both prediction and observation
        Args:
            dt: Time step between measurements
            process_noise: Process noise standard deviation
            measurement_noise: Measurement noise standard deviation
        """
        self.dt = dt
        self.state_dim = 6  # [x,y,z,roll,pitch,yaw] - no velocity terms needed
        self.measurement_dim = 6 # [x,y,z,roll,pitch,yaw]
        
        # State transition matrix (identity - prediction comes from GICP)
        self.F = np.eye(self.state_dim)
        
        # Measurement matrix (direct observation of pose)
        self.H = np.eye(self.measurement_dim)
        
        # Process noise covariance (uncertainty in GICP prediction)
        self.Q = np.eye(self.state_dim) * (process_noise ** 2)
        
        # Measurement noise covariance (uncertainty in GICP observation)
        self.R = np.eye(self.measurement_dim) * (measurement_noise ** 2)
        
        # Initialize state
        self.kalman_state = None
        
        # Store previous clouds for prediction
        self.previous_observation_cloud = None
        self.previous_prediction_cloud = None
        self.prediction_pose = None

    def transformation_to_pose(self, transformation_matrix: np.ndarray) -> np.ndarray:
        """Convert 4x4 transformation matrix to pose vector [x,y,z,roll,pitch,yaw]"""
        # Extract translation
        translation = transformation_matrix[:3, 3]
        
        # Extract rotation matrix and convert to Euler angles
        rotation_matrix = transformation_matrix[:3, :3]
        
        # Convert rotation matrix to Euler angles (roll, pitch, yaw)
        sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            yaw = 0
            
        return np.array([translation[0], translation[1], translation[2], roll, pitch, yaw])

    def pose_to_transformation(self, pose: np.ndarray) -> np.ndarray:
        """Convert pose vector [x,y,z,roll,pitch,yaw] to 4x4 transformation matrix"""
        x, y, z, roll, pitch, yaw = pose
        
        # Create rotation matrix from Euler angles
        R_x = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
        
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
        
        R = R_z @ R_y @ R_x
        
        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        
        return T

    def predict_with_gicp(self, source_cloud, target_cloud, apply_gicp_func, cumulative_transform=None):
        """
        Predict step using GICP between source and target clouds
        Args:
            source_cloud: Source point cloud (current observation)
            target_cloud: Target point cloud (prediction cloud)
            apply_gicp_func: GICP function to use
            cumulative_transform: Previous cumulative transformation matrix (4x4)
        """
        if self.kalman_state is None:
            # No prediction for first step
            self.prediction_pose = None
            return cumulative_transform
            
        try:
            # Get GICP transformation from source to target (relative transform)
            current_transformation = apply_gicp_func(source_cloud, target_cloud)
            
            # Accumulate transformation: cumulative = cumulative * current
            if cumulative_transform is not None:
                cumulative_transformation = cumulative_transform @ current_transformation
            else:
                cumulative_transformation = current_transformation
            
            self.prediction_pose = self.transformation_to_pose(cumulative_transformation)
            
            # Predict state using cumulative GICP result
            self.kalman_state.state = self.prediction_pose.copy()
            
            # Predict covariance
            self.kalman_state.covariance = self.F @ self.kalman_state.covariance @ self.F.T + self.Q
            
            print(f"GICP Prediction pose (cumulative): {self.prediction_pose}")
            
            return cumulative_transformation
            
        except Exception as e:
            print(f"Error in GICP prediction: {e}")
            # Fall back to previous state
            if self.kalman_state is not None:
                self.prediction_pose = self.kalman_state.state.copy()
            return cumulative_transform

    def update_with_gicp(self, source_cloud, target_cloud, apply_gicp_func, cumulative_transform=None):
        """
        Update step using GICP between source and target clouds
        Args:
            source_cloud: Source point cloud (current observation)
            target_cloud: Target point cloud (next observation or reference)
            apply_gicp_func: GICP function to use
            cumulative_transform: Previous cumulative transformation matrix (4x4)
        """
        try:
            # Get GICP transformation from source to target (relative transform)
            current_transformation = apply_gicp_func(source_cloud, target_cloud)
            
            # Accumulate transformation: cumulative = cumulative * current
            if cumulative_transform is not None:
                cumulative_transformation = cumulative_transform @ current_transformation
            else:
                cumulative_transformation = current_transformation
            
            observation_pose = self.transformation_to_pose(cumulative_transformation)
            
            print(f"GICP Observation pose (cumulative): {observation_pose}")
            
            if self.kalman_state is None:
                # Initialize state with first observation
                initial_covariance = np.eye(self.state_dim) * 1.0
                self.kalman_state = KalmanState(observation_pose.copy(), initial_covariance)
                print("Initialized Kalman state with first observation")
                return observation_pose, cumulative_transformation
            
            # Standard Kalman update
            # Innovation (difference between observation and prediction)
            if self.prediction_pose is not None:
                y = observation_pose - self.prediction_pose
            else:
                y = observation_pose - self.kalman_state.state
                
            # Innovation covariance
            S = self.H @ self.kalman_state.covariance @ self.H.T + self.R
            
            # Kalman gain
            K = self.kalman_state.covariance @ self.H.T @ np.linalg.inv(S)
            
            # Update state
            self.kalman_state.state = self.kalman_state.state + K @ y
            
            # Update covariance
            I_KH = np.eye(self.state_dim) - K @ self.H
            self.kalman_state.covariance = I_KH @ self.kalman_state.covariance
            
            print(f"Kalman filtered pose: {self.kalman_state.state}")
            return observation_pose, cumulative_transformation
            
        except Exception as e:
            print(f"Error in GICP observation: {e}")
            return None, cumulative_transform

    def get_current_pose(self) -> Optional[np.ndarray]:
        """Get current estimated pose"""
        if self.kalman_state is None:
            return None
        return self.kalman_state.state

def compare_to_ground_truth_positions(filtered_poses: List[np.ndarray], ground_truth_csv_path: str):
    """
    Compare Kalman filtered positions to ground truth (x, y, z only).
    Args:
        filtered_poses: List of 6-DOF pose arrays [x,y,z,roll,pitch,yaw]
        ground_truth_csv_path: Path to CSV file containing columns: time, x, y, z
    """
    # Load ground truth
    gt_df = pd.read_csv(ground_truth_csv_path)

    if not all(col in gt_df.columns for col in ["X", "Y", "Z"]):
        print("Error: CSV must have columns: time, x, y, z")
        return

    gt_positions = gt_df[["X", "Y", "Z"]].to_numpy()
    filtered_positions = np.array([pose[:3] if pose is not None else [np.nan, np.nan, np.nan] for pose in filtered_poses])

    # Truncate to shortest length
    min_len = min(len(filtered_positions), len(gt_positions))
    gt_positions = gt_positions[:min_len]
    filtered_positions = filtered_positions[:min_len]

    # Compute Euclidean position error
    position_errors = np.linalg.norm(filtered_positions - gt_positions, axis=1)

    # Summary statistics
    print("\n=== Position Error vs Ground Truth ===")
    print(f"Compared {min_len} frames")
    print(f"Mean error: {np.nanmean(position_errors):.4f} m")
    print(f"Max error: {np.nanmax(position_errors):.4f} m")
    print(f"Std deviation: {np.nanstd(position_errors):.4f} m")

    # Plot error
    plt.figure(figsize=(10, 4))
    plt.plot(position_errors, label="Position Error (m)")
    plt.xlabel("Frame Index")
    plt.ylabel("Error (m)")
    plt.title("Kalman Filter Position Error vs Ground Truth")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def generate_prediction_cloud(previous_cloud, current_cloud, method="extrapolation"):
    """
    Generate a prediction cloud for the next time step
    Args:
        previous_cloud: Point cloud from previous time step
        current_cloud: Point cloud from current time step  
        method: Method to generate prediction ("extrapolation", "copy", "motion_model")
    Returns:
        Predicted point cloud for next time step
    """
    if previous_cloud is None:
        # No previous cloud, return current as prediction
        return o3d.geometry.PointCloud(current_cloud)
    
    if method == "copy":
        # Simple approach: predict next cloud will be same as current
        return o3d.geometry.PointCloud(current_cloud)
    
    elif method == "extrapolation":
        # Extrapolate motion from previous to current
        try:
            # Compute transformation from previous to current
            import small_gicp
            prev_points = np.asarray(previous_cloud.points, dtype=np.float64)
            curr_points = np.asarray(current_cloud.points, dtype=np.float64)
            
            # Get transformation
            result = small_gicp.align(prev_points, curr_points)
            transformation = result.T_target_source
            
            # Apply same transformation again to extrapolate
            predicted_cloud = o3d.geometry.PointCloud(current_cloud)
            predicted_cloud.transform(np.linalg.inv(transformation))
            
            return predicted_cloud
            
        except Exception as e:
            print(f"Error in extrapolation: {e}, falling back to copy method")
            return o3d.geometry.PointCloud(current_cloud)
    
    else:
        # Default to copy
        return o3d.geometry.PointCloud(current_cloud)

def load_pcd_files(folder_path: str,
                   step_size: int = 1,
                   start_index: int = 0,
                   max_clouds: int = None,
                   voxel_size: float = 0.5,
                   apply_sor: bool = True,
                   sor_nb_neighbors: int = 50,
                   sor_std_ratio: float = 1.0) -> List[Tuple[str, o3d.geometry.PointCloud]]:
    """
    Load PCD files from a folder with optional downsampling and SOR filtering.

    Args:
        folder_path: Path to folder containing PCD files
        step_size: Load every Nth file (e.g., step_size=5 loads every 5th file)
        start_index: Starting index for sampling (0-based)
        max_clouds: Maximum number of clouds to load (None for no limit)
        voxel_size: Voxel size for downsampling (0 disables downsampling)
        apply_sor: Whether to apply Statistical Outlier Removal
        sor_nb_neighbors: Number of neighbors to analyze for SOR
        sor_std_ratio: Standard deviation multiplier for SOR
    Returns:
        List of (filename, pointcloud) tuples
    """
    pcd_files = glob.glob(os.path.join(folder_path, "*.pcd"))
    pcd_files.sort()  # Sort to ensure consistent ordering

    # Apply sampling
    sampled_files = pcd_files[start_index::step_size]

    # Apply max_clouds limit
    if max_clouds is not None and max_clouds > 0:
        sampled_files = sampled_files[:max_clouds]

    print(f"Found {len(pcd_files)} total PCD files")
    print(f"Sampling every {step_size} files starting from index {start_index}")
    if max_clouds is not None:
        print(f"Limited to first {max_clouds} clouds after sampling")
    print(f"Will process {len(sampled_files)} files: {[os.path.basename(f) for f in sampled_files[:5]]}{'...' if len(sampled_files) > 5 else ''}")

    point_clouds = []
    for file_path in sampled_files:
        try:
            pcd = o3d.io.read_point_cloud(file_path)
            if len(pcd.points) == 0:
                print(f"Warning: Empty point cloud in {file_path}")
                continue

            # Apply voxel downsampling
            if voxel_size > 0:
                pcd = pcd.voxel_down_sample(voxel_size)

            # Apply SOR filtering
            if apply_sor:
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=sor_nb_neighbors,
                                                        std_ratio=sor_std_ratio)

            point_clouds.append((os.path.basename(file_path), pcd))
            print(f"Loaded {file_path}: {len(pcd.points)} points")

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return point_clouds


def apply_gicp_wrapper(source_cloud, target_cloud, apply_gicp_func, voxel_size=0.05):
    """
    Wrapper function to adapt Open3D PointClouds for your apply_gicp function
    Args:
        source_cloud: Open3D PointCloud object
        target_cloud: Open3D PointCloud object
        apply_gicp_func: Your apply_gicp function
        voxel_size: voxel size parameter for your function
    Returns:
        4x4 transformation matrix
    """
    import numpy as np
    
    # Create simple wrapper class that mimics your expected input
    class PCDWrapper:
        def __init__(self, o3d_cloud):
            self.pcd = o3d_cloud
    
    # Wrap the Open3D clouds
    source_wrapped = PCDWrapper(source_cloud)
    target_wrapped = PCDWrapper(target_cloud)
    
    # Call your function
    return apply_gicp_func(source_wrapped, target_wrapped, voxel_size)

def apply_gicp_direct(source_cloud, target_cloud, voxel_size=0.05):
    """
    Direct implementation using small_gicp only
    Args:
        source_cloud: Open3D PointCloud object
        target_cloud: Open3D PointCloud object
        voxel_size: voxel size for downsampling (unused but kept for compatibility)
    Returns:
        4x4 transformation matrix
    """
    import numpy as np
    import small_gicp
    
    # Extract raw point data directly from Open3D PointClouds
    target_raw_numpy = np.asarray(target_cloud.points, dtype=np.float64)
    source_raw_numpy = np.asarray(source_cloud.points, dtype=np.float64)
    
    # Check if clouds have points
    if len(source_raw_numpy) == 0 or len(target_raw_numpy) == 0:
        print("Warning: Empty point cloud detected, returning identity matrix")
        return np.eye(4)
    
    # Perform alignment using small_gicp
    result = small_gicp.align(source_raw_numpy, target_raw_numpy)
    return result.T_target_source

def apply_gicp_open3d_fallback(source_cloud, target_cloud, voxel_size=0.05):
    """
    Fallback GICP implementation using Open3D's built-in ICP
    Args:
        source_cloud: Open3D PointCloud object
        target_cloud: Open3D PointCloud object
        voxel_size: voxel size for downsampling
    Returns:
        4x4 transformation matrix
    """
    import numpy as np
    import open3d as o3d
    
    try:
        # Create copies to avoid modifying originals
        source_copy = o3d.geometry.PointCloud(source_cloud)
        target_copy = o3d.geometry.PointCloud(target_cloud)
        
        # Downsample for efficiency
        if voxel_size > 0:
            source_copy = source_copy.voxel_down_sample(voxel_size)
            target_copy = target_copy.voxel_down_sample(voxel_size)
        
        # Estimate normals for better ICP
        source_copy.estimate_normals()
        target_copy.estimate_normals()
        
        # Initial alignment using global registration if available
        try:
            # Try FPFH-based global registration first
            radius_feature = voxel_size * 5
            source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                source_copy, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
            target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                target_copy, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
            
            # Global registration
            distance_threshold = voxel_size * 1.5
            global_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_copy, target_copy, source_fpfh, target_fpfh, True,
                distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3, [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
                ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
            
            initial_transform = global_result.transformation
        except:
            # Fallback to identity if global registration fails
            initial_transform = np.eye(4)
        
        # Refined ICP registration
        distance_threshold = voxel_size * 2.0
        icp_result = o3d.pipelines.registration.registration_icp(
            source_copy, target_copy, distance_threshold, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        
        if icp_result.fitness > 0.1:  # Reasonable fitness threshold
            return icp_result.transformation
        else:
            print(f"Warning: Low ICP fitness ({icp_result.fitness:.3f}), returning identity")
            return np.eye(4)
            
    except Exception as e:
        print(f"Error in Open3D ICP fallback: {e}, returning identity matrix")
        return np.eye(4)

def visualize_point_clouds(source_cloud, target_cloud, registered_cloud, title="Point Cloud Registration"):
    """
    Visualize source, target, and registered point clouds using Open3D
    Args:
        source_cloud: Original source point cloud
        target_cloud: Target point cloud  
        registered_cloud: Transformed source point cloud
        title: Window title
    """
    # Create copies to avoid modifying originals
    source_vis = o3d.geometry.PointCloud(source_cloud)
    target_vis = o3d.geometry.PointCloud(target_cloud)
    registered_vis = o3d.geometry.PointCloud(registered_cloud)
    
    # Color the point clouds
    source_vis.paint_uniform_color([1, 0, 0])      # Red for source
    target_vis.paint_uniform_color([0, 1, 0])      # Green for target
    registered_vis.paint_uniform_color([0, 0, 1])  # Blue for registered
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1200, height=800)
    
    # Add point clouds
    vis.add_geometry(source_vis)
    vis.add_geometry(target_vis)
    vis.add_geometry(registered_vis)
    
    # Set viewing parameters
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark background
    opt.point_size = 2.0
    
    # Run visualizer
    vis.run()
    vis.destroy_window()

def visualize_final_registered_cloud(observation_pcds, target_cloud, filtered_poses, kalman_filter):
    """
    Create and visualize the final registered point cloud using Kalman filtered poses
    Args:
        observation_pcds: List of (name, point_cloud) tuples
        target_cloud: Reference target point cloud (first observation)
        filtered_poses: List of Kalman filtered poses
        kalman_filter: KalmanFilter instance for pose conversion
    """
    print("\nCreating final registered point cloud using Kalman filtered poses...")

    # Combined global map
    combined_registered = o3d.geometry.PointCloud()
    
    for i, ((obs_name, obs_cloud), filtered_pose) in enumerate(zip(observation_pcds, filtered_poses)):
        if filtered_pose is not None:
            # Convert Kalman filtered pose to transformation matrix
            filtered_transform = kalman_filter.pose_to_transformation(filtered_pose)

            # Apply the Kalman filtered transformation to the observation cloud
            transformed_cloud = o3d.geometry.PointCloud(obs_cloud)
            transformed_cloud.transform(filtered_transform)

            # Add to the global map
            combined_registered += transformed_cloud

            print(f"Added {obs_name} with Kalman filtered pose: {filtered_pose[:3]}")
        else:
            print(f"Skipping {obs_name} due to missing Kalman filtered pose.")

    # Downsample the final map for visualization
    voxel_size = 0.02
    combined_registered = combined_registered.voxel_down_sample(voxel_size)

    print(f"Final Kalman filtered map contains {len(combined_registered.points)} points")

    # Visualize
    visualize_point_clouds(
        source_cloud=o3d.geometry.PointCloud(),  # Empty
        target_cloud=target_cloud,
        registered_cloud=combined_registered,
        title="Final Map (Kalman Filtered Poses)"
    )

    return combined_registered


def plot_pose_evolution(filtered_poses, raw_poses):
    """
    Plot the evolution of poses over time (raw vs filtered)
    Args:
        filtered_poses: List of Kalman filtered poses
        raw_poses: List of raw GICP poses
    """
    # Filter out None values
    valid_filtered = [pose for pose in filtered_poses if pose is not None]
    valid_raw = [pose for pose in raw_poses if pose is not None]
    
    if len(valid_filtered) == 0 or len(valid_raw) == 0:
        print("No valid poses to plot")
        return
    
    # Convert to numpy arrays
    filtered_array = np.array(valid_filtered)
    raw_array = np.array(valid_raw)
    
    # Create time axis
    time_steps = np.arange(len(valid_filtered))
    
    # Create subplots
    fig, axes = plt.subplot(2, 3, figsize=(15, 10))
    fig.suptitle('Pose Evolution: Raw vs Kalman Filtered', fontsize=16)
    
    pose_labels = ['X (m)', 'Y (m)', 'Z (m)', 'Roll (rad)', 'Pitch (rad)', 'Yaw (rad)']
    
    for i in range(6):
        ax = axes[i//3, i%3]
        ax.plot(time_steps, raw_array[:, i], 'r-', label='Raw GICP', alpha=0.7, linewidth=2)
        ax.plot(time_steps, filtered_array[:, i], 'b-', label='Kalman Filtered', linewidth=2)
        ax.set_xlabel('Time Step')
        ax.set_ylabel(pose_labels[i])
        ax.set_title(f'{pose_labels[i]} Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def save_results(results, transformations, filtered_poses, combined_cloud=None, output_dir="output"):
    """
    Save all results to files
    Args:
        results: Processing results
        transformations: Raw transformation matrices
        filtered_poses: Kalman filtered poses
        combined_cloud: Final registered point cloud
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save numerical results
    np.save(os.path.join(output_dir, 'kalman_results.npy'), results)
    np.save(os.path.join(output_dir, 'transformations.npy'), transformations)
    np.save(os.path.join(output_dir, 'filtered_poses.npy'), filtered_poses)
    
    # Save final registered point cloud if provided
    if combined_cloud is not None:
        o3d.io.write_point_cloud(os.path.join(output_dir, 'final_registered_cloud.pcd'), combined_cloud)
        print(f"Saved final registered cloud to {output_dir}/final_registered_cloud.pcd")
    
    print(f"Results saved to {output_dir}/")

def process_point_clouds_with_kalman(observation_folder: str,
                                   prediction_folder: str = None,
                                   apply_gicp_func=None,
                                   visualize=True,
                                   prediction_method="extrapolation",
                                   observation_step_size=None,
                                   observation_start_index=0,
                                   prediction_step_size=None,
                                   prediction_start_index=0,
                                   use_separate_prediction_folder=False,
                                   max_observation_clouds=None,
                                   max_prediction_clouds=None):
    """
    Main function to process point clouds with dual GICP Kalman filtering
    
    UPDATED GICP Registration Strategy:
    - Prediction GICP: cloud_i from observation → cloud_i from prediction (same file index, starts from first)
    - Observation GICP: cloud_i from observation → reference_cloud (CUMULATIVE from first observation)
    
    This change ensures that observation_transformation gives cumulative poses from the beginning,
    not just consecutive transformations.
    
    Args:
        observation_folder: Path to observation PCD files
        prediction_folder: Path to prediction PCD files (if using separate folder, can be None)
        apply_gicp_func: Your GICP function that takes (source, target) and returns transformation
        visualize: Whether to show visualizations
        prediction_method: Method for generating prediction clouds ("extrapolation", "copy", "separate_folder")
        observation_step_size: Load every Nth observation file
        observation_start_index: Starting index for observation sampling
        prediction_step_size: Load every Nth prediction file (when using separate folder)
        prediction_start_index: Starting index for prediction sampling
        use_separate_prediction_folder: Whether to use a separate folder for prediction clouds
        max_observation_clouds: Maximum number of observation clouds to process (None for no limit)
        max_prediction_clouds: Maximum number of prediction clouds to load (None for no limit)
    """
    # Use default GICP function if none provided
    if apply_gicp_func is None:
        apply_gicp_func = apply_gicp_direct
    
    # Load observation clouds with sampling and limit
    print("Loading observation point clouds...")
    observation_pcds = load_pcd_files(observation_folder, observation_step_size, observation_start_index, max_observation_clouds)
    
    if len(observation_pcds) == 0:
        print("Error: No observation point clouds found")
        return
    
    # Load prediction clouds if using separate folder
    prediction_pcds = None
    if use_separate_prediction_folder and prediction_folder:
        print(f"\nLoading prediction point clouds...")
        prediction_pcds = load_pcd_files(prediction_folder, prediction_step_size, prediction_start_index, max_prediction_clouds)
        
        # Ensure we have enough prediction clouds
        if len(prediction_pcds) < len(observation_pcds):
            print(f"Warning: Only {len(prediction_pcds)} prediction clouds but {len(observation_pcds)} observation clouds")
    elif use_separate_prediction_folder and not prediction_folder:
        print("Error: use_separate_prediction_folder=True but no prediction_folder provided")
        return
    
    # Print processing summary
    print(f"\n=== Processing Summary ===")
    print(f"Will process {len(observation_pcds)} observation clouds")
    if max_observation_clouds is not None:
        print(f"  - Limited to first {max_observation_clouds} after sampling")
    if use_separate_prediction_folder and prediction_pcds:
        print(f"Will use {len(prediction_pcds)} prediction clouds")
        if max_prediction_clouds is not None:
            print(f"  - Limited to first {max_prediction_clouds} after sampling")
    print(f"Observation sampling: every {observation_step_size}th file starting from index {observation_start_index}")
    if use_separate_prediction_folder:
        print(f"Prediction sampling: every {prediction_step_size}th file starting from index {prediction_start_index}")
    
    # Use first observation as reference cloud for CUMULATIVE transformations
    reference_cloud = observation_pcds[0][1]
    print(f"\nUsing first observation cloud '{observation_pcds[0][0]}' as REFERENCE for cumulative transformations")
    
    # Adjust Kalman filter parameters
    # Use the larger of the two step sizes for process noise scaling
    max_step_size = max(observation_step_size, prediction_step_size if use_separate_prediction_folder else 1)
    process_noise = 0.2 * max_step_size
    print(f"Using process noise: {process_noise} (scaled for max step size {max_step_size})")
    
    # Initialize Kalman filter
    kalman_filter = PointCloudKalmanFilter(dt=observation_step_size, process_noise=process_noise, measurement_noise=0.5)
    
    # Store results
    results = []
    prediction_transformations = []
    observation_transformations = []
    filtered_poses = []
    prediction_poses = []
    observation_poses = []
    
    print(f"\nProcessing {len(observation_pcds)} observation clouds (every {observation_step_size}th) with CUMULATIVE dual GICP Kalman filtering...")
    if use_separate_prediction_folder:
        print(f"Using {len(prediction_pcds) if prediction_pcds else 0} prediction clouds (every {prediction_step_size}th)")
    
    # Track previous clouds for prediction generation and cumulative transforms
    previous_obs_cloud = None
    cumulative_observation_transform = None
    cumulative_prediction_transform = None
    
    for i, (obs_name, obs_cloud) in enumerate(observation_pcds):
        print(f"\nProcessing observation {obs_name} ({i+1}/{len(observation_pcds)})")
        
        try:
            # Generate or select prediction cloud
            prediction_cloud = None
            
            # Always try to get prediction cloud (including for first observation)
            if use_separate_prediction_folder and prediction_pcds and i < len(prediction_pcds):
                # Use corresponding prediction cloud from separate folder
                # Match based on the actual file indices, not array indices
                pred_name, prediction_cloud = prediction_pcds[i]
                print(f"Using prediction cloud: {pred_name}")
                
            elif prediction_method == "extrapolation" and i > 0:
                # Generate prediction cloud via extrapolation (need previous cloud)
                prediction_cloud = generate_prediction_cloud(
                    previous_obs_cloud, 
                    observation_pcds[i-1][1], 
                    method="extrapolation"
                )
                print("Generated prediction cloud via extrapolation")
                
            elif prediction_method == "copy" and i > 0:
                # Use previous observation as prediction
                prediction_cloud = generate_prediction_cloud(
                    previous_obs_cloud, 
                    observation_pcds[i-1][1], 
                    method="copy"
                )
                print("Generated prediction cloud via copy method")
            
            # GICP-based prediction step with cumulative transformation: current observation vs corresponding prediction cloud
            # This registers cloud_i from observation to cloud_i from prediction (same index)
            if prediction_cloud is not None:
                if use_separate_prediction_folder:
                    pred_name = prediction_pcds[i][0] if i < len(prediction_pcds) else "generated"
                    print(f"Prediction GICP (cumulative): {obs_name} -> {pred_name}")
                else:
                    print(f"Prediction GICP (cumulative): {obs_name} -> generated prediction")
                cumulative_prediction_transform = kalman_filter.predict_with_gicp(obs_cloud, prediction_cloud, apply_gicp_func, cumulative_prediction_transform)
                if kalman_filter.prediction_pose is not None:
                    prediction_poses.append(kalman_filter.prediction_pose.copy())
                else:
                    prediction_poses.append(None)
            else:
                prediction_poses.append(None)
            
            # MODIFIED: GICP-based observation step with cumulative transformation
            if i < len(observation_pcds) - 1:  # Not the last observation
                next_obs_name, next_obs_cloud = observation_pcds[i + 1]
                print(f"Observation GICP (cumulative): {obs_name} -> {next_obs_name}")
                observation_pose, cumulative_observation_transform = kalman_filter.update_with_gicp(
                    obs_cloud, next_obs_cloud, apply_gicp_func, cumulative_observation_transform)
                observation_poses.append(observation_pose.copy() if observation_pose is not None else None)
            else:
                # For the last observation, skip observation GICP since no next cloud
                print(f"Observation GICP: {obs_name} -> skipped (last observation)")
                # Use the last available observation pose or initialize with identity
                if len(observation_poses) > 0 and observation_poses[-1] is not None:
                    observation_poses.append(observation_poses[-1].copy())
                else:
                    observation_poses.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))  # Identity pose
            
            # Get filtered pose
            filtered_pose = kalman_filter.get_current_pose()
            filtered_poses.append(filtered_pose.copy() if filtered_pose is not None else None)
            
            # Store results
            result = {
                'observation_file': obs_name,
                'observation_file_index': observation_start_index + i * observation_step_size,
                'prediction_file': prediction_pcds[i][0] if (use_separate_prediction_folder and prediction_pcds and i < len(prediction_pcds)) else None,
                'prediction_file_index': prediction_start_index + i * prediction_step_size if use_separate_prediction_folder else None,
                'observation_step_size': observation_step_size,
                'prediction_step_size': prediction_step_size if use_separate_prediction_folder else None,
                'max_observation_clouds': max_observation_clouds,
                'max_prediction_clouds': max_prediction_clouds,
                'prediction_method': prediction_method,
                'prediction_pose': kalman_filter.prediction_pose.copy() if kalman_filter.prediction_pose is not None else None,
                'observation_pose': observation_pose.copy() if observation_pose is not None else None,
                'filtered_pose': filtered_pose.copy() if filtered_pose is not None else None,
                'covariance': kalman_filter.kalman_state.covariance.copy() if kalman_filter.kalman_state else None,
                'registration_type': 'cumulative_consecutive'  # Updated to reflect the change
            }
            results.append(result)
            
            # Update previous cloud
            previous_obs_cloud = obs_cloud
            
            # Print summary
            if kalman_filter.prediction_pose is not None:
                print(f"Prediction pose: {kalman_filter.prediction_pose}")
            if observation_pose is not None:
                print(f"Observation pose (cumulative): {observation_pose}")
            if filtered_pose is not None:
                print(f"Filtered pose: {filtered_pose}")
                
        except Exception as e:
            print(f"Error processing {obs_name}: {e}")
            results.append({
                'observation_file': obs_name,
                'observation_file_index': observation_start_index + i * observation_step_size,
                'observation_step_size': observation_step_size,
                'prediction_step_size': prediction_step_size if use_separate_prediction_folder else None,
                'max_observation_clouds': max_observation_clouds,
                'max_prediction_clouds': max_prediction_clouds,
                'error': str(e)
            })
            prediction_poses.append(None)
            observation_poses.append(None)
            filtered_poses.append(None)
    
    # Visualization and final processing
    if visualize and len(filtered_poses) > 0:
        # Create and visualize final registered cloud
        combined_cloud = visualize_final_registered_cloud(
            observation_pcds, reference_cloud, filtered_poses, kalman_filter
        )
        
        # Plot pose evolution with step size information
        step_info = f"Obs:{observation_step_size}"
        if use_separate_prediction_folder:
            step_info += f", Pred:{prediction_step_size}"
        if max_observation_clouds is not None:
            step_info += f", MaxObs:{max_observation_clouds}"
        if max_prediction_clouds is not None:
            step_info += f", MaxPred:{max_prediction_clouds}"
        
        plot_dual_gicp_pose_evolution(prediction_poses, observation_poses, filtered_poses, 
                                    observation_step_size, step_info)
        
        # Additional detailed trajectory analysis
        plot_trajectory_comparison(prediction_poses, observation_poses, filtered_poses, step_info)
        
        # Save results including the combined cloud
        save_results(results, prediction_transformations, filtered_poses, combined_cloud)
        
        return results, prediction_transformations, observation_transformations, filtered_poses, combined_cloud
    else:
        save_results(results, prediction_transformations, filtered_poses)
        return results, prediction_transformations, observation_transformations, filtered_poses

def plot_dual_gicp_pose_evolution(prediction_poses, observation_poses, filtered_poses, step_size=1, step_info=""):
    """
    Plot the evolution of poses over time (prediction vs observation vs filtered) + X-Y trajectory
    NOTE: observation_poses now represent CUMULATIVE transformations from reference frame
    Args:
        prediction_poses: List of GICP prediction poses
        observation_poses: List of CUMULATIVE GICP observation poses (from reference)
        filtered_poses: List of Kalman filtered poses
        step_size: Step size used for sampling (for proper time axis labeling)
        step_info: Additional step size information for title
    """
    # Filter out None values and align arrays
    valid_indices = []
    for i in range(len(filtered_poses)):
        if (filtered_poses[i] is not None and 
            observation_poses[i] is not None):
            valid_indices.append(i)
    
    if len(valid_indices) == 0:
        print("No valid poses to plot")
        return
    
    # Extract valid poses
    valid_filtered = [filtered_poses[i] for i in valid_indices]
    valid_observation = [observation_poses[i] for i in valid_indices]
    valid_prediction = [prediction_poses[i] if i < len(prediction_poses) and prediction_poses[i] is not None else None for i in valid_indices]
    
    # Convert to numpy arrays
    filtered_array = np.array(valid_filtered)
    observation_array = np.array(valid_observation)
    
    # Handle prediction array (may have None values)
    prediction_array = []
    prediction_indices = []
    for i, pred in enumerate(valid_prediction):
        if pred is not None:
            prediction_array.append(pred)
            prediction_indices.append(i)
    
    if len(prediction_array) > 0:
        prediction_array = np.array(prediction_array)
    
    # Create time axis accounting for step size
    time_steps = np.arange(len(valid_filtered)) * step_size
    
    # Create figure with more subplots (3x3 grid)
    fig = plt.figure(figsize=(18, 12))
    title = f'CUMULATIVE Dual GICP Kalman Filter ({step_info}): Prediction vs Observation vs Filtered'
    fig.suptitle(title, fontsize=16)
    
    pose_labels = ['X (m)', 'Y (m)', 'Z (m)', 'Roll (rad)', 'Pitch (rad)', 'Yaw (rad)']
    
    # Plot individual pose components (2x3 grid for 6 DOF)
    for i in range(6):
        ax = plt.subplot(3, 3, i+1)  # First 6 positions in 3x3 grid
        
        # Plot observation (cumulative GICP measurements from reference)
        ax.plot(time_steps, observation_array[:, i], 'r-', label='GICP Observation (Cumulative)', alpha=0.7, linewidth=2, marker='o', markersize=4)
        
        # Plot prediction (if available)
        if len(prediction_array) > 0:
            pred_time_steps = [time_steps[j] for j in prediction_indices]
            ax.plot(pred_time_steps, prediction_array[:, i], 'g--', label='GICP Prediction', alpha=0.7, linewidth=2, marker='s', markersize=4)
        
        # Plot filtered result
        ax.plot(time_steps, filtered_array[:, i], 'b-', label='Kalman Filtered', linewidth=3, marker='^', markersize=5)
        
        ax.set_xlabel(f'File Index')
        ax.set_ylabel(pose_labels[i])
        ax.set_title(f'{pose_labels[i]} Evolution (Cumulative)')
        if i == 0:  # Only show legend on first plot to avoid clutter
            ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Add X-Y trajectory plot (position 7 in 3x3 grid)
    ax_traj = plt.subplot(3, 3, 7)
    
    # Extract X and Y coordinates
    obs_x, obs_y = observation_array[:, 0], observation_array[:, 1]
    filt_x, filt_y = filtered_array[:, 0], filtered_array[:, 1]
    
    # Plot trajectories
    ax_traj.plot(obs_x, obs_y, 'r-', label='GICP Observation (Cumulative)', alpha=0.7, linewidth=2, marker='o', markersize=6)
    ax_traj.plot(filt_x, filt_y, 'b-', label='Kalman Filtered', linewidth=3, marker='^', markersize=7)
    
    # Plot prediction trajectory if available
    if len(prediction_array) > 0:
        pred_x, pred_y = prediction_array[:, 0], prediction_array[:, 1]
        ax_traj.plot(pred_x, pred_y, 'g--', label='GICP Prediction', alpha=0.7, linewidth=2, marker='s', markersize=6)
    
    # Mark start and end points
    ax_traj.plot(obs_x[0], obs_y[0], 'ko', markersize=10, label='Start (Reference)')
    ax_traj.plot(obs_x[-1], obs_y[-1], 'k*', markersize=12, label='End')
    
    # Add trajectory point numbers for better understanding
    for i in range(0, len(obs_x), max(1, len(obs_x)//5)):  # Show numbers for every ~5th point
        ax_traj.annotate(f'{i}', (obs_x[i], obs_y[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8, alpha=0.7)
    
    ax_traj.set_xlabel('X (m)')
    ax_traj.set_ylabel('Y (m)')
    ax_traj.set_title('X-Y Trajectory (Cumulative from Reference)')
    ax_traj.legend()
    ax_traj.grid(True, alpha=0.3)
    ax_traj.axis('equal')  # Equal aspect ratio for true trajectory shape
    
    # Add 3D trajectory plot (position 8 in 3x3 grid)
    ax_3d = plt.subplot(3, 3, 8, projection='3d')
    
    # Extract X, Y, Z coordinates
    obs_z = observation_array[:, 2]
    filt_z = filtered_array[:, 2]
    
    # Plot 3D trajectories
    ax_3d.plot(obs_x, obs_y, obs_z, 'r-', label='GICP Observation (Cumulative)', alpha=0.7, linewidth=2, marker='o', markersize=4)
    ax_3d.plot(filt_x, filt_y, filt_z, 'b-', label='Kalman Filtered', linewidth=3, marker='^', markersize=5)
    
    if len(prediction_array) > 0:
        pred_z = prediction_array[:, 2]
        ax_3d.plot(pred_x, pred_y, pred_z, 'g--', label='GICP Prediction', alpha=0.7, linewidth=2, marker='s', markersize=4)
    
    # Mark start and end points
    ax_3d.plot([obs_x[0]], [obs_y[0]], [obs_z[0]], 'ko', markersize=8, label='Start (Reference)')
    ax_3d.plot([obs_x[-1]], [obs_y[-1]], [obs_z[-1]], 'k*', markersize=10, label='End')
    
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    ax_3d.set_title('3D Trajectory (Cumulative)')
    ax_3d.legend()
    
    # Add pose uncertainty visualization (position 9 in 3x3 grid)
    if len(valid_filtered) > 1:
        ax_uncert = plt.subplot(3, 3, 9)
        
        # Calculate pose differences (innovation) between prediction and observation
        innovations = []
        for i in range(len(valid_filtered)):
            if (i < len(prediction_poses) and prediction_poses[i] is not None and 
                i < len(observation_poses) and observation_poses[i] is not None):
                # Calculate Euclidean distance between prediction and observation
                pred_pos = prediction_poses[i][:3]  # X, Y, Z
                obs_pos = observation_poses[i][:3]
                distance = np.linalg.norm(obs_pos - pred_pos)
                innovations.append(distance)
            else:
                innovations.append(0)
        
        if len(innovations) > 0:
            innovation_time = time_steps[:len(innovations)]
            ax_uncert.plot(innovation_time, innovations, 'purple', linewidth=2, marker='d', markersize=5)
            ax_uncert.set_xlabel('File Index')
            ax_uncert.set_ylabel('Position Innovation (m)')
            ax_uncert.set_title('Prediction vs Observation\nPosition Difference')
            ax_uncert.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_trajectory_comparison(prediction_poses, observation_poses, filtered_poses, step_info=""):
    """
    Dedicated trajectory comparison plot with detailed analysis
    NOTE: observation_poses now represent CUMULATIVE transformations from reference frame
    Args:
        prediction_poses: List of GICP prediction poses
        observation_poses: List of CUMULATIVE GICP observation poses (from reference)
        filtered_poses: List of Kalman filtered poses
        step_info: Additional step size information for title
    """
    # Filter out None values
    valid_indices = []
    for i in range(len(filtered_poses)):
        if (filtered_poses[i] is not None and 
            observation_poses[i] is not None):
            valid_indices.append(i)
    
    if len(valid_indices) == 0:
        print("No valid poses for trajectory plot")
        return
    
    # Extract valid poses
    valid_filtered = [filtered_poses[i] for i in valid_indices]
    valid_observation = [observation_poses[i] for i in valid_indices]
    valid_prediction = [prediction_poses[i] if i < len(prediction_poses) and prediction_poses[i] is not None else None for i in valid_indices]
    
    # Convert to numpy arrays
    filtered_array = np.array(valid_filtered)
    observation_array = np.array(valid_observation)
    
    # Create figure for detailed trajectory analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'CUMULATIVE Trajectory Analysis ({step_info})', fontsize=16)
    
    # Extract coordinates
    obs_x, obs_y, obs_z = observation_array[:, 0], observation_array[:, 1], observation_array[:, 2]
    filt_x, filt_y, filt_z = filtered_array[:, 0], filtered_array[:, 1], filtered_array[:, 2]
    
    # Plot 1: X-Y trajectory with error ellipses
    ax1 = axes[0, 0]
    ax1.plot(obs_x, obs_y, 'r-', label='GICP Observation (Cumulative)', linewidth=2, marker='o', markersize=6, alpha=0.7)
    ax1.plot(filt_x, filt_y, 'b-', label='Kalman Filtered', linewidth=3, marker='^', markersize=7)
    
    # Add prediction if available
    if len(valid_prediction) > 0 and any(p is not None for p in valid_prediction):
        pred_array = np.array([p for p in valid_prediction if p is not None])
        if len(pred_array) > 0:
            pred_x, pred_y = pred_array[:, 0], pred_array[:, 1]
            ax1.plot(pred_x, pred_y, 'g--', label='GICP Prediction', linewidth=2, marker='s', markersize=6, alpha=0.7)
    
    # Mark waypoints
    for i in range(0, len(obs_x), max(1, len(obs_x)//8)):
        ax1.annotate(f'{i}', (obs_x[i], obs_y[i]), xytext=(8, 8), 
                    textcoords='offset points', fontsize=9, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax1.plot(obs_x[0], obs_y[0], 'go', markersize=12, label='Start (Reference)')
    ax1.plot(obs_x[-1], obs_y[-1], 'ro', markersize=12, label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('X-Y Trajectory with Waypoints (Cumulative)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Z trajectory over time
    ax2 = axes[0, 1]
    time_steps = np.arange(len(valid_filtered))
    ax2.plot(time_steps, obs_z, 'r-', label='GICP Observation (Cumulative)', linewidth=2, marker='o', markersize=5)
    ax2.plot(time_steps, filt_z, 'b-', label='Kalman Filtered', linewidth=3, marker='^', markersize=6)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Altitude Profile (Cumulative)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Trajectory speed analysis
    ax3 = axes[1, 0]
    if len(obs_x) > 1:
        # Calculate speeds between consecutive points
        obs_speeds = []
        filt_speeds = []
        for i in range(1, len(obs_x)):
            obs_dist = np.sqrt((obs_x[i] - obs_x[i-1])**2 + (obs_y[i] - obs_y[i-1])**2 + (obs_z[i] - obs_z[i-1])**2)
            filt_dist = np.sqrt((filt_x[i] - filt_x[i-1])**2 + (filt_y[i] - filt_y[i-1])**2 + (filt_z[i] - filt_z[i-1])**2)
            obs_speeds.append(obs_dist)
            filt_speeds.append(filt_dist)
        
        speed_time = time_steps[1:]
        ax3.plot(speed_time, obs_speeds, 'r-', label='GICP Observation (Cumulative)', linewidth=2, marker='o', markersize=5)
        ax3.plot(speed_time, filt_speeds, 'b-', label='Kalman Filtered', linewidth=3, marker='^', markersize=6)
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Step Distance (m)')
        ax3.set_title('Movement Speed Between Steps')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Filtering error analysis
    ax4 = axes[1, 1]
    position_errors = []
    for i in range(len(valid_filtered)):
        obs_pos = observation_array[i, :3]
        filt_pos = filtered_array[i, :3]
        error = np.linalg.norm(filt_pos - obs_pos)
        position_errors.append(error)
    
    ax4.plot(time_steps, position_errors, 'purple', linewidth=2, marker='d', markersize=5)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Position Error (m)')
    ax4.set_title('Kalman Filter Position Error')
    ax4.grid(True, alpha=0.3)
    
    # Add statistics
    mean_error = np.mean(position_errors)
    max_error = np.max(position_errors)
    ax4.axhline(y=mean_error, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_error:.3f}m')
    ax4.axhline(y=max_error, color='orange', linestyle='--', alpha=0.7, label=f'Max: {max_error:.3f}m')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print trajectory statistics
    total_obs_distance = np.sum([np.sqrt((obs_x[i] - obs_x[i-1])**2 + (obs_y[i] - obs_y[i-1])**2 + (obs_z[i] - obs_z[i-1])**2) 
                                for i in range(1, len(obs_x))])
    total_filt_distance = np.sum([np.sqrt((filt_x[i] - filt_x[i-1])**2 + (filt_y[i] - filt_y[i-1])**2 + (filt_z[i] - filt_z[i-1])**2) 
                                 for i in range(1, len(filt_x))])
    
    print(f"\n=== CUMULATIVE Trajectory Statistics ===")
    print(f"Total observed path length: {total_obs_distance:.3f} m")
    print(f"Total filtered path length: {total_filt_distance:.3f} m")
    print(f"Path smoothing: {((total_obs_distance - total_filt_distance) / total_obs_distance * 100):.1f}% reduction")
    print(f"Mean position error: {mean_error:.3f} m")
    print(f"Max position error: {max_error:.3f} m")
    print(f"Start position (reference): ({obs_x[0]:.3f}, {obs_y[0]:.3f}, {obs_z[0]:.3f})")
    print(f"End position: ({obs_x[-1]:.3f}, {obs_y[-1]:.3f}, {obs_z[-1]:.3f})")
    print(f"Total displacement: {np.sqrt((obs_x[-1] - obs_x[0])**2 + (obs_y[-1] - obs_y[0])**2 + (obs_z[-1] - obs_z[0])**2):.3f} m")

# Example usage function
def main():
    """
    Example usage - replace with your actual folder paths and GICP function
    """
    # Replace these paths with your actual folder paths

    observation_folder = "/home/robotics/testdata/BA6I"
    prediction_folder = "/home/robotics/testdata/BA6Irtest"
    #observation_folder = "/home/robotics/testdata/927"
    #prediction_folder = "/home/robotics/testdata/927pred"  # Optional, only needed for separate prediction method
    
    # The apply_gicp_direct function now uses only small_gicp
    # Make sure small_gicp is installed: pip install small_gicp
    
    # Example usage scenarios with cloud limits:
    
    # Scenario 1: Limited to first 10 observation clouds and 15 prediction clouds (separate folder)
    print("=== Scenario 1: CUMULATIVE Limited clouds with separate prediction folder (10 obs, 15 pred) ===")
    results1, pred_transforms1, obs_transforms1, filtered_poses1, combined_cloud1 = process_point_clouds_with_kalman(
        observation_folder,
        prediction_folder,
        apply_gicp_direct,  # Uses only small_gicp
        visualize=True,
        prediction_method="separate_folder",
        observation_step_size=5,
        observation_start_index=500,
        prediction_step_size=5,
        prediction_start_index=501,
        use_separate_prediction_folder=True,
        max_observation_clouds=100,          # Limit to first 10 observation clouds
        max_prediction_clouds=100           # Limit to first 15 prediction clouds
    )

    compare_to_ground_truth_positions(filtered_poses1, "./output/eval/test1gt.csv")
    
    print(f"\nCUMULATIVE Processing complete!")
    print(f"Scenario 1: {len([r for r in results1 if 'error' not in r])} clouds processed")
    print("Note: Observation transformations are now CUMULATIVE from the reference frame")
    

if __name__ == "__main__":
    main()