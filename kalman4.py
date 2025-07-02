import numpy as np
import open3d as o3d
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import pandas as pd
from scipy import ndimage
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import savgol_filter

@dataclass
class ScanState:
    """State for non-repetitive LiDAR scan processing"""
    pose: np.ndarray  # [x, y, z, yaw]
    uncertainty: np.ndarray  # 4x4 covariance matrix
    confidence: float  # Confidence in this pose estimate
    scan_features: Dict  # Geometric features of the scan

class ExtremePathSmoother:
    """Extreme path smoothing for LiDAR trajectories"""
    
    def __init__(self, 
                 smoothing_strength: float = 0.95,
                 min_trajectory_length: int = 5,
                 preserve_endpoints: bool = True,
                 angle_smoothing_factor: float = 0.9):
        """
        Initialize extreme path smoother
        
        Args:
            smoothing_strength: Strength of smoothing (0.0 = no smoothing, 1.0 = maximum)
            min_trajectory_length: Minimum points needed for smoothing
            preserve_endpoints: Whether to preserve start/end points
            angle_smoothing_factor: Factor for angle smoothing (0.0 = no smoothing, 1.0 = maximum)
        """
        self.smoothing_strength = smoothing_strength
        self.min_trajectory_length = min_trajectory_length
        self.preserve_endpoints = preserve_endpoints
        self.angle_smoothing_factor = angle_smoothing_factor
    
    def apply_gaussian_smoothing(self, poses: np.ndarray) -> np.ndarray:
        """Apply Gaussian smoothing to poses"""
        if len(poses) < self.min_trajectory_length:
            return poses.copy()
        
        smoothed_poses = poses.copy()
        
        # Calculate sigma based on smoothing strength
        sigma = self.smoothing_strength * len(poses) * 0.1
        
        # Smooth x, y, z coordinates
        for dim in range(3):
            smoothed_poses[:, dim] = ndimage.gaussian_filter1d(
                poses[:, dim], sigma=sigma, mode='nearest'
            )
        
        # Special handling for angles (yaw)
        angles = poses[:, 3]
        
        # Unwrap angles to handle discontinuities
        unwrapped_angles = np.unwrap(angles)
        
        # Smooth unwrapped angles
        angle_sigma = sigma * self.angle_smoothing_factor
        smoothed_unwrapped = ndimage.gaussian_filter1d(
            unwrapped_angles, sigma=angle_sigma, mode='nearest'
        )
        
        # Wrap back to [-π, π]
        smoothed_poses[:, 3] = np.arctan2(
            np.sin(smoothed_unwrapped), 
            np.cos(smoothed_unwrapped)
        )
        
        # Preserve endpoints if requested
        if self.preserve_endpoints and len(poses) > 2:
            smoothed_poses[0] = poses[0]
            smoothed_poses[-1] = poses[-1]
        
        return smoothed_poses
    
    def apply_spline_smoothing(self, poses: np.ndarray) -> np.ndarray:
        """Apply spline-based smoothing"""
        if len(poses) < self.min_trajectory_length:
            return poses.copy()
        
        smoothed_poses = poses.copy()
        t = np.arange(len(poses))
        
        # Smoothing parameter based on strength
        s = (1.0 - self.smoothing_strength) * len(poses)
        
        try:
            # Smooth x, y, z coordinates
            for dim in range(3):
                spline = UnivariateSpline(t, poses[:, dim], s=s)
                smoothed_poses[:, dim] = spline(t)
            
            # Handle angles with unwrapping
            angles = poses[:, 3]
            unwrapped_angles = np.unwrap(angles)
            
            angle_s = s * self.angle_smoothing_factor
            angle_spline = UnivariateSpline(t, unwrapped_angles, s=angle_s)
            smoothed_unwrapped = angle_spline(t)
            
            smoothed_poses[:, 3] = np.arctan2(
                np.sin(smoothed_unwrapped), 
                np.cos(smoothed_unwrapped)
            )
            
            # Preserve endpoints if requested
            if self.preserve_endpoints and len(poses) > 2:
                smoothed_poses[0] = poses[0]
                smoothed_poses[-1] = poses[-1]
                
        except Exception as e:
            print(f"Spline smoothing failed, using Gaussian: {e}")
            return self.apply_gaussian_smoothing(poses)
        
        return smoothed_poses
    
    def apply_savgol_smoothing(self, poses: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay smoothing"""
        if len(poses) < self.min_trajectory_length:
            return poses.copy()
        
        smoothed_poses = poses.copy()
        
        # Calculate window length based on smoothing strength
        window_length = max(5, int(self.smoothing_strength * len(poses) * 0.3))
        if window_length >= len(poses):
            window_length = len(poses) - 1
        if window_length % 2 == 0:  # Must be odd
            window_length -= 1
        
        polyorder = min(3, window_length - 1)
        
        try:
            # Smooth x, y, z coordinates
            for dim in range(3):
                smoothed_poses[:, dim] = savgol_filter(
                    poses[:, dim], window_length, polyorder
                )
            
            # Handle angles
            angles = poses[:, 3]
            unwrapped_angles = np.unwrap(angles)
            
            # Use smaller window for angles
            angle_window = max(3, int(window_length * self.angle_smoothing_factor))
            if angle_window % 2 == 0:
                angle_window -= 1
            angle_poly = min(2, angle_window - 1)
            
            smoothed_unwrapped = savgol_filter(
                unwrapped_angles, angle_window, angle_poly
            )
            
            smoothed_poses[:, 3] = np.arctan2(
                np.sin(smoothed_unwrapped), 
                np.cos(smoothed_unwrapped)
            )
            
            # Preserve endpoints if requested
            if self.preserve_endpoints and len(poses) > 2:
                smoothed_poses[0] = poses[0]
                smoothed_poses[-1] = poses[-1]
                
        except Exception as e:
            print(f"Savgol smoothing failed, using Gaussian: {e}")
            return self.apply_gaussian_smoothing(poses)
        
        return smoothed_poses
    
    def apply_multi_pass_smoothing(self, poses: np.ndarray, num_passes: int = 3) -> np.ndarray:
        """Apply multiple passes of smoothing for extreme effect"""
        if len(poses) < self.min_trajectory_length:
            return poses.copy()
        
        smoothed = poses.copy()
        
        # Reduce smoothing strength per pass to avoid over-smoothing
        pass_strength = self.smoothing_strength ** (1.0 / num_passes)
        original_strength = self.smoothing_strength
        
        for pass_num in range(num_passes):
            # Alternate between different smoothing methods
            self.smoothing_strength = pass_strength
            
            if pass_num % 3 == 0:
                smoothed = self.apply_gaussian_smoothing(smoothed)
            elif pass_num % 3 == 1:
                smoothed = self.apply_spline_smoothing(smoothed)
            else:
                smoothed = self.apply_savgol_smoothing(smoothed)
            
            print(f"Applied smoothing pass {pass_num + 1}/{num_passes}")
        
        # Restore original strength
        self.smoothing_strength = original_strength
        
        return smoothed
    
    def apply_extreme_smoothing(self, poses: np.ndarray, method: str = 'multi_pass') -> np.ndarray:
        """
        Apply extreme smoothing to trajectory
        
        Args:
            poses: Array of poses [N, 4] where each pose is [x, y, z, yaw]
            method: Smoothing method ('gaussian', 'spline', 'savgol', 'multi_pass')
            
        Returns:
            Extremely smoothed poses
        """
        if len(poses) < self.min_trajectory_length:
            print(f"Trajectory too short for smoothing ({len(poses)} < {self.min_trajectory_length})")
            return poses.copy()
        
        print(f"Applying extreme {method} smoothing to {len(poses)} poses...")
        print(f"Smoothing strength: {self.smoothing_strength}")
        print(f"Angle smoothing factor: {self.angle_smoothing_factor}")
        
        if method == 'gaussian':
            return self.apply_gaussian_smoothing(poses)
        elif method == 'spline':
            return self.apply_spline_smoothing(poses)
        elif method == 'savgol':
            return self.apply_savgol_smoothing(poses)
        elif method == 'multi_pass':
            return self.apply_multi_pass_smoothing(poses)
        else:
            print(f"Unknown method {method}, using multi_pass")
            return self.apply_multi_pass_smoothing(poses)
    
    def calculate_smoothness_metrics(self, original_poses: np.ndarray, smoothed_poses: np.ndarray) -> Dict:
        """Calculate metrics to quantify smoothing effect"""
        if len(original_poses) < 3 or len(smoothed_poses) < 3:
            return {}
        
        metrics = {}
        
        # Calculate curvature for both trajectories
        def calculate_curvature(poses):
            if len(poses) < 3:
                return []
            
            curvatures = []
            for i in range(1, len(poses) - 1):
                # Use 2D curvature (x, y only)
                p1, p2, p3 = poses[i-1][:2], poses[i][:2], poses[i+1][:2]
                
                # Vector from p1 to p2 and p2 to p3
                v1 = p2 - p1
                v2 = p3 - p2
                
                # Cross product for curvature
                cross = np.cross(v1, v2)
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                
                if norm_v1 > 1e-6 and norm_v2 > 1e-6:
                    curvature = abs(cross) / (norm_v1 * norm_v2)
                    curvatures.append(curvature)
            
            return curvatures
        
        # Calculate angular velocity changes
        def calculate_angular_changes(poses):
            if len(poses) < 2:
                return []
            
            angular_changes = []
            for i in range(1, len(poses)):
                angle_diff = poses[i][3] - poses[i-1][3]
                # Handle angle wrapping
                while angle_diff > np.pi:
                    angle_diff -= 2*np.pi
                while angle_diff < -np.pi:
                    angle_diff += 2*np.pi
                angular_changes.append(abs(angle_diff))
            
            return angular_changes
        
        # Calculate acceleration changes
        def calculate_acceleration_changes(poses):
            if len(poses) < 3:
                return []
            
            accelerations = []
            for i in range(1, len(poses) - 1):
                # Calculate velocities
                v1 = poses[i][:3] - poses[i-1][:3]
                v2 = poses[i+1][:3] - poses[i][:3]
                
                # Calculate acceleration (change in velocity)
                acceleration = v2 - v1
                acc_magnitude = np.linalg.norm(acceleration)
                accelerations.append(acc_magnitude)
            
            return accelerations
        
        # Calculate metrics
        orig_curvature = calculate_curvature(original_poses)
        smooth_curvature = calculate_curvature(smoothed_poses)
        
        orig_angular = calculate_angular_changes(original_poses)
        smooth_angular = calculate_angular_changes(smoothed_poses)
        
        orig_acceleration = calculate_acceleration_changes(original_poses)
        smooth_acceleration = calculate_acceleration_changes(smoothed_poses)
        
        if orig_curvature and smooth_curvature:
            metrics['curvature_reduction'] = 1.0 - (np.mean(smooth_curvature) / np.mean(orig_curvature))
            metrics['max_curvature_original'] = np.max(orig_curvature)
            metrics['max_curvature_smoothed'] = np.max(smooth_curvature)
        
        if orig_angular and smooth_angular:
            metrics['angular_change_reduction'] = 1.0 - (np.mean(smooth_angular) / np.mean(orig_angular))
            metrics['max_angular_change_original'] = np.max(orig_angular)
            metrics['max_angular_change_smoothed'] = np.max(smooth_angular)
        
        if orig_acceleration and smooth_acceleration:
            metrics['acceleration_reduction'] = 1.0 - (np.mean(smooth_acceleration) / np.mean(orig_acceleration))
            metrics['max_acceleration_original'] = np.max(orig_acceleration)
            metrics['max_acceleration_smoothed'] = np.max(smooth_acceleration)
        
        # Calculate path length change
        def calculate_path_length(poses):
            length = 0
            for i in range(1, len(poses)):
                segment_length = np.linalg.norm(poses[i][:3] - poses[i-1][:3])
                length += segment_length
            return length
        
        orig_length = calculate_path_length(original_poses)
        smooth_length = calculate_path_length(smoothed_poses)
        
        if orig_length > 0:
            metrics['path_length_change'] = (smooth_length - orig_length) / orig_length
            metrics['original_path_length'] = orig_length
            metrics['smoothed_path_length'] = smooth_length
        
        return metrics

class NonRepetitiveLiDARProcessor:
    def __init__(self, 
                 adaptive_threshold: float = 0.9,
                 feature_weight: float = 0.3,
                 geometric_weight: float = 0.4,
                 temporal_weight: float = 0.3,
                 force_z_zero: bool = False,
                 z_redistribution_method: str = 'prediction',
                 enable_extreme_smoothing: bool = True,
                 smoothing_strength: float = 0.95,
                 smoothing_method: str = 'multi_pass'):
        """
        Processor for non-repetitive LiDAR scans with extreme path smoothing
        
        Args:
            adaptive_threshold: Threshold for switching prediction strategies
            feature_weight: Weight for feature-based matching
            geometric_weight: Weight for geometric consistency
            temporal_weight: Weight for temporal smoothness
            force_z_zero: If True, forces z coordinate to 0 and redistributes z values
            z_redistribution_method: Method for redistributing z values
            enable_extreme_smoothing: Enable extreme path smoothing
            smoothing_strength: Strength of smoothing (0.0 - 1.0)
            smoothing_method: Smoothing method ('gaussian', 'spline', 'savgol', 'multi_pass')
        """
        self.adaptive_threshold = adaptive_threshold
        self.feature_weight = feature_weight
        self.geometric_weight = geometric_weight
        self.temporal_weight = temporal_weight
        self.force_z_zero = force_z_zero
        self.z_redistribution_method = z_redistribution_method
        
        # Extreme smoothing parameters
        self.enable_extreme_smoothing = enable_extreme_smoothing
        self.smoothing_method = smoothing_method
        
        if self.enable_extreme_smoothing:
            self.path_smoother = ExtremePathSmoother(
                smoothing_strength=smoothing_strength,
                preserve_endpoints=True,
                angle_smoothing_factor=0.9
            )
        
        # State tracking
        self.scan_states = []
        self.feature_database = []
        self.motion_patterns = []
        
        # Store original poses for comparison
        self.original_poses = []
        self.smoothed_poses = []
        
        # Adaptive parameters
        self.current_strategy = "feature_based"
        self.confidence_threshold = 0.7
        
        # Feature extraction parameters
        self.voxel_size = 0.1
        self.normal_radius = 0.5
        self.fpfh_radius = 1.0

    def redistribute_z_component(self, pose: np.ndarray, predicted_pose: Optional[np.ndarray] = None) -> np.ndarray:
        """Redistribute z component to x and y coordinates"""
        if not self.force_z_zero or abs(pose[2]) < 1e-6:
            return pose.copy()
        
        modified_pose = pose.copy()
        z_value = modified_pose[2]
        
        if self.z_redistribution_method == 'prediction' and predicted_pose is not None:
            if len(self.scan_states) >= 1:
                last_pose = self.scan_states[-1].pose
                predicted_movement = predicted_pose[:3] - last_pose[:3]
                modified_pose[0] += 1 * -z_value + predicted_movement[0]
        
        modified_pose[2] = 0.0
        return modified_pose

    def extract_scan_features(self, cloud: o3d.geometry.PointCloud) -> Dict:
        """Extract geometric features from LiDAR scan"""
        features = {}
        
        try:
            points = np.asarray(cloud.points)
            if len(points) == 0:
                return features
            
            # Basic geometric properties
            features['point_count'] = len(points)
            features['centroid'] = np.mean(points, axis=0)
            features['std_dev'] = np.std(points, axis=0)
            features['bounding_box'] = {
                'min': np.min(points, axis=0),
                'max': np.max(points, axis=0),
                'extent': np.max(points, axis=0) - np.min(points, axis=0)
            }
            
            # Downsampling for feature extraction
            if len(points) > 1000:
                cloud_ds = cloud.voxel_down_sample(self.voxel_size)
            else:
                cloud_ds = cloud
            
            # Normal estimation
            if len(cloud_ds.points) > 10:
                cloud_ds.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=self.normal_radius, max_nn=30
                    )
                )
                
                normals = np.asarray(cloud_ds.normals)
                if len(normals) > 0:
                    features['normal_distribution'] = {
                        'mean': np.mean(normals, axis=0),
                        'std': np.std(normals, axis=0)
                    }
            
            # FPFH features
            if len(cloud_ds.points) > 50 and cloud_ds.has_normals():
                fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    cloud_ds,
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=self.fpfh_radius, max_nn=100
                    )
                )
                features['fpfh_histogram'] = np.asarray(fpfh.data).mean(axis=1)
            
            # Height profile
            z_coords = points[:, 2]
            features['height_profile'] = {
                'min_height': np.min(z_coords),
                'max_height': np.max(z_coords),
                'mean_height': np.mean(z_coords),
                'height_variance': np.var(z_coords)
            }
            
            # Density analysis
            if len(points) > 100:
                sample_indices = np.random.choice(len(points), min(100, len(points)), replace=False)
                sample_points = points[sample_indices]
                
                distances = cdist(sample_points, points)
                k_nearest_dists = np.sort(distances, axis=1)[:, 1:6]
                avg_density = np.mean(k_nearest_dists)
                features['local_density'] = avg_density
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            features['extraction_error'] = str(e)
        
        return features

    def compute_feature_similarity(self, features1: Dict, features2: Dict) -> float:
        """Compute similarity between two feature sets"""
        if not features1 or not features2:
            return 0.0
        
        similarities = []
        
        try:
            # Point count similarity
            if 'point_count' in features1 and 'point_count' in features2:
                count_ratio = min(features1['point_count'], features2['point_count']) / \
                             max(features1['point_count'], features2['point_count'])
                similarities.append(count_ratio)
            
            # Centroid distance
            if 'centroid' in features1 and 'centroid' in features2:
                if self.force_z_zero:
                    centroid_dist = np.linalg.norm(features1['centroid'][:2] - features2['centroid'][:2])
                else:
                    centroid_dist = np.linalg.norm(features1['centroid'] - features2['centroid'])
                centroid_sim = max(0, 1 - centroid_dist / 50.0)
                similarities.append(centroid_sim)
            
            # FPFH feature similarity
            if ('fpfh_histogram' in features1 and 'fpfh_histogram' in features2):
                fpfh1, fpfh2 = features1['fpfh_histogram'], features2['fpfh_histogram']
                if len(fpfh1) == len(fpfh2):
                    dot_product = np.dot(fpfh1, fpfh2)
                    norm_product = np.linalg.norm(fpfh1) * np.linalg.norm(fpfh2)
                    if norm_product > 0:
                        fpfh_sim = dot_product / norm_product
                        similarities.append(max(0, fpfh_sim))
            
        except Exception as e:
            print(f"Error computing similarity: {e}")
        
        if similarities:
            return np.mean(similarities)
        else:
            return 0.0

    def predict_pose_adaptive(self, current_features: Dict) -> Tuple[np.ndarray, float]:
        """Adaptive pose prediction combining multiple strategies"""
        if len(self.scan_states) < 2:
            return None, 0.0
        
        # Simple extrapolation for prediction
        last_pose = self.scan_states[-1].pose
        prev_pose = self.scan_states[-2].pose
        extrapolated_pose = last_pose + 0.3 * (last_pose - prev_pose)
        extrapolated_pose[3] = np.arctan2(np.sin(extrapolated_pose[3]), np.cos(extrapolated_pose[3]))
        
        if self.force_z_zero:
            extrapolated_pose[2] = 0.0
        
        return extrapolated_pose, 0.6

    def update_with_observation(self, 
                              observed_pose: np.ndarray, 
                              scan_features: Dict,
                              registration_confidence: float = 1.0,
                              predicted_pose: Optional[np.ndarray] = None):
        """Update state with new observation"""
        # Apply z redistribution if required
        final_pose = self.redistribute_z_component(observed_pose, predicted_pose)
        
        # Store original pose before any smoothing
        self.original_poses.append(final_pose.copy())
        
        # Create new scan state
        base_uncertainty = 0.1
        uncertainty_matrix = np.eye(4) * (base_uncertainty / registration_confidence) ** 2
        
        new_state = ScanState(
            pose=final_pose.copy(),
            uncertainty=uncertainty_matrix,
            confidence=registration_confidence,
            scan_features=scan_features
        )
        
        # Add to history
        self.scan_states.append(new_state)
        self.feature_database.append(scan_features)
        
        # Apply extreme smoothing if enabled and we have enough poses
        if self.enable_extreme_smoothing and len(self.original_poses) >= 5:
            self._apply_extreme_smoothing()
        
        # Keep limited history
        max_history = 20
        if len(self.scan_states) > max_history:
            self.scan_states.pop(0)
            self.feature_database.pop(0)
            self.original_poses.pop(0)
            if self.smoothed_poses:
                self.smoothed_poses.pop(0)

    def _apply_extreme_smoothing(self):
        """Apply extreme smoothing to the trajectory"""
        if len(self.original_poses) < 5:
            self.smoothed_poses = [pose.copy() for pose in self.original_poses]
            return
        
        # Convert to numpy array
        poses_array = np.array(self.original_poses)
        
        # Apply extreme smoothing
        smoothed_array = self.path_smoother.apply_extreme_smoothing(
            poses_array, method=self.smoothing_method
        )
        
        # Update scan states with smoothed poses
        self.smoothed_poses = [pose.copy() for pose in smoothed_array]
        
        # Update scan states with smoothed poses
        for i, smoothed_pose in enumerate(smoothed_array):
            if i < len(self.scan_states):
                self.scan_states[i].pose = smoothed_pose.copy()
        
        print(f"Applied extreme smoothing to {len(smoothed_array)} poses")

    def get_current_state(self) -> Optional[ScanState]:
        """Get current scan state"""
        if self.scan_states:
            return self.scan_states[-1]
        return None

    def get_motion_analysis(self) -> Dict:
        """Get motion analysis summary including smoothing metrics"""
        if len(self.scan_states) < 2:
            return {}
        
        poses = [state.pose for state in self.scan_states]
        confidences = [state.confidence for state in self.scan_states]
        
        # Calculate statistics
        position_moves = []
        for i in range(1, len(poses)):
            if self.force_z_zero:
                move = np.linalg.norm(poses[i][:2] - poses[i-1][:2])
            else:
                move = np.linalg.norm(poses[i][:3] - poses[i-1][:3])
            position_moves.append(move)
        
        analysis = {
            'scan_count': len(self.scan_states),
            'avg_movement': np.mean(position_moves) if position_moves else 0,
            'movement_std': np.std(position_moves) if position_moves else 0,
            'avg_confidence': np.mean(confidences),
            'current_weights': {
                'feature': self.feature_weight,
                'geometric': self.geometric_weight,
                'temporal': self.temporal_weight
            },
            'force_z_zero': self.force_z_zero,
            'z_redistribution_method': self.z_redistribution_method,
            'extreme_smoothing_enabled': self.enable_extreme_smoothing,
            'smoothing_method': self.smoothing_method if self.enable_extreme_smoothing else 'none'
        }
        
        # Add smoothing metrics if available
        if (self.enable_extreme_smoothing and 
            len(self.original_poses) > 5 and 
            len(self.smoothed_poses) > 5):
            
            orig_array = np.array(self.original_poses)
            smooth_array = np.array(self.smoothed_poses)
            
            smoothing_metrics = self.path_smoother.calculate_smoothness_metrics(
                orig_array, smooth_array
            )
            analysis['smoothing_metrics'] = smoothing_metrics
        
        return analysis

    def get_original_poses(self) -> List[np.ndarray]:
        """Get original poses before smoothing"""
        return self.original_poses.copy()
    
    def get_smoothed_poses(self) -> List[np.ndarray]:
        """Get smoothed poses"""
        if self.enable_extreme_smoothing and self.smoothed_poses:
            return self.smoothed_poses.copy()
        else:
            return [state.pose.copy() for state in self.scan_states]

def process_non_repetitive_lidar_scans_with_extreme_smoothing(
    observation_folder: str,
    apply_gicp_func=None,
    visualize=True,
    observation_step_size=1,
    observation_start_index=0,
    max_observation_clouds=None,
    force_z_zero=False,
    z_redistribution_method='prediction',
    enable_extreme_smoothing=True,
    smoothing_strength=0.95,
    smoothing_method='multi_pass'):
    """
    Process non-repetitive LiDAR scans with extreme path smoothing
    """
    # Load clouds
    if apply_gicp_func is None:
        from Kalman import apply_gicp_direct
        apply_gicp_func = apply_gicp_direct
    
    from Kalman import load_pcd_files
    observation_pcds = load_pcd_files(observation_folder, observation_step_size, 
                                     observation_start_index, max_observation_clouds)
    
    if len(observation_pcds) == 0:
        print("Error: No observation point clouds found")
        return
    
    print(f"\n=== Non-Repetitive LiDAR Processing with Extreme Smoothing ===")
    print(f"Processing {len(observation_pcds)} scans")
    print(f"Force Z=0: {force_z_zero}")
    print(f"Extreme smoothing: {enable_extreme_smoothing}")
    if enable_extreme_smoothing:
        print(f"Smoothing method: {smoothing_method}")
        print(f"Smoothing strength: {smoothing_strength}")
    
    # Initialize processor with extreme smoothing
    processor = NonRepetitiveLiDARProcessor(
        force_z_zero=force_z_zero,
        z_redistribution_method=z_redistribution_method,
        enable_extreme_smoothing=enable_extreme_smoothing,
        smoothing_strength=smoothing_strength,
        smoothing_method=smoothing_method
    )
    
    # Store results
    results = []
    predicted_poses = []
    observed_poses = []
    final_poses = []
    original_poses = []
    
    cumulative_transform = None
    
    for i, (scan_name, scan_cloud) in enumerate(observation_pcds):
        print(f"\n=== Processing scan {i+1}/{len(observation_pcds)}: {scan_name} ===")
        
        try:
            # Extract features from current scan
            print("Extracting scan features...")
            current_features = processor.extract_scan_features(scan_cloud)
            
            # Predict pose using adaptive method
            print("Predicting pose...")
            predicted_pose, prediction_confidence = processor.predict_pose_adaptive(current_features)
            predicted_poses.append(predicted_pose.copy() if predicted_pose is not None else None)
            
            if predicted_pose is not None:
                print(f"Predicted pose: {predicted_pose} (confidence: {prediction_confidence:.3f})")
            else:
                print("No prediction available")
            
            # GICP registration for observation
            observed_pose = None
            if i < len(observation_pcds) - 1:
                next_scan_name, next_scan_cloud = observation_pcds[i + 1]
                print(f"GICP registration: {scan_name} -> {next_scan_name}")
                
                # Get transformation from GICP
                transformation = apply_gicp_func(scan_cloud, next_scan_cloud)
                
                # Accumulate transformation
                if cumulative_transform is not None:
                    cumulative_transform = cumulative_transform @ transformation
                else:
                    cumulative_transform = transformation
                
                # Convert to pose
                observed_pose = transformation_to_pose(cumulative_transform)
                print(f"Observed pose (before processing): {observed_pose}")
                
                # Estimate registration confidence
                reg_confidence = estimate_registration_confidence(scan_cloud, next_scan_cloud, transformation)
                print(f"Registration confidence: {reg_confidence:.3f}")
                
                # Update processor with observation (includes smoothing)
                processor.update_with_observation(observed_pose, current_features, reg_confidence, predicted_pose)
                
                # Get the final pose after processing and smoothing
                current_state = processor.get_current_state()
                if current_state is not None:
                    final_observed_pose = current_state.pose
                    print(f"Final pose (after smoothing): {final_observed_pose}")
                else:
                    final_observed_pose = observed_pose
                
            else:
                print("Last scan - no registration")
                if len(observed_poses) > 0 and observed_poses[-1] is not None:
                    observed_pose = observed_poses[-1].copy()
                else:
                    observed_pose = np.array([0.0, 0.0, 0.0, 0.0])
                final_observed_pose = observed_pose
            
            observed_poses.append(final_observed_pose.copy() if final_observed_pose is not None else None)
            
            # Get final pose estimate
            current_state = processor.get_current_state()
            if current_state is not None:
                final_pose = current_state.pose
                final_poses.append(final_pose.copy())
                print(f"Final smoothed pose: {final_pose}")
            else:
                final_poses.append(None)
                print("No final pose estimate")
            
            # Store original pose before smoothing
            original_poses.append(observed_pose.copy() if observed_pose is not None else None)
            
            # Store results
            motion_analysis = processor.get_motion_analysis()
            
            result = {
                'scan_file': scan_name,
                'scan_index': observation_start_index + i * observation_step_size,
                'predicted_pose': predicted_pose.copy() if predicted_pose is not None else None,
                'observed_pose': final_observed_pose.copy() if final_observed_pose is not None else None,
                'final_pose': final_pose.copy() if current_state is not None else None,
                'original_pose': observed_pose.copy() if observed_pose is not None else None,
                'prediction_confidence': prediction_confidence if predicted_pose is not None else 0.0,
                'features': current_features,
                'motion_analysis': motion_analysis,
                'processing_method': 'extreme_smoothed_z_zero' if force_z_zero else 'extreme_smoothed',
                'extreme_smoothing_applied': enable_extreme_smoothing
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {scan_name}: {e}")
            import traceback
            traceback.print_exc()
            
            results.append({
                'scan_file': scan_name,
                'error': str(e)
            })
            predicted_poses.append(None)
            observed_poses.append(None)
            final_poses.append(None)
            original_poses.append(None)
    
    # Final analysis with smoothing metrics
    motion_analysis = processor.get_motion_analysis()
    print(f"\n=== Processing Summary ===")
    print(f"Successfully processed: {len([r for r in results if 'error' not in r])} scans")
    if motion_analysis:
        print(f"Average movement: {motion_analysis['avg_movement']:.3f} m")
        print(f"Average confidence: {motion_analysis['avg_confidence']:.3f}")
        print(f"Extreme smoothing: {motion_analysis['extreme_smoothing_enabled']}")
        print(f"Smoothing method: {motion_analysis['smoothing_method']}")
        
        # Print smoothing metrics if available
        if 'smoothing_metrics' in motion_analysis:
            print(f"\n=== Extreme Smoothing Metrics ===")
            metrics = motion_analysis['smoothing_metrics']
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric_name}: {value:.4f}")
                else:
                    print(f"  {metric_name}: {value}")
    
    # Visualization with comparison
    if visualize and len(final_poses) > 0:
        # Create combined clouds for both original and smoothed
        original_cloud = create_combined_cloud(observation_pcds, original_poses)
        smoothed_cloud = create_combined_cloud(observation_pcds, final_poses)
        
        if original_cloud is not None and smoothed_cloud is not None:
            print("\nOpening combined LiDAR maps...")
            
            # Color the clouds differently
            original_cloud.paint_uniform_color([1.0, 0.7, 0.7])  # Light red for original
            smoothed_cloud.paint_uniform_color([0.7, 0.7, 1.0])  # Light blue for smoothed
            
            window_name = "LiDAR Map Comparison: Red=Original, Blue=Extreme Smoothed"
            o3d.visualization.draw_geometries([original_cloud, smoothed_cloud], 
                                            window_name=window_name,
                                            width=1400, height=900)
        
        # Plot detailed analysis with smoothing comparison
        plot_extreme_smoothing_analysis(predicted_poses, observed_poses, final_poses, 
                                      original_poses, processor)
    
    return results, predicted_poses, observed_poses, final_poses, original_poses

def transformation_to_pose(transformation_matrix: np.ndarray) -> np.ndarray:
    """Convert 4x4 transformation matrix to pose [x,y,z,yaw]"""
    translation = transformation_matrix[:3, 3]
    rotation_matrix = transformation_matrix[:3, :3]
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return np.array([translation[0], translation[1], translation[2], yaw])

def estimate_registration_confidence(cloud1, cloud2, transformation, sample_size=100):
    """Estimate confidence in GICP registration result"""
    try:
        points1 = np.asarray(cloud1.points)
        points2 = np.asarray(cloud2.points)
        
        if len(points1) == 0 or len(points2) == 0:
            return 0.1
        
        sample_indices = np.random.choice(len(points1), min(sample_size, len(points1)), replace=False)
        sample_points = points1[sample_indices]
        
        sample_points_hom = np.column_stack([sample_points, np.ones(len(sample_points))])
        transformed_points = (transformation @ sample_points_hom.T).T[:, :3]
        
        from scipy.spatial import cKDTree
        tree = cKDTree(points2)
        distances, _ = tree.query(transformed_points)
        
        avg_distance = np.mean(distances)
        confidence = max(0.1, min(1.0, 1.0 - avg_distance / 2.0))
        
        return confidence
        
    except Exception as e:
        print(f"Error estimating confidence: {e}")
        return 0.5

def create_combined_cloud(observation_pcds, poses):
    """Create combined point cloud from poses"""
    if not observation_pcds or not poses:
        return None
    
    combined_cloud = o3d.geometry.PointCloud()
    
    for i, ((scan_name, scan_cloud), pose) in enumerate(zip(observation_pcds, poses)):
        if pose is None:
            continue
        
        if i == 0:
            combined_cloud += scan_cloud
        else:
            T = np.eye(4)
            T[0, 3] = pose[0]  # x
            T[1, 3] = pose[1]  # y
            T[2, 3] = pose[2]  # z
            
            # Yaw rotation
            yaw = pose[3]
            T[0, 0] = np.cos(yaw)
            T[0, 1] = -np.sin(yaw)
            T[1, 0] = np.sin(yaw)
            T[1, 1] = np.cos(yaw)
            
            transformed_cloud = o3d.geometry.PointCloud(scan_cloud)
            transformed_cloud.transform(T)
            combined_cloud += transformed_cloud
    
    return combined_cloud

def plot_extreme_smoothing_analysis(predicted_poses, observed_poses, final_poses, 
                                   original_poses, processor):
    """Plot comprehensive analysis including smoothing comparison"""
    
    # Filter valid poses
    valid_indices = [i for i in range(len(final_poses)) if final_poses[i] is not None and original_poses[i] is not None]
    
    if not valid_indices:
        print("No valid poses to plot")
        return
    
    valid_final = [final_poses[i] for i in valid_indices]
    valid_original = [original_poses[i] for i in valid_indices]
    valid_predicted = [predicted_poses[i] if i < len(predicted_poses) and predicted_poses[i] is not None else None for i in valid_indices]
    
    final_array = np.array(valid_final)
    original_array = np.array(valid_original)
    
    # Create comprehensive analysis plots
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Extreme Path Smoothing Analysis for Non-Repetitive LiDAR', fontsize=16)
    
    time_steps = np.arange(len(valid_final))
    pose_labels = ['X (m)', 'Y (m)', 'Z (m)', 'Yaw (rad)']
    
    # Plot pose evolution comparison for each dimension (top row)
    for i in range(4):
        if i < 3:  # Only plot first 3 dimensions in top row
            ax = axes[0, i]
            
            # Plot original poses
            ax.plot(time_steps, original_array[:, i], 'r-', linewidth=2, marker='o', 
                   markersize=4, alpha=0.7, label='Original')
            
            # Plot smoothed poses
            ax.plot(time_steps, final_array[:, i], 'b-', linewidth=3, marker='^', 
                   markersize=5, label='Extreme Smoothed')
            
            # Plot predictions if available
            pred_array = np.array([pred for pred in valid_predicted if pred is not None])
            if len(pred_array) > 0:
                pred_time = time_steps[:len(pred_array)]
                ax.plot(pred_time, pred_array[:, i], 'g--', linewidth=2, marker='s', 
                       markersize=3, alpha=0.6, label='Predicted')
            
            # Special handling for Z coordinate
            if i == 2 and processor.force_z_zero:
                ax.axhline(y=0, color='black', linestyle=':', alpha=0.7, label='Forced Z=0')
                ax.set_ylim(-0.1, 0.1)
            
            ax.set_xlabel('Scan Index')
            ax.set_ylabel(pose_labels[i])
            ax.set_title(f'{pose_labels[i]} Evolution Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Trajectory comparison (middle row, left)
    ax_traj = axes[1, 0]
    ax_traj.plot(original_array[:, 0], original_array[:, 1], 'r-', linewidth=2, 
                marker='o', markersize=4, alpha=0.7, label='Original Trajectory')
    ax_traj.plot(final_array[:, 0], final_array[:, 1], 'b-', linewidth=3, 
                marker='^', markersize=5, label='Smoothed Trajectory')
    
    # Mark start and end points
    ax_traj.plot(original_array[0, 0], original_array[0, 1], 'go', markersize=8, label='Start')
    ax_traj.plot(original_array[-1, 0], original_array[-1, 1], 'ro', markersize=8, label='End')
    
    ax_traj.set_xlabel('X (m)')
    ax_traj.set_ylabel('Y (m)')
    ax_traj.set_title('2D Trajectory Comparison')
    ax_traj.legend()
    ax_traj.grid(True, alpha=0.3)
    ax_traj.axis('equal')
    
    # Movement magnitude comparison (middle row, center)
    ax_movement = axes[1, 1]
    
    # Calculate movement magnitudes
    orig_movements = []
    smooth_movements = []
    
    for i in range(1, len(valid_final)):
        if processor.force_z_zero:
            orig_movement = np.linalg.norm(original_array[i, :2] - original_array[i-1, :2])
            smooth_movement = np.linalg.norm(final_array[i, :2] - final_array[i-1, :2])
        else:
            orig_movement = np.linalg.norm(original_array[i, :3] - original_array[i-1, :3])
            smooth_movement = np.linalg.norm(final_array[i, :3] - final_array[i-1, :3])
        
        orig_movements.append(orig_movement)
        smooth_movements.append(smooth_movement)
    
    movement_time = time_steps[1:]
    ax_movement.plot(movement_time, orig_movements, 'r-', linewidth=2, marker='o', 
                    markersize=4, alpha=0.7, label='Original Movement')
    ax_movement.plot(movement_time, smooth_movements, 'b-', linewidth=3, marker='^', 
                    markersize=5, label='Smoothed Movement')
    
    ax_movement.set_xlabel('Scan Index')
    ax_movement.set_ylabel('Movement Magnitude (m)')
    ax_movement.set_title('Movement Magnitude Comparison')
    ax_movement.legend()
    ax_movement.grid(True, alpha=0.3)
    
    # Angular velocity comparison (middle row, right)
    ax_angular = axes[1, 2]
    
    # Calculate angular changes
    orig_angular = []
    smooth_angular = []
    
    for i in range(1, len(valid_final)):
        orig_angle_diff = original_array[i, 3] - original_array[i-1, 3]
        smooth_angle_diff = final_array[i, 3] - final_array[i-1, 3]
        
        # Handle angle wrapping
        for angle_diff in [orig_angle_diff, smooth_angle_diff]:
            while angle_diff > np.pi:
                angle_diff -= 2*np.pi
            while angle_diff < -np.pi:
                angle_diff += 2*np.pi
        
        orig_angular.append(abs(orig_angle_diff))
        smooth_angular.append(abs(smooth_angle_diff))
    
    ax_angular.plot(movement_time, np.degrees(orig_angular), 'r-', linewidth=2, 
                   marker='o', markersize=4, alpha=0.7, label='Original Angular')
    ax_angular.plot(movement_time, np.degrees(smooth_angular), 'b-', linewidth=3, 
                   marker='^', markersize=5, label='Smoothed Angular')
    
    ax_angular.set_xlabel('Scan Index')
    ax_angular.set_ylabel('Angular Change (degrees)')
    ax_angular.set_title('Angular Velocity Comparison')
    ax_angular.legend()
    ax_angular.grid(True, alpha=0.3)
    
    # Smoothing metrics visualization (bottom row)
    motion_analysis = processor.get_motion_analysis()
    
    # Curvature analysis (bottom row, left)
    ax_curvature = axes[2, 0]
    
    def calculate_curvature(poses):
        curvatures = []
        for i in range(1, len(poses) - 1):
            p1, p2, p3 = poses[i-1][:2], poses[i][:2], poses[i+1][:2]
            v1 = p2 - p1
            v2 = p3 - p2
            cross = np.cross(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 > 1e-6 and norm_v2 > 1e-6:
                curvature = abs(cross) / (norm_v1 * norm_v2)
                curvatures.append(curvature)
        return curvatures
    
    if len(valid_final) > 3:
        orig_curvature = calculate_curvature(original_array)
        smooth_curvature = calculate_curvature(final_array)
        
        if orig_curvature and smooth_curvature:
            curvature_time = time_steps[1:-1]
            ax_curvature.plot(curvature_time, orig_curvature, 'r-', linewidth=2, 
                             marker='o', markersize=4, alpha=0.7, label='Original Curvature')
            ax_curvature.plot(curvature_time, smooth_curvature, 'b-', linewidth=3, 
                             marker='^', markersize=5, label='Smoothed Curvature')
    
    ax_curvature.set_xlabel('Scan Index')
    ax_curvature.set_ylabel('Curvature')
    ax_curvature.set_title('Path Curvature Comparison')
    ax_curvature.legend()
    ax_curvature.grid(True, alpha=0.3)
    
    # Smoothing effectiveness metrics (bottom row, center)
    ax_metrics = axes[2, 1]
    
    if 'smoothing_metrics' in motion_analysis:
        metrics = motion_analysis['smoothing_metrics']
        metric_names = []
        metric_values = []
        
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and 'reduction' in name:
                metric_names.append(name.replace('_', ' ').title())
                metric_values.append(value * 100)  # Convert to percentage
        
        if metric_names:
            bars = ax_metrics.bar(range(len(metric_names)), metric_values, 
                                 color=['skyblue', 'lightcoral', 'lightgreen'][:len(metric_names)])
            ax_metrics.set_xlabel('Metric')
            ax_metrics.set_ylabel('Reduction (%)')
            ax_metrics.set_title('Smoothing Effectiveness')
            ax_metrics.set_xticks(range(len(metric_names)))
            ax_metrics.set_xticklabels(metric_names, rotation=45, ha='right')
            ax_metrics.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax_metrics.text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{value:.1f}%', ha='center', va='bottom')
    
    # Path smoothness comparison (bottom row, right)
    ax_smoothness = axes[2, 2]
    
    # Calculate path length ratios
    def calculate_path_length(poses):
        length = 0
        for i in range(1, len(poses)):
            segment_length = np.linalg.norm(poses[i][:3] - poses[i-1][:3])
            length += segment_length
        return length
    
    orig_length = calculate_path_length(original_array)
    smooth_length = calculate_path_length(final_array)
    
    # Create summary statistics
    summary_stats = {
        'Original Path Length': f'{orig_length:.2f} m',
        'Smoothed Path Length': f'{smooth_length:.2f} m',
        'Length Change': f'{((smooth_length - orig_length) / orig_length * 100):.1f}%',
        'Avg Original Movement': f'{np.mean(orig_movements):.3f} m',
        'Avg Smoothed Movement': f'{np.mean(smooth_movements):.3f} m',
        'Movement Std Reduction': f'{((np.std(orig_movements) - np.std(smooth_movements)) / np.std(orig_movements) * 100):.1f}%'
    }
    
    # Display as text
    ax_smoothness.axis('off')
    y_pos = 0.9
    ax_smoothness.text(0.1, y_pos, 'Path Smoothing Summary:', fontsize=14, fontweight='bold')
    y_pos -= 0.15
    
    for key, value in summary_stats.items():
        ax_smoothness.text(0.1, y_pos, f'{key}: {value}', fontsize=12)
        y_pos -= 0.12
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis
    print_extreme_smoothing_analysis(original_array, final_array, processor)

def print_extreme_smoothing_analysis(original_array, final_array, processor):
    """Print detailed analysis of extreme smoothing results"""
    print(f"\n=== Extreme Smoothing Analysis ===")
    
    if len(original_array) > 1 and len(final_array) > 1:
        # Calculate comprehensive metrics
        def analyze_trajectory(poses, label):
            print(f"\n{label} Trajectory:")
            
            # Distance metrics
            total_distance = 0
            movements = []
            for i in range(1, len(poses)):
                if processor.force_z_zero:
                    movement = np.linalg.norm(poses[i, :2] - poses[i-1, :2])
                else:
                    movement = np.linalg.norm(poses[i, :3] - poses[i-1, :3])
                movements.append(movement)
                total_distance += movement
            
            print(f"  Total distance: {total_distance:.3f} m")
            print(f"  Average movement: {np.mean(movements):.3f} m")
            print(f"  Movement std dev: {np.std(movements):.3f} m")
            print(f"  Movement consistency (CV): {np.std(movements) / (np.mean(movements) + 1e-6):.3f}")
            
            # Angular metrics
            yaw_changes = []
            for i in range(1, len(poses)):
                yaw_change = abs(poses[i, 3] - poses[i-1, 3])
                if yaw_change > np.pi:
                    yaw_change = 2*np.pi - yaw_change
                yaw_changes.append(yaw_change)
            
            if yaw_changes:
                print(f"  Total yaw change: {np.degrees(np.sum(yaw_changes)):.1f}°")
                print(f"  Average yaw change: {np.degrees(np.mean(yaw_changes)):.1f}°")
                print(f"  Yaw std dev: {np.degrees(np.std(yaw_changes)):.1f}°")
            
            return movements, yaw_changes
        
        # Analyze both trajectories
        orig_movements, orig_yaw_changes = analyze_trajectory(original_array, "Original")
        smooth_movements, smooth_yaw_changes = analyze_trajectory(final_array, "Extreme Smoothed")
        
        # Comparison metrics
        print(f"\n=== Smoothing Comparison ===")
        
        if orig_movements and smooth_movements:
            movement_reduction = (np.std(orig_movements) - np.std(smooth_movements)) / np.std(orig_movements)
            print(f"Movement variability reduction: {movement_reduction * 100:.1f}%")
            
            avg_movement_change = (np.mean(smooth_movements) - np.mean(orig_movements)) / np.mean(orig_movements)
            print(f"Average movement change: {avg_movement_change * 100:.1f}%")
        
        if orig_yaw_changes and smooth_yaw_changes:
            yaw_reduction = (np.std(orig_yaw_changes) - np.std(smooth_yaw_changes)) / np.std(orig_yaw_changes)
            print(f"Angular variability reduction: {yaw_reduction * 100:.1f}%")
        
        # Endpoint preservation
        start_diff = np.linalg.norm(original_array[0][:3] - final_array[0][:3])
        end_diff = np.linalg.norm(original_array[-1][:3] - final_array[-1][:3])
        print(f"Start point preservation: {start_diff:.6f} m difference")
        print(f"End point preservation: {end_diff:.6f} m difference")
        
        # Overall smoothness assessment
        orig_cv = np.std(orig_movements) / (np.mean(orig_movements) + 1e-6)
        smooth_cv = np.std(smooth_movements) / (np.mean(smooth_movements) + 1e-6)
        
        print(f"\nOverall Smoothness Assessment:")
        print(f"Original trajectory smoothness: {get_smoothness_rating(orig_cv)}")
        print(f"Smoothed trajectory smoothness: {get_smoothness_rating(smooth_cv)}")
        print(f"Improvement factor: {orig_cv / smooth_cv:.2f}x smoother")

def get_smoothness_rating(cv):
    """Get smoothness rating based on coefficient of variation"""
    if cv < 0.2:
        return "Very Smooth"
    elif cv < 0.4:
        return "Smooth" 
    elif cv < 0.6:
        return "Moderately Smooth"
    elif cv < 0.8:
        return "Somewhat Rough"
    else:
        return "Very Rough"

def save_extreme_smoothing_results(results, final_poses, original_poses, 
                                  combined_cloud=None, force_z_zero=False):
    """Save results from extreme smoothing processing"""
    import os
    
    # Create output directory
    output_dir = "./output/extreme_smoothed_lidar"
    if force_z_zero:
        output_dir += "_z_zero"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save poses comparison to CSV
    poses_file = os.path.join(output_dir, "smoothing_comparison.csv")
    pose_data = []
    
    for i, (result, final_pose, orig_pose) in enumerate(zip(results, final_poses, original_poses)):
        if final_pose is not None and orig_pose is not None:
            pose_data.append({
                'scan_index': i,
                'scan_file': result.get('scan_file', f'scan_{i}'),
                'orig_x': orig_pose[0],
                'orig_y': orig_pose[1],
                'orig_z': orig_pose[2],
                'orig_yaw_rad': orig_pose[3],
                'smooth_x': final_pose[0],
                'smooth_y': final_pose[1],
                'smooth_z': final_pose[2],
                'smooth_yaw_rad': final_pose[3],
                'x_diff': final_pose[0] - orig_pose[0],
                'y_diff': final_pose[1] - orig_pose[1],
                'z_diff': final_pose[2] - orig_pose[2],
                'yaw_diff_rad': final_pose[3] - orig_pose[3],
                'extreme_smoothing_applied': result.get('extreme_smoothing_applied', False)
            })
    
    if pose_data:
        df = pd.DataFrame(pose_data)
        df.to_csv(poses_file, index=False)
        print(f"Saved smoothing comparison to: {poses_file}")
    
    # Save combined cloud
    if combined_cloud is not None:
        cloud_filename = "extreme_smoothed_map_z_zero.pcd" if force_z_zero else "extreme_smoothed_map.pcd"
        cloud_file = os.path.join(output_dir, cloud_filename)
        success = o3d.io.write_point_cloud(cloud_file, combined_cloud)
        if success:
            print(f"Saved extreme smoothed cloud to: {cloud_file}")
    
    # Save detailed results with smoothing metrics
    results_file = os.path.join(output_dir, "extreme_smoothing_results.json")
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = []
    for result in results:
        json_result = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                json_result[key] = value.tolist()
            elif key == 'features':
                json_result[key] = 'extracted' if value else 'failed'
            elif key == 'motion_analysis' and isinstance(value, dict):
                # Handle nested dict with numpy arrays
                json_motion = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        json_motion[k] = v.tolist()
                    elif isinstance(v, dict):
                        json_motion[k] = v  # Nested dicts should be serializable
                    else:
                        json_motion[k] = v
                json_result[key] = json_motion
            else:
                json_result[key] = value
        json_results.append(json_result)
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved extreme smoothing results to: {results_file}")

# Example usage function
def main_extreme_smoothing_lidar():
    """Example usage for extreme smoothing LiDAR processing"""
    
    # Configure paths
    observation_folder = "/home/robotics/testdata/BIA6"  # Replace with your path
    
    print("=== Extreme Path Smoothing for Non-Repetitive LiDAR ===")
    print("Features:")
    print("- Extreme path smoothing with multiple methods")
    print("- Multi-pass smoothing for ultra-smooth trajectories")
    print("- Gaussian, Spline, and Savitzky-Golay smoothing")
    print("- Preserves trajectory endpoints")
    print("- Comprehensive smoothing metrics")
    print("- Adaptive feature-based prediction")
    print("- Optional Z=0 mode with redistribution")
    print("////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////")
    
    try:
        # Process the scans with extreme smoothing enabled
        results, pred_poses, obs_poses, final_poses, orig_poses = process_non_repetitive_lidar_scans_with_extreme_smoothing(
            observation_folder,
            visualize=True,
            observation_step_size=5,  # Process every 5th scan for efficiency
            observation_start_index=0,
            max_observation_clouds=250,  # Limit for testing
            force_z_zero=True,  # Enable Z=0 mode
            z_redistribution_method='prediction',  # Use prediction-based redistribution
            enable_extreme_smoothing=True,  # Enable extreme smoothing
            smoothing_strength=0.95,  # Very high smoothing (0.0 = none, 1.0 = maximum)
            smoothing_method='multi_pass'  # Use multi-pass smoothing for best results
        )
        
        print(f"\n=== Extreme Smoothing Processing Complete ===")
        successful_scans = len([r for r in results if 'error' not in r])
        print(f"Successfully processed: {successful_scans}/{len(results)} scans")
        
        if final_poses and any(pose is not None for pose in final_poses):
            # Create combined clouds for both original and smoothed
            from Kalman import load_pcd_files
            observation_pcds = load_pcd_files(observation_folder, 5, 0, 250)
            
            original_cloud = create_combined_cloud(observation_pcds, orig_poses)
            smoothed_cloud = create_combined_cloud(observation_pcds, final_poses)
            
            # Save results
            save_extreme_smoothing_results(results, final_poses, orig_poses, 
                                         smoothed_cloud, force_z_zero=True)
            
            print(f"\nKey achievements:")
            print(f"- Applied extreme path smoothing to LiDAR trajectory")
            print(f"- Multi-pass smoothing with 95% strength")
            print(f"- Z-coordinate forced to 0 with intelligent redistribution")
            print(f"- Generated ultra-smooth 2D trajectory map")
            print(f"- Comprehensive smoothing metrics calculated")
            print(f"- Results saved to ./output/extreme_smoothed_lidar_z_zero/")
            
            # Calculate and display final smoothing statistics
            if orig_poses and final_poses:
                valid_orig = [p for p in orig_poses if p is not None]
                valid_final = [p for p in final_poses if p is not None]
                
                if len(valid_orig) > 1 and len(valid_final) > 1:
                    orig_array = np.array(valid_orig)
                    final_array = np.array(valid_final)
                    
                    # Calculate movement variability reduction
                    orig_movements = []
                    final_movements = []
                    
                    for i in range(1, len(orig_array)):
                        orig_move = np.linalg.norm(orig_array[i, :2] - orig_array[i-1, :2])
                        final_move = np.linalg.norm(final_array[i, :2] - final_array[i-1, :2])
                        orig_movements.append(orig_move)
                        final_movements.append(final_move)
                    
                    if orig_movements and final_movements:
                        orig_std = np.std(orig_movements)
                        final_std = np.std(final_movements)
                        reduction = (orig_std - final_std) / orig_std * 100
                        
                        print(f"\n=== Final Smoothing Statistics ===")
                        print(f"Movement variability reduction: {reduction:.1f}%")
                        print(f"Original movement std: {orig_std:.4f} m")
                        print(f"Smoothed movement std: {final_std:.4f} m")
                        print(f"Smoothness improvement: {orig_std / final_std:.2f}x")
        
        return results, pred_poses, obs_poses, final_poses, orig_poses
        
    except Exception as e:
        print(f"Error in extreme smoothing processing: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None

# Additional utility functions for advanced smoothing analysis

def analyze_smoothing_effectiveness(original_poses, smoothed_poses, method_name="Unknown"):
    """Analyze the effectiveness of the smoothing algorithm"""
    if not original_poses or not smoothed_poses:
        return {}
    
    # Filter valid poses
    valid_pairs = [(orig, smooth) for orig, smooth in zip(original_poses, smoothed_poses) 
                   if orig is not None and smooth is not None]
    
    if len(valid_pairs) < 3:
        return {}
    
    orig_array = np.array([pair[0] for pair in valid_pairs])
    smooth_array = np.array([pair[1] for pair in valid_pairs])
    
    metrics = {}
    metrics['method'] = method_name
    metrics['trajectory_length'] = len(valid_pairs)
    
    # Calculate trajectory smoothness metrics
    def calculate_jerk(poses):
        """Calculate jerk (third derivative of position)"""
        if len(poses) < 4:
            return []
        
        jerks = []
        for i in range(2, len(poses) - 1):
            # Calculate second differences (approximation of jerk)
            vel1 = poses[i] - poses[i-1]
            vel2 = poses[i+1] - poses[i]
            accel1 = vel1 - (poses[i-1] - poses[i-2])
            accel2 = vel2 - vel1
            jerk = accel2 - accel1
            jerk_magnitude = np.linalg.norm(jerk[:3])
            jerks.append(jerk_magnitude)
        
        return jerks
    
    # Calculate jerk reduction
    orig_jerks = calculate_jerk(orig_array)
    smooth_jerks = calculate_jerk(smooth_array)
    
    if orig_jerks and smooth_jerks:
        orig_jerk_mean = np.mean(orig_jerks)
        smooth_jerk_mean = np.mean(smooth_jerks)
        if orig_jerk_mean > 0:
            jerk_reduction = (orig_jerk_mean - smooth_jerk_mean) / orig_jerk_mean
            metrics['jerk_reduction'] = jerk_reduction
            metrics['jerk_improvement_factor'] = orig_jerk_mean / smooth_jerk_mean if smooth_jerk_mean > 0 else float('inf')
    
    # Calculate path deviation
    max_deviation = 0
    mean_deviation = 0
    deviations = []
    
    for orig, smooth in zip(orig_array, smooth_array):
        deviation = np.linalg.norm(orig[:3] - smooth[:3])
        deviations.append(deviation)
        max_deviation = max(max_deviation, deviation)
    
    if deviations:
        mean_deviation = np.mean(deviations)
        metrics['max_path_deviation'] = max_deviation
        metrics['mean_path_deviation'] = mean_deviation
        metrics['path_deviation_std'] = np.std(deviations)
    
    # Calculate frequency domain metrics (if trajectory is long enough)
    if len(orig_array) > 10:
        from scipy.fft import fft, fftfreq
        
        # Analyze frequency content of x and y coordinates
        for dim, dim_name in enumerate(['x', 'y']):
            orig_fft = fft(orig_array[:, dim])
            smooth_fft = fft(smooth_array[:, dim])
            
            # Calculate power spectral density
            orig_power = np.abs(orig_fft) ** 2
            smooth_power = np.abs(smooth_fft) ** 2
            
            # High frequency attenuation
            n = len(orig_fft)
            high_freq_idx = n // 4  # Consider upper 75% as high frequency
            
            orig_high_power = np.sum(orig_power[high_freq_idx:])
            smooth_high_power = np.sum(smooth_power[high_freq_idx:])
            
            if orig_high_power > 0:
                high_freq_attenuation = (orig_high_power - smooth_high_power) / orig_high_power
                metrics[f'{dim_name}_high_freq_attenuation'] = high_freq_attenuation
    
    return metrics

def compare_smoothing_methods(poses, methods=['gaussian', 'spline', 'savgol', 'multi_pass']):
    """Compare different smoothing methods on the same trajectory"""
    if not poses or len(poses) < 5:
        print("Insufficient poses for method comparison")
        return {}
    
    # Filter valid poses
    valid_poses = [p for p in poses if p is not None]
    if len(valid_poses) < 5:
        print("Insufficient valid poses for method comparison")
        return {}
    
    poses_array = np.array(valid_poses)
    comparison_results = {}
    
    print(f"\n=== Comparing Smoothing Methods on {len(valid_poses)} poses ===")
    
    for method in methods:
        print(f"Testing {method} smoothing...")
        try:
            smoother = ExtremePathSmoother(
                smoothing_strength=0.8,  # Moderate strength for comparison
                preserve_endpoints=True,
                angle_smoothing_factor=0.8
            )
            
            smoothed = smoother.apply_extreme_smoothing(poses_array, method=method)
            
            # Calculate metrics
            metrics = analyze_smoothing_effectiveness(valid_poses, smoothed.tolist(), method)
            metrics['success'] = True
            
            comparison_results[method] = metrics
            
            print(f"  {method}: Success")
            if 'jerk_reduction' in metrics:
                print(f"    Jerk reduction: {metrics['jerk_reduction']*100:.1f}%")
            if 'mean_path_deviation' in metrics:
                print(f"    Mean deviation: {metrics['mean_path_deviation']:.4f} m")
                
        except Exception as e:
            print(f"  {method}: Failed - {e}")
            comparison_results[method] = {'success': False, 'error': str(e)}
    
    # Determine best method
    successful_methods = {k: v for k, v in comparison_results.items() if v.get('success', False)}
    
    if successful_methods:
        # Rank by jerk reduction (primary) and low path deviation (secondary)
        def rank_method(method_data):
            jerk_score = method_data.get('jerk_reduction', 0) * 100  # Higher is better
            deviation_score = -method_data.get('mean_path_deviation', 1) * 1000  # Lower is better (negative)
            return jerk_score + deviation_score
        
        ranked_methods = sorted(successful_methods.items(), key=lambda x: rank_method(x[1]), reverse=True)
        best_method = ranked_methods[0][0]
        
        print(f"\n=== Method Ranking ===")
        for i, (method, data) in enumerate(ranked_methods):
            print(f"{i+1}. {method}")
            if 'jerk_reduction' in data:
                print(f"   Jerk reduction: {data['jerk_reduction']*100:.1f}%")
            if 'mean_path_deviation' in data:
                print(f"   Mean deviation: {data['mean_path_deviation']:.4f} m")
        
        print(f"\nRecommended method: {best_method}")
        comparison_results['recommended'] = best_method
    
    return comparison_results

if __name__ == "__main__":
    # Run the extreme smoothing example
    main_extreme_smoothing_lidar()
    
    # Optionally, run method comparison on sample data
    print("\n" + "="*60)
    print("Running smoothing method comparison on sample trajectory...")
    
    # Generate sample noisy trajectory for demonstration
    t = np.linspace(0, 10, 50)
    sample_poses = []
    for i, time in enumerate(t):
        # Create a curved path with noise
        x = time + 0.1 * np.sin(time * 2) + np.random.normal(0, 0.05)
        y = 0.5 * np.sin(time) + 0.1 * np.cos(time * 3) + np.random.normal(0, 0.03)
        z = 0.0  # 2D trajectory
        yaw = 0.1 * time + 0.05 * np.sin(time * 4) + np.random.normal(0, 0.02)
        
        sample_poses.append(np.array([x, y, z, yaw]))
    
    # Compare methods
    method_comparison = compare_smoothing_methods(sample_poses)
    
    if 'recommended' in method_comparison:
        print(f"\nFor your data, we recommend using: {method_comparison['recommended']} smoothing")
    
    print("\nExtreme path smoothing analysis complete!")