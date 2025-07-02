import numpy as np
import open3d as o3d
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import pandas as pd

@dataclass
class ScanState:
    """State for non-repetitive LiDAR scan processing"""
    pose: np.ndarray  # [x, y, z, yaw]
    uncertainty: np.ndarray  # 4x4 covariance matrix
    confidence: float  # Confidence in this pose estimate
    scan_features: Dict  # Geometric features of the scan
    prediction_error: Optional[np.ndarray] = None  # Error between prediction and observation
    fusion_weight: float = 0.5  # Weight used in prediction-observation fusion

class NonRepetitiveLiDARProcessor:
    def __init__(self, 
                 adaptive_threshold: float = 0.9,
                 feature_weight: float = 0.3,
                 geometric_weight: float = 0.4,
                 temporal_weight: float = 0.3,
                 force_z_zero: bool = False,
                 z_redistribution_method: str = 'prediction',
                 prediction_observation_fusion: str = 'adaptive'):  # 'adaptive', 'weighted', 'confidence_based'
        """
        Processor for non-repetitive LiDAR scans without velocity assumptions
        
        Args:
            adaptive_threshold: Threshold for switching prediction strategies
            feature_weight: Weight for feature-based matching
            geometric_weight: Weight for geometric consistency
            temporal_weight: Weight for temporal smoothness
            force_z_zero: If True, forces z coordinate to 0 and redistributes z values
            z_redistribution_method: Method for redistributing z values ('prediction', 'dominant_axis', 'equal')
            prediction_observation_fusion: Method for fusing prediction and observation ('adaptive', 'weighted', 'confidence_based')
        """
        self.adaptive_threshold = adaptive_threshold
        self.feature_weight = feature_weight
        self.geometric_weight = geometric_weight
        self.temporal_weight = temporal_weight
        self.force_z_zero = force_z_zero
        self.z_redistribution_method = z_redistribution_method
        self.prediction_observation_fusion = prediction_observation_fusion
        
        # State tracking
        self.scan_states = []  # History of scan states
        self.feature_database = []  # Database of scan features
        self.motion_patterns = []  # Detected motion patterns
        
        # Prediction-observation fusion tracking
        self.prediction_accuracy_history = []  # Track prediction accuracy over time
        self.observation_reliability_history = []  # Track observation reliability
        self.fusion_weights_history = []  # Track how fusion weights evolve
        
        # Adaptive parameters
        self.current_strategy = "feature_based"
        self.confidence_threshold = 0.7
        
        # Feature extraction parameters
        self.voxel_size = 0.1
        self.normal_radius = 0.5
        self.fpfh_radius = 1.0

    def redistribute_z_component(self, pose: np.ndarray, predicted_pose: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Redistribute z component to x and y coordinates based on the specified method
        
        Args:
            pose: Original pose [x, y, z, yaw]
            predicted_pose: Predicted pose for direction guidance
            
        Returns:
            Modified pose with z=0 and redistributed values
        """
        if not self.force_z_zero or abs(pose[2]) < 1e-6:
            return pose.copy()
        
        modified_pose = pose.copy()
        z_value = modified_pose[2]
        
        if self.z_redistribution_method == 'prediction' and predicted_pose is not None:
            # Use prediction to determine dominant movement direction
            if len(self.scan_states) >= 1:
                last_pose = self.scan_states[-1].pose
                predicted_movement = predicted_pose[:3] - last_pose[:3]
                
                # Determine dominant movement axis based on prediction
                modified_pose[0] += 1 * -z_value + predicted_movement[0]
                
        elif self.z_redistribution_method == 'dominant_axis':
            # Use historical movement to determine dominant axis
            if len(self.scan_states) >= 2:
                recent_poses = [state.pose for state in self.scan_states[-3:]]
                x_movements = []
                y_movements = []
                
                for i in range(1, len(recent_poses)):
                    x_movements.append(abs(recent_poses[i][0] - recent_poses[i-1][0]))
                    y_movements.append(abs(recent_poses[i][1] - recent_poses[i-1][1]))
                
                avg_x_movement = np.mean(x_movements)
                avg_y_movement = np.mean(y_movements)
                
                if avg_x_movement > avg_y_movement:
                    modified_pose[0] += z_value
                    print(f"Redistributing z={z_value:.3f} to x based on dominant axis")
                else:
                    modified_pose[1] += z_value
                    print(f"Redistributing z={z_value:.3f} to y based on dominant axis")
            else:
                # Equal distribution for early scans
                modified_pose[0] += z_value * 0.5
                modified_pose[1] += z_value * 0.5
                print(f"Redistributing z={z_value:.3f} equally (insufficient history)")
                
        elif self.z_redistribution_method == 'equal':
            # Distribute equally between x and y
            modified_pose[0] += z_value * 0.5
            modified_pose[1] += z_value * 0.5
            print(f"Redistributing z={z_value:.3f} equally to x and y")
        
        # Force z to zero
        modified_pose[2] = 0.0
        
        return modified_pose

    def extract_scan_features(self, cloud: o3d.geometry.PointCloud) -> Dict:
        """
        Extract geometric features from LiDAR scan for non-repetitive matching
        
        Args:
            cloud: Input point cloud
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        try:
            # Basic geometric properties
            points = np.asarray(cloud.points)
            if len(points) == 0:
                return features
            
            # 1. Statistical features
            features['point_count'] = len(points)
            features['centroid'] = np.mean(points, axis=0)
            features['std_dev'] = np.std(points, axis=0)
            features['bounding_box'] = {
                'min': np.min(points, axis=0),
                'max': np.max(points, axis=0),
                'extent': np.max(points, axis=0) - np.min(points, axis=0)
            }
            
            # 2. Downsampling for feature extraction
            if len(points) > 1000:
                cloud_ds = cloud.voxel_down_sample(self.voxel_size)
            else:
                cloud_ds = cloud
            
            # 3. Normal estimation
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
            
            # 4. FPFH features for distinctive geometric signatures
            if len(cloud_ds.points) > 50 and cloud_ds.has_normals():
                fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    cloud_ds,
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=self.fpfh_radius, max_nn=100
                    )
                )
                features['fpfh_histogram'] = np.asarray(fpfh.data).mean(axis=1)
            
            # 5. Planar structures detection
            if len(cloud_ds.points) > 100:
                plane_model, inliers = cloud_ds.segment_plane(
                    distance_threshold=0.1,
                    ransac_n=3,
                    num_iterations=1000
                )
                
                if len(inliers) > 50:
                    features['dominant_plane'] = {
                        'normal': plane_model[:3],
                        'distance': plane_model[3],
                        'inlier_ratio': len(inliers) / len(cloud_ds.points)
                    }
            
            # 6. Height distribution (for outdoor LiDAR)
            z_coords = points[:, 2]
            features['height_profile'] = {
                'min_height': np.min(z_coords),
                'max_height': np.max(z_coords),
                'mean_height': np.mean(z_coords),
                'height_variance': np.var(z_coords)
            }
            
            # 7. Density analysis
            if len(points) > 100:
                # Sample points for density estimation
                sample_indices = np.random.choice(len(points), min(100, len(points)), replace=False)
                sample_points = points[sample_indices]
                
                distances = cdist(sample_points, points)
                k_nearest_dists = np.sort(distances, axis=1)[:, 1:6]  # 5 nearest neighbors
                avg_density = np.mean(k_nearest_dists)
                features['local_density'] = avg_density
            
            # 8. Shape complexity
            if len(points) > 20:
                # PCA analysis for shape understanding
                pca = PCA(n_components=3)
                pca.fit(points)
                features['shape_complexity'] = {
                    'explained_variance_ratio': pca.explained_variance_ratio_,
                    'linearity': pca.explained_variance_ratio_[0],
                    'planarity': pca.explained_variance_ratio_[1],
                    'sphericity': pca.explained_variance_ratio_[2]
                }
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            features['extraction_error'] = str(e)
        
        return features

    def compute_feature_similarity(self, features1: Dict, features2: Dict) -> float:
        """
        Compute similarity between two feature sets
        
        Args:
            features1, features2: Feature dictionaries
            
        Returns:
            Similarity score [0, 1]
        """
        if not features1 or not features2:
            return 0.0
        
        similarities = []
        
        try:
            # 1. Point count similarity
            if 'point_count' in features1 and 'point_count' in features2:
                count_ratio = min(features1['point_count'], features2['point_count']) / \
                             max(features1['point_count'], features2['point_count'])
                similarities.append(count_ratio)
            
            # 2. Centroid distance (normalized) - only consider x,y if force_z_zero is True
            if 'centroid' in features1 and 'centroid' in features2:
                if self.force_z_zero:
                    centroid_dist = np.linalg.norm(features1['centroid'][:2] - features2['centroid'][:2])
                else:
                    centroid_dist = np.linalg.norm(features1['centroid'] - features2['centroid'])
                # Normalize by typical scan range (assume 50m max)
                centroid_sim = max(0, 1 - centroid_dist / 50.0)
                similarities.append(centroid_sim)
            
            # 3. Bounding box similarity
            if ('bounding_box' in features1 and 'bounding_box' in features2):
                bb1, bb2 = features1['bounding_box'], features2['bounding_box']
                if self.force_z_zero:
                    # Only consider x,y extents
                    extent1 = bb1['extent'][:2]
                    extent2 = bb2['extent'][:2]
                else:
                    extent1 = bb1['extent']
                    extent2 = bb2['extent']
                extent_ratio = np.prod(np.minimum(extent1, extent2)) / \
                              np.prod(np.maximum(extent1, extent2))
                similarities.append(extent_ratio)
            
            # 4. FPFH feature similarity
            if ('fpfh_histogram' in features1 and 'fpfh_histogram' in features2):
                fpfh1, fpfh2 = features1['fpfh_histogram'], features2['fpfh_histogram']
                if len(fpfh1) == len(fpfh2):
                    # Cosine similarity
                    dot_product = np.dot(fpfh1, fpfh2)
                    norm_product = np.linalg.norm(fpfh1) * np.linalg.norm(fpfh2)
                    if norm_product > 0:
                        fpfh_sim = dot_product / norm_product
                        similarities.append(max(0, fpfh_sim))
            
            # 5. Height profile similarity (modified for z=0 mode)
            if ('height_profile' in features1 and 'height_profile' in features2):
                hp1, hp2 = features1['height_profile'], features2['height_profile']
                if not self.force_z_zero:
                    height_range_ratio = min(hp1['max_height'] - hp1['min_height'],
                                           hp2['max_height'] - hp2['min_height']) / \
                                       max(hp1['max_height'] - hp1['min_height'],
                                           hp2['max_height'] - hp2['min_height'])
                    similarities.append(height_range_ratio)
                else:
                    # In z=0 mode, consider height variance instead of range
                    var_ratio = min(hp1['height_variance'], hp2['height_variance']) / \
                               max(hp1['height_variance'], hp2['height_variance'])
                    similarities.append(var_ratio)
            
            # 6. Density similarity
            if ('local_density' in features1 and 'local_density' in features2):
                density_ratio = min(features1['local_density'], features2['local_density']) / \
                               max(features1['local_density'], features2['local_density'])
                similarities.append(density_ratio)
            
        except Exception as e:
            print(f"Error computing similarity: {e}")
        
        # Return weighted average of similarities
        if similarities:
            return np.mean(similarities)
        else:
            return 0.0

    def predict_pose_feature_based(self, current_features: Dict) -> Tuple[np.ndarray, float]:
        """
        Predict pose based on feature matching with previous scans
        
        Args:
            current_features: Features of current scan
            
        Returns:
            (predicted_pose, confidence)
        """
        if len(self.feature_database) < 2:
            return None, 0.0
        
        # Find most similar previous scans
        similarities = []
        for i, (features, state) in enumerate(zip(self.feature_database, self.scan_states)):
            sim = self.compute_feature_similarity(current_features, features)
            similarities.append((sim, i, state))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        if len(similarities) < 2:
            return None, 0.0
        
        # Use top matches for prediction
        best_sim, best_idx, best_state = similarities[0]
        second_sim, second_idx, second_state = similarities[1]
        
        if best_sim < 0.3:  # Low similarity threshold
            return None, 0.0
        
        # Weighted prediction based on similarity
        weight1 = best_sim / (best_sim + second_sim)
        weight2 = second_sim / (best_sim + second_sim)
        
        predicted_pose = weight1 * best_state.pose + weight2 * second_state.pose
        confidence = (best_sim + second_sim) / 2
        
        return predicted_pose, confidence

    def predict_pose_geometric_consistency(self) -> Tuple[np.ndarray, float]:
        """
        Predict pose based on geometric consistency with recent scans
        
        Returns:
            (predicted_pose, confidence)
        """
        if len(self.scan_states) < 3:
            return None, 0.0
        
        # Use last 3-5 poses for geometric consistency
        recent_states = self.scan_states[-min(5, len(self.scan_states)):]
        poses = [state.pose for state in recent_states]
        
        # Fit smooth trajectory through recent poses
        if len(poses) >= 3:
            # Simple polynomial fit for each dimension
            predicted_pose = np.zeros(4)
            confidence_scores = []
            
            # Handle z coordinate based on force_z_zero setting
            dimensions_to_predict = 4 if not self.force_z_zero else [0, 1, 3]  # Skip z if forced to zero
            
            for dim in dimensions_to_predict if self.force_z_zero else range(4):
                values = [pose[dim] for pose in poses]
                x = np.arange(len(values))
                
                # Fit polynomial (degree depends on number of points)
                degree = min(2, len(values) - 1)
                if degree > 0:
                    coeffs = np.polyfit(x, values, degree)
                    next_x = len(values)
                    predicted_pose[dim] = np.polyval(coeffs, next_x)
                    
                    # Estimate confidence based on fit quality
                    fitted_values = np.polyval(coeffs, x)
                    residuals = np.array(values) - fitted_values
                    mse = np.mean(residuals**2)
                    confidence_scores.append(max(0, 1 - mse))
                else:
                    predicted_pose[dim] = values[-1]
                    confidence_scores.append(0.5)
            
            # Force z to zero if required
            if self.force_z_zero:
                predicted_pose[2] = 0.0
                
            # Handle angle wrapping for yaw
            predicted_pose[3] = np.arctan2(np.sin(predicted_pose[3]), np.cos(predicted_pose[3]))
            
            avg_confidence = np.mean(confidence_scores)
            return predicted_pose, avg_confidence
        
        return None, 0.0

    def predict_pose_adaptive(self, current_features: Dict) -> Tuple[np.ndarray, float]:
        """
        Adaptive pose prediction combining multiple strategies
        
        Args:
            current_features: Features of current scan
            
        Returns:
            (predicted_pose, confidence)
        """
        predictions = []
        
        # 1. Feature-based prediction
        feature_pose, feature_conf = self.predict_pose_feature_based(current_features)
        if feature_pose is not None:
            predictions.append((feature_pose, feature_conf, 'feature'))
        
        # 2. Geometric consistency prediction
        geom_pose, geom_conf = self.predict_pose_geometric_consistency()
        if geom_pose is not None:
            predictions.append((geom_pose, geom_conf, 'geometric'))
        
        # 3. Simple extrapolation (fallback)
        if len(self.scan_states) >= 2:
            last_pose = self.scan_states[-1].pose
            prev_pose = self.scan_states[-2].pose
            extrapolated_pose = last_pose + 0.3 * (last_pose - prev_pose)  # Damped extrapolation
            extrapolated_pose[3] = np.arctan2(np.sin(extrapolated_pose[3]), np.cos(extrapolated_pose[3]))
            
            # Force z to zero if required
            if self.force_z_zero:
                extrapolated_pose[2] = 0.0
                
            predictions.append((extrapolated_pose, 0.4, 'extrapolation'))
        
        if not predictions:
            return None, 0.0
        
        # Adaptive fusion based on confidence and strategy performance
        if len(predictions) == 1:
            return predictions[0][0], predictions[0][1]
        
        # Weight predictions by confidence and strategy reliability
        total_weight = 0
        weighted_pose = np.zeros(4)
        
        for pose, conf, strategy in predictions:
            # Strategy-specific weights
            if strategy == 'feature':
                strategy_weight = self.feature_weight
            elif strategy == 'geometric':
                strategy_weight = self.geometric_weight
            else:
                strategy_weight = self.temporal_weight
            
            weight = conf * strategy_weight
            weighted_pose += weight * pose
            total_weight += weight
        
        if total_weight > 0:
            final_pose = weighted_pose / total_weight
            final_confidence = total_weight / len(predictions)
            
            # Force z to zero if required
            if self.force_z_zero:
                final_pose[2] = 0.0
                
            return final_pose, final_confidence
        
        return None, 0.0

    def compute_prediction_accuracy(self, predicted_pose: np.ndarray, observed_pose: np.ndarray) -> float:
        """
        Compute accuracy of prediction compared to observation
        
        Args:
            predicted_pose: Predicted pose [x, y, z, yaw]
            observed_pose: Observed pose [x, y, z, yaw]
            
        Returns:
            Accuracy score [0, 1] where 1 is perfect prediction
        """
        if predicted_pose is None or observed_pose is None:
            return 0.0
        
        # Position error (consider only x,y if force_z_zero)
        if self.force_z_zero:
            position_error = np.linalg.norm(predicted_pose[:2] - observed_pose[:2])
        else:
            position_error = np.linalg.norm(predicted_pose[:3] - observed_pose[:3])
        
        # Yaw error (handle angle wrapping)
        yaw_error = abs(predicted_pose[3] - observed_pose[3])
        if yaw_error > np.pi:
            yaw_error = 2 * np.pi - yaw_error
        
        # Normalize errors and compute accuracy
        # Assume 5m position error and 180° yaw error represent 0 accuracy
        position_accuracy = max(0, 1 - position_error / 5.0)
        yaw_accuracy = max(0, 1 - yaw_error / np.pi)
        
        # Weighted combination (position is typically more important)
        overall_accuracy = 0.7 * position_accuracy + 0.3 * yaw_accuracy
        
        return overall_accuracy

    def estimate_observation_reliability(self, observed_pose: np.ndarray, registration_confidence: float, scan_features: Dict) -> float:
        """
        Estimate reliability of the observation based on multiple factors
        
        Args:
            observed_pose: Observed pose from registration
            registration_confidence: Confidence from GICP registration
            scan_features: Features of the current scan
            
        Returns:
            Reliability score [0, 1]
        """
        reliability_factors = []
        
        # 1. Registration confidence
        reliability_factors.append(registration_confidence)
        
        # 2. Feature quality (more features = more reliable)
        feature_quality = 0.5  # Base quality
        if scan_features:
            if 'fpfh_histogram' in scan_features:
                feature_quality += 0.2
            if 'dominant_plane' in scan_features:
                feature_quality += 0.1
            if scan_features.get('point_count', 0) > 1000:
                feature_quality += 0.1
            if 'local_density' in scan_features:
                feature_quality += 0.1
        
        reliability_factors.append(min(1.0, feature_quality))
        
        # 3. Consistency with recent observations
        consistency_score = 0.5  # Default
        if len(self.scan_states) >= 2:
            recent_poses = [state.pose for state in self.scan_states[-2:]]
            
            # Check if current observation is consistent with recent motion
            if len(recent_poses) >= 2:
                prev_movement = recent_poses[-1] - recent_poses[-2]
                if len(self.scan_states) >= 1:
                    current_movement = observed_pose - self.scan_states[-1].pose
                    
                    # Movement consistency (considering only x,y if force_z_zero)
                    if self.force_z_zero:
                        movement_similarity = 1.0 - min(1.0, np.linalg.norm(current_movement[:2] - prev_movement[:2]) / 2.0)
                    else:
                        movement_similarity = 1.0 - min(1.0, np.linalg.norm(current_movement[:3] - prev_movement[:3]) / 2.0)
                    
                    consistency_score = movement_similarity
        
        reliability_factors.append(consistency_score)
        
        # 4. Temporal stability (observations close in time should be more reliable)
        temporal_score = 1.0  # Assume all observations are temporally close for now
        reliability_factors.append(temporal_score)
        
        # Weighted combination of reliability factors
        weights = [0.4, 0.3, 0.2, 0.1]  # Registration, features, consistency, temporal
        weighted_reliability = sum(w * f for w, f in zip(weights, reliability_factors))
        
        return min(1.0, max(0.1, weighted_reliability))

    def compute_fusion_weights(self, predicted_pose: np.ndarray, prediction_confidence: float,
                             observed_pose: np.ndarray, observation_reliability: float) -> Tuple[float, float]:
        """
        Compute optimal fusion weights for prediction and observation
        
        Args:
            predicted_pose: Predicted pose
            prediction_confidence: Confidence in prediction
            observed_pose: Observed pose
            observation_reliability: Reliability of observation
            
        Returns:
            (prediction_weight, observation_weight) - weights sum to 1.0
        """
        if self.prediction_observation_fusion == 'weighted':
            # Simple weighted average based on confidence/reliability
            total_confidence = prediction_confidence + observation_reliability
            if total_confidence > 0:
                pred_weight = prediction_confidence / total_confidence
                obs_weight = observation_reliability / total_confidence
            else:
                pred_weight, obs_weight = 0.5, 0.5
                
        elif self.prediction_observation_fusion == 'confidence_based':
            # Use confidence/reliability with exponential weighting
            pred_exp = np.exp(2 * prediction_confidence)
            obs_exp = np.exp(2 * observation_reliability)
            total_exp = pred_exp + obs_exp
            
            pred_weight = pred_exp / total_exp
            obs_weight = obs_exp / total_exp
            
        else:  # adaptive
            # Adaptive weighting based on historical performance
            if len(self.prediction_accuracy_history) >= 3:
                recent_pred_accuracy = np.mean(self.prediction_accuracy_history[-3:])
                recent_obs_reliability = np.mean(self.observation_reliability_history[-3:])
                
                # Adjust weights based on recent performance
                pred_weight = 0.3 + 0.4 * recent_pred_accuracy
                obs_weight = 0.3 + 0.4 * recent_obs_reliability
                
                # Normalize
                total_weight = pred_weight + obs_weight
                pred_weight /= total_weight
                obs_weight /= total_weight
            else:
                # Default balanced weighting for early scans
                pred_weight = 0.4 + 0.1 * prediction_confidence
                obs_weight = 0.4 + 0.1 * observation_reliability
                
                # Normalize
                total_weight = pred_weight + obs_weight
                pred_weight /= total_weight
                obs_weight /= total_weight
        
        return pred_weight, obs_weight

    def fuse_prediction_observation(self, predicted_pose: np.ndarray, prediction_confidence: float,
                                  observed_pose: np.ndarray, observation_reliability: float) -> Tuple[np.ndarray, float]:
        """
        Fuse predicted pose and observed pose to get final state estimate
        
        Args:
            predicted_pose: Predicted pose [x, y, z, yaw]
            prediction_confidence: Confidence in prediction
            observed_pose: Observed pose [x, y, z, yaw]
            observation_reliability: Reliability of observation
            
        Returns:
            (fused_pose, fused_confidence)
        """
        if predicted_pose is None and observed_pose is None:
            return None, 0.0
        elif predicted_pose is None:
            return observed_pose.copy(), observation_reliability
        elif observed_pose is None:
            return predicted_pose.copy(), prediction_confidence
        
        # Compute fusion weights
        pred_weight, obs_weight = self.compute_fusion_weights(
            predicted_pose, prediction_confidence, observed_pose, observation_reliability
        )
        
        # Store weights for analysis
        self.fusion_weights_history.append((pred_weight, obs_weight))
        if len(self.fusion_weights_history) > 20:
            self.fusion_weights_history.pop(0)
        
        # Fuse poses with special handling for yaw angle
        fused_pose = np.zeros(4)
        
        # Position components (x, y, z)
        for i in range(3):
            fused_pose[i] = pred_weight * predicted_pose[i] + obs_weight * observed_pose[i]
        
        # Yaw angle (handle circular nature)
        # Convert to complex numbers for proper averaging
        pred_complex = np.exp(1j * predicted_pose[3])
        obs_complex = np.exp(1j * observed_pose[3])
        fused_complex = pred_weight * pred_complex + obs_weight * obs_complex
        fused_pose[3] = np.angle(fused_complex)
        
        # Force z to zero if required
        if self.force_z_zero:
            fused_pose[2] = 0.0
        
        # Compute fused confidence
        fused_confidence = pred_weight * prediction_confidence + obs_weight * observation_reliability
        
        return fused_pose, fused_confidence
    
    def update_with_observation(self, 
                              observed_pose: np.ndarray, 
                              scan_features: Dict,
                              registration_confidence: float = 1.0):
        """
        Update state with new observation
        
        Args:
            observed_pose: Observed pose from registration
            scan_features: Features of the scan
            registration_confidence: Confidence in the registration
        """
        # Create new scan state
        # Simple uncertainty model - could be made more sophisticated
        base_uncertainty = 0.1
        uncertainty_matrix = np.eye(4) * (base_uncertainty / registration_confidence) ** 2
        
        new_state = ScanState(
            pose=observed_pose.copy(),
            uncertainty=uncertainty_matrix,
            confidence=registration_confidence,
            scan_features=scan_features
        )
        
        # Add to history
        self.scan_states.append(new_state)
        self.feature_database.append(scan_features)
        
        # Keep limited history
        max_history = 20
        if len(self.scan_states) > max_history:
            self.scan_states.pop(0)
            self.feature_database.pop(0)
        
        # Analyze motion patterns
        self._analyze_motion_patterns()