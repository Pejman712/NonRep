#!/usr/bin/env python3
"""
Complete Example: Non-Repetitive LiDAR Processing with Prediction-Observation Fusion

This example demonstrates how to use the enhanced LiDAR processor that fuses
prediction and observation poses without velocity assumptions.

Features demonstrated:
1. Basic usage with different fusion methods
2. Custom configuration for different scenarios
3. Performance analysis and visualization
4. Real-time adaptation monitoring
5. Error analysis and diagnostics
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import time

# Import the modified processor (assuming it's in the same directory)
from modified_lidar_processor import NonRepetitiveLiDARProcessor
from kalman2 import process_non_repetitive_lidar_scans
from kalman2 import main_non_repetitive_lidar

def example_1_basic_usage():
    """
    Example 1: Basic usage with default adaptive fusion
    """
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage with Adaptive Fusion")
    print("=" * 60)
    
    # Configure your data path
    observation_folder = "/home/robotics/testdata/BIA6"  # Replace with your actual path
    
    try:
        # Basic usage with adaptive fusion (recommended for most cases)
        results, pred_poses, obs_poses, final_poses = process_non_repetitive_lidar_scans (
            observation_folder=observation_folder,
            visualize=True,
            observation_step_size=3,  # Process every 3rd scan
            observation_start_index=0,
            max_observation_clouds=100,
            force_z_zero=True,  # Force 2D processing
            z_redistribution_method='prediction',
            fusion_method='adaptive'  # Let system adapt automatically
        )
        
        print(f"Processed {len(results)} scans successfully")
        return results, pred_poses, obs_poses, final_poses
        
    except Exception as e:
        print(f"Error in basic example: {e}")
        print("Note: Make sure to update 'observation_folder' path")
        return None, None, None, None

def example_2_custom_processor():
    """
    Example 2: Using the processor directly with custom configuration
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Custom Processor Configuration")
    print("=" * 60)
    
    # Initialize processor with custom settings
    processor = NonRepetitiveLiDARProcessor(
        adaptive_threshold=0.85,
        feature_weight=0.4,
        geometric_weight=0.3,
        temporal_weight=0.3,
        force_z_zero=True,
        z_redistribution_method='prediction',
        prediction_observation_fusion='confidence_based'  # Start with confidence-based
    )
    
    # Simulate processing loop (replace with your actual data loading)
    print("Simulating scan processing...")
    
    # Example: Create synthetic point clouds for demonstration
    synthetic_results = []
    
    for i in range(10):
        print(f"\nProcessing synthetic scan {i+1}/10")
        
        # Create a synthetic point cloud (replace with actual cloud loading)
        cloud = create_synthetic_cloud(i)
        
        # Extract features
        features = processor.extract_scan_features(cloud)
        print(f"Extracted {len(features)} feature types")
        
        # Get predictionprocess_non_repetitive_lidar_scans
        predicted_pose, pred_confidence = processor.predict_pose_adaptive(features)
        
        if predicted_pose is not None:
            print(f"Prediction: {predicted_pose} (confidence: {pred_confidence:.3f})")
        else:
            print("No prediction available")
        
        # Simulate observation (replace with actual GICP registration)
        observed_pose = simulate_observation(i)
        registration_confidence = 0.7 + 0.3 * np.random.random()
        
        print(f"Observation: {observed_pose} (confidence: {registration_confidence:.3f})")
        
        # Update processor with fusion
        processor.update_with_observation(
            observed_pose=observed_pose,
            scan_features=features,
            registration_confidence=registration_confidence,
            predicted_pose=predicted_pose,
            prediction_confidence=pred_confidence if predicted_pose is not None else 0.0
        )
        
        # Get final state
        current_state = processor.get_current_state()
        if current_state:
            print(f"Final fused pose: {current_state.pose}")
            print(f"Fusion weight (prediction): {current_state.fusion_weight:.3f}")
            
            synthetic_results.append({
                'scan_id': i,
                'predicted_pose': predicted_pose.copy() if predicted_pose is not None else None,
                'observed_pose': observed_pose.copy(),
                'final_pose': current_state.pose.copy(),
                'prediction_confidence': pred_confidence if predicted_pose is not None else 0.0,
                'observation_confidence': registration_confidence,
                'fusion_weight': current_state.fusion_weight
            })
    
    # Analyze results
    analyze_fusion_performance(processor, synthetic_results)
    
    return processor, synthetic_results

def example_3_fusion_method_comparison():
    """
    Example 3: Compare different fusion methods
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Fusion Method Comparison")
    print("=" * 60)
    
    fusion_methods = ['adaptive', 'weighted', 'confidence_based']
    results_comparison = {}
    
    for method in fusion_methods:
        print(f"\nTesting fusion method: {method}")
        
        processor = NonRepetitiveLiDARProcessor(
            force_z_zero=True,
            prediction_observation_fusion=method
        )
        
        method_results = []
        
        # Process same synthetic data with different fusion methods
        for i in range(15):
            cloud = create_synthetic_cloud(i)
            features = processor.extract_scan_features(cloud)
            
            predicted_pose, pred_conf = processor.predict_pose_adaptive(features)
            observed_pose = simulate_observation(i)
            obs_conf = 0.6 + 0.4 * np.random.random()
            
            processor.update_with_observation(
                observed_pose, features, obs_conf, predicted_pose, pred_conf or 0.0
            )
            
            state = processor.get_current_state()
            if state:
                method_results.append({
                    'final_pose': state.pose.copy(),
                    'fusion_weight': state.fusion_weight,
                    'confidence': state.confidence
                })
        
        results_comparison[method] = method_results
        
        # Get fusion analysis
        analysis = processor.get_fusion_analysis()
        print(f"Method: {method}")
        if 'prediction_accuracy' in analysis:
            print(f"  Prediction accuracy: {analysis['prediction_accuracy']['mean']:.3f}")
        if 'fusion_weights' in analysis:
            weights = analysis['fusion_weights']
            print(f"  Avg weights - Pred: {weights['avg_prediction_weight']:.3f}, Obs: {weights['avg_observation_weight']:.3f}")
            print(f"  Weight stability: {weights['weight_stability']:.3f}")
    
    # Plot comparison
    plot_fusion_comparison(results_comparison)
    
    return results_comparison

def example_4_real_time_monitoring():
    """
    Example 4: Real-time fusion performance monitoring
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Real-Time Fusion Performance Monitoring")
    print("=" * 60)
    
    processor = NonRepetitiveLiDARProcessor(
        force_z_zero=True,
        prediction_observation_fusion='adaptive'
    )
    
    # Setup real-time monitoring
    monitoring_data = {
        'scan_ids': [],
        'prediction_accuracy': [],
        'observation_reliability': [],
        'fusion_weights_pred': [],
        'fusion_weights_obs': [],
        'fusion_method': []
    }
    
    print("Starting real-time monitoring simulation...")
    
    for i in range(25):
        # Simulate varying conditions
        if i < 10:
            # Good predictions, poor observations
            pred_noise = 0.1
            obs_noise = 0.5
        elif i < 20:
            # Poor predictions, good observations  
            pred_noise = 0.5
            obs_noise = 0.1
        else:
            # Balanced conditions
            pred_noise = 0.2
            obs_noise = 0.2
        
        cloud = create_synthetic_cloud(i)
        features = processor.extract_scan_features(cloud)
        
        predicted_pose, pred_conf = processor.predict_pose_adaptive(features)
        observed_pose = simulate_observation(i, noise_level=obs_noise)
        obs_conf = max(0.3, 1.0 - obs_noise)
        
        # Add noise to prediction
        if predicted_pose is not None:
            predicted_pose += np.random.normal(0, pred_noise, 4)
            pred_conf *= max(0.3, 1.0 - pred_noise)
        
        processor.update_with_observation(
            observed_pose, features, obs_conf, predicted_pose, pred_conf or 0.0
        )
        
        # Monitor performance
        analysis = processor.get_motion_analysis()
        
        monitoring_data['scan_ids'].append(i)
        
        if 'prediction_accuracy' in analysis:
            monitoring_data['prediction_accuracy'].append(
                analysis['prediction_accuracy']['recent']
            )
        else:
            monitoring_data['prediction_accuracy'].append(0)
        
        if 'observation_reliability' in analysis:
            monitoring_data['observation_reliability'].append(
                analysis['observation_reliability']['recent']
            )
        else:
            monitoring_data['observation_reliability'].append(0)
        
        if 'fusion_weights' in analysis:
            weights = analysis['fusion_weights']
            monitoring_data['fusion_weights_pred'].append(weights['avg_prediction_weight'])
            monitoring_data['fusion_weights_obs'].append(weights['avg_observation_weight'])
        else:
            monitoring_data['fusion_weights_pred'].append(0.5)
            monitoring_data['fusion_weights_obs'].append(0.5)
        
        monitoring_data['fusion_method'].append(analysis.get('fusion_method', 'adaptive'))
        
        # Print periodic updates
        if (i + 1) % 5 == 0:
            current_method = analysis.get('fusion_method', 'adaptive')
            print(f"Scan {i+1}: Method={current_method}, "
                  f"PredAcc={monitoring_data['prediction_accuracy'][-1]:.3f}, "
                  f"ObsRel={monitoring_data['observation_reliability'][-1]:.3f}")
    
    # Plot monitoring results
    plot_real_time_monitoring(monitoring_data)
    
    return monitoring_data

def example_5_error_analysis():
    """
    Example 5: Detailed error analysis and diagnostics
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Error Analysis and Diagnostics")
    print("=" * 60)
    
    processor = NonRepetitiveLiDARProcessor(
        force_z_zero=True,
        prediction_observation_fusion='adaptive'
    )
    
    # Generate data with known ground truth
    ground_truth_poses = []
    processing_results = []
    
    print("Generating data with known ground truth...")
    
    for i in range(30):
        # Create ground truth trajectory (circular motion)
        t = i * 0.2
        gt_pose = np.array([
            3 * np.cos(t),      # x
            3 * np.sin(t),      # y  
            0.0,                # z (forced to 0)
            t                   # yaw
        ])
        ground_truth_poses.append(gt_pose)
        
        # Generate noisy observations
        cloud = create_synthetic_cloud(i)
        features = processor.extract_scan_features(cloud)
        
        predicted_pose, pred_conf = processor.predict_pose_adaptive(features)
        
        # Add realistic noise to observation
        obs_noise = np.array([0.1, 0.1, 0.0, 0.05])  # x, y, z, yaw noise
        observed_pose = gt_pose + np.random.normal(0, obs_noise)
        observed_pose[2] = 0.0  # Force z=0
        obs_conf = 0.8
        
        processor.update_with_observation(
            observed_pose, features, obs_conf, predicted_pose, pred_conf or 0.0
        )
        
        state = processor.get_current_state()
        if state:
            processing_results.append({
                'scan_id': i,
                'ground_truth': gt_pose.copy(),
                'predicted': predicted_pose.copy() if predicted_pose is not None else None,
                'observed': observed_pose.copy(),
                'final': state.pose.copy(),
                'pred_error': None if predicted_pose is None else np.linalg.norm(predicted_pose[:2] - gt_pose[:2]),
                'obs_error': np.linalg.norm(observed_pose[:2] - gt_pose[:2]),
                'final_error': np.linalg.norm(state.pose[:2] - gt_pose[:2])
            })
    
    # Analyze errors
    analyze_errors(processing_results, processor)
    
    return processing_results

# Helper functions for examples

def create_synthetic_cloud(scan_id):
    """Create a synthetic point cloud for testing"""
    np.random.seed(scan_id)  # Reproducible results
    
    # Create a simple geometric structure
    points = []
    
    # Add some planar surfaces
    for _ in range(200):
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        z = np.random.uniform(0, 2)
        points.append([x, y, z])
    
    # Add some linear features
    for i in range(50):
        t = i / 50.0
        points.append([t * 10 - 5, 0, 1])
        points.append([0, t * 10 - 5, 0.5])
    
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.array(points))
    
    return cloud

def simulate_observation(scan_id, noise_level=0.2):
    """Simulate a GICP observation with realistic noise"""
    # Simple forward motion with some rotation
    base_pose = np.array([
        scan_id * 0.5,           # x: moving forward
        0.1 * np.sin(scan_id * 0.3),  # y: slight sinusoidal motion
        0.0,                     # z: always 0
        scan_id * 0.05           # yaw: gradual rotation
    ])
    
    # Add noise
    noise = np.random.normal(0, noise_level, 4)
    noise[2] = 0.0  # No z noise since z is forced to 0
    
    return base_pose + noise

def analyze_fusion_performance(processor, results):
    """Analyze the fusion performance"""
    print("\n--- Fusion Performance Analysis ---")
    
    # Get detailed analysis
    analysis = processor.get_motion_analysis()
    
    if 'prediction_accuracy' in analysis:
        pred_acc = analysis['prediction_accuracy']
        print(f"Prediction Accuracy:")
        print(f"  Mean: {pred_acc['mean']:.3f}")
        print(f"  Std: {pred_acc['std']:.3f}")
        print(f"  Recent: {pred_acc['recent']:.3f}")
        print(f"  Trend: {pred_acc['trend']}")
    
    if 'observation_reliability' in analysis:
        obs_rel = analysis['observation_reliability']
        print(f"Observation Reliability:")
        print(f"  Mean: {obs_rel['mean']:.3f}")
        print(f"  Recent: {obs_rel['recent']:.3f}")
    
    if 'fusion_weights' in analysis:
        weights = analysis['fusion_weights']
        print(f"Fusion Weights:")
        print(f"  Avg Prediction Weight: {weights['avg_prediction_weight']:.3f}")
        print(f"  Avg Observation Weight: {weights['avg_observation_weight']:.3f}")
        print(f"  Weight Stability: {weights['weight_stability']:.3f}")
        print(f"  Current Method: {weights['current_fusion_method']}")
    
    if 'prediction_errors' in analysis:
        errors = analysis['prediction_errors']
        print(f"Prediction Errors:")
        print(f"  Position RMSE: {errors['position_rmse']:.3f} m")
        print(f"  Yaw RMSE: {np.degrees(errors['yaw_rmse']):.1f}°")
        print(f"  Position Bias: ({errors['bias_x']:.3f}, {errors['bias_y']:.3f}) m")

def plot_fusion_comparison(results_comparison):
    """Plot comparison of different fusion methods"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Fusion Method Comparison', fontsize=16)
    
    for method, results in results_comparison.items():
        if not results:
            continue
            
        poses = np.array([r['final_pose'] for r in results])
        weights = [r['fusion_weight'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        # Trajectory plot
        axes[0, 0].plot(poses[:, 0], poses[:, 1], 'o-', label=method, alpha=0.7)
        
        # Fusion weights over time
        axes[0, 1].plot(weights, 'o-', label=method, alpha=0.7)
        
        # Confidence over time
        axes[1, 0].plot(confidences, 'o-', label=method, alpha=0.7)
        
        # Position vs confidence scatter
        position_norms = np.linalg.norm(poses[:, :2], axis=1)
        axes[1, 1].scatter(position_norms, confidences, label=method, alpha=0.7)
    
    axes[0, 0].set_title('Trajectories')
    axes[0, 0].set_xlabel('X (m)')
    axes[0, 0].set_ylabel('Y (m)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].set_title('Fusion Weights (Prediction)')
    axes[0, 1].set_xlabel('Scan Index')
    axes[0, 1].set_ylabel('Prediction Weight')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].set_title('Confidence Over Time')
    axes[1, 0].set_xlabel('Scan Index')
    axes[1, 0].set_ylabel('Confidence')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].set_title('Position vs Confidence')
    axes[1, 1].set_xlabel('Distance from Origin (m)')
    axes[1, 1].set_ylabel('Confidence')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_real_time_monitoring(monitoring_data):
    """Plot real-time monitoring results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Real-Time Fusion Performance Monitoring', fontsize=16)
    
    scan_ids = monitoring_data['scan_ids']
    
    # Prediction accuracy over time
    axes[0, 0].plot(scan_ids, monitoring_data['prediction_accuracy'], 'b-', linewidth=2, label='Prediction Accuracy')
    axes[0, 0].plot(scan_ids, monitoring_data['observation_reliability'], 'r-', linewidth=2, label='Observation Reliability')
    axes[0, 0].set_title('Performance Over Time')
    axes[0, 0].set_xlabel('Scan Index')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Fusion weights over time
    axes[0, 1].plot(scan_ids, monitoring_data['fusion_weights_pred'], 'g-', linewidth=2, label='Prediction Weight')
    axes[0, 1].plot(scan_ids, monitoring_data['fusion_weights_obs'], 'orange', linewidth=2, label='Observation Weight')
    axes[0, 1].set_title('Fusion Weights Over Time')
    axes[0, 1].set_xlabel('Scan Index')
    axes[0, 1].set_ylabel('Weight')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Method switching
    methods = monitoring_data['fusion_method']
    method_changes = []
    current_method = methods[0]
    for i, method in enumerate(methods):
        if method != current_method:
            method_changes.append(i)
            current_method = method
    
    axes[1, 0].plot(scan_ids, monitoring_data['prediction_accuracy'], 'b-', alpha=0.5)
    for change_point in method_changes:
        axes[1, 0].axvline(x=change_point, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_title('Method Switching Points')
    axes[1, 0].set_xlabel('Scan Index')
    axes[1, 0].set_ylabel('Prediction Accuracy')
    axes[1, 0].grid(True)
    
    # Performance correlation
    axes[1, 1].scatter(monitoring_data['prediction_accuracy'], monitoring_data['observation_reliability'], alpha=0.6)
    axes[1, 1].set_title('Prediction vs Observation Performance')
    axes[1, 1].set_xlabel('Prediction Accuracy')
    axes[1, 1].set_ylabel('Observation Reliability')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def analyze_errors(results, processor):
    """Analyze errors against ground truth"""
    print("\n--- Error Analysis Against Ground Truth ---")
    
    # Extract error data
    pred_errors = [r['pred_error'] for r in results if r['pred_error'] is not None]
    obs_errors = [r['obs_error'] for r in results]
    final_errors = [r['final_error'] for r in results]
    
    print(f"Prediction Errors:")
    if pred_errors:
        print(f"  Mean: {np.mean(pred_errors):.3f} m")
        print(f"  Std: {np.std(pred_errors):.3f} m")
        print(f"  Max: {np.max(pred_errors):.3f} m")
    
    print(f"Observation Errors:")
    print(f"  Mean: {np.mean(obs_errors):.3f} m")
    print(f"  Std: {np.std(obs_errors):.3f} m")
    print(f"  Max: {np.max(obs_errors):.3f} m")
    
    print(f"Final Fused Errors:")
    print(f"  Mean: {np.mean(final_errors):.3f} m")
    print(f"  Std: {np.std(final_errors):.3f} m")
    print(f"  Max: {np.max(final_errors):.3f} m")
    
    # Improvement analysis
    if pred_errors:
        improvements_over_pred = [f - p for f, p in zip(final_errors[:len(pred_errors)], pred_errors)]
        improvements_over_obs = [f - o for f, o in zip(final_errors, obs_errors)]
        
        print(f"Fusion Improvements:")
        print(f"  Vs Prediction: {np.mean(improvements_over_pred):.3f} m (negative = better)")
        print(f"  Vs Observation: {np.mean(improvements_over_obs):.3f} m (negative = better)")
    
    # Plot error comparison
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    scan_ids = [r['scan_id'] for r in results]
    if pred_errors:
        plt.plot(scan_ids[:len(pred_errors)], pred_errors, 'g-', label='Prediction Error')
    plt.plot(scan_ids, obs_errors, 'r-', label='Observation Error')
    plt.plot(scan_ids, final_errors, 'b-', linewidth=2, label='Final Fused Error')
    plt.title('Error Over Time')
    plt.xlabel('Scan Index')
    plt.ylabel('Position Error (m)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.hist([obs_errors, final_errors], bins=15, alpha=0.7, label=['Observation', 'Final Fused'])
    plt.title('Error Distribution')
    plt.xlabel('Position Error (m)')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    ground_truth = np.array([r['ground_truth'] for r in results])
    final_poses = np.array([r['final'] for r in results])
    plt.plot(ground_truth[:, 0], ground_truth[:, 1], 'k-', linewidth=3, label='Ground Truth')
    plt.plot(final_poses[:, 0], final_poses[:, 1], 'b-', linewidth=2, label='Final Fused')
    plt.title('Trajectory Comparison')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    if pred_errors:
        plt.scatter(pred_errors, final_errors, alpha=0.6, label='Pred vs Final')
    plt.scatter(obs_errors, final_errors, alpha=0.6, label='Obs vs Final')
    plt.plot([0, max(max(obs_errors), max(final_errors))], [0, max(max(obs_errors), max(final_errors))], 'k--', alpha=0.5)
    plt.title('Error Comparison')
    plt.xlabel('Input Error (m)')
    plt.ylabel('Final Error (m)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """Run all examples"""
    print("Non-Repetitive LiDAR Fusion Examples")
    print("====================================")
    print("This script demonstrates various usage patterns of the enhanced")
    print("LiDAR processor with prediction-observation fusion.")
    print("\nNote: Update the observation_folder path in example_1_basic_usage()")
    print("to point to your actual LiDAR scan data.")
    
    # Run examples
    try:
        # Example 1: Basic usage (requires real data)
        print("\nSkipping Example 1 (requires real data path)")
        # example_1_basic_usage()
        
        # Example 2: Custom processor with synthetic data
        processor, results = example_2_custom_processor()
        
        # Example 3: Compare fusion methods
        comparison = example_3_fusion_method_comparison()
        
        # Example 4: Real-time monitoring
        monitoring = example_4_real_time_monitoring()
        
        # Example 5: Error analysis
        error_analysis = example_5_error_analysis()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey takeaways:")
        print("1. The fusion system automatically adapts to data quality")
        print("2. Different fusion methods work better for different scenarios")
        print("3. Real-time monitoring helps track system performance")
        print("4. Error analysis validates fusion effectiveness")
        print("5. No velocity assumptions are needed")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()