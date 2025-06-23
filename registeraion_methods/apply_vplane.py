import numpy as np
import os
import glob
import open3d as o3d
import small_gicp

from numpy.linalg import inv

def load_point_cloud(pcd_path):
    if not os.path.exists(pcd_path):
        raise FileNotFoundError(f"File not found: {pcd_path}")
    with open(pcd_path, 'r') as f:
        lines = f.readlines()

    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('DATA'):
            data_start = i + 1
            break

    data = np.loadtxt(lines[data_start:])
    header = lines[:data_start]
    points = data[:, :3]
    intensity = data[:, 3]
    return points, intensity, header

def save_pcd_with_intensity(filepath, points, intensities):
    assert points.shape[0] == intensities.shape[0], "Mismatch between points and intensities"
    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {points.shape[0]}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {points.shape[0]}
DATA ascii
"""
    with open(filepath, 'w') as f:
        f.write(header)
        for point, intensity in zip(points, intensities):
            f.write(f"{point[0]} {point[1]} {point[2]} {intensity}\n")

def cartesian_to_spherical(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.linalg.norm(points, axis=1)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return r, theta, phi

def spherical_to_cartesian(radius, theta, phi):
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.stack((x, y, z), axis=-1)

def project_to_dome(points):
    r, theta, phi = cartesian_to_spherical(points)
    mask = phi <= (np.pi / 2)
    return r[mask], theta[mask], phi[mask], mask

def apply_gicp(source_points, target_points):
    result = small_gicp.align(np.asarray(source_points, dtype=np.float64),
                              np.asarray(target_points, dtype=np.float64))
    transformation = result.T_target_source
    rotation_only = transformation[:3, :3]
    angle_z_rad = np.arctan2(rotation_only[1, 0], rotation_only[0, 0])
    angle_z_deg = np.degrees(angle_z_rad)
    print(f"Estimated Z-axis rotation (degrees): {angle_z_deg:.2f}")
    rotated_source = (rotation_only @ source_points.T).T
    return rotated_source, transformation

def apply_manual_rotation(source_points, degrees):
    radians = np.deg2rad(degrees)
    rotation_matrix = np.array([
        [np.cos(radians), -np.sin(radians), 0],
        [np.sin(radians),  np.cos(radians), 0],
        [0,               0,                1]
    ])
    return (rotation_matrix @ source_points.T).T

def apply_statistical_outlier_removal(points, nb_neighbors=20, std_ratio=2.0):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return np.asarray(cl.points)

def process_and_align(input_folder, output_folder, use_icp=False, manual_degrees=0):
    os.makedirs(output_folder, exist_ok=True)
    pcd_files = sorted(glob.glob(os.path.join(input_folder, '*.pcd')))

    for i in range(len(pcd_files) - 1):
        source_path = pcd_files[i]
        target_path = pcd_files[i + 1]

        print(f"Processing pair: {source_path} -> {target_path}")

        source_points, source_intensity, _ = load_point_cloud(source_path)
        target_points, _, _ = load_point_cloud(target_path)

        # Use only points below 2 meters for registration
        source_filter = source_points[:, 2] < 2.0
        target_filter = target_points[:, 2] < 2.0

        _, s_theta, s_phi, s_mask = project_to_dome(source_points[source_filter])
        _, t_theta, t_phi, t_mask = project_to_dome(target_points[target_filter])

        source_dome = spherical_to_cartesian(np.ones_like(s_theta), s_theta, s_phi)
        target_dome = spherical_to_cartesian(np.ones_like(t_theta), t_theta, t_phi)

        if use_icp:
            aligned_source, transformation = apply_gicp(source_dome, target_dome)
        else:
            aligned_source = apply_manual_rotation(source_dome, manual_degrees)

        restored_points = spherical_to_cartesian(np.linalg.norm(source_points[source_filter][s_mask], axis=1),
                                                 np.arctan2(aligned_source[:,1], aligned_source[:,0]),
                                                 np.arccos(aligned_source[:,2] / np.linalg.norm(aligned_source, axis=1)))

        filtered_points = apply_statistical_outlier_removal(restored_points)

        indices = np.isin(restored_points[:, 0], filtered_points[:, 0]) & \
                  np.isin(restored_points[:, 1], filtered_points[:, 1]) & \
                  np.isin(restored_points[:, 2], filtered_points[:, 2])
        filtered_intensity = source_intensity[source_filter][s_mask][indices]

        base_name = os.path.splitext(os.path.basename(source_path))[0]
        restored_path = os.path.join(output_folder, f"{base_name}.pcd")
        save_pcd_with_intensity(restored_path, filtered_points, filtered_intensity)
        print(f"Saved aligned point cloud: {restored_path}")

        # Visualize registration
        vis_source = o3d.geometry.PointCloud()
        vis_source.points = o3d.utility.Vector3dVector(source_dome)
        vis_source.paint_uniform_color([1, 0, 0])

        vis_target = o3d.geometry.PointCloud()
        vis_target.points = o3d.utility.Vector3dVector(target_dome)
        vis_target.paint_uniform_color([0, 1, 0])

        vis_aligned = o3d.geometry.PointCloud()
        vis_aligned.points = o3d.utility.Vector3dVector(aligned_source)
        vis_aligned.paint_uniform_color([0, 0, 1])

        print("Visualizing registration result...")
        o3d.visualization.draw_geometries([vis_source, vis_target, vis_aligned],
                                          window_name="Source (Red), Target (Green), Aligned (Blue)")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Align consecutive dome scans using GICP or fixed rotation.")
    parser.add_argument("--folder", type=str, nargs=2, metavar=('INPUT_FOLDER', 'OUTPUT_FOLDER'),
                        help="Input folder and output folder")
    parser.add_argument("--use_icp", action="store_true", help="Use GICP for registration instead of manual rotation")
    parser.add_argument("--rotation_deg", type=float, default=0.0, help="Manual rotation in degrees around Z axis")
    args = parser.parse_args()

    input_folder, output_folder = args.folder
    process_and_align(input_folder, output_folder, use_icp=args.use_icp, manual_degrees=args.rotation_deg)
