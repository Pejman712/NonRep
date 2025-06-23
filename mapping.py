import numpy as np
import open3d as o3d
import copy
import os
from scipy.spatial.transform import Rotation
import small_gicp
from registeraion_methods.gicp import apply_gicp
from registeraion_methods.cpd import apply_cpd
from registeraion_methods.fpfh_icp import apply_fpfh_icp
from registeraion_methods.ndt import apply_ndt
from numpy.linalg import inv

class PCDWithIntensity:
    def __init__(self, points=None, intensity=None):
        self.pcd = o3d.geometry.PointCloud()
        if points is not None:
            self.pcd.points = o3d.utility.Vector3dVector(points)
        self.intensity = intensity
        self.update_colors()

    def update_colors(self):
        if self.pcd is not None and self.intensity is not None and len(self.intensity) > 0:
            min_i = np.min(self.intensity)
            max_i = np.max(self.intensity)
            norm_i = (self.intensity - min_i) / (max_i - min_i) if max_i > min_i else np.zeros_like(self.intensity)
            colors = np.tile(norm_i[:, None], (1, 3))
            self.pcd.colors = o3d.utility.Vector3dVector(colors)

    def transform(self, transformation):
        self.pcd.transform(transformation)
        return self

    def voxel_down_sample(self, voxel_size):
        points = np.asarray(self.pcd.points)
        xyzi = np.column_stack((points, self.intensity))
        downsampled_pcd = self.pcd.voxel_down_sample(voxel_size)
        downsampled_points = np.asarray(downsampled_pcd.points)
        voxel_grid = {}

        for i in range(len(points)):
            voxel_idx = tuple(np.floor(points[i] / voxel_size).astype(int))
            voxel_grid.setdefault(voxel_idx, []).append((points[i], self.intensity[i]))

        downsampled_intensity = np.zeros(len(downsampled_points))
        for i, point in enumerate(downsampled_points):
            voxel_idx = tuple(np.floor(point / voxel_size).astype(int))
            if voxel_idx in voxel_grid:
                intensities = [p[1] for p in voxel_grid[voxel_idx]]
                downsampled_intensity[i] = np.mean(intensities)

        return PCDWithIntensity(downsampled_points, downsampled_intensity)

    def __add__(self, other):
        combined_pcd = self.pcd + other.pcd
        combined_intensity = np.concatenate([self.intensity, other.intensity])
        result = PCDWithIntensity()
        result.pcd = combined_pcd
        result.intensity = combined_intensity
        result.update_colors()
        return result


def read_pcd_with_intensity(filename):
    pcd = o3d.io.read_point_cloud(filename)
    with open(filename, 'r') as f:
        header = []
        for line in f:
            header.append(line.strip())
            if line.startswith('DATA'):
                break
        has_intensity = False
        fields = []
        for line in header:
            if line.startswith('FIELDS'):
                fields = line.split()[1:]
                if 'intensity' in fields:
                    has_intensity = True
                    intensity_idx = fields.index('intensity')
                    break
        if has_intensity:
            print(f"Found intensity field in {filename}")
            data = [list(map(float, line.strip().split())) for line in f]
            if not data:
                return PCDWithIntensity(np.asarray(pcd.points), np.zeros(len(pcd.points)))
            data_np = np.array(data)
            xyz = data_np[:, :3]
            intensity = data_np[:, intensity_idx]
            return PCDWithIntensity(xyz, intensity)
    return PCDWithIntensity(np.asarray(pcd.points), np.zeros(len(pcd.points)))


def rotation_matrix_to_quaternion(rotation_matrix):
    r = Rotation.from_matrix(rotation_matrix)
    return r.as_quat()


def save_pcd_with_intensity(pcd_with_intensity, filename):
    points = np.asarray(pcd_with_intensity.pcd.points)
    intensity = pcd_with_intensity.intensity
    if len(intensity) != len(points):
        if len(intensity) > len(points):
            intensity = intensity[:len(points)]
        else:
            intensity = np.concatenate([intensity, np.zeros(len(points) - len(intensity))])
    with open(filename, 'w') as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z intensity\n")
        f.write("SIZE 4 4 4 4\n")
        f.write("TYPE F F F F\n")
        f.write("COUNT 1 1 1 1\n")
        f.write(f"WIDTH {len(points)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\n")
        f.write("DATA ascii\n")
        for i in range(len(points)):
            f.write(f"{points[i][0]} {points[i][1]} {points[i][2]} {intensity[i]}\n")
    print(f"Saved point cloud with intensity to {filename}")
    return True


def mapping(pcd_files, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    cumulative_transform = np.eye(4)
    trajectory = [cumulative_transform]
    global_map = None

    step = 1
    max_regs = 750
    total_regs = min(max_regs, (len(pcd_files) - step) // step + 1)

    for reg_idx in range(total_regs):
        i = reg_idx * step
        j = i + step
        if j >= len(pcd_files):
            break
        print(f"Processing registration {reg_idx+1}/{total_regs}: clouds {i} and {j}")
        source_cloud = read_pcd_with_intensity(pcd_files[i])
        target_cloud = read_pcd_with_intensity(pcd_files[j])
        transformation = apply_gicp(source_cloud, target_cloud)
        cumulative_transform = cumulative_transform @ transformation
        trajectory.append(copy.deepcopy(cumulative_transform))
        current_cloud = copy.deepcopy(target_cloud)
        current_cloud.transform(cumulative_transform)
        if global_map is None:
            first_cloud = copy.deepcopy(source_cloud)
            first_cloud.transform(trajectory[0])
            global_map = first_cloud
        else:
            global_map = global_map + current_cloud

    # Save trajectory as CSV (X,Y,Z only)
    trajectory_csv = os.path.join(output_dir, "trajectory_xyz.csv")
    positions = np.array([pose[:3, 3] for pose in trajectory])
    np.savetxt(trajectory_csv, positions, delimiter=',', header='x,y,z', comments='', fmt='%.6f')
    print(f"Trajectory saved as CSV: {trajectory_csv}")

    # Downsample and save global map
    for voxel_size in [0.05, 0.1, 0.2]:
        try:
            global_map_down = global_map.voxel_down_sample(voxel_size)
            break
        except RuntimeError as e:
            if "voxel_size is too small" in str(e):
                print(f"Voxel size {voxel_size} too small, trying larger...")
                continue
            else:
                raise
    else:
        print("All voxel sizes failed, using original global map")
        global_map_down = global_map

    map_path = os.path.join(output_dir, "global_map.pcd")
    save_pcd_with_intensity(global_map_down, map_path)

    csv_path = os.path.join(output_dir, "global_map_xyz.csv")
    points = np.asarray(global_map_down.pcd.points)
    np.savetxt(csv_path, points, delimiter=',', header='x,y,z', comments='', fmt='%.6f')
    print(f"Global map saved to CSV: {csv_path}")

    return global_map_down, trajectory


def main():
    pcd_dir = "/home/robotics/testdata/Dumpareaa"  # Change this path as needed
    pcd_files = sorted([os.path.join(pcd_dir, f) for f in os.listdir(pcd_dir) if f.endswith('.pcd')])
    if len(pcd_files) < 2:
        print("Need at least two PCD files for mapping")
        return
    print(f"Found {len(pcd_files)} PCD files")
    global_map, trajectory = mapping(pcd_files)
    o3d.visualization.draw_geometries([global_map.pcd])


if __name__ == "__main__":
    main()
