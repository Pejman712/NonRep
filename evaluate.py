import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

def compute_metrics(traj_df, gt_df):
    errors = np.sqrt((traj_df['x'] - gt_df['x'])**2 + (traj_df['y'] - gt_df['y'])**2)
    return {
        'mean': errors.mean(),
        'std': errors.std(),
        'rmse': np.sqrt(np.mean(errors ** 2)),
        'max': errors.max()
    }

def resample(df, length, x_col='x', y_col='y'):
    df = df.dropna(subset=[x_col, y_col])
    return pd.DataFrame({
        'x': np.interp(np.linspace(0, len(df)-1, length), np.arange(len(df)), df[x_col]),
        'y': np.interp(np.linspace(0, len(df)-1, length), np.arange(len(df)), df[y_col])
    })

# Set folder path
folder_path = "./output/eval"
all_csvs = glob.glob(os.path.join(folder_path, "*.csv"))

gt_files = [f for f in all_csvs if f.endswith("gt.csv")]
trajectory_files = [f for f in all_csvs if f not in gt_files]

# Group by prefix
grouped = {}
for gt in gt_files:
    prefix = os.path.basename(gt)[:-6]  # remove 'gt.csv'
    grouped[prefix] = {
        "gt": gt,
        "trajectories": [f for f in trajectory_files if os.path.basename(f).startswith(prefix)]
    }

results = []

for prefix, files in grouped.items():
    print(f"Processing group: {prefix}")
    traj_files = files["trajectories"]
    abcde_files = [f for f in traj_files if os.path.basename(f)[-5] in ['a', 'b', 'c', 'd', 'e']]
    r_file = next((f for f in traj_files if f.endswith("r.csv")), None)
    rr_file = next((f for f in traj_files if f.endswith("rr.csv")), None)

    df_gt = pd.read_csv(files["gt"])
    df_gt = df_gt.dropna(subset=['X', 'Y'])
    if len(df_gt) < 2:
        print(f"Skipping group {prefix}: GT too short.")
        continue

    # Interpolate GT
    length = len(df_gt)
    df_gt_interp = resample(df_gt, length, x_col='X', y_col='Y')
    df_gt_interp.columns = ['x', 'y']

    def prepare_and_eval(traj_file):
        df = pd.read_csv(traj_file)
        df = df.dropna(subset=['x', 'y'])
        if len(df) < 2:
            return None, None
        df_interp = resample(df, len(df_gt_interp))
        return df_interp, compute_metrics(df_interp, df_gt_interp)

    # Average a-e
    avg_df, avg_metrics = None, None
    if abcde_files:
        dfs = []
        for f in abcde_files:
            df = pd.read_csv(f).dropna(subset=['x', 'y'])
            dfs.append(df[['x', 'y']])
        max_len = max(len(df) for df in dfs)
        dfs = [df.reindex(range(max_len)).reset_index(drop=True) for df in dfs]
        concat_df = pd.concat(dfs, axis=1)
        avg_df = pd.DataFrame({
            'x': concat_df.filter(like='x').mean(axis=1),
            'y': concat_df.filter(like='y').mean(axis=1)
        }).dropna()
        if len(avg_df) >= 2:
            avg_df = resample(avg_df, len(df_gt_interp))
            avg_metrics = compute_metrics(avg_df, df_gt_interp)

    # r and rr
    df_r, r_metrics = prepare_and_eval(r_file) if r_file else (None, None)
    df_rr, rr_metrics = prepare_and_eval(rr_file) if rr_file else (None, None)

    # Plot
    plt.figure(figsize=(8, 6))
    if avg_df is not None:
        plt.plot(avg_df['x'], avg_df['y'], label="Our Method", linewidth=2)
    #if df_r is not None:
    #    plt.plot(df_r['x'], df_r['y'], label="r", linestyle='--')
    if df_rr is not None:
        plt.plot(df_rr['x'], df_rr['y'], label="Single scan GICP", linestyle='--')
    if df_gt_interp is not None:
        plt.plot(df_gt_interp['x'], df_gt_interp['y'], label="GT", color='black', linewidth=2)

    plt.title(f"Trajectory Group: {prefix}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, f"{prefix}_trajectory_plot.png"),dpi=900)
    plt.show()

    # Save results
    results.append({
        "group": prefix,
        **{f"avg_{k}": v for k, v in (avg_metrics or {}).items()},
        **{f"r_{k}": v for k, v in (r_metrics or {}).items()},
        **{f"rr_{k}": v for k, v in (rr_metrics or {}).items()},
    })

# Save all metrics to CSV
results_df = pd.DataFrame(results)
output_csv = os.path.join(folder_path, "trajectory_metrics_summary.csv")
results_df.to_csv(output_csv, index=False)
print(f"\nSaved all metrics to {output_csv}")
