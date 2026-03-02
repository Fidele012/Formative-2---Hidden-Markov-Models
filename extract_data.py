import os
import zipfile
import csv
from io import TextIOWrapper

def merge_and_save(accel_data, gyro_data, output_path, prefix):
    common_times = sorted(set(accel_data.keys()) & set(gyro_data.keys()))
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'seconds_elapsed', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'])
        for t in common_times:
            if t in accel_data and t in gyro_data:
                a = accel_data[t]
                g = gyro_data[t]
                writer.writerow([t, a['seconds_elapsed'], a['x'], a['y'], a['z'], g['x'], g['y'], g['z']])
    return len(common_times)

def process_zips(source_dir, output_dir, prefix):
    if not os.path.exists(source_dir): return
    os.makedirs(output_dir, exist_ok=True)
    zip_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.zip')])
    for i, zip_name in enumerate(zip_files, 1):
        if zip_name.startswith('standing-'): continue
        zip_path = os.path.join(source_dir, zip_name)
        with zipfile.ZipFile(zip_path, 'r') as z:
            with z.open('Accelerometer.csv') as af:
                accel_data = {row['time']: row for row in csv.DictReader(TextIOWrapper(af, encoding='utf-8'))}
            with z.open('Gyroscope.csv') as gf:
                gyro_data = {row['time']: row for row in csv.DictReader(TextIOWrapper(gf, encoding='utf-8'))}
            output_path = os.path.join(output_dir, f"{prefix}_{i:03d}.csv")
            rows = merge_and_save(accel_data, gyro_data, output_path, prefix)
            print(f"Processed ZIP {zip_name} -> {output_path} ({rows} rows)")

def process_flat_csvs(source_dir, output_dir, prefix):
    if not os.path.exists(source_dir): return
    os.makedirs(output_dir, exist_ok=True)
    all_files = os.listdir(source_dir)
    indices = sorted(list(set([f.split('_')[-1].split('.')[0] for f in all_files if f.endswith('.csv')])))
    for idx_str in indices:
        try:
            accel_file = [f for f in all_files if 'Accelerometer' in f and f"_{idx_str}.csv" in f][0]
            gyro_file = [f for f in all_files if 'Gyroscope' in f and f"_{idx_str}.csv" in f][0]
            with open(os.path.join(source_dir, accel_file), 'r') as af:
                accel_data = {row['time']: row for row in csv.DictReader(af)}
            with open(os.path.join(source_dir, gyro_file), 'r') as gf:
                gyro_data = {row['time']: row for row in csv.DictReader(gf)}
            output_path = os.path.join(output_dir, f"{prefix}_{int(idx_str):03d}.csv")
            rows = merge_and_save(accel_data, gyro_data, output_path, prefix)
            print(f"Processed Flat CSV {idx_str} -> {output_path} ({rows} rows)")
        except Exception as e:
            print(f"Skipping index {idx_str} due to error: {e}")

if __name__ == "__main__":
    base_repo = "/home/muhirwa/alu/Formative-2---Hidden-Markov-Models"
    
    # Process Jumping (User ZIPs)
    process_zips(os.path.join(base_repo, "Jumping"), os.path.join(base_repo, "dataset", "raw", "jumping"), "jumping")
    # Process Walking (User ZIPs)
    process_zips(os.path.join(base_repo, "Walking"), os.path.join(base_repo, "dataset", "raw", "walking"), "walking")
    # Process Standing (Partner Flat CSVs)
    process_flat_csvs(os.path.join(base_repo, "Standing_Acceleromenter_and_Gyroscope"), os.path.join(base_repo, "dataset", "raw", "standing"), "standing")
    # Process Still (Partner Flat CSVs)
    process_flat_csvs(os.path.join(base_repo, "Still_Accelerometer_and_Gyroscope"), os.path.join(base_repo, "dataset", "raw", "still"), "still")
