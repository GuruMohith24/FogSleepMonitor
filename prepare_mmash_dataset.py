"""
MMASH Dataset Preprocessor for FogSleepMonitor
================================================
Converts the raw MMASH (PhysioNet) multi-subject data into a unified CSV
that matches our LSTM training pipeline's expected format.

Source: Multilevel Monitoring of Activity and Sleep in Healthy People (MMASH)
        https://physionet.org/content/mmash/1.0.0/

Output columns: timestamp, acc_x, acc_y, acc_z, heart_rate, hrv, sleep_duration,
                movement_level, sleep_quality, sleep_score
"""

import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MMASH_Preprocessor")

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MMASH_DIR = os.path.join(SCRIPT_DIR, "data", "MMASH", "DataPaper")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "data", "sleep_sensor_dataset.csv")


def load_sleep_info(user_dir):
    """
    Load sleep quality data for a user.
    Returns sleep_duration (hours) and sleep_score (0-100).
    """
    sleep_path = os.path.join(user_dir, "sleep.csv")
    if not os.path.exists(sleep_path):
        return None, None

    sleep_df = pd.read_csv(sleep_path)
    if len(sleep_df) == 0:
        return None, None

    # Use the first night's sleep data
    row = sleep_df.iloc[0]

    # Total Sleep Time in minutes -> hours
    tst_minutes = float(row.get("Total Sleep Time (TST)", 0))
    sleep_duration = tst_minutes / 60.0

    # Sleep Efficiency (%) directly maps to quality
    efficiency = float(row.get("Efficiency", 0))

    # WASO (Wake After Sleep Onset) - higher = worse
    waso = float(row.get("Wake After Sleep Onset (WASO)", 0))

    # Number of Awakenings
    awakenings = float(row.get("Number of Awakenings", 0))

    # Movement Index
    movement_idx = float(row.get("Movement Index", 0))

    # Sleep Fragmentation Index
    frag_idx = float(row.get("Sleep Fragmentation Index", 0))

    # Compute a composite sleep score (0-100):
    # Weighted formula based on clinical sleep metrics
    score = efficiency * 0.40                          # Efficiency is primary
    score += max(0, (1 - waso / tst_minutes)) * 30     # Penalty for WASO relative to TST
    score += max(0, (1 - awakenings / 20)) * 15        # Penalty for frequent awakenings
    score += max(0, (1 - frag_idx / 50)) * 15          # Penalty for high fragmentation
    score = np.clip(score, 0, 100)

    return sleep_duration, score


def compute_hrv_from_rr(user_dir, target_times=None):
    """
    Compute HRV (RMSSD-based) from RR interval data.
    Returns a dict mapping time-string -> HRV value.
    """
    rr_path = os.path.join(user_dir, "RR.csv")
    if not os.path.exists(rr_path):
        return {}

    rr_df = pd.read_csv(rr_path)
    if len(rr_df) == 0:
        return {}

    # Filter out artifact beats (IBI > 2s or < 0.3s are likely noise)
    rr_df = rr_df[(rr_df['ibi_s'] >= 0.3) & (rr_df['ibi_s'] <= 2.0)].copy()

    # Group by time (second-level) and compute RMSSD per time window
    hrv_dict = {}

    if 'time' in rr_df.columns:
        # Group by minute to get HRV estimates
        rr_df['minute'] = rr_df['time'].str[:5]  # HH:MM

        for minute, group in rr_df.groupby('minute'):
            ibis = group['ibi_s'].values
            if len(ibis) >= 3:
                diffs = np.diff(ibis)
                rmssd = np.sqrt(np.mean(diffs ** 2)) * 1000  # Convert to ms
                hrv_dict[minute] = np.clip(rmssd, 5, 200)

    return hrv_dict


def process_user(user_dir, user_id):
    """
    Process a single user's data into the unified format.
    """
    actigraph_path = os.path.join(user_dir, "Actigraph.csv")
    if not os.path.exists(actigraph_path):
        logger.warning(f"  Skipping {user_id}: No Actigraph.csv")
        return None

    # Load actigraph data
    acti_df = pd.read_csv(actigraph_path)

    # Check required columns
    required_cols = ['Axis1', 'Axis2', 'Axis3', 'HR', 'time']
    for col in required_cols:
        if col not in acti_df.columns:
            logger.warning(f"  Skipping {user_id}: missing column '{col}'")
            return None

    # Remove rows with zero/NaN heart rate (sensor disconnected)
    acti_df = acti_df[acti_df['HR'] > 0].copy()
    if len(acti_df) == 0:
        logger.warning(f"  Skipping {user_id}: no valid HR data")
        return None

    # Load sleep quality info
    sleep_duration, sleep_score = load_sleep_info(user_dir)
    if sleep_duration is None:
        logger.warning(f"  Skipping {user_id}: no sleep data")
        return None

    # Compute HRV from RR intervals
    hrv_dict = compute_hrv_from_rr(user_dir)

    # --- Normalize accelerometer data ---
    # MMASH Actigraph axes are in activity counts (0-1000+)
    # Our Arduino outputs in g-force units (~-2 to +2)
    # Normalize: divide by 1000 to get approximate g-force scale
    acti_df['acc_x'] = acti_df['Axis1'] / 1000.0
    acti_df['acc_y'] = acti_df['Axis2'] / 1000.0
    acti_df['acc_z'] = acti_df['Axis3'] / 1000.0

    # Heart rate is already in BPM
    acti_df['heart_rate'] = acti_df['HR'].astype(float)

    # Map HRV values by minute
    acti_df['minute'] = acti_df['time'].str[:5]
    acti_df['hrv'] = acti_df['minute'].map(hrv_dict)
    # Fill missing HRV with reasonable default (40ms RMSSD is normal resting)
    acti_df['hrv'] = acti_df['hrv'].fillna(40.0)

    # Sleep duration is constant per subject
    acti_df['sleep_duration'] = sleep_duration

    # Movement level classification
    movement_mag = np.sqrt(acti_df['acc_x']**2 + acti_df['acc_y']**2 + acti_df['acc_z']**2)
    acti_df['movement_level'] = ['high' if m > 0.15 else 'low' for m in movement_mag]

    # --- Per-second sleep score ---
    # Base it on the subject's overall sleep quality, but modulate by instantaneous activity
    # High movement and elevated HR at a given second → worse score for that moment
    base_score = sleep_score
    hr_penalty = np.abs(acti_df['heart_rate'] - 65) * 0.3
    movement_penalty = movement_mag * 50
    instant_score = base_score - hr_penalty - movement_penalty
    acti_df['sleep_score'] = np.clip(instant_score, 0, 100)

    # Sleep quality label
    acti_df['sleep_quality'] = ['good' if s >= 70 else 'poor' for s in acti_df['sleep_score']]

    # Use the original time as timestamp
    acti_df['timestamp'] = acti_df['time']

    # Select output columns
    output_cols = [
        'timestamp', 'acc_x', 'acc_y', 'acc_z', 'heart_rate',
        'hrv', 'sleep_duration', 'movement_level', 'sleep_quality', 'sleep_score'
    ]

    result = acti_df[output_cols].copy()
    logger.info(f"  {user_id}: {len(result)} rows, sleep_duration={sleep_duration:.1f}h, "
                f"base_score={sleep_score:.1f}")

    return result


def main():
    logger.info("=" * 60)
    logger.info("MMASH Dataset Preprocessor for FogSleepMonitor")
    logger.info("=" * 60)

    if not os.path.exists(MMASH_DIR):
        logger.error(f"MMASH data not found at: {MMASH_DIR}")
        logger.error("Please download from: https://physionet.org/content/mmash/1.0.0/")
        return

    # Get all user directories
    user_dirs = sorted([
        d for d in os.listdir(MMASH_DIR)
        if os.path.isdir(os.path.join(MMASH_DIR, d)) and d.startswith("user_")
    ])

    logger.info(f"Found {len(user_dirs)} subjects")

    all_data = []
    for user_id in user_dirs:
        user_path = os.path.join(MMASH_DIR, user_id)
        logger.info(f"Processing {user_id}...")
        result = process_user(user_path, user_id)
        if result is not None:
            all_data.append(result)

    if not all_data:
        logger.error("No data was processed!")
        return

    # Combine all subjects
    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"\nTotal combined rows: {len(combined)}")

    # Summary statistics
    logger.info(f"Heart Rate  — mean: {combined['heart_rate'].mean():.1f}, "
                f"std: {combined['heart_rate'].std():.1f}")
    logger.info(f"Sleep Score — mean: {combined['sleep_score'].mean():.1f}, "
                f"std: {combined['sleep_score'].std():.1f}")
    logger.info(f"HRV         — mean: {combined['hrv'].mean():.1f}, "
                f"std: {combined['hrv'].std():.1f}")

    quality_counts = combined['sleep_quality'].value_counts()
    logger.info(f"Quality distribution: {quality_counts.to_dict()}")

    # Save
    combined.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"\n✅ Dataset saved to: {OUTPUT_PATH}")
    logger.info(f"   Shape: {combined.shape}")
    logger.info(f"   Size: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")

    # Show sample
    print("\n--- Sample Data (first 5 rows) ---")
    print(combined.head().to_string(index=False))


if __name__ == "__main__":
    main()
