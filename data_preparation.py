import pandas as pd
import os
import glob
import re

# Define the path to the data directory
data_dir = "Data"
bio_file = os.path.join(data_dir, "bio.csv")
cgm_pattern = os.path.join(data_dir, "CGMacros-*.csv")

print(f"Looking for bio file at: {bio_file}")
print(f"Looking for CGM files matching: {cgm_pattern}")

# --- 1. Load Bio Data ---
try:
    bio_df = pd.read_csv(bio_file)
    print(f"Successfully loaded bio data: {bio_df.shape}")
    # Rename the first column if it's unnamed or incorrect
    if 'subject' not in bio_df.columns:
        bio_df.rename(columns={bio_df.columns[0]: 'subject'}, inplace=True)
    print("Bio data columns:", bio_df.columns.tolist())
except FileNotFoundError:
    print(f"Error: Bio file not found at {bio_file}")
    exit()
except Exception as e:
    print(f"Error loading bio file: {e}")
    exit()

# --- 2. Load and Combine CGM Data ---
cgm_files = glob.glob(cgm_pattern)
if not cgm_files:
    print(f"Error: No CGM files found matching pattern {cgm_pattern}")
    exit()

print(f"Found {len(cgm_files)} CGM files.")

all_cgm_data = []

for cgm_file in cgm_files:
    try:
        # Extract subject ID from filename (e.g., "CGMacros-001.csv" -> 1)
        match = re.search(r"CGMacros-(\d+)\.csv", os.path.basename(cgm_file))
        if match:
            subject_id = int(match.group(1))
        else:
            print(f"Warning: Could not extract subject ID from {cgm_file}. Skipping.")
            continue

        # Load CGM data
        df = pd.read_csv(cgm_file)

        # Add subject ID column
        df['subject'] = subject_id

        # Keep track of original filename
        df['original_file'] = os.path.basename(cgm_file)

        all_cgm_data.append(df)
        # print(f"Loaded {os.path.basename(cgm_file)} for subject {subject_id}, shape: {df.shape}")

    except Exception as e:
        print(f"Error processing {cgm_file}: {e}")

if not all_cgm_data:
    print("Error: No CGM data could be loaded.")
    exit()

# Concatenate all CGM data
combined_cgm_df = pd.concat(all_cgm_data, ignore_index=True)
print(f"Combined CGM data shape: {combined_cgm_df.shape}")
print("Combined CGM columns:", combined_cgm_df.columns.tolist())

# --- 3. Merge Bio and CGM Data ---
# Ensure subject columns are of the same type if necessary
# bio_df['subject'] = bio_df['subject'].astype(int) # Adjust if needed based on bio.csv structure
# combined_cgm_df['subject'] = combined_cgm_df['subject'].astype(int)

try:
    # Using left merge to keep all CGM data and add bio info
    merged_df = pd.merge(combined_cgm_df, bio_df, on='subject', how='left')
    print(f"Merged data shape: {merged_df.shape}")
    print("Merged data columns:", merged_df.columns.tolist())

    # Display info and head
    print("\nMerged DataFrame Info:")
    merged_df.info()

    print("\nMerged DataFrame Head:")
    print(merged_df.head())

    # --- 4. Save Merged Data (Optional but recommended) ---
    output_file = "merged_cgm_bio_data.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"\nSuccessfully merged data saved to {output_file}")

except KeyError as e:
    print(f"Error merging: Column {e} not found. Check 'subject' column name in both dataframes.")
except Exception as e:
    print(f"Error during merge or save: {e}") 