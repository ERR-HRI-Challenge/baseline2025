import os
import pandas as pd
import numpy as np
from datetime import timedelta
from glob import glob
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Base Path to Data
BASE_PATH = "/data/scao/ERR-HRI-2025/ACM-MM-ERR-HRI-2025-Dataset"

# Window Size and Stride
WINDOW_SIZE = 5 # seconds
STRIDE = 0.5 # seconds

# Data Output File
OUTPUT_FILE = f"rf_data_window_size_{WINDOW_SIZE}_stride_{STRIDE}.csv"

# Naming Patterns for Tasks
SYSTEMS = {
    "voice_assistant": ["medical", "trip", "police"],
    "social_robot": ["survival", "discussion"]
}
# SYSTEMS = {
#     "voice_assistant": ["medical"],
#     "social_robot": ["survival"]
# }

# Naming Patterns for Feature Files
FILE_PATTERNS = {
    "face": "{trial}-{task}-openface-output.csv",
    "audio": "{trial}-{task}-eGeMAPSv02-features.csv",
    "transcript": "transcript-{trial}-{task}_embeddings.csv"
}

# Naming Pattern for Label Files
LABEL_FILE_PATTERNS = {
    "robot_errors": "challenge1_robot_error_labels_{task}_train.csv",
    "human_reactions_ch1": "challenge1_user_reaction_labels_{task}_train.csv",
    "human_reactions_ch2": "challenge2_user_reaction_labels_{task}_train.csv"
}

def fix_timestamp_format(ts):
    if isinstance(ts, str):
        parts = ts.split(':')
        if len(parts) == 4:
            return ':'.join(parts[:3]) + '.' + parts[3]
    return ts

def load_and_preprocess_labels(task):
    """Load all label files for a specific task and preprocess timestamps"""
    labels = {}    
    
    try:
        # Load robot error labels
        robot_errors = pd.read_csv(
            f"{BASE_PATH}/labels_train/challenge1_train/{LABEL_FILE_PATTERNS['robot_errors'].format(task=task)}"
        )
        robot_errors['error_onset'] = robot_errors['error_onset'].apply(fix_timestamp_format)
        robot_errors['error_onset'] = pd.to_timedelta(robot_errors['error_onset'])
        robot_errors['error_offset'] = robot_errors['error_offset'].apply(fix_timestamp_format)
        robot_errors['error_offset'] = pd.to_timedelta(robot_errors['error_offset'])
        robot_errors['trial_num'] = robot_errors['trial_name'].apply(lambda s: s.split('-', 1)[0])
        labels['robot_errors'] = robot_errors        
        
        # Load human reaction labels
        human_reactions_ch1 = pd.read_csv(
            f"{BASE_PATH}/labels_train/challenge1_train/{LABEL_FILE_PATTERNS['human_reactions_ch1'].format(task=task)}"
        )
        human_reactions_ch1['reaction_onset'] = human_reactions_ch1['reaction_onset'].apply(fix_timestamp_format)
        human_reactions_ch1['reaction_onset'] = pd.to_timedelta(human_reactions_ch1['reaction_onset'])
        human_reactions_ch1['reaction_offset'] = human_reactions_ch1['reaction_offset'].apply(fix_timestamp_format)
        human_reactions_ch1['reaction_offset'] = pd.to_timedelta(human_reactions_ch1['reaction_offset'])
        human_reactions_ch1['trial_num'] = human_reactions_ch1['trial_name'].apply(lambda s: s.split('-', 1)[0])
        labels['human_reactions_ch1'] = human_reactions_ch1        
        
        human_reactions_ch2 = pd.read_csv(
            f"{BASE_PATH}/labels_train/challenge2_train/{LABEL_FILE_PATTERNS['human_reactions_ch2'].format(task=task)}"
        )
        human_reactions_ch2['reaction_onset'] = human_reactions_ch2['reaction_onset'].apply(fix_timestamp_format)
        human_reactions_ch2['reaction_onset'] = pd.to_timedelta(human_reactions_ch2['reaction_onset'])
        human_reactions_ch2['reaction_offset'] = human_reactions_ch2['reaction_offset'].apply(fix_timestamp_format)
        human_reactions_ch2['reaction_offset'] = pd.to_timedelta(human_reactions_ch2['reaction_offset'])
        human_reactions_ch2['trial_num'] = human_reactions_ch2['trial_name'].apply(lambda s: s.split('-', 1)[0])
        labels['human_reactions_ch2'] = human_reactions_ch2    
    
    except Exception as e:
        print(f"Error loading label files for task {task}: {str(e)}")
        return None    
    
    return labels, robot_errors, human_reactions_ch1, human_reactions_ch2

def is_error_window(start_td, end_td, trial_num, labels_robot_errors, labels_human_reactions_ch1, labels_human_reactions_ch2):

    """Check if window contains any labels"""
    labels = {
        'robot_error': 0,
        'reaction_ch1': 0,
        'reaction_ch2': 0,
        'reaction_type': None
    }    
    
    # Check robot errors
    trial_errors = labels_robot_errors[labels_robot_errors['trial_num'] == trial_num]
    for _, row in trial_errors.iterrows():
        if (row['error_onset'] <= start_td and row['error_offset'] >= start_td) or (row['error_onset'] <= end_td and row['error_offset'] >= end_td):
            labels['robot_error'] = 1
            break    
            
    # Check human reactions (Challenge 1)
    trial_reactions_ch1 = labels_human_reactions_ch1[labels_human_reactions_ch1['trial_num'] == trial_num]
    for _, row in trial_reactions_ch1.iterrows():
        if (row['reaction_onset'] <= start_td and row['reaction_offset'] >= start_td) or (row['reaction_onset'] <= end_td and row['reaction_offset'] >= end_td):
            labels['reaction_ch1'] = 1
            break    
    
    # Check human reactions (Challenge 2)
    trial_reactions_ch2 = labels_human_reactions_ch2[labels_human_reactions_ch2['trial_num'] == trial_num]
    for _, row in trial_reactions_ch2.iterrows():
        if (row['reaction_onset'] <= start_td and row['reaction_offset'] >= start_td) or (row['reaction_onset'] <= end_td and row['reaction_offset'] >= end_td):
            labels['reaction_ch2'] = 1
            labels['reaction_type'] = row['reaction_type']
            break    
    
    return labels

def extract_windows(face_df, audio_df, transcript_df, trial_name, labels_robot_errors, labels_human_reactions_ch1, labels_human_reactions_ch2, window_size, stride):
    
    """Extract time windows with features and labels, flattening agg_ vectors into columns."""
    # print(f"      Extracting windows for: {trial_name}")
    # print(f"      Face max: {face_df['timestamp'].max()}")
    # print(f"      Audio end max: {audio_df['end'].max()}")
    # print(f"      Transcript end max: {transcript_df['end'].max()}")

    rows = []
    win_delta = timedelta(seconds=window_size)
    str_delta = timedelta(seconds=stride)

    task_duration = max(
        face_df['timestamp'].max(),
        audio_df['end'].max(),
        transcript_df['end'].max()
    )

    start = timedelta(seconds=0)
    while start + win_delta <= task_duration:
        end = start + win_delta

        # slice
        fw = face_df[(face_df['timestamp'] >= start) & (face_df['timestamp'] < end)]
        aw = audio_df[(audio_df['start'] < end) & (audio_df['end'] > start)]
        tw = transcript_df[(transcript_df['start'] < end) & (transcript_df['end'] > start)]

        # aggregate or zero‐pad
        if not fw.empty:
            agg_face_mean = fw.drop(columns=['frame','face_id','timestamp']).mean().add_suffix('_mean')
            agg_face_min = fw.drop(columns=['frame','face_id','timestamp']).min().add_suffix('_min')
            agg_face_max = fw.drop(columns=['frame','face_id','timestamp']).max().add_suffix('_max')
            agg_face_std = fw.drop(columns=['frame','face_id','timestamp']).std().add_suffix('_std')
        else:
            cols_mean = face_df.drop(columns=['frame','face_id','timestamp']).add_suffix('_mean').columns
            agg_face_mean = pd.Series(0, index=cols_mean)

            cols_min = face_df.drop(columns=['frame','face_id','timestamp']).add_suffix('_min').columns
            agg_face_min = pd.Series(0, index=cols_min)

            cols_max = face_df.drop(columns=['frame','face_id','timestamp']).add_suffix('_max').columns
            agg_face_max = pd.Series(0, index=cols_max)

            cols_std = face_df.drop(columns=['frame','face_id','timestamp']).add_suffix('_std').columns
            agg_face_std = pd.Series(0, index=cols_std)

        if not aw.empty:
            agg_audio_mean = aw.drop(columns=['start','end']).mean().add_suffix('_mean')
            agg_audio_min = aw.drop(columns=['start','end']).min().add_suffix('_min')
            agg_audio_max = aw.drop(columns=['start','end']).max().add_suffix('_max')
            agg_audio_std = aw.drop(columns=['start','end']).std().add_suffix('_std')
        else:
            cols_mean = audio_df.drop(columns=['start','end']).add_suffix('_mean').columns
            agg_audio_mean = pd.Series(0, index=cols_mean)

            cols_min = audio_df.drop(columns=['start','end']).add_suffix('_min').columns
            agg_audio_mean = pd.Series(0, index=cols_min)

            cols_max = audio_df.drop(columns=['start','end']).add_suffix('_max').columns
            agg_audio_max = pd.Series(0, index=cols_max)

            cols_std = audio_df.drop(columns=['start','end']).add_suffix('_std').columns
            agg_audio_std = pd.Series(0, index=cols_std)


        if not tw.empty:
            agg_transcript_mean = tw.drop(columns=['start','end','confidence','speaker','word_count']).mean().add_suffix('_mean')
            agg_transcript_min = tw.drop(columns=['start','end','confidence','speaker','word_count']).mean().add_suffix('_min')
            agg_transcript_max = tw.drop(columns=['start','end','confidence','speaker','word_count']).mean().add_suffix('_max')
            agg_transcript_std = tw.drop(columns=['start','end','confidence','speaker','word_count']).mean().add_suffix('_std')
        else:
            cols_mean = transcript_df.drop(columns=['start','end','confidence','speaker','word_count']).add_suffix('_mean').columns
            agg_transcript_mean = pd.Series(0, index=cols_mean)

            cols_min = transcript_df.drop(columns=['start','end','confidence','speaker','word_count']).add_suffix('_min').columns
            agg_transcript_min = pd.Series(0, index=cols_min)

            cols_max = transcript_df.drop(columns=['start','end','confidence','speaker','word_count']).add_suffix('_max').columns
            agg_transcript_max = pd.Series(0, index=cols_max)

            cols_std = transcript_df.drop(columns=['start','end','confidence','speaker','word_count']).add_suffix('_std').columns
            agg_transcript_std = pd.Series(0, index=cols_std)

        # labels
        labels = is_error_window(start, end, trial_name,
                                 labels_robot_errors,
                                 labels_human_reactions_ch1,
                                 labels_human_reactions_ch2)

        # build flat row
        row = {
            'start': start.total_seconds(),
            'end':   end.total_seconds(),
            'robot_error': labels['robot_error'],
            'reaction_ch1': labels['reaction_ch1'],
            'reaction_ch2': labels['reaction_ch2'],
            'reaction_type': labels['reaction_type'],
        }
        # prefix & merge each agg_ series
        row.update(agg_face_mean.add_prefix('face_').to_dict())
        row.update(agg_audio_mean.add_prefix('audio_').to_dict())
        row.update(agg_transcript_mean.add_prefix('transcript_').to_dict())

        row.update(agg_face_min.add_prefix('face_').to_dict())
        row.update(agg_audio_min.add_prefix('audio_').to_dict())
        row.update(agg_transcript_min.add_prefix('transcript_').to_dict())

        row.update(agg_face_max.add_prefix('face_').to_dict())
        row.update(agg_audio_max.add_prefix('audio_').to_dict())
        row.update(agg_transcript_max.add_prefix('transcript_').to_dict())

        row.update(agg_face_std.add_prefix('face_').to_dict())
        row.update(agg_audio_std.add_prefix('audio_').to_dict())
        row.update(agg_transcript_std.add_prefix('transcript_').to_dict())

        rows.append(row)
        start += str_delta

    return pd.DataFrame(rows)

def process_all_data(WINDOW_SIZE, STRIDE):
    all_windows = []
    
    for system, tasks in SYSTEMS.items():
        print(f"\nProcessing system: {system.upper()}")        
        
        for task in tasks:
            print(f"\n  Task: {task}")            
            
            # Load label files for this task
            labels, robot_errors, human_reactions_ch1, human_reactions_ch2 = load_and_preprocess_labels(task)
            if labels is None:
                continue            
                
            face_files = glob(f"{BASE_PATH}/face_head_features_train/{system}/{task}/*-{task}-openface-output.csv")
            audio_files = glob(f"{BASE_PATH}/audio_features_train/{system}/{task}/*-{task}-eGeMAPSv02-features.csv")
            transcript_files = glob(f"{BASE_PATH}/transcript_features_train/{system}/{task}/transcript-*-{task}_embeddings.csv")            
            
            print(f"    Found {len(face_files)} face files")
            print(f"    Found {len(audio_files)} audio files")
            print(f"    Found {len(transcript_files)} transcript files")    
    
            # Process each trial
            for face_file in sorted(face_files):
                trial_name = os.path.basename(face_file).split('-')[0]
                # print(f"    Processing trial: {trial_name}")             
    
                try:
                    # Find matching files
                    matching_audio = [f for f in audio_files if os.path.basename(f).startswith(trial_name)]
                    matching_transcript = [f for f in transcript_files if f"transcript-{trial_name}-{task}" in os.path.basename(f)]                    
                    
                    if not matching_audio or not matching_transcript:
                        print(f"      Missing matching files")
                        continue         
    
                    # Load face data
                    try:
                        face_df = pd.read_csv(face_file)
                        face_df.columns = face_df.columns.str.lstrip()
                        if face_df.empty:
                            print(f"      Skipping empty face file: {face_file}")
                            continue
                    except Exception as e:
                        print(f"      Could not read face file: {face_file} — {e}")
                        continue           
    
                    # Check for timestamp column
                    if 'timestamp' not in face_df.columns:
                        print(f"      :x: 'timestamp' column missing in: {face_file}")
                        continue
    
                    # Load audio data
                    audio_df = pd.read_csv(matching_audio[0])
                    audio_df.columns = audio_df.columns.str.strip()
    
                    # Load transcript data
                    transcript_df = pd.read_csv(matching_transcript[0])
                    transcript_df.columns = transcript_df.columns.str.strip()                    
                    
                    # Fix timestamps
                    face_df['timestamp'] = face_df['timestamp'].apply(fix_timestamp_format)
                    face_df['timestamp'] = pd.to_timedelta(face_df['timestamp'])                    
    
                    audio_df['start'] = audio_df['start'].apply(fix_timestamp_format)
                    audio_df['start'] = pd.to_timedelta(audio_df['start'])
                    audio_df['end'] = audio_df['end'].apply(fix_timestamp_format)
                    audio_df['end'] = pd.to_timedelta(audio_df['end'])                    
                    
                    transcript_df['start'] = transcript_df['start'].apply(fix_timestamp_format)
                    transcript_df['start'] = pd.to_timedelta(transcript_df['start'])
                    transcript_df['end'] = transcript_df['end'].apply(fix_timestamp_format)
                    transcript_df['end'] = pd.to_timedelta(transcript_df['end'])                    
    
                    # Extract windows
                    windows = extract_windows(face_df, audio_df, transcript_df, trial_name, robot_errors, human_reactions_ch1, human_reactions_ch2, WINDOW_SIZE, STRIDE)                    
                        
                    # Add metadata
                    windows['system'] = system
                    windows['task'] = task
                    windows['trial'] = trial_name                    
                    
                    all_windows.append(windows)
                    # print(f"      Extracted {len(windows)} windows")                
                
                except Exception as e:
                    print(f"      Error processing trial: {str(e)}")
                    continue    
    
    final_df = pd.concat(all_windows, ignore_index=True)
    print(f"\nTOTAL WINDOWS EXTRACTED: {len(final_df)}")
    return final_df

def main():
    windows = process_all_data(WINDOW_SIZE, STRIDE)
    windows.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(windows)} windows to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
