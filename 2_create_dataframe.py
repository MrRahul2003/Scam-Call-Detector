# ========================================
# 📊 Create DataFrame from Transcripts
# ========================================

import pandas as pd
import os

def read_transcripts(directory, label):
    """
    Read transcripts from a directory and assign a label
    
    Args:
        directory: Path to transcript folder
        label: Label to assign (0 for real, 1 for scam)
    
    Returns:
        List of dictionaries containing file_id, conversation, and label
    """
    data = []
    
    if os.path.exists(directory):
        for transcription_file in os.listdir(directory):
            if transcription_file.endswith("_transcription.txt"):
                file_id = os.path.splitext(transcription_file)[0].replace("_transcription", "")
                file_path = os.path.join(directory, transcription_file)
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        conversation = f.read()
                    data.append({
                        "file_id": file_id,
                        "conversation": conversation,
                        "label": label
                    })
                except Exception as e:
                    print(f"Error reading {transcription_file}: {str(e)}")
        
        print(f"Loaded {len(data)} transcripts from {directory}")
    else:
        print(f"Warning: Directory not found - {directory}")
    
    return data


def create_dataframe(real_dir, scam_dir, output_csv="transcripts_dataset.csv"):
    """
    Create a DataFrame from real and scam transcripts
    
    Args:
        real_dir: Path to real transcripts folder
        scam_dir: Path to scam transcripts folder
        output_csv: Path to save the DataFrame as CSV
    
    Returns:
        pandas DataFrame
    """
    print("=" * 50)
    print("Creating DataFrame from Transcripts")
    print("=" * 50)
    
    data = []
    
    # Read real transcripts (label = 0)
    print("\nReading REAL transcripts...")
    data.extend(read_transcripts(real_dir, 0))
    
    # Read scam transcripts (label = 1)
    print("\nReading SCAM transcripts...")
    data.extend(read_transcripts(scam_dir, 1))
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    if df.empty:
        print("\n❌ No data found! Please check your transcript directories.")
        return None
    
    # Display summary
    print("\n" + "=" * 50)
    print("DataFrame Summary")
    print("=" * 50)
    print(f"Total samples: {len(df)}")
    print(f"Real samples (label=0): {len(df[df['label']==0])}")
    print(f"Scam samples (label=1): {len(df[df['label']==1])}")
    print(f"\nDataFrame shape: {df.shape}")
    
    # Save to CSV
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n✅ DataFrame saved to '{output_csv}'")
    
    # Display first few rows
    print("\nFirst few rows:")
    print(df.head())
    
    return df


if __name__ == "__main__":
    # Configuration
    REAL_TRANSCRIPTS_DIR = "Real transcripts"
    SCAM_TRANSCRIPTS_DIR = "Scam transcripts"
    OUTPUT_CSV = "transcripts_dataset.csv"
    
    # Create DataFrame
    df = create_dataframe(REAL_TRANSCRIPTS_DIR, SCAM_TRANSCRIPTS_DIR, OUTPUT_CSV)
