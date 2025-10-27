# ========================================
# 🔊 Speech-to-Text Transcription Script
# Using OpenAI Whisper
# ========================================

import whisper
import torch
import os
import shutil

def generate_transcripts(audio_folder, output_folder, model_size="base"):
    """
    Generate transcripts from audio files using Whisper
    
    Args:
        audio_folder: Path to folder containing audio files
        output_folder: Path to save transcriptions
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
    """
    
    # Load Whisper model
    print(f"Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of audio files
    audio_files = [f for f in os.listdir(audio_folder) 
                   if f.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.ogg'))]
    
    if not audio_files:
        print(f"No audio files found in {audio_folder}")
        return
    
    print(f"Found {len(audio_files)} audio files to transcribe")
    
    # Transcribe each audio file
    for i, audio_file in enumerate(audio_files, 1):
        audio_path = os.path.join(audio_folder, audio_file)
        print(f"\n[{i}/{len(audio_files)}] Transcribing {audio_file}...")
        
        try:
            result = model.transcribe(audio_path)
            
            # Save transcription
            output_filename = os.path.splitext(audio_file)[0] + "_transcription.txt"
            output_path = os.path.join(output_folder, output_filename)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result["text"])
            
            print(f"✅ Transcription saved: {output_filename}")
            print(f"Preview: {result['text'][:100]}...")
            
        except Exception as e:
            print(f"❌ Error transcribing {audio_file}: {str(e)}")
    
    print(f"\n✅ All transcriptions completed! Saved to '{output_folder}'")


if __name__ == "__main__":
    # Configuration
    REAL_AUDIO_FOLDER = "./audio/Real"
    SCAM_AUDIO_FOLDER = "./audio/Scam"
    
    REAL_TRANSCRIPTS_FOLDER = "./Real transcripts"
    SCAM_TRANSCRIPTS_FOLDER = "./Scam transcripts"
    
    MODEL_SIZE = "base"  # Options: 'tiny', 'base', 'small', 'medium', 'large'
    
    # Generate transcripts for real audio files
    if os.path.exists(REAL_AUDIO_FOLDER):
        print("=" * 50)
        print("Generating REAL transcripts...")
        print("=" * 50)
        generate_transcripts(REAL_AUDIO_FOLDER, REAL_TRANSCRIPTS_FOLDER, MODEL_SIZE)
    else:
        print(f"Warning: {REAL_AUDIO_FOLDER} not found. Skipping real transcripts.")
    
    # Generate transcripts for scam audio files
    if os.path.exists(SCAM_AUDIO_FOLDER):
        print("\n" + "=" * 50)
        print("Generating SCAM transcripts...")
        print("=" * 50)
        generate_transcripts(SCAM_AUDIO_FOLDER, SCAM_TRANSCRIPTS_FOLDER, MODEL_SIZE)
    else:
        print(f"Warning: {SCAM_AUDIO_FOLDER} not found. Skipping scam transcripts.")
