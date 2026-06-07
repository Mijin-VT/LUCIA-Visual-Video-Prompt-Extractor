import streamlit as st
import os
from moviepy.editor import VideoFileClip
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa
import tempfile
from datetime import timedelta

# Load the pre-trained Wav2Vec2 model and processor from Hugging Face
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def convert_video_to_audio(video_path, audio_path):
    """Convert video file to audio file."""
    with VideoFileClip(video_path) as video:
        video.audio.write_audiofile(audio_path)

def transcribe_audio_with_timestamps(audio_path):
    """Transcribe audio file to text with timestamps using Hugging Face's Wav2Vec2 model."""
    # Load audio
    speech, rate = librosa.load(audio_path, sr=16000)
    
    # Process audio
    input_values = processor(speech, sampling_rate=rate, return_tensors="pt").input_values
    
    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits
    
    # Decode the predicted text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    
    # Generate timestamps
    duration = librosa.get_duration(y=speech, sr=rate)
    timestamped_transcript = generate_timestamps(transcription[0], duration)
    
    return transcription[0], timestamped_transcript

def generate_timestamps(transcription, duration):
    """Generate timestamps for each word in the transcription."""
    words = transcription.split()
    words_per_second = len(words) / duration
    timestamped_transcript = []

    for i, word in enumerate(words):
        time = timedelta(seconds=i / words_per_second)
        timestamped_transcript.append(f"[{str(time)}] {word}")

    # Group words into paragraphs
    paragraph_length = 50  # Number of words per paragraph
    paragraphed_transcript = "\n\n".join(
        " ".join(timestamped_transcript[i:i + paragraph_length])
        for i in range(0, len(timestamped_transcript), paragraph_length)
    )

    return paragraphed_transcript

def save_transcripts(plain_transcript, timestamped_transcript, base_filename):
    """Save transcripts in two versions: plain and with timestamps."""
    # Ensure transcripts directory exists
    if not os.path.exists("transcripts"):
        os.makedirs("transcripts")
    
    # Save plain transcript
    plain_text_path = os.path.join("transcripts", f"{base_filename}_plain.txt")
    with open(plain_text_path, "w") as f:
        f.write(plain_transcript)
    
    # Save timestamped transcript
    timestamped_text_path = os.path.join("transcripts", f"{base_filename}_timestamped.txt")
    with open(timestamped_text_path, "w") as f:
        f.write(timestamped_transcript)

# Streamlit app
st.title("Video Transcription App")

# Step 1: Upload video
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Create a temporary directory to save uploaded video and audio
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, uploaded_file.name)
        
        # Check if a file with the same name already exists
        if os.path.exists(video_path):
            st.warning("A file with the same name already exists. Upload a different file or rename your file.")
        else:
            # Save video file to the temporary directory
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.write("Video uploaded successfully!")
            
            # Convert video to audio
            audio_path = os.path.splitext(video_path)[0] + ".wav"
            
            # Call the function to convert video to audio
            convert_video_to_audio(video_path, audio_path)
            
            st.write("Audio extracted from video successfully!")
            
            # Transcribe audio
            plain_transcript, timestamped_transcript = transcribe_audio_with_timestamps(audio_path)
            
            # Display transcription in paragraph format
            st.write("Plain Transcription:")
            st.text(plain_transcript)
            
            st.write("Transcription with Timestamps:")
            st.text(timestamped_transcript)
            
            # Save transcripts to files
            base_filename = os.path.splitext(uploaded_file.name)[0]
            save_transcripts(plain_transcript, timestamped_transcript, base_filename)
            
            st.write("Transcripts saved successfully in the 'transcripts' folder.")
