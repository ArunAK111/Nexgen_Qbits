import streamlit as st
import webrtcvad
import wave
from speech_processing import extract_audio, read_wave, frame_generator, vad_audio, upload_blob, transcribe_gcs
import os

# Setup Google Cloud Credentials
# Make sure to set your Google Cloud credentials in your environment variables or in your session secrets if using Streamlit sharing
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/google-credentials.json"

st.title("Audio Processing and Transcription App")

# Setup for file upload
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
if uploaded_file is not None:
    # Save uploaded video to a temporary file
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("File uploaded successfully.")

    # Extract audio from video
    audio_path = extract_audio(temp_video_path)
    st.audio(audio_path)

    # Initialize VAD
    vad = webrtcvad.Vad(1)  # Set aggressiveness from 0 to 3

    # Read the wave file, perform VAD, and save voiced audio
    pcm_data, sample_rate, channels = read_wave(audio_path)
    frames = list(frame_generator(30, pcm_data, sample_rate))
    voiced_audio = vad_audio(vad, frames, sample_rate)
    voiced_audio_path = "voiced_audio.wav"
    with wave.open(voiced_audio_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(voiced_audio)
    st.success("Voiced audio extracted.")

    # Play the processed (voiced) audio
    st.audio(voiced_audio_path)

    # Specify your GCS bucket and the destination filename
    bucket_name = 'nexgen_storage'  # Change this to your actual bucket name
    destination_blob_name = 'voiced_audio.wav'

    # Upload the voiced audio file to GCS
    upload_blob(bucket_name, voiced_audio_path, destination_blob_name)
    st.success("Voiced audio uploaded to Google Cloud Storage.")

    # Prepare the GCS URI for the Google Speech-to-Text API
    gcs_uri = f'gs://{bucket_name}/{destination_blob_name}'

    # Transcribe the voiced audio
    transcripts = transcribe_gcs(gcs_uri, sample_rate=sample_rate)
    if transcripts:
        st.subheader("Transcriptions")
        for transcript in transcripts:
            st.write(transcript)
    else:
        st.error("Could not transcribe the audio.")