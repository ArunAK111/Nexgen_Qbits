# audio_processing.py

import wave
import webrtcvad
from moviepy.editor import VideoFileClip
from google.cloud import speech, storage

def extract_audio(video_path, output_path="temp_audio.wav", sample_rate=16000):
    """Extracts audio from a video file and saves it as a WAV file."""
    video_clip = VideoFileClip(video_path)
    audio = video_clip.audio
    audio.write_audiofile(output_path, codec='pcm_s16le', fps=sample_rate)
    return output_path

def read_wave(path):
    """Reads a WAV file and returns its properties."""
    with wave.open(path, 'rb') as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate, channels

def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from the WAV file data."""
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)  # 2 bytes per sample
    offset = 0
    while offset + n < len(audio):
        yield audio[offset:offset + n]
        offset += n

def vad_audio(vad, frames, sample_rate):
    """Performs VAD on the audio frames."""
    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame, sample_rate)
        if is_speech:
            voiced_frames.append(frame)
    return b''.join(voiced_frames)

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

def transcribe_gcs(gcs_uri, sample_rate=16000):
    """Transcribes audio from a GCS URI using Google Cloud Speech-to-Text."""
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code="en-US",
        use_enhanced=True,
        model='video',
        enable_automatic_punctuation=True
    )
    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=90)  # Adjust timeout as needed
    return [result.alternatives[0].transcript for result in response.results]
