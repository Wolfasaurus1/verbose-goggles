import speech_recognition as sr
import openai
import json
from sentence_transformers import SentenceTransformer, util
import torch
import io

openai.api_key = "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"

# Initialize recognizer and microphone
recognizer = sr.Recognizer()

# Initialize the sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# get the list of questions
with open("database.json", 'r') as f:
    contents = json.load(f)
    questions = list(contents.keys())
    sentences_embeddings = model.encode(questions, convert_to_tensor=True)

# Capture audio from the microphone
with sr.Microphone() as source:
    recognizer.adjust_for_ambient_noise(source)
    print("Please speak something...")
    audio_data = recognizer.listen(source)

# see if we can improve response time even further by not saving the audio data
audio_bytes = audio_data.get_wav_data()
audio_file_like = io.BytesIO(audio_bytes)
audio_file_like.name = "captured_audio.wav"

# transcribe the saved audio
audio_file = open("captured_audio.wav", "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file_like)

transcript_embedding = model.encode(str(transcript), convert_to_tensor=True)

# Compute the cosine similarities
cosine_scores = util.pytorch_cos_sim(transcript_embedding, sentences_embeddings)[0]

# Find the index with the highest similarity score
most_similar_index = torch.argmax(cosine_scores).item()
best_score = cosine_scores[most_similar_index].item()

print(contents[questions[most_similar_index]])

from pydub import playback
import pydub

path = "audio/" + contents[questions[most_similar_index]]
sound = pydub.AudioSegment.from_file(path, format="mp3")
playback.play(sound)