from transformers import pipeline
import librosa
import torch

from datasets import load_dataset, Audio

data_files = ["thai_data/train.tsv","thai_data/dev.tsv"]
from datasets import load_dataset, Audio, ClassLabel

#minds = load_dataset("csv", data_files=data_files,delimiter='\t', split="train", keep_default_na=False)
#f = minds[1]
#print(f)
#audio_file = f"thai_data/clips/{f['path']}"
#print(audio_file)
#audio_file = 'thai_data/clips/common_voice_th_26707251.mp3'
audio_file = '595.wav'

def infer(audio_file):
    
    
    #classifier = pipeline("audio-classification", model="my_awesome_mind_model/checkpoint-1176")
    #a = classifier(audio_file)
    #
    #print(a)
    
    
    
    from transformers import AutoFeatureExtractor
    #
    feature_extractor = AutoFeatureExtractor.from_pretrained("my_awesome_mind_model/checkpoint-2750")
    
    
    def read_audio_file(wav_path: str, dest_fs: int = 16000):
        audio, fs = librosa.load(wav_path, sr=dest_fs)
        return audio
    
    audio = read_audio_file(audio_file)
    sampling_rate = 16000
    
    inputs = feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt")
    
    
    
    from transformers import AutoModelForAudioClassification
    
    model = AutoModelForAudioClassification.from_pretrained("my_awesome_mind_model/checkpoint-2750")
    with torch.no_grad():
        logits = model(**inputs).logits
    
    
    
    predicted_class_ids = torch.argmax(logits).item()
    predicted_label = model.config.id2label[predicted_class_ids]
    print(predicted_label)

from pathlib import Path

test_file_list = Path("aishell3/SSB1918/").rglob("*.wav")
for audio_file in test_file_list:
    print(audio_file)
    infer(audio_file)
