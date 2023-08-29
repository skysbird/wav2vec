from transformers import AutoFeatureExtractor
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
feature_extractor.save_pretrained("feature_extractor")

from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

model = AutoModelForAudioClassification.from_pretrained(
    "facebook/wav2vec2-base"
)

model.save_pretrained("model")


