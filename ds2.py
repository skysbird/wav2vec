from datasets import load_dataset,Value
import pandas as pd
import numpy as np
import librosa



from transformers import AutoFeatureExtractor

#feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
feature_extractor = AutoFeatureExtractor.from_pretrained("feature_extractor")



# Load dataset
#dataset = load_dataset("./thai_data")
SAMPLING_RATE = 16000

def read_audio_file(wav_path: str, dest_fs: int = 16000):
    audio, fs = librosa.load(wav_path, sr=dest_fs)
    return audio


data_files = ["wav.tsv"]


#dataset = load_dataset("csv", data_files=data_files,delimiter='\t')['train']
#
#dataset = dataset.train_test_split(test_size=0.3)
#
#



from datasets import load_dataset, Audio, ClassLabel

minds = load_dataset("csv", data_files=data_files,delimiter='\t', split="train", keep_default_na=False)
print(minds)


labels = minds["client_id"]
labels = sorted(list(set(labels)))

label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = str(label)


minds = minds.train_test_split(test_size=0.2)






#print(dataset['train'][0:2])
#client_id\tpath\tsentence\tup_votes\tdown_votes\tage\tgender\taccents\tlocale\tsegment

# Add labels for each audio file
# Train data
#x_train_data = [f for f in dataset['train']]
#x_train_data = [read_audio_file(f"thai_data/clips/{f['path']}",SAMPLING_RATE) for f in dataset['train']]
#y_train_data = [f['client_id'] for f in dataset['train']]
#
#x_test_data = [read_audio_file(f"thai_data/clips/{f['path']}",SAMPLING_RATE) for f in dataset['test']]
#y_test_data = [f['client_id'] for f in dataset['test']]


def preprocess_function(examples):

    audio_arrays = []
    for wav_path in examples['path']:
        audio = read_audio_file(wav_path, SAMPLING_RATE) 
        audio_arrays.append(audio)

    inputs = feature_extractor(
        audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
    )
    return inputs

encoded_minds = minds.map(preprocess_function,  batched=True)
encoded_minds = encoded_minds.rename_column("client_id", "label")

lb = ClassLabel(num_classes=len(labels), names=list(id2label.values()))
encoded_minds = encoded_minds.cast_column("label", lb)

#print(encoded_minds['train'][0])


import evaluate

accuracy = evaluate.load("./metrics/accuracy")


import numpy as np





def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)


from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

num_labels = len(id2label)

print(id2label)
print("l=",num_labels)
#model = AutoModelForAudioClassification.from_pretrained(
#    "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label
#)

model = AutoModelForAudioClassification.from_pretrained(
    "model", num_labels=num_labels, label2id=label2id, id2label=id2label,ignore_mismatched_sizes=True
)

training_args = TrainingArguments(
    output_dir="my_awesome_mind_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    # use_cpu=True,
)


# model = model.to('cpu')

trainer = Trainer(
    model = model,
    args=training_args,
    train_dataset=encoded_minds["train"],
    eval_dataset=encoded_minds["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,

)

trainer.train()

#x_train_data = [read_audio_file(f"thai_data/clips/{f}",SAMPLING_RATE) for f in dataset['train'][:100]['path']]
#y_train_data = [f for f in dataset['train'][:100]['client_id']]


#x_test_data = [read_audio_file(f"thai_data/clips/{f['path']}",SAMPLING_RATE) for f in dataset['test']]
#y_test_data = [f['client_id'] for f in dataset['test']]




#print(x_train_data)
#print(y_train_data)




#y_train_data = [str(int(i/10)) for i in range(len(dataset['train']))]


# Test data
#x_test_data = [f['path'] for f in dataset['test']]
#y_test_data = [str(int(i/10)) for i in range(len(dataset['test']))]

# Let's count the speakers and audio files
#speakers_id, c = np.unique(y_train_data, return_counts=True)
#df = pd.DataFrame({"Speaker": speakers_id, "wav_files_count": c})
##print(df)
#
#
#import torch
#from tqdm import tqdm
#from transformers import Wav2Vec2FeatureExtractor
#from transformers import WavLMForXVector
#
#device = "cuda" if torch.cuda.is_available() else "cpu"
#feature_extractor_wav2vec = Wav2Vec2FeatureExtractor.from_pretrained(
#    "feature_extractor")
#model_wav_lm = WavLMForXVector.from_pretrained(
#    "model").to(device)
#
#
#def extract_embeddings(model, feature_extractor, data, device,sampling_rate):
#    """Use WavLM model to extract embeddings for audio segments"""
#    emb_train = list()
#    for i in tqdm(range(len(data))):
#        inputs = feature_extractor(
#            data[i], 
#            sampling_rate=sampling_rate, 
#            return_tensors="pt", 
#            padding=True
#        ).to(device)
#        with torch.no_grad():
#            embeddings = model(**inputs).embeddings
#
#        emb_train += torch.nn.functional.normalize(
#            embeddings.cpu(), dim=-1).cpu()
#
#    return torch.stack(emb_train)
#
#
#N_SECONDS_SEGMENT = 4  # Seconds
#
#def segment_audio(x, y, n_sec, samp_rate):
#    """Segment each array in list of audio arrays to N seconds segments"""
#    x_segment = list()
#    y_segment = list()
#    for x, y in zip(x, y):
#        segments = np.array_split(x, round(x.shape[0] / (samp_rate * n_sec)))
#        x_segment += segments
#        y_segment += [y] * len(segments)
#
#    return x_segment, y_segment
#
#x_train, y_train = x_train_data, y_train_data
#
#x_test, y_test = x_test_data, y_test_data
## Segment train and test sets
##x_train, y_train = segment_audio(
##    x_train_data, y_train_data, N_SECONDS_SEGMENT, SAMPLING_RATE)
##x_test, y_test = segment_audio(
##    x_test_data, y_test_data, N_SECONDS_SEGMENT, SAMPLING_RATE)
#
#
## Extract embeddings for train and test set
#x_train_emb = extract_embeddings(
#    model=model_wav_lm, 
#    feature_extractor=feature_extractor_wav2vec,
#    data=x_train, 
#    device=device,
#    sampling_rate=SAMPLING_RATE
#)
#
#
#from sklearn.neighbors import KNeighborsClassifier
#
## Fit k-NN model
#N_NEIGHBORS = 5
#model_knn = KNeighborsClassifier(
#    n_neighbors=N_NEIGHBORS,
#    metric='cosine',
#    algorithm='brute'
#)
#model_knn.fit(x_train_emb, y_train)
#
#
#from sklearn.metrics import confusion_matrix, f1_score
#
##print(x_train_emb.size())
##print(x_train_emb[0,:][None].size())
## Predict
#
##print(y_train)
#x_test_emb = extract_embeddings(
#    model=model_wav_lm, 
#    feature_extractor=feature_extractor_wav2vec,
#    data=x_test, 
#    device=device,
#    sampling_rate=SAMPLING_RATE
#)
#
#y_pred = model_knn.predict(x_test_emb[3].reshape(1,-1))
#
#print(x_test[3])
#print(y_pred)
#
#n = model_knn.kneighbors(x_test_emb[-1].reshape(1,-1),5,False) 
#
#print(n)
#
#
#
## Print F1 score
#y_pred = model_knn.predict(x_test_emb)
#
#f1 = f1_score(y_test, y_pred, average="macro")
#print("k-NN test set F1 score={:.3f}".format(f1))






