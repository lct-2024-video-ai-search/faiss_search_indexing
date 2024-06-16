from get_data import get_videos
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import faiss
from math import sqrt
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_ckpt = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt).to(device)

def create_index_index_videos():
    result, columns = get_videos()
    index, ids_index = _create_index(len(result))

    training_data = _get_training_data(result)
    index.train(training_data)

    for row in result:
        with torch.no_grad():
            text = ' '.join([row[1], row[2], row[3]])
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            vector = model(**inputs.to(device)).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        index.add(np.expand_dims(vector, axis=0))
        ids_index.append(row[0])
    return index, ids_index


def index_video(video_index, video_description, video_movement_desc, video_speech_description, index, index_ids):
    with torch.no_grad():
        text = ' '.join([video_speech_description, video_description, video_movement_desc])
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        vector = model(**inputs.to(device)).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    index.add(np.expand_dims(vector, axis=0))
    index_ids.append(video_index)
    logging.info(f"Обработка данных для создания индекса окончена")
    return index, index_ids

def _create_index(N):
    dim = 1024
    nlist = int(sqrt(N))
    m = 16
    bits = 8

    quantiser = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantiser, dim, nlist, m, bits)
    index.nprobe = 100

    ids_index = []
    return index, ids_index


def _get_training_data(result):
    training_data = []
    for row in result:
        with torch.no_grad():
            text = ' '.join([row[1], row[2], row[3]])
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            vector = model(**inputs.to(device)).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            training_data.append(vector)
    return np.array(training_data).astype('float32')


def search(query, index, ids_index):
    with torch.no_grad():
        inputs = tokenizer(query, return_tensors='pt', truncation=True, padding=True, max_length=512)
        query_vector = model(**inputs.to(device)).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    D, I = index.search(np.expand_dims(query_vector, axis=0), k=1000)
    return D, I, [ids_index[idx] for idx in I[0]]
