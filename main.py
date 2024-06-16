import logging
import torch

from typing import Union
from pydantic import BaseModel, ValidationError
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModel
import numpy as np
from index_creation import create_index_index_videos, index_video

class IndexInfo(BaseModel):
    VideoDescription: str
    VideoMovementDesc: str
    SpeechDescription: str
    Index: int

class Object(BaseModel):
    video_id: int

# return models
class CreateVideoIndexResponse(BaseModel):
    status: str

class SearchResponse(BaseModel):
    ids: list[int]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.INFO)
logging.info(f"Доступна ли видеокарта? Ответ: {torch.cuda.is_available()}")

index, index_ids = None, None
model_ckpt = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt).to(device)

@app.on_event("startup")
async def startup_event():
    global index, index_ids
    index, index_ids = create_index_index_videos()
    logging.info("Index creation complete")

@app.post("/create_video_index", response_model=CreateVideoIndexResponse)
def create_video_index(indexInfo: IndexInfo):
    global index, index_ids

    try:
        video_description = indexInfo.VideoDescription
        video_movement_desc = indexInfo.VideoMovementDesc
        video_speech_description = indexInfo.SpeechDescription
        video_index = indexInfo.Index

        if index is None:
            raise HTTPException(status_code=503, detail="Index not yet initialized.")

        index, index_ids = index_video(video_index, video_description, video_movement_desc, video_speech_description, index, index_ids)
        return CreateVideoIndexResponse(status="Success")
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search", response_model=SearchResponse)
def search(query: str):
    global index, index_ids

    with torch.no_grad():
        inputs = tokenizer(query, return_tensors='pt', truncation=True, padding=True, max_length=512)
        query_vector = model(**inputs.to(device)).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    if index is None:
        raise HTTPException(status_code=503, detail="Index not yet initialized.")

    D, I = index.search(np.expand_dims(query_vector, axis=0), k=1000)
    ids = [index_ids[idx] for idx in I[0]]

    return SearchResponse(
        ids=ids
    )

def __main__():
    global index, index_ids

    if index is None:
        index, index_ids = create_index_index_videos()

