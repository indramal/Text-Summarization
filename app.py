import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from transformers import pipeline
import torch

from fastapi import FastAPI
from mangum import Mangum
import uvicorn

import os

#cache_dir = "/tmp/"
#os.makedirs(cache_dir, exist_ok=True)
#os.environ['TRANSFORMERS_CACHE'] = cache_dir

#from transformers_cache import TransformersCache
#cache = TransformersCache(cache_dir=cache_dir)

from aws_lambda_powertools import Logger 
logger = Logger()

model_checkpoint = "./Model" # NLP Model Location
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

def summeriesf(textdata):
  logger.info("predictcat function running...")
  sumtext = pipeline("summarization", model=model, tokenizer=tokenizer)
  logger.info("Models loaded...")
  outputsum = sumtext(textdata, max_length=130, min_length=30, do_sample=False)
  return outputsum

app = FastAPI()
handler = Mangum(app)

@app.get("/")
async def index():
    return "Home"

@app.get("/api")
async def student_data(tinput:str,token:str):

    key = "RRshJy4beYdlNbu"

    if(token == key):
      logger.info("INFO: Program is Running...")

      sumoutput = summeriesf(tinput)
      return {"Status":"Done","Summery":sumoutput}
    else:
      return {"Status":"Error"}

if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)
