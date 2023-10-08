from transformers import pipeline
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import time
from translate import Translator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

generator = pipeline('text-generation', model='openai-gpt')

def translate_text(text, target_language):
    print(text)
    # Reverse lookup to get the language code from the language name
    target_code = target_language
    translator = Translator(to_lang=target_code)
    translation = translator.translate(text)
    return translation

app = FastAPI()

class Body(BaseModel):
    text: str


@app.get('/')
def root():
    return HTMLResponse("<h1>A self-documenting API to interact with a GPT2 model and generate text</h1>")


@app.post('/translate')
def predict(body: Body):
    print(body.text)
    return translate_text(body.text, 'de')
