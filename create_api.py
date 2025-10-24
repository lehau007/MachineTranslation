from fastapi import FastAPI
from model.machine_translation import MachineTranslation
from translate.translate import generate_translation_beam_search
from transformers import AutoTokenizer
app = FastAPI()

model = MachineTranslation() 
model.load_state_dict("")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

@app.post("/translate")
def translate(text: str):
    return generate_translation_beam_search(model, tokenizer, text, device="cpu")