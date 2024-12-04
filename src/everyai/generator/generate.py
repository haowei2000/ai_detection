import logging
from pathlib import Path

from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer


def openai_generate(message: str,base_url:str,model_name:str,api_key:str="0") -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)
    result = client.chat.completions.create(
        messages=message, model=model_name
    )
    return result.choices[0].message.text

def hugggingface_generate(message:str,model_path_or_name:Path|str,) -> str:
    generated_text = ""
    if model_path_or_name is not None:
        logging.info(f"Try to load model form: {model_path_or_name}")
        model = AutoModelForCausalLM.from_pretrained(model_path_or_name)
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
        inputs = tokenizer(message, return_tensors="pt")
        outputs = model.generate(**inputs)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        logging.error(f"Model path or name is None: {model_path_or_name}")
        return generated_text

class Generator:
    def __init__(self, config,Template):
        self.config = config
        self.generator_type: str = config["generator_type"]
        self.model_name: str = config["model_name"]
        self.model_path: str|Path = config["model_path"]
        self.api: str = config["api"]
        self.Tempalte = Template

    def generate(self, message: str) -> str:
        response=""
        match self.generator_type:
            case "openai":
                message= self.Tempalte.format(message)
                response = openai_generate(message=message,base_url=self.api,model_name=self.model_name)
            case "huggingface":
                if self.model_path is not None:
                    response= hugggingface_generate(message=message,model_path_or_name=self.model_path)
                elif self.model_name is not None:
                    response= hugggingface_generate(message=message,model_path_or_name=self.model_name)
                else:
                    logging.error("Model path and model name is None")
            case _:
                logging.error(f"Generator type is not supported {self.generator_type}")
        return response

if __name__=="__main__":
    config = {
        "generator_type": "openai",
        "model_name": "gpt-3.5-turbo",
        "model_path": None,
        "api": "https://api.openai.com",
    }