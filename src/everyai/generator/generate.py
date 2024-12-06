import logging
from pathlib import Path

from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from everyai.config.config import get_config
from everyai.everyai_path import GENERATE_CONFIG_PATH


class Generator:
    def __init__(self, config: dict, template: None = None):
        self.config = config
        self.generator_type: str = config["generator_type"]
        self.model_name: str = config["model_name"]
        self.model_path: str | Path = config["model_path"]
        self.config = config
        self.gen_kwargs: dict = {}
        self.tempalte = template
        self.model = None
        self.tokenizer = None

    def _openai_generate(
        self, input: str, base_url: str, model_name: str, api_key: str = "0"
    ) -> str:
        client = OpenAI(api_key=api_key, base_url=base_url)
        messages = [{"role": "user", "content": input}]
        result = client.chat.completions.create(
            messages=messages, model=model_name
        )
        return result.choices[0].message.content

    def _hugggingface_generate(
        self,
        input: str,
        model_path_or_name: Path | str,
        gen_kwargs: dict,
    ) -> str:
        generated_text = ""
        if model_path_or_name is not None:
            match model_path_or_name:
                case _ if "glm-4-9b-chat" in model_path_or_name:
                    logging.info("Using glm-4-9b-chatI model")
                    logging.info(f"load model from {model_path_or_name}")
                    if self.tokenizer is None:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            model_path_or_name
                        )
                    else:
                        self.tokenizer = self.tokenizer
                        logging.info("Use existing tokenizer")
                    if self.model is None:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_path_or_name, device_map="auto"
                        )
                    else:
                        self.model = self.model
                        logging.info("Use existing model")
                    message = [
                        {
                            "role": "system",
                            "content": "Answer the following question.",
                        },
                        {
                            "role": "user",
                            "content": input,
                        },
                    ]

                    inputs = self.tokenizer.apply_chat_template(
                        message,
                        return_tensors="pt",
                        add_generation_prompt=True,
                        return_dict=True,
                    ).to(self.model.device)

                    input_len = inputs["input_ids"].shape[1]
                    gen_kwargs = gen_kwargs | {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"],
                    }
                    out = self.model.generate(**gen_kwargs)
                    generated_text = self.tokenizer.decode(
                        out[0][input_len:], skip_special_tokens=True
                    )
                case _:
                    logging.error(f"Unsupport model: {model_path_or_name}")
        else:
            logging.error(f"Model path or name is None: {model_path_or_name}")
        return generated_text

    def generate(self, message: str) -> str:
        response = ""
        if self.tempalte is not None:
            message = self.tempalte.format(message)
        match self.generator_type:
            case "openai":
                if "api" in self.config.keys():
                    self.api: str = self.config["api"]
                else:
                    logging.info("API is not provided")
                if "api_key" in self.config.keys():
                    self.api_key: str = self.config["api_key"]
                else:
                    self.api_key = "0"
                logging.info("API key is not provided and was set to 0")
                response = self._openai_generate(
                    input=message,
                    base_url=self.api,
                    model_name=self.model_name,
                    api_key=self.api_key,
                )
            case "huggingface":
                logging.info("Huggingface generator")
                if "gen_kwargs" in self.config.keys():
                    self.gen_kwargs: dict = self.config["gen_kwargs"]
                else:
                    logging.info("Generation kwargs is not provided")
                if self.model_path is not None:
                    response = self._hugggingface_generate(
                        input=message,
                        model_path_or_name=self.model_path,
                        gen_kwargs=self.gen_kwargs,
                    )
                elif self.model_name is not None:
                    response = self._hugggingface_generate(
                        input=message,
                        model_path_or_name=self.model_name,
                        gen_kwargs=self.gen_kwargs,
                    )
                else:
                    logging.error("Model path and model name is None")
            case _:
                logging.error(
                    f"Generator type is not supported {self.generator_type}"
                )
        return response


if __name__ == "__main__":
    generate_config = get_config(GENERATE_CONFIG_PATH)
    print(generate_config)
    generator = Generator(config=generate_config)
    print(generator.generate("Hello"))
