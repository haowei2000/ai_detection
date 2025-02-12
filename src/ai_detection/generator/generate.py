import logging
from collections.abc import Callable
from pathlib import Path

from ai_detection.generator.openai_generate import openai_generate
from ai_detection.generator.huggingface_generate import glm4, llama, qwen2_5
from ai_detection.utils.everyai_path import GENERATE_CONFIG_PATH
from ai_detection.utils.load_args import set_attrs_2class
from ai_detection.utils.load_config import get_config


class Generator:
    def __init__(self, config: dict, formatter: Callable[[str], str] = None):
        # TODO: set param auto type
        self.generator_type = config["generator_type"]
        if self.generator_type == "huggingface":
            default_params = [
                "model_name",
                "model_path",
                "model",
                "tokenizer",
                "gen_kwargs",
            ]
            allowed_keys = [
                "generator_type",
                "model_name",
                "model_path",
                "gen_kwargs",
            ]
        elif self.generator_type == "openai":
            allowed_keys = [
                "generator_type",
                "base_url",
                "model_name",
                "api_key",
                "proxy",
            ]
            default_params = ["model_name", "base_url", "api_key"]
        else:
            raise ValueError(
                "Unsupported generator type: %s", self.generator_type
            )
        self.formatter = formatter
        set_attrs_2class(self, config, allowed_keys, default_params)

    def _huggingface_generate(
        self,
        user_input: str,
        model_path_or_name: Path | str,
        gen_kwargs: dict,
    ) -> str:
        match model_path_or_name:
            case _ if "glm-4" in model_path_or_name.lower():
                logging.info("glm-4 generator: %s", model_path_or_name)
                self.model, self.tokenizer, response = glm4(
                    user_input=user_input,
                    model_path_or_name=model_path_or_name,
                    gen_kwargs=gen_kwargs,
                    model=self.model,
                    tokenizer=self.tokenizer,
                )
            case _ if "qwen" in model_path_or_name.lower():
                logging.info("qwen2.5 generator: %s", model_path_or_name)
                self.model, self.tokenizer, response = qwen2_5(
                    user_input=user_input,
                    model_path_or_name=model_path_or_name,
                    gen_kwargs=gen_kwargs,
                    model=self.model,
                    tokenizer=self.tokenizer,
                )
            case _ if "llama" in model_path_or_name.lower():
                logging.info("llama generator: %s", model_path_or_name)
                self.model, self.tokenizer, response = llama(
                    user_input=user_input,
                    model_path_or_name=model_path_or_name,
                    gen_kwargs=gen_kwargs,
                    model=self.model,
                    tokenizer=self.tokenizer,
                )
            case _:
                raise ValueError("Unsupported model: %s", model_path_or_name)
        return response

    def generate(self, message: str) -> str:
        if self.formatter is not None:
            message = self.formatter(message)
            logging.info("Formatted input: %s", message)
        else:
            logging.info("Input was not formatted, using original input")
        match self.generator_type:
            case "openai":
                response = openai_generate(
                    user_input=message,
                    base_url=self.base_url,
                    model_name=self.model_name,
                    api_key=self.api_key,
                    proxy=self.proxy,
                )
            case "huggingface":
                logging.info("Huggingface generator")
                if self.model_path is not None:
                    response = self._huggingface_generate(
                        user_input=message,
                        model_path_or_name=self.model_path,
                        gen_kwargs=self.gen_kwargs,
                    )
                elif self.model_name is not None:
                    response = self._huggingface_generate(
                        user_input=message,
                        model_path_or_name=self.model_name,
                        gen_kwargs=self.gen_kwargs,
                    )
                else:
                    logging.error("Model path and model name is None")
            case _:
                logging.error(
                    "Generator type '%s' is not supported", self.generator_type
                )
        return response


if __name__ == "__main__":
    generate_config = get_config(GENERATE_CONFIG_PATH)
    print(generate_config)
    generator = Generator(config=generate_config)
    print(generator.generate("Hello"))
