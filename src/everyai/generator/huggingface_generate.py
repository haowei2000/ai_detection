import logging
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def glm4(
    user_input: str,
    model_path_or_name: str | Path = None,
    gen_kwargs: dict = None,
    tokenizer=None,
    model=None,
):
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(  # type: ignore
            model_path_or_name, device_map="auto"
        )
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    message = [
        {
            "role": "system",
            "content": "Answer the following question.",
        },
        {
            "role": "user",
            "content": user_input,
        },
    ]
    inputs = tokenizer.apply_chat_template(
        message,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]
    gen_kwargs |= {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }
    out = model.generate(**gen_kwargs)
    return (
        model,
        tokenizer,
        tokenizer.decode(out[0][input_len:], skip_special_tokens=True),
    )


def qwen2_5(
    user_input: str,
    model_path_or_name: str | Path = None,
    gen_kwargs: dict = None,
    tokenizer=None,
    model=None,
):
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(  # type: ignore
            model_path_or_name, device_map="auto"
        )
    else:
        logging.info("Use existing model")
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    else:
        logging.info("Use existing tokenizer")
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {"role": "user", "content": user_input},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, **gen_kwargs)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return (
        model,
        tokenizer,
        tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0],
    )


def llama(
    user_input: str,
    model_path_or_name: str | Path = None,
    gen_kwargs: dict = None,
    tokenizer=None,
    model=None,
):
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(  # type: ignore
            model_path_or_name, device_map="auto"
        )
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )

    messages = [
        {
            "role": "system",
            "content": "You are a chatbot assistant.",
        },
        {"role": "user", "content": user_input},
    ]

    outputs = pipeline(
        messages,
        **gen_kwargs,
    )
    response = outputs[0]["generated_text"][-1]
    return (model, tokenizer, response)
