import logging

from openai import OpenAI
from everyai.utils.proxy import TempProxy


def openai_generate(
    user_input: str,
    base_url: str,
    model_name: str,
    api_key: str = "0",
    proxy: str = None,
) -> str:
    if proxy is not None:
        temp_proxy = TempProxy()
        temp_proxy.start_proxy(proxy)
        logging.info("Proxy started: %s", proxy)
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [{"role": "user", "content": user_input}]
    result = client.chat.completions.create(
        messages=messages, model=model_name
    )
    if proxy is not None:
        temp_proxy.reset_proxy()
        logging.info("Proxy reset")
    return result.choices[0].message.content
