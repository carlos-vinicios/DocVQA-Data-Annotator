from openai import OpenAI
from .base import LanguageModelInterface

GPT3_PRICE = {'input': 0.0005, 'output': 0.0015}
GPT4_PRICE = {'input': 0.01, 'output': 0.03}
GPT4O_PRICE = {'input': 0.005, 'output': 0.015}
GPT4O_MINI_PRICE = {'input': 0.00015, 'output': 0.0006}

class OpenAIModel(LanguageModelInterface):

    def __init__(self, model: str, prices: dict, temperature: float = 0.1) -> None:
        self.client = OpenAI()
        self.model = model
        self.prices = prices
        self.temperature = temperature

    def call(self, prompt_pipe: list):
        self.convert_prompt_pipe(prompt_pipe)
        self.response = self.client.chat.completions.create(
            model=self.model,
            messages=self.convert_prompt_pipe(prompt_pipe),
            temperature=self.temperature
        )
        text = self.get_response_text(self.response.choices[0].message.content)
        usage = self.get_response_usage(self.response.usage.to_dict())
        return text, usage
