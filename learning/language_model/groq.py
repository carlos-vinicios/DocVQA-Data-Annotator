import os
from groq import Groq
from .base import LanguageModelInterface

LLAMA3_PRICE = {'input': 0.0006, 'output': 0.0008}

class GroqModel(LanguageModelInterface):

    def __init__(self, model: str, prices: dict, temperature: float = 0.1) -> None:
        self.client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )
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
