import os
from maritalk import MariTalk
from .base import LanguageModelInterface

SABIA_SMALL_PRICE = {"input": 0.0002, "output": 0.0006}
SABIA_MEDIUM_PRICE = {"input": 0.0010, "output": 0.0029}
MARITALK_MAX_TOKENS = 4000

class MaritalkModel(LanguageModelInterface):

    def __init__(self, model: str, prices: dict, temperature: float = 0.1) -> None:
        self.client = MariTalk(
            os.environ.get('MARITALK_API_KEY'),
            model = model
        )
        self.model = model
        self.prices = prices
        self.temperature = temperature

    def call(self, prompt_pipe: list):
        self.convert_prompt_pipe(prompt_pipe)
        self.response = self.client.generate(
            messages=self.convert_prompt_pipe(prompt_pipe),
            max_tokens=MARITALK_MAX_TOKENS,
            temperature=self.temperature
        )
        text = self.get_response_text(self.response["answer"])
        usage = self.get_response_usage(self.response["usage"])
        return text, usage