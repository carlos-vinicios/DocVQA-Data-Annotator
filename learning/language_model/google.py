import os
import google.generativeai as genai
from utils.enums import PromptRoleType
from .base import LanguageModelInterface

# GEMINI_FLASH_PRICE = {'input': 0.0001, 'output': 0.0004}
# GEMINI_PRICE = {'input': 0.0012, 'output': 0.0038}

GEMINI_FLASH_PRICE = {'input': 0.00035, 'output': 0.00105}
GEMINI_PRICE = {'input': 0.0035, 'output': 0.0105}

class GeminiModel(LanguageModelInterface):

    def __init__(self, model: str='gemini-pro', prices: dict = {}, temperature: float = 0.1) -> None:
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        self.model_name = model
        self.prices = prices
        self.temperature = temperature

    def convert_prompt_pipe(self, prompt_pipe) -> str:
        user_messages = []
        system_message = "" #o Claude processa a entrada de System de forma separada
        for i, prompt in enumerate(prompt_pipe.pipeline):
            if i == 0 and prompt.get_role() == PromptRoleType.SYSTEM:
                system_message = prompt.get_text()
            else:
                user_messages.append(prompt.get_text())
        
        return '\n'.join(user_messages), system_message

    def get_response_usage(self, response):
        return None
    
    def call(self, prompt_pipe: list):
        message, system_message = self.convert_prompt_pipe(prompt_pipe)
        
        if len(system_message) > 0:
            self.model = genai.GenerativeModel(
                self.model_name, 
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature
                ),
                system_instruction=system_message
            )
        else:
            self.model = genai.GenerativeModel(
                self.model_name, 
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature
                )
            )
        
        self.response = self.model.generate_content(message)
        text = self.get_response_text(self.response.text)
        usage = self.get_response_usage(self.response)
        return text, usage