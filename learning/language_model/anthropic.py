from anthropic import Anthropic
from .base import LanguageModelInterface
from utils.enums import PromptRoleType

#"claude-3-opus-20240229"
CLAUDE_3_HAIKU_PRICE = {'input': 0.00025, 'output': 0.00125}
CLAUDE_3_SONNET_PRICE = {'input': 0.0030, 'output': 0.0150}
CLAUDE_3_5_SONNET_PRICE = {'input': 0.0030, 'output': 0.0150}
CLAUDE_3_OPUS_PRICE = {'input': 0.0150, 'output': 0.0750}
CLAUDE_2_PRICE = {'input': 0.0080, 'output': 0.0240}
CLAUDE_INSTANT = {'input': 0.0008, 'output': 0.0024}

CLAUDE_MAX_TOKENS = 1024

class AnthropicModel(LanguageModelInterface):
    
    def __init__(self, model: str, prices: dict, temperature: float = 0.1) -> None:
        self.client = Anthropic()
        self.model = model
        self.prices = prices
        self.temperature = temperature

    def convert_prompt_pipe(self, prompt_pipe):
        user_messages = []
        system_message = "" #o Claude processa a entrada de System de forma separada
        for i, prompt in enumerate(prompt_pipe.pipeline):
            if i == 0 and prompt.get_role() == PromptRoleType.SYSTEM:
                system_message = prompt.get_text()
            else:
                user_messages.append({
                    "role": prompt.get_role().value.lower(),
                    "content": prompt.get_text()
                })
        return user_messages, system_message

    def call(self, prompt_pipe: list):
        user_messages, system_message = self.convert_prompt_pipe(prompt_pipe)
        self.response = self.client.messages.create(
            model=self.model,
            max_tokens=CLAUDE_MAX_TOKENS,
            system=system_message,
            messages=user_messages,
            temperature=self.temperature
        )
        text = self.get_response_text(self.response.content[0].text)

        #renomeando as chaves de uso da saída do modelo
        model_usage = self.response.usage.to_dict()
        model_usage['prompt_tokens'] = model_usage.pop('input_tokens')
        model_usage['completion_tokens'] = model_usage.pop('output_tokens')
        usage = self.get_response_usage(model_usage)
        
        return text, usage