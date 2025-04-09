import re
from model.language_model import ModelUsage

# GEMINI_FLASH_PRICE = {'input': 0.0001, 'output': 0.0004}
# GEMINI_PRICE = {'input': 0.0012, 'output': 0.0038}

GEMINI_FLASH_PRICE = {'input': 0.00035, 'output': 0.00105}
GEMINI_PRICE = {'input': 0.0035, 'output': 0.0105}

CLAUDE_3_HAIKU_PRICE = {'input': 0.00025, 'output': 0.00125}
CLAUDE_3_SONNET_PRICE = {'input': 0.0030, 'output': 0.0150}
CLAUDE_3_5_SONNET_PRICE = {'input': 0.0030, 'output': 0.0150}
CLAUDE_3_OPUS_PRICE = {'input': 0.0150, 'output': 0.0750}
CLAUDE_2_PRICE = {'input': 0.0080, 'output': 0.0240}
CLAUDE_INSTANT = {'input': 0.0008, 'output': 0.0024}

GPT3_PRICE = {'input': 0.0005, 'output': 0.0015}
GPT4_PRICE = {'input': 0.01, 'output': 0.03}
GPT4O_PRICE = {'input': 0.005, 'output': 0.015}
GPT4O_MINI_PRICE = {'input': 0.00015, 'output': 0.0006}

LLAMA3_70B_PRICE = {'input': 0.0006, 'output': 0.0008}
LLAMA31_405B_PRICE = {'input': 0.003, 'output': 0.003}
MIXTRAL_MOE_8X22B_PRICE = {'input': 0.0013, 'output': 0.0013}

SABIA_SMALL_PRICE = {"input": 0.0002, "output": 0.0006}
SABIA_MEDIUM_PRICE = {"input": 0.0010, "output": 0.0029}

class LanguageModelInterface():

    def __init__(self, lang_model, prices: dict) -> None:
        self.model = lang_model
        self.prices = prices

    def clean_response_anomalies(self, response_string: str) -> str:
        wrong_break_line_pattern = re.compile(r'\n\s*\|')
        multiples_break_line_pattern = re.compile(r'\n\s*\n')
        
        response_string = multiples_break_line_pattern.sub('\n', response_string)
        response_string = wrong_break_line_pattern.sub(' |', response_string)
        
        return response_string

    def convert_prompt_pipe(self, prompt_pipe) -> str:
        messages = []
        for prompt in prompt_pipe.pipeline:
            messages.append((
                prompt.get_role().value.lower(),
                prompt.get_text()
            ))
        
        return messages

    def get_response_usage(self, usage: dict):
        return ModelUsage(
            usage['input_tokens'],
            usage['output_tokens'],
            self.prices
        )
    
    def get_response_text(self, response_str: str) -> str:
        # salva na classe para debug futuro
        response = self.clean_response_anomalies(response_str)
        return response.split("\n")
    
    def call(self, prompt_pipe: list):
        messages = self.convert_prompt_pipe(prompt_pipe)

        self.response = self.model.invoke(messages)
        text = self.get_response_text(self.response.content)
        usage = self.get_response_usage(self.response.usage_metadata)
        return text, usage