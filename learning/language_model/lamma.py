from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from .base import LanguageModelInterface

#TESTED MODEL:
# repo_id = "TheBloke/Llama-2-13B-chat-GGML"
# filename = "llama-2-13b-chat.ggmlv3.q5_1.bin" # the model is in bin format

LLAMA_PRICE = {'input': 0.0, 'output': 0.0}
LAMMA_CONTEXT = 4096
LAMMA_BATCH = 512
LAMMA_THREADS = 2
LAMMA_GPU_LAYERS = 32 
LAMMA_MAX_TOKENS = 1024

class OpenAIModel(LanguageModelInterface):

    def __init__(self, model: dict, prices: dict, temperature: float = 0.1) -> None:
        model_path = self.__download_model(model["repo_id"], model["filename"])
        self.model = Llama(
            model_path=model_path,
            n_threads=LAMMA_THREADS,        # CPU cores
            n_ctx=LAMMA_CONTEXT,            # Model context window (4096 is the maximum).
            n_batch=LAMMA_BATCH,            # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
            n_gpu_layers=LAMMA_GPU_LAYERS,  # Change this value based on your model and your GPU VRAM pool.
        )
        self.prices = prices
        self.temperature = temperature

    def __download_model(self, repo_id: str, filename: str):
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename
        )
        return model_path

    def call(self, prompt_pipe: list):
        self.convert_prompt_pipe(prompt_pipe)
        self.response = self.model.create_chat_completion(
            messages = self.convert_prompt_pipe(prompt_pipe),
            max_tokens = LAMMA_MAX_TOKENS,
            temperature= self.temperature
        )
        text = self.get_response_text(self.response['choices'][0]['message']['content'])
        usage = self.get_response_usage(self.response.usage.to_dict())
        return text, usage
