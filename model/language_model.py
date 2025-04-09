
class ModelUsage():

    def __init__(self, prompt_tokens, output_tokens, prices: dict = None):
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.prices = prices

    def to_dict(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "prices": self.prices,
            "total": self.calculate_cost()
        }

    def get_total_output_tokens(self):
        """Retorna o total de tokens consumidos pela chamada realizada"""
        return self.prompt_tokens + self.output_tokens

    def calculate_cost(self):
        """Calcula o preço para a realização da chamada com base em 1K de tokens
        consumidos. Dessa forma, o atributo price, deve considerar o mesmo padrão."""
        input_price = self.prompt_tokens / 1000 * self.prices['input']
        output_price = self.output_tokens / 1000 * self.prices['output']
        return input_price + output_price

