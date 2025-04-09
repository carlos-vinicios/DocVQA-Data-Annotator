
class InvalidOCRConfig(Exception):

    def __init__(self, *args):
        super().__init__(*args)

class InvalidAzureOCRConfig(Exception):

    def __init__(self, *args):
        super().__init__(*args)