import os
from model.image_data import ImageData

class TableDetection():

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def run(self, page_data: ImageData, table_id: int) -> str:
        filename = page_data.filename.split(".")[0]
        with open(os.path.join(self.model_path, f"{filename}_{table_id-1}.html"), "rb") as rf:
            structure = rf.read()
        
        return self.to_markdown(structure)