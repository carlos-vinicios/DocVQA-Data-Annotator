import os
from utils.image import load_document_image

class ImageData:

    def __init__(self, page_path) -> None:
        self.page_path = page_path
        self.image = load_document_image(page_path)
        self.table_image = None
    
    @property
    def filename(self):
        return os.path.basename(self.page_path)

    