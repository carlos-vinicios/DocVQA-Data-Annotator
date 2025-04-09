from PIL import Image, ImageDraw
#PIL -> image.size

def load_document_image(page_path: str) -> Image:
    """Carrega a imagem"""
    image = Image.open(page_path).convert("RGB")
    return image

def crop_image(page_image: Image, crop_bbox: tuple) -> Image:
    """Faz o corte da imagem de acordo com a box passada"""
    #recebe: (x1, y1, x2, y2)
    return page_image.crop(crop_bbox)