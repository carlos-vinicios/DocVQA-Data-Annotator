from PIL import Image
from ultralytics import YOLO

from utils.enums import DocumentSegments
from model.doc_segment import DocSegment
from model.image_data import ImageData


class SegmenterModel():  # pode ser mudado a vontade

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_map = {
            0: 'Caption', 1: 'Footnote', 2: 'Formula', 3: 'List-item',
            4: 'Page-footer', 5: 'Page-header', 6: 'Picture', 7: 'Section-header',
            8: 'Table', 9: 'Text', 10: 'Title'
        }

    def __map_regions(self, bbox, cls):
        if cls == 8:
            return DocSegment(DocumentSegments.TABLE, bbox)
        if cls == 2 or cls == 6:
            return DocSegment(DocumentSegments.IMAGE, bbox)
        if cls == 5 or cls == 10:
            return DocSegment(DocumentSegments.PAGE_HEADER, bbox)
        if cls == 7:
            return DocSegment(DocumentSegments.SECTION_HEADER, bbox)

        return DocSegment(DocumentSegments.TEXT, bbox)
    
    def run(self, page_data: ImageData):
        blocks = []
        width, height = page_data.image.size
        # results = self.model(page_image, save=True, show_labels=True, show_conf=True, show_boxes=True)
        results = self.model(page_data.image, save=False)
        for entry in results:
            coords = entry.boxes.xyxy.numpy()
            coords[:, 2] -= coords[:, 0]
            coords[:, 3] -= coords[:, 1]
            cls = entry.boxes.cls.numpy()
            for co, cl in zip(coords, cls):
                blocks.append(self.__map_regions(co, cl))

        return blocks
