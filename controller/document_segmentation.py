# aqui ficarão todos os códigos para segmentação do layout do documento
from utils.enums import DocumentSegments

from learning.layout_analysis.yoloV8 import SegmenterModel
from controller.ocr import OcrData
from model.doc_segment import DocSegment
from model.image_data import ImageData


class DocumentSegmentation():

    def __init__(self, page_data: ImageData, segmenter: SegmenterModel,
                 ocr_data: OcrData):
        self.ocr_data = ocr_data
        self.page_data = page_data
        self.segmenter = segmenter
        self.segments = self.__segment()
        self.__sort_segments()
        self.__map_text()

    def __segment(self) -> list[DocSegment]:
        """Chama o fluxo de segmentação do layout do documento"""
        return self.segmenter.run(self.page_data)

    def __map_text(self):
        """Monta a relação entre os segmentos de layout dos documentos e os textos do OCR"""
        for seg in self.segments:
            seg.set_texts(self.ocr_data.text_in_roi(
                self.page_data.image.size, seg.get_bbox(), True
            ))

    def __sort_segments(self):
        self.segments = sorted(
            self.segments, 
            key=lambda s: s.bbox[1]
        )

    def get_segments(self):
        return self.segments

    def filter_segments(self, filter: DocumentSegments) -> list[DocSegment]:
        """Filtra os segmentos com base nos segmentos disponives no sistema"""
        result = []
        for seg in self.segments:
            if seg.seg_type == filter:
                result.append(seg)

        return result
