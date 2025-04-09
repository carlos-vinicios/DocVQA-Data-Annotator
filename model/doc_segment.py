import numpy as np


class DocSegment():

    def __init__(self, seg_type, bbox, score):
        self.seg_type = seg_type
        self.bbox = bbox
        self.score = score
        self.texts = []

    def get_bbox(self):
        """Retonra a bounding boxes do tipo xywh"""
        return self.bbox

    def get_type(self):
        return self.seg_type

    def get_texts(self):
        return self.texts

    def set_texts(self, texts: list):
        self.texts = texts

    def __pad_texts_blocks(self, value):
        """Adiciona o espaçamento nas bounding boxes relativas
        das caixas de textos presentes no segmento"""
        for text in self.texts:
            text.pad_relative_bbox(value)

    def pad_block(self, value: int):
        """Adiciona espaçamento uma região na bounding boxes do segmento"""
        self.bbox[:2] -= value
        self.bbox[2:] += value * 2
        self.__pad_texts_blocks(value)

    def get_crop_bbox(self):
        """Retorna a bounding box no padrão para recorte da imagem"""
        bbox = np.copy(self.bbox)
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        return bbox
