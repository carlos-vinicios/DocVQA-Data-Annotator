import os, pickle
import numpy as np

from utils.enums import DocumentSegments
from utils.bboxes import calculate_overlap_ratio
from model.doc_segment import DocSegment
from model.image_data import ImageData


class SegmenterModel():  # pode ser mudado a vontade

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.class_map = {
            0: 'Caption', 1: 'Footnote', 2: 'Formula', 3: 'List-item',
            4: 'Page-footer', 5: 'Page-header', 6: 'Picture', 7: 'Section-header',
            8: 'Table', 9: 'Text', 10: 'Title'
        }

    def __map_regions(self, bbox, cls, score):
        if cls == 8:
            return DocSegment(DocumentSegments.TABLE, bbox, score)
        if cls == 2 or cls == 6:
            return DocSegment(DocumentSegments.IMAGE, bbox, score)
        if cls == 5 or cls == 10:
            return DocSegment(DocumentSegments.PAGE_HEADER, bbox, score)
        if cls == 7:
            return DocSegment(DocumentSegments.SECTION_HEADER, bbox, score)
        # if cls == 3:
            # print("Tem list item")
        
        return DocSegment(DocumentSegments.TEXT, bbox, score)

    def __overlap_filter(self, coords) -> list:
        overlaps_ids = []
        for i, c1 in enumerate(coords):
            if i in overlaps_ids:
                continue            
            for j, c2 in enumerate(coords):
                if i == j or j in overlaps_ids:
                    continue
                if calculate_overlap_ratio(c1[:4], c2[:4]) > 0.3:
                    if c1[4] > c2[4]:
                        overlaps_ids.append(i)
                    else:
                        overlaps_ids.append(j)
        
        return overlaps_ids

    def run(self, page_data: ImageData):
        blocks = []
        results = []

        filename = page_data.filename.split(".")[0]
        with open(os.path.join(self.model_path, f"{filename}.pickle"), "rb") as rf:
            results = pickle.load(rf)
        labels = results['labels']
        coords = results['bboxes']

        # filtrando os scores abaixo de 0.3
        mask = coords[:, -1] >= 0.3
        coords = coords[mask]
        labels = labels[mask]
        
        #transformado de (x, y, x, y) -> (x, y, w, h)
        coords[:, 2] -= coords[:, 0]
        coords[:, 3] -= coords[:, 1]

        #removendo os elementos que apresentam overlap
        overlaps_ids = self.__overlap_filter(coords)
        coords = np.delete(coords, overlaps_ids, axis=0)
        labels = np.delete(labels, overlaps_ids)
        
        #removendo os valores das probs
        scores = coords[:, 4]
        coords = coords[:, :4]
        for co, cl, scr in zip(coords, labels, scores):
            blocks.append(self.__map_regions(co, cl, scr))

        return blocks
