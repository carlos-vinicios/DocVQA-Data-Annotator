
from transformers import DetrFeatureExtractor
from transformers import TableTransformerForObjectDetection
from PIL import Image, ImageDraw
import torch
import numpy as np

from learning.table_detection.TATR_utils.table_parser import objects_to_structures, structure_to_cells, cells_to_html

from utils.enums import TableSegments
from model.table import TableBlock
from model.image_data import ImageData

COLORS = [
    (0, 122, 204, 128), (255, 106, 0, 128), (76, 175, 80, 128),
    (255, 193, 7, 128), (156, 39, 176, 128), (233, 30, 99, 128)
]


class TableDetection():

    def __init__(self, model_path: str):
        self.model = TableTransformerForObjectDetection.from_pretrained(
            model_path)
        self.model_labels = self.model.config.id2label
        self.feature_extractor = DetrFeatureExtractor()
        self.model_classes = {
            'table': 0,
            'table column': 1,
            'table row': 2,
            'table column header': 3,
            'table projected row header': 4,
            'table spanning cell': 5,
            'no object': 6
        }
        self.class_idx2name = {v:k for k, v in self.model_classes.items()}

    def plot_results(self, image: Image, results: tuple, desired_labels: list):
        """Cria a imagem para a visualização dos resultados obtidos da execução do modelo.
        Pode-se escolher quais as labels visualizaveis, para reduzir a quantidade de ruído
        visual e melhorar a visualização"""
        new_image = image.copy()
        img_drawed = ImageDraw.Draw(new_image, 'RGBA')

        for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
            if label in desired_labels:
                fix_box = [(box[0], box[1]), (box[2], box[3])]
                img_drawed.rectangle(fix_box, fill=COLORS[label])
                img_drawed.text(
                    (box[0], box[1]),
                    f"{self.model_labels[int(label)]}: {round(float(score), 2)}",
                    fill=(0, 0, 0)
                )

        return new_image

    def _map_tokens(self, table_text: list):
        tokens = []

        for idx, t_text in enumerate(table_text):
            bbox = [float(v) for v in t_text.relative_bbox]
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]

            tokens.append({
                'span_num': idx,
                'bbox': bbox,
                'text': t_text.text,
                'line_num': 0,
                'block_num': 0
            })
        return tokens
    
    def calculate_structure_confidence_micro_avg_score(self, table_structure):
        structure_scores = []

        if "rows" in table_structure:
            structure_scores += [element['score'] for element in table_structure["rows"]]
        if "columns" in table_structure:
            structure_scores += [element['score'] for element in table_structure["columns"]]
        if "spanning cells" in table_structure:
            structure_scores += [element['score'] for element in table_structure["spanning cells"]] 
        
        return np.average(structure_scores)

    def run(self, table_data: ImageData, table_text: list) -> list[TableBlock]:
        """Executa a extração de features e a segmentação com o modelo informado"""
        tokens = self._map_tokens(table_text)
        encoding = self.feature_extractor(
            table_data.table_image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**encoding)

        # Fazemos a substituição da altura e largura pois o modelo espera uma entrada: (height, width)
        target_sizes = [table_data.table_image.size[::-1]]

        # Utilizamos o extrator de features, aplicando um limiar de 0.6 as detecções realizadas
        results = self.feature_extractor.post_process_object_detection(
            outputs,
            threshold=0.4,
            # threshold=0.7,
            target_sizes=target_sizes)[0]

        objects = []
        for score, label, bbox in zip(results['scores'], results['labels'], results['boxes']):
            class_label = self.class_idx2name[int(label)]
            objects.append({
                'label': class_label,
                'score': float(score),
                'bbox': [float(elem) for elem in bbox]
            })
        
        # Further process the detected objects so they correspond to a consistent table
        table_structure = objects_to_structures(objects, tokens)
        confidence_score = self.calculate_structure_confidence_micro_avg_score(table_structure)

        # Enumerate all table cells: grid cells and spanning cells
        table_cells = structure_to_cells(table_structure, tokens)[0]

        # list(range(1, 6))
        # self.plot_results(table_data.table_image, results, [2]).show()
        # self.plot_results(table_data.table_image, results, [1]).show()
        # self.plot_results(table_data.table_image, results, [3]).show()
        # self.plot_results(table_data.table_image, results, [5]).show()

        return table_cells, cells_to_html(table_cells), confidence_score