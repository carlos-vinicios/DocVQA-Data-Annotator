import numpy as np
from utils.enums import TableSegments


class TableBlock():

    def __init__(self, block_type: TableSegments, score: float, bbox: np.array):
        self.block_type = block_type
        self.score = score
        self.bbox = bbox.round()


class TableCell():

    def __init__(self, bbox: np.array, row_id: int, col_id: int):
        self.text = ""
        self.bbox = bbox.round()
        self.rows_id = [row_id]
        self.cols_id = [col_id]
        self.span_id = None

    def set_text(self):
        # TODO: implementar a captação do texto
        return

    def set_span_id(self, span_id: int):
        self.span_id = span_id

    def append_row_id(self, row_id: int):
        self.rows_id.append(row_id)

    def append_col_id(self, col_id: int):
        self.cols_id.append(col_id)

    def merge(self, bbox):
        # Inicialize os limites da nova bounding box combinada
        x_min = float('inf')
        y_min = float('inf')
        x_max = float('-inf')
        y_max = float('-inf')

        # Encontre os limites extremos entre todas as bounding boxes
        cells = [bbox, self.bbox]
        for box in cells:
            x, y, w, h = box
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        # Calcule a largura e a altura da nova bounding box combinada
        combined_w = x_max - x_min
        combined_h = y_max - y_min

        # Retorne a nova bounding box combinada no formato (x_min, y_min,
        # combined_w, combined_h)
        self.bbox = np.array([x_min, y_min, combined_w, combined_h])
