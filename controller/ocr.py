
import json
import pandas as pd
import numpy as np

from utils.bboxes import intersect_box, same_line, group_ocr_lines
from utils.excepts import InvalidOCRConfig, InvalidAzureOCRConfig
from model.text import Text

class OcrData():

    def __init__(self, ocr_data_path: str, configs: dict):
        """
        As configs devem conter o tipo de extrator de texto utilizado:
            - model: Modelo extrator utilizado para gerar o arquivo de OCR (Azure | TextExtract | Tesseract)
            - page: Para o caso de modelos que salvam multiplas páginas em um só JSON. A página deve ser informada
                    em contagem padrão (1 - tamanho do arquivo).
        """
        #TODO: Vamos configurar para que todos os extratores utilizam o padrão de somente um arquivo para extração de texto
        self.__validate_configs(configs)

        self.configs = configs
        self.ocr_data = self.__load_data(ocr_data_path)

        if configs["model"] == "Azure":
            self.text_data = self.__convert_Azure_text_bbox()
        if configs["model"] == "TextExtract":
            self.text_data = self.__convert_AWS_text_bbox()
        elif configs["model"] == "Tesseract":
            self.text_data = self.__convert_Tesseract_text_bbox()
            self.text_data = self.__build_lines_from_single_words()
    
    def __validate_configs(self, configs):
        if "model" not in configs:
            raise InvalidOCRConfig("A configuração para carregamento do arquivo de OCR é inválida.")

        if "model" == "Azure" and "page" not in configs:
            #TODO: no futuro essa configuração é invalida
            raise InvalidAzureOCRConfig("A configuração para carregamento do arquivo do Azure é inválida.")

    def __load_data(self, ocr_data_path: str) -> dict:
        """Carrega os dados do arquivo vindo do TextExtract"""
        with open(ocr_data_path) as data_file:
            ocr_data = json.load(data_file)
        return ocr_data

    def __convert_AWS_text_bbox(self):
        """Carrega e organiza todas as bounding boxes de textos
        presentes no arquivo de saida do TextExtract e converte
        para um DataFrame do Pandas."""
        text_data = []

        for block in self.ocr_data['Blocks']:
            if block['BlockType'] == 'LINE':
                x = block['Geometry']['BoundingBox']['Left']
                y = block['Geometry']['BoundingBox']['Top']
                w = block['Geometry']['BoundingBox']['Width']
                h = block['Geometry']['BoundingBox']['Height']

                text_data.append({
                    "text": block['Text'].strip(),
                    "x": x, "y": y, "w": w, "h": h
                })
        return pd.DataFrame(text_data)

    def __convert_Azure_text_bbox(self):
        """Carrega e organiza todas as bounding boxes de textos
        presentes no arquivo de saida do TextExtract e converte
        para um DataFrame do Pandas."""
        text_data = []
        for line in self.ocr_data['pages'][self.configs["page"]-1]['lines']:
            x, y, w, h = tuple(line['bbox'])
            
            text_data.append({
                "text": line['text'].strip(),
                "x": x, "y": y, "w": w, "h": h
            })
        return pd.DataFrame(text_data)

    def __convert_Tesseract_text_bbox(self):
        """Carrega e organiza todas as bounding boxes de textos
        presentes no arquivo de saida do Tesseract e converte
        para um DataFrame do Pandas."""
        text_data = []

        for bbox, text in zip(self.ocr_data['boxes'], self.ocr_data['texts']):
            if len(text.strip()) == 0:
                #limpando detecções vazias do Tesseract
                continue
            
            x, y, w, h = tuple(bbox)
            text_data.append({
                "text": text.strip(),
                "x": x, "y": y, "w": w, "h": h
            })
        
        return pd.DataFrame(text_data)

    def __build_lines_from_single_words(self) -> pd.DataFrame:
        """Transforma o resultado de OCR (single words) para o formato de linhas, baseado
        em uma tolerância vertical e horizontal para agrupamento dos valores."""
        df = self.text_data.copy()
        
        # Identificar grupos verticais
        y_tolerance = 0.005  # Tolerância para considerar que encontra-se na mesma linha
        df["line_group"] = (df["y"].diff().abs() > y_tolerance).cumsum()
        result = df.groupby("line_group").apply(group_ocr_lines).reset_index(drop=True)
        return result
    
    def calc_relative_bbox(self, roi_bbox: np.array, text_bbox: np.array) -> np.array:
        """Calcula a bounding boxes relativa a uma região de interesse
        informada como parâmetro"""
        return np.array([
            text_bbox[0] - roi_bbox[0],
            text_bbox[1] - roi_bbox[1],
            text_bbox[2], text_bbox[3]
        ])

    def __sort_elements(self, elements: list):
        # Ordena as bounding boxes pelo elemento mais à esquerda e mais ao topo
        line_group = []
        for elem in elements:
            inserted = False
            for group in line_group:
                if same_line(elem.true_bbox, group[0].true_bbox):
                    group.append(elem)
                    inserted = True
                    break
            if not inserted:
                line_group.append([elem])

        line_group = [
            sorted(group, key=lambda elem: (elem.true_bbox[0], elem.true_bbox[1])) 
                for group in line_group
        ]
        #transformando a lista em um dimensão e retornando
        return [x for xs in line_group for x in xs] 

    def text_in_roi(self, page_size: tuple, roi_bbox: np.array, sort_elements=False) -> list[Text]:
        """Extrai todas as caixas de texto presente em uma região de interesse."""
        texts = []
        width, height = page_size
        t_data = self.text_data.copy()
        t_data['x'] = round(self.text_data['x'] * width, 2)
        t_data['w'] = round(self.text_data['w'] * width, 2)
        t_data['y'] = round(self.text_data['y'] * height, 2)
        t_data['h'] = round(self.text_data['h'] * height, 2)

        for row in t_data.itertuples():
            true_bbox = np.array([row.x, row.y, row.w, row.h])        
            if intersect_box(roi_bbox, true_bbox, 0.85):
                # esta contido totalmente dentro da região de interesse
                relative_bbox = self.calc_relative_bbox(roi_bbox, true_bbox)
                t = Text(row.text, true_bbox, relative_bbox)
                texts.append(t)
        
        if sort_elements:
            texts = self.__sort_elements(texts)
        return texts
