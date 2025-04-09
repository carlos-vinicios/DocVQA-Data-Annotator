import numpy as np
from utils.enums import PromptRoleType, BBOX

class PromptData():

    def __init__(self, role: PromptRoleType, text: str):
        self.role = role
        self.text = text

    def get_role(self) -> str:
        return self.role

    def get_text(self) -> str:
        return self.text

    def replace_marker(self, key, value):
        """Atualiza a string do prompt com os dados externos"""
        self.text = self.text.replace(key, value)

class PromptResponse():

    def __init__(self, question: str, answer: str, region: str):
        self.question = question
        self.answer = answer
        self.region = region
        self.base_text = ""
        self.question_bboxes = []
        self.answer_bboxes = []
    
    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "region": self.region,
            "text": self.base_text,
            "answer_bboxes": [[int(v) for v in bboxes] for bboxes in self.answer_bboxes]
        }
    
    def add_answer_bbox(self, bbox):
        if bbox is not None:
            if type(bbox) == list:
                self.answer_bboxes += bbox
            else:
                self.answer_bboxes.append(bbox)
    
    def add_base_text(self, text):
        self.base_text = text

    def normalize_bboxes(self, page_size: tuple):
        normalized_bboxes = []
        width, height = page_size
        for bbox in self.answer_bboxes:
            normalized_bbox = np.array([
                bbox[BBOX.X.value] / width,
                bbox[BBOX.Y.value] / height,
                bbox[BBOX.W.value] / width,
                bbox[BBOX.H.value] / height
            ])
            normalized_bboxes.append(normalized_bbox)
        
        return normalized_bboxes
