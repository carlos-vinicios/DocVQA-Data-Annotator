import numpy as np

class Text():

  def __init__(self, text, true_bbox: np.array, relative_bbox: np.array):
    self.text = text
    self.true_bbox = true_bbox.round()
    self.relative_bbox = relative_bbox.round()

  def pad_relative_bbox(self, value: int):
    """Adiciona um espaçamento na bounding box relativa
    a região da imagem que o texto se encontra"""
    self.relative_bbox[0] += value
    self.relative_bbox[1] += value