import pandas as pd

def calculate_overlap_ratio(bbox1, bbox2):
    # Extrair coordenadas das bounding boxes
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Calcular coordenadas dos cantos dos retângulos
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    # Calcular área de interseção
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Calcular áreas das bounding boxes
    area1 = w1 * h1
    area2 = w2 * h2

    # Calcular área da menor bounding box
    min_area = min(area1, area2)

    # Calcular proporção da área de interseção em relação à área da menor bounding box
    overlap_ratio = intersection_area / min_area if min_area > 0 else 0

    return overlap_ratio

def intersect_box(bbox1, bbox2, threshold=0.6):
    return calculate_overlap_ratio(bbox1, bbox2) >= threshold

def same_line(bbox1, bbox2):
    # Intervalo vertical da primeira bounding box
    y_min_1 = bbox1[1]
    y_max_1 = bbox1[1] + bbox1[3]

    # Intervalo vertical da segunda bounding box
    y_min_2 = bbox2[1]
    y_max_2 = bbox2[1] + bbox2[3]

    # Se os intervalos verticais se sobrepõem, as bounding boxes estão na mesma linha
    return (y_min_1 <= y_max_2 and y_max_1 >= y_min_2) or (y_min_2 <= y_max_1 and y_max_2 >= y_min_1)

def group_ocr_lines(line_df: pd.Series) -> pd.DataFrame:
    # Função para agrupar horizontalmente dentro de cada linha

    x_tolerance = 0.01  # Tolerância para considerar elementos próximos em uma mesma linha    
    line_df = line_df.sort_values(by="x").reset_index(drop=True)
    grouped_texts = []
    current_group = []
    
    for i, row in line_df.iterrows():
        if current_group:
            # Verificar se o texto atual se sobrepõe ou está próximo ao anterior
            prev_row = line_df.iloc[current_group[-1]]
            if row["x"] <= prev_row["x"] + prev_row["w"] + x_tolerance:
                current_group.append(i)
            else:
                grouped_texts.append(current_group)
                current_group = [i]
        else:
            current_group = [i]
    
    if current_group:
        grouped_texts.append(current_group)
    
    # Concatenar textos para cada grupo
    concatenated_rows = []
    for group in grouped_texts:
        group_rows = line_df.iloc[group]
        new_row = {
            "text": " ".join(group_rows["text"]),
            "x": group_rows["x"].min(),
            "y": group_rows["y"].mean(),
            "w": group_rows["x"].max() + group_rows["w"].max() - group_rows["x"].min(),
            "h": group_rows["h"].max()
        }
        concatenated_rows.append(new_row)
    
    return pd.DataFrame(concatenated_rows)