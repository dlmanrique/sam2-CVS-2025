import os
import cv2
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils

from main import process_json_data


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def visualize_and_save_masks(image_path, rle_masks, output_path="output.png", colors=None, alpha=0.5):
    """
    Visualiza y guarda las máscaras de segmentación sobre una imagen con sus bounding boxes.

    Args:
        image_path (str): Ruta de la imagen original.
        rle_masks (list): Lista de máscaras en formato RLE (COCO).
        output_path (str): Ruta para guardar la imagen resultante.
        colors (list): Lista de colores RGB (ej: [[255,0,0],[0,255,0]]) para cada máscara.
        alpha (float): Transparencia de las máscaras (0.0 a 1.0).
    """
    # Leer imagen
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Crear una copia para dibujar
    overlay = image.copy()

    for idx, rle in enumerate(rle_masks):
        # Decodificar RLE a máscara binaria
        mask = maskUtils.decode(rle)

        # Obtener bounding box (x, y, w, h)
        bbox = maskUtils.toBbox(rle).astype(int)
        x, y, w, h = bbox
        print(f"📦 BBox {idx}: x={x}, y={y}, w={w}, h={h}")

        # Asignar color
        if colors is not None and idx < len(colors):
            color = colors[idx]
        else:
            color = [random.randint(0, 255) for _ in range(3)]

        # Aplicar color donde la máscara es 1
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask == 1,
                overlay[:, :, c] * (1 - alpha) + alpha * color[c],
                overlay[:, :, c]
            )

        # Dibujar bounding box en overlay
        cv2.rectangle(
            overlay,
            (x, y),
            (x + w, y + h),
            color=color,
            thickness=2
        )

    # Convertir a uint8
    overlay = overlay.astype(np.uint8)

    # Guardar resultado
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.imsave(output_path, overlay)

    print(f"✅ Imagen con máscaras y bboxes guardada en: {output_path}")





if __name__ == "__main__":
    
    
    categories_dict = {"categories": [
        {
            "id": 1,
            "name": "cystic_plate",
            "supercategory": "anatomy",
            "color": [
                248,
                231,
                28
            ]
        },
        {
            "id": 2,
            "name": "calot_triangle",
            "supercategory": "anatomy",
            "color": [
                74,
                144,
                226
            ]
        },
        {
            "id": 3,
            "name": "cystic_artery",
            "supercategory": "anatomy",
            "color": [
                218,
                13,
                15
            ]
        },
        {
            "id": 4,
            "name": "cystic_duct",
            "supercategory": "anatomy",
            "color": [
                65,
                117,
                6
            ]
        },
        {
            "id": 5,
            "name": "gallbladder",
            "supercategory": "anatomy",
            "color": [
                126,
                211,
                33
            ]
        },
        {
            "id": 6,
            "name": "tool",
            "supercategory": "tool",
            "color": [
                245,
                166,
                35
            ]
        }]}
    
    annots_path_train = 'Dataset/segmentation_labels/fold1/train_annotation_coco.json'
    annots_path_test = 'Dataset/segmentation_labels/fold1/test_annotation_coco.json'
    train_data = load_json(annots_path_train)
    test_data = load_json(annots_path_test)

    complete_info = process_json_data(train_data, test_data)
    frame_info = complete_info[0]
    frame_path = f'Dataset/frames/{frame_info["file_name"]}'
    
    rle_list = []
    colors = []

    for seg in frame_info['segmentations_list']:
        rle_list.append(seg['segmentation'])
        colors.append(categories_dict['categories'][seg['category_id']-1]['color'])
    
    visualize_and_save_masks(frame_path, rle_list, "./resultado.png", colors)

