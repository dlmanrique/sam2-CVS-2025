# This file plots the data un the extended jsons for a desired video

import json
import os
import cv2
import pandas as pd
from PIL import Image
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils
import torch
from matplotlib.patches import Rectangle


colors = { 1:[248,231,28], 2: [74, 144, 226], 3: [218, 13, 15],
                       4: [65, 117, 6], 5: [126, 211, 33], 6: [245, 166, 35]}

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def rle_to_tensor(rle, dtype=torch.float32, device="cpu"):
    """
    Convierte una máscara RLE a un tensor de PyTorch.

    Args:
        rle (dict): codificación RLE en formato COCO.
        dtype (torch.dtype): tipo de dato del tensor.
        device (str): dispositivo donde colocar el tensor ("cpu" o "cuda").

    Returns:
        torch.Tensor: máscara binaria (H, W).
    """
    # Decodificar RLE a numpy (uint8, con 0 y 1)
    mask = mask_utils.decode(rle)

    # Convertir a tensor
    mask_tensor = torch.as_tensor(mask, dtype=dtype, device=device)

    return mask_tensor


def create_visualizations_coco(img_files, annots_masks, output_dir="viz_masks"):
    """
    Visualiza y guarda imágenes con sus máscaras superpuestas.

    Args:
        img_files (list[str]): lista de nombres de imágenes.
        annots_masks (list[list[dict]]): lista de anotaciones por imagen.
        output_dir (str): carpeta donde guardar las visualizaciones.
    """
    colors = { 1:[248,231,28], 2: [74, 144, 226], 3: [218, 13, 15],
                       4: [65, 117, 6], 5: [126, 211, 33], 6: [245, 166, 35]}

    for img_name, annot_mask in zip(img_files, annots_masks):
        # Cargar imagen original
        complete_img_name = f"Dataset/frames/{img_name}"
        img = np.array(Image.open(complete_img_name).convert("RGB"))

        # Crear figura
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        ax.axis("off")

        # Superponer máscaras
        for annot in annot_mask:
            mask = rle_to_tensor(annot["segmentation"]).cpu().numpy()

            # Crear un color aleatorio (RGB con transparencia)
            color = colors[annot['category_id']]
            ax.imshow(
                np.dstack([mask * color[0]/255, mask * color[1]/255, mask * color[2]/255, mask * 0.5])
            )

            # Bounding box (si viene en el dict, usarlo; si no, calcular del mask)
            if "bbox" in annot:
                x, y, w, h = annot["bbox"]
            else:
                ys, xs = np.where(mask > 0)
                if len(xs) > 0 and len(ys) > 0:
                    x, y = xs.min(), ys.min()
                    w, h = xs.max() - xs.min(), ys.max() - ys.min()
                else:
                    continue  # máscara vacía → no dibujar

            rect = Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor=np.array(color) / 255,
                facecolor="none"
            )
            ax.add_patch(rect)

        breakpoint()
        # Guardar resultado
        os.makedirs(f'{output_dir}/{img_name.split("/")[0]}', exist_ok=True)
        save_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_masks.png")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        print(f"✅ Guardado: {save_path}")
        




data = load_json('extend_annotations/video_005/extend_annotations.json')

images_names = []
img_ids = []
for info in data['images']:
    images_names.append(info['file_name'])
    img_ids.append(info['id'])

annot_mask = []
df_annots = pd.DataFrame(data['annotations'])
for img_idx in img_ids:
    df_img = df_annots[df_annots['image_id'] == img_idx]
    lista_annots_masks = df_img.to_dict(orient="records")
    annot_mask.append(lista_annots_masks)


create_visualizations_coco(images_names, annot_mask)
