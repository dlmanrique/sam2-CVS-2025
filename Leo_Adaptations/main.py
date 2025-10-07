# Main file for SAM2 annotations extension
import os
import cv2
import json
import random
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils
from sam2.build_sam import build_sam2_video_predictor



def visualize_and_save_sam2(image_path, mask_logits, bboxes, output_path="output.png", colors=None, alpha=0.5):
    """
    Visualiza y guarda las máscaras de segmentación de SAM2 sobre una imagen con sus bounding boxes.

    Args:
        image_path (str): Ruta de la imagen original.
        mask_logits (list or tensor): Lista de tensores de máscaras (salida de SAM2).
        bboxes (list): Lista de bounding boxes [x, y, w, h].
        output_path (str): Ruta para guardar la imagen resultante.
        colors (list): Lista de colores RGB (ej: [[255,0,0],[0,255,0]]) para cada máscara.
        alpha (float): Transparencia de las máscaras (0.0 a 1.0).
    """
    # Leer imagen
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Crear una copia para dibujar
    overlay = image.copy()

    for idx, (logit, box) in enumerate(zip(mask_logits, bboxes)):
        # Convertir logits a máscara binaria
        if isinstance(logit, torch.Tensor):
            mask = (logit > 0).cpu().numpy().astype(np.uint8)
        else:
            mask = (logit > 0).astype(np.uint8)

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

        # Dibujar bounding box si existe
        if box is not None:
            try:
                x0, y0, x1, y1 = map(int, box)
                cv2.rectangle(
                    overlay,
                    (x0, y0),
                    (x1, y1),  # esquina opuesta
                    color=color,
                    thickness=2
                )
            except Exception as e:
                pass
        else:
            pass

    # Convertir a uint8
    overlay = overlay.astype(np.uint8)

    # Guardar resultado
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.imsave(output_path, overlay)



def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data



def process_json_data(train_data: dict, test_data: dict):
    complete_info = []

    df_train = pd.DataFrame(train_data['annotations'])
    for img_info in train_data['images']:
        file_name = img_info['file_name']
        img_id = img_info['id']
        video_id = img_info['video_id']

        df_image_masks = df_train[df_train['image_id'] == img_id]
        df_image_masks = df_image_masks.drop(columns=['id', 'image_id'])
        masks_info_list = df_image_masks.to_dict(orient="records")

        complete_info.append({'file_name': file_name,
                              'image_id': img_id,
                              'video_id': video_id,
                               'segmentations_list': masks_info_list})



    df_test = pd.DataFrame(test_data['annotations'])
    for img_info in test_data['images']:
        file_name = img_info['file_name']
        img_id = img_info['id']
        video_id = img_info['video_id']

        df_image_masks = df_test[df_test['image_id'] == img_id]
        df_image_masks = df_image_masks.drop(columns=['id', 'image_id'])
        masks_info_list = df_image_masks.to_dict(orient="records")

        complete_info.append({'file_name': file_name,
                              'image_id': img_id,
                              'video_id': video_id,
                               'segmentations_list': masks_info_list})

    return complete_info


def xywh_to_xyxy(bbox):
    """
    Convierte una caja del formato [x, y, w, h] 
    al formato [x_min, y_min, x_max, y_max].
    """
    x, y, w, h = bbox
    return [x, y, x + w, y + h]

def obtain_bboxes(info: dict):
    bboxes = []
    categories = []
    for seg_info in info['segmentations_list']:
        bbox = seg_info['bbox']
        category = seg_info['category_id']
        bboxes.append(xywh_to_xyxy(bbox))
        categories.append(category)

    return bboxes, categories


def mask_to_bbox(mask):
    """
    Convierte un mask binario de SAM a bbox en formato [x, y, w, h].
    """
    # Encuentra las posiciones donde hay píxeles "True" (o >0)
    mask = np.squeeze(mask)
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None  # no hay objeto

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()


    return [int(x_min), int(y_min), int(x_max), int(y_max)]




def rle_list_to_masks(rle_list):
    """
    Convierte una lista de dicts con RLEs (COCO-style)
    en una lista de máscaras binarias (H, W).
    
    Args:
        rle_list (list[dict]): lista de RLEs (cada uno en formato COCO).
        
    Returns:
        masks (list[np.ndarray]): lista de máscaras binarias (H, W).
    """

    masks = []
    categories = []
    object_ids = []
    count_instances = {1: 1, 2: 1, 3:1, 4: 1, 5:1, 6:1}
    M = 10 #maximo de instancias para una clase
    for rle in rle_list['segmentations_list']:
        mask = mask_utils.decode(rle['segmentation'])  # (H, W, 1) o (H, W)
        if mask.ndim == 3:             # a veces viene con última dim = 1
            mask = mask[..., 0]
        mask = mask.astype(bool)       # asegurar formato booleano
        masks.append(mask)
        categories.append(rle['category_id'])
        object_id_instance = (rle['category_id'] - 1) * M + count_instances[rle['category_id']]
        object_ids.append(object_id_instance)
        count_instances[rle['category_id']] += 1


    return masks, object_ids, categories 


def return_linear_transform(num: int, M: int = 10):
    temp = num - 1
    category_id = temp // M + 1
    return category_id



def sam_predictions_to_coco(masks, image_id, category_ids=None, start_ann_id=1):
    """
    Convierte predicciones de SAM en anotaciones estilo COCO.

    Args:
        masks (list[Tensor o np.ndarray]): Lista de máscaras binarias o logits (>0 se considera foreground).
        image_id (int): ID de la imagen.
        category_ids (list[int]): Lista con el category_id de cada máscara.
        start_ann_id (int): ID inicial para las anotaciones.

    Returns:
        list[dict]: lista de anotaciones estilo COCO.
    """
    annotations = []
    ann_id = start_ann_id

    for idx, mask_pred in enumerate(masks):
        # Tensor -> numpy binario
        if isinstance(mask_pred, torch.Tensor):
            mask = (mask_pred > 0).cpu().numpy().astype(np.uint8)
        else:
            mask = (mask_pred > 0).astype(np.uint8)

        # Asegurar 2D
        if mask.ndim == 3:
            mask = mask[0]

        # RLE encoding
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("utf-8")  # JSON friendly

        # Área y bbox
        area = int(mask_utils.area(rle))
        bbox = mask_utils.toBbox(rle).tolist()  # [x, y, w, h]

        # Category ID
        category_id = category_ids[idx] if category_ids is not None else 1

        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "segmentation": rle,
            "iscrowd": 0,
            "bbox": bbox,
            "area": area,
            "category_id": category_id
        })

        ann_id += 1

    return annotations


def parse_args():
    parser = argparse.ArgumentParser(description="Video selection")
    # Argumentos obligatorios
    parser.add_argument('--initial_video', type=int, default=0)
    parser.add_argument('--final_video', type=int, default=700)
    return parser.parse_args()




if __name__ == '__main__':

    args = parse_args()
    print('Videos a procesar: ')
    print(f'Initial Video: {args.initial_video}')
    print(f'Final Video: {args.final_video}')



    categories_key =  [
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
        }
    ]

    categories_dict = { 1:[248,231,28], 2: [74, 144, 226], 3: [218, 13, 15],
                       4: [65, 117, 6], 5: [126, 211, 33], 6: [245, 166, 35]}
        
    annots_path_train = 'Dataset/segmentation_labels/fold1/train_annotation_coco.json'
    annots_path_test = 'Dataset/segmentation_labels/fold1/test_annotation_coco.json'
    train_data = load_json(annots_path_train)
    test_data = load_json(annots_path_test)

    complete_info = process_json_data(train_data, test_data)
    complete_info = sorted(complete_info, key=lambda x: x["file_name"])

    device = torch.device("cuda")

    sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    ann_id = 1  # contador global para anotaciones
    image_id_count = 1
    #breakpoint()
    for info in complete_info[args.initial_video: args.final_video + 1]:
        video_dir = info['video_id']
        frame_names = [
            p for p in os.listdir(f'Dataset/frames_SAM/{video_dir}')
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        inference_state = predictor.init_state(video_path=f'Dataset/frames_SAM/{video_dir}')
        ann_frame_idx = 75 # Mi key_frame siempre va a estar en esta posicion
        bboxes, _ = obtain_bboxes(info)
        colors = []
        
        initial_masks, object_ids, categories = rle_list_to_masks(info)

        for i in range(len(categories)):
            _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_ids[i],
                mask=initial_masks[i],
            )
            colors.append(categories_dict[categories[i]])

        image_path = os.path.join('Dataset', 'frames_SAM', video_dir, frame_names[ann_frame_idx])
        

        print('Predicciones hacia el futuro')
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state, start_frame_idx=ann_frame_idx, reverse=False):
            video_segments[out_frame_idx] = {
                out_obj_id: out_mask_logits[i] for i, out_obj_id in enumerate(out_obj_ids)
            }

        print('Predicciones hacia el pasado')
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state, start_frame_idx=ann_frame_idx, reverse=True):
            video_segments[out_frame_idx] = {
                out_obj_id: out_mask_logits[i] for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Folder for saving all the json
        os.makedirs(f'extend_annotations/{video_dir}', exist_ok=True)

        all_video_annotations = []
        all_image_info = []

        ordered_video_segments = {0: video_segments[0]}
        for k in sorted(video_segments.keys()):
            if k != 0:
                ordered_video_segments[k] = video_segments[k]


        for frame_idx, obj_dict in ordered_video_segments.items():
            # Ruta original del frame
            original_frame_path = os.path.realpath(f"Dataset/frames_SAM/{video_dir}/{frame_names[frame_idx]}")

            # ID de la imagen (puedes usar frame_idx o algo más robusto)
            image_id = image_id_count  
            image_id_count += 1

            # Images info
            image_info = {
                "file_name": '/'.join(original_frame_path.split('/')[-2:]),
                "height": info['segmentations_list'][0]['segmentation']['size'][0],
                "width": info['segmentations_list'][0]['segmentation']['size'][1],
                "id": image_id,
                "video_name": video_dir,
                "frame_id": '',
                "is_det_keyframe": True,
                "ds": [],
                "video_id": video_dir,
                "is_ds_keyframe": True
            }

            all_image_info.append(image_info)
            # Prepara máscaras y categorías para esta imagen
            masks = []
            category_ids = []
            for obj_id, logit in obj_dict.items():
                # Convertir a binaria
                if isinstance(logit, torch.Tensor):
                    mask = (logit > 0).cpu().numpy().astype(np.uint8)
                else:
                    mask = (logit > 0).astype(np.uint8)

                if mask.ndim == 3:
                    mask = mask[0]

                if mask is not None and mask.any():
                    masks.append(mask)
                    category_ids.append(return_linear_transform(obj_id))

            # Obtener anotaciones con la nueva función
            annotations = sam_predictions_to_coco(
                masks=masks,
                image_id=image_id,
                category_ids=category_ids,
                start_ann_id=ann_id
            )

            # Actualizar contador global
            ann_id += len(annotations)

            # Guardar
            all_video_annotations.extend(annotations)

        
        # Guardar como JSON
        with open(f"extend_annotations/{video_dir}/extend_annotations.json", "w") as f:
            json.dump({'images': all_image_info, 'annotations': all_video_annotations, 'categories': categories_key}, f, indent=4)
        
        print(f'Numero de imagenes: {len(all_image_info)} \n Numero de anotaciones: {len(all_video_annotations)}' )

        # render the segmentation results every few frames
        vis_frame_stride = 3
        plt.close("all")
        
        os.makedirs(f'visualizations/{video_dir}', exist_ok=True)
        
        print('Creando visualizaciones ... ')
        for out_frame_idx in tqdm(range(0, len(frame_names), vis_frame_stride)):
            image_path = os.path.join('Dataset', 'frames_SAM', video_dir, frame_names[out_frame_idx])

            # organizamos logits y colores para visualize_and_save_sam2
            out_obj_ids = list(video_segments[out_frame_idx].keys())
            out_mask_logits = [video_segments[out_frame_idx][oid] for oid in out_obj_ids]
            out_colors = [categories_dict[return_linear_transform(oid)] for oid in out_obj_ids]  # usa mismo mapeo


            # calculamos bboxes a partir de las máscaras
            out_bboxes = []
            for logit in out_mask_logits:
                mask = (logit > 0)  # binariza
                bbox = mask_to_bbox(mask.cpu().numpy())
                out_bboxes.append(bbox)

            
            visualize_and_save_sam2(
                image_path=image_path,
                mask_logits=out_mask_logits,
                bboxes=out_bboxes,  # si no quieres bboxes durante propagación
                output_path=f"visualizations/{video_dir}/prueba_{out_frame_idx}.png",
                colors=out_colors
            )

        print(f'Videeo {video_dir} procesado con exito')