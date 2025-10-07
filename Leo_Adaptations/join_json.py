import os
import re
import glob
import json
import pandas as pd
from tqdm import tqdm


def sort_by_video_id(paths: list[str]) -> list[str]:
    """
    Ordena una lista de paths por el número de video_id extraído del string.
    
    Args:
        paths (list[str]): lista de rutas como strings.
    
    Returns:
        list[str]: lista ordenada por video_id.
    """
    def extract_id(path: str) -> int:
        match = re.search(r"video_(\d+)", path)
        return int(match.group(1)) if match else -1  # -1 si no encuentra nada

    return sorted(paths, key=extract_id)



def load_json(path: str) -> dict:
    """
    Carga un archivo JSON desde disco.
    
    Args:
        path (str): ruta del archivo JSON.
    
    Returns:
        dict: contenido del JSON como diccionario.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: str, indent: int = 4) -> None:
    """
    Guarda un diccionario en un archivo JSON.
    
    Args:
        data (dict): diccionario a guardar.
        path (str): ruta del archivo de salida.
        indent (int): número de espacios para indentar el JSON (default=4).
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def fix_image_ids(data, last_img_id, last_annots_id):

    df_images = pd.DataFrame(data['images'])
    df_annots = pd.DataFrame(data['annotations'])

    mapping_img_id = {}
    for idx, img_original_id in enumerate(df_images['id']):
        new_id = idx + last_img_id + 1
        mapping_img_id[img_original_id] = new_id
    df_images['id'] = df_images["id"].apply(lambda x: mapping_img_id[x])
    df_annots["image_id"] = df_annots["image_id"].apply(lambda x: mapping_img_id[x])

    mapping_annots_id = {}
    for idx, annot_original_id in enumerate(df_annots['id']):
        new_id = idx + last_annots_id + 1
        mapping_annots_id[annot_original_id] = new_id
    df_annots["id"] = df_annots["id"].apply(lambda x: mapping_annots_id[x])

    list_images = df_images.to_dict(orient="records")
    list_annots = df_annots.to_dict(orient="records")

    return {'images': list_images, 'annotations': list_annots, 'categories': data['categories']}

def join_json_files(list_json: list, output_path="complete_extended_annots.json"):
    all_images = []
    all_annots = []
    categories = None  # se toma de uno de los archivos
    img_id = 0
    annots_id = 0

    for json_file in tqdm(list_json):
        data = load_json(json_file)
        data_id_fixed = fix_image_ids(data, img_id, annots_id)
        all_images.extend(data_id_fixed['images'])
        all_annots.extend(data_id_fixed['annotations'])
        img_id += len(data_id_fixed['images'])
        annots_id += len(data_id_fixed['annotations'])
        
        # categories siempre es igual en COCO, lo tomamos del primero
        if categories is None:
            categories = data_id_fixed['categories']
    
    all_info = {
        'images': all_images,
        'annotations': all_annots,
        'categories': categories
    }

    save_json(all_info, output_path)


def filter_paths_by_videos(paths, videos):
    """
    Filtra los paths que pertenecen a los videos en `videos`.

    Args:
        paths (list[str]): lista de rutas como '.../video_695/extend_annotations.json'
        videos (list[str]): lista de videos como ['video_003', 'video_567']

    Returns:
        list[str]: lista filtrada con solo los paths que corresponden a `videos`.
    """
    return [p for p in paths if any(v in p for v in videos)]



if __name__ == '__main__':

    all_json_files = glob.glob('extend_annotations/**/*.json')
    all_json_files_sorted = sort_by_video_id(all_json_files)

    # Divide the json paths depending on fold splits
    fold1_splits_data = load_json('Dataset/Splits_partition/fold1_video_splits.json')
    fold2_splits_data = load_json('Dataset/Splits_partition/fold2_video_splits.json')

    os.makedirs('Extended_annots', exist_ok=True)
    fold1_train_json_files_sorted = filter_paths_by_videos(all_json_files_sorted, fold1_splits_data['train'])
    fold1_test_json_files_sorted = filter_paths_by_videos(all_json_files_sorted, fold1_splits_data['test'])

    fold2_train_json_files_sorted = filter_paths_by_videos(all_json_files_sorted, fold2_splits_data['train'])
    fold2_test_json_files_sorted = filter_paths_by_videos(all_json_files_sorted, fold2_splits_data['test'])
    print('Complete data json')
    join_json_files(all_json_files_sorted, 'Extended_annots/complete_extended_annots.json')
    print('Fold1 train data json')
    join_json_files(fold1_train_json_files_sorted, 'Extended_annots/fold1_train_extended_annots.json')
    print('Fold1 test data json')
    join_json_files(fold1_test_json_files_sorted, 'Extended_annots/fold1_test_extended_annots.json')
    print('Fold2 train data json')
    join_json_files(fold2_train_json_files_sorted, 'Extended_annots/fold2_train_extended_annots.json')
    print('Fold2 test data json')
    join_json_files(fold2_test_json_files_sorted, 'Extended_annots/fold2_test_extended_annots.json')



