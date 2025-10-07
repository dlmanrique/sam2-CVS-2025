import os
import json
import pandas as pd
from tqdm import tqdm

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


if __name__ == '__main__':

    annots_path_train = 'Dataset/segmentation_labels/fold1/train_annotation_coco.json'
    annots_path_test = 'Dataset/segmentation_labels/fold1/test_annotation_coco.json'
    train_data = load_json(annots_path_train)
    test_data = load_json(annots_path_test)

    complete_info = process_json_data(train_data, test_data)

    
    for info in tqdm(complete_info):
        video_dir = info['video_id']
        os.makedirs(f'/media/SSD3/leoshared/Dataset/frames_SAM/{video_dir}', exist_ok=True)

        file_name = info['file_name']
        file_id = int(info['file_name'].split('/')[-1][:-4])
        ids_range = list(range(file_id-75, file_id+76))

        count_frames = 0
        for idx in ids_range:
            complete_original_path = f'/media/SSD3/leoshared/Dataset/frames/{video_dir}/{str(idx).zfill(5)}.jpg'
            new_path = f'/media/SSD3/leoshared/Dataset/frames_SAM/{video_dir}/{str(count_frames).zfill(5)}.jpg'

            if not os.path.exists(new_path):  # Evitar sobreescrituras
                os.symlink(complete_original_path, new_path)
                count_frames += 1

    





