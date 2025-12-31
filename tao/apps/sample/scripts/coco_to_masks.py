import json
import os
import random
import copy
import cv2
import shutil
import numpy as np
from tqdm import tqdm


filename_coco = "../data/coco/result_coco.json"
images_coco_filepath = "../data/coco/images"
mask_filepath = "../data/dataset"

split_config = {
    "train": {"ratio": 0.7, "sub_dir_images": "images/train", "sub_dir_masks": "masks/train"},
    "val": {"ratio": 0.15, "sub_dir_images": "images/val", "sub_dir_masks": "masks/val"},
    "test": {"ratio": 0.15, "sub_dir_images": "images/test", "sub_dir_masks": "masks/test"}
}


coco_dict = None

with open(filename_coco) as f:
    coco_dict = json.load(f)

if(coco_dict is None):
    print("COCO annotation is none")
    exit


# =========================
# Train / Val / Test split
# =========================

fileimages = coco_dict["images"]

for fileimage in fileimages:
    file_name = fileimage["file_name"]
    fileimage["path"] = f"{images_coco_filepath }/{file_name}"

image_ids = [img["id"] for img in fileimages]
random.shuffle(image_ids)

n_total = len(image_ids)
n_train = int(n_total * split_config["train"]["ratio"])
n_val = int(n_total * split_config["val"]["ratio"])

train_ids = set(image_ids[:n_train])
val_ids = set(image_ids[n_train:n_train + n_val])
test_ids = set(image_ids[n_train + n_val:])

splits = {
    "train": train_ids,
    "val": val_ids,
    "test": test_ids
}

def get_annotations_for_image(image_id, annotations):
    return [ann for ann in annotations if ann["image_id"] == image_id]

for split_name, ids in splits.items():
    # Creamos las rutas
    img_dest_dir = os.path.join(mask_filepath, split_config[split_name]["sub_dir_images"])
    mask_dest_dir = os.path.join(mask_filepath, split_config[split_name]["sub_dir_masks"])
    
    os.makedirs(img_dest_dir, exist_ok=True)
    os.makedirs(mask_dest_dir, exist_ok=True)

    # Añadimos la barra de progreso aquí
    # desc: Texto que aparece a la izquierda
    # unit: La unidad que está procesando
    pbar = tqdm(ids, desc=f"Procesando {split_name}", unit="img")

    for img_id in pbar:
        # 1. Buscar info de la imagen
        img_info = next(img for img in fileimages if img["id"] == img_id)
        img_name = img_info["file_name"]
        h, w = img_info["height"], img_info["width"]
        
        # 2. Copiar la imagen original
        source_path = img_info["path"]
        target_path = os.path.join(img_dest_dir, img_name)
        if os.path.exists(source_path):
            shutil.copy(source_path, target_path)
        
        # 3. Generar la máscara
        mask = np.zeros((h, w), dtype=np.uint8)
        img_anns = get_annotations_for_image(img_id, coco_dict["annotations"])
        
        
        for ann in img_anns:
            class_id = ann["category_id"]
            for seg in ann["segmentation"]:
                # Manejo de seguridad por si la segmentación no es una lista
                if isinstance(seg, list):
                    poly = np.array(seg, dtype=np.int32).reshape((-1, 2))
                    cv2.fillPoly(mask, [poly], int(class_id))

        # 4. Guardar la máscara
        mask_name = os.path.splitext(img_name)[0] + ".png"
        cv2.imwrite(os.path.join(mask_dest_dir, mask_name), mask)
        
        # Opcional: puedes actualizar el texto de la barra con el nombre del archivo actual
        pbar.set_postfix(file=img_name[:15])

print("\n¡Proceso finalizado! Los datos están organizados en:", mask_filepath)