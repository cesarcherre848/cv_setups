import json
import os
import random
import copy


filename = "../data/annotations/result_coco.json"
base_path = "data/images/"

split_config = {
    "train": {"ratio": 0.7, "dir": "../data/annotations/train"},
    "val": {"ratio": 0.15, "dir": "../data/annotations/val"},
    "test": {"ratio": 0.15, "dir": "../data/annotations/test"}
}

coco_dic = None

with open(filename) as f:
    coco_dic = json.load(f)

if(coco_dic is None):
    print("COCO annotation is none")
    exit


fileimages = coco_dic["images"]

for fileimage in fileimages:
    file_name = fileimage["file_name"]
    fileimage["path"] = f"{base_path}{file_name}"

# =========================
# Train / Val / Test split
# =========================
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

# =========================
# Generate split COCO files
# =========================
for split_name, id_set in splits.items():

    split_coco = {
        "info": coco_dic.get("info", {}),
        "licenses": coco_dic.get("licenses", []),
        "categories": coco_dic["categories"],
        "images": [],
        "annotations": []
    }

    split_coco["images"] = [
        img for img in coco_dic["images"] if img["id"] in id_set
    ]

    split_coco["annotations"] = [
        ann for ann in coco_dic["annotations"] if ann["image_id"] in id_set
    ]

    os.makedirs(split_config[split_name]["dir"], exist_ok=True)

    out_file = os.path.join(
        split_config[split_name]["dir"],
        f"instances_{split_name}.json"
    )

    with open(out_file, "w") as f:
        json.dump(split_coco, f, indent=2)

    print(f"{split_name}: {len(split_coco['images'])} images, "
          f"{len(split_coco['annotations'])} annotations")
