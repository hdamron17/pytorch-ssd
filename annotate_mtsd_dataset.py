import sys
from tqdm import tqdm
import cv2
import numpy as np

from torch.utils.data import DataLoader

from vision.datasets.mtsd_dataset import MapillaryTrafficSignsDataset

if len(sys.argv) < 4:
    print('Usage: python annotate_mtsd_dataset.py <dataset path> <split path> <label path>')
    sys.exit(0)
dataset_path = sys.argv[1]
split_path = sys.argv[2]
label_path = sys.argv[3]

class_names = [name.strip() for name in open(label_path).readlines()]

dataset = MapillaryTrafficSignsDataset(dataset_path, split_file=split_path)

for i in tqdm(range(len(dataset))):
    id = dataset.ids[i]
    orig_image, boxes, labels = dataset[i]
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)  # Convert back

    for i, (box, label) in enumerate(zip(boxes, labels)):
        box = box.astype(np.int32)
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 8)
        text = f"{class_names[labels[i]]}"
        cv2.putText(orig_image, text,
                    (box[0] + 20, box[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,  # font scale
                    (255, 0, 255),
                    8)  # line type
    path = f"models/annotated/{id}.jpg"
    cv2.imwrite(path, orig_image)
    print(f"Found {boxes.shape[0]} objects. The output image is {path}")
