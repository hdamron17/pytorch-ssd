import json
import pathlib
from copy import copy
import numpy as np
import cv2

from . import mtsd_default_classes as dflt

class MapillaryTrafficSignsDataset:
    def __init__(self, root,
                 transform=None, target_transform=None,
                 dataset_type="train",
                 categories=dflt.CATEGORIES, convert_label=dflt.convert_label,
                 split_file=None):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type.lower()

        self.class_names = categories
        self.convert_label = convert_label

        if split_file is None:
            # split_file = f"{self.root}/splits/{self.dataset_type}.txt"
            split_file = f"models/{self.dataset_type}_relevant.txt"
        with open(split_file, 'r') as f:
            self.ids = list(map(str.strip, f.readlines()))

    def _getitem(self, index):
        id = self.ids[index]
        json_fname = self.root / "annotations" / f"{id}.json"
        with open(json_fname) as json_file:
            data = json.load(json_file)

        boxes = []
        labels = []

        for obj in data['objects']:
            label = self.convert_label(obj['label'])
            if label is None:
                continue
            b = copy(obj['bbox'])
            boxes.append(np.array([b['xmin'], b['ymin'], b['xmax'], b['ymax']], dtype=np.float32))
            labels.append(copy(label))

        boxes = np.array(boxes)
        labels = np.array(labels, dtype=np.int64)

        image = self._read_image(id)

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def __getitem__(self, index):
        image, boxes, labels = self._getitem(index)
        return image, boxes, labels

    def get_annotation(self, index):
        """To conform the eval_ssd implementation that is based on the VOC dataset."""
        _, boxes, labels = self._getitem(index)
        is_difficult = np.zeros(boxes.shape[0], dtype=np.uint8)
        return self.ids[index], (boxes, labels, is_difficult)

    def get_image(self, index):
        image = self._read_image(self.ids[index])
        if self.transform:
            image, _ = self.transform(image)
        return image

    def __len__(self):
        return len(self.ids)

    def _read_image(self, image_id):
        image_file = self.root / "images" / f"{image_id}.jpg"
        image = cv2.imread(str(image_file))
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
