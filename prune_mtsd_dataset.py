import vision.datasets.mtsd_default_classes as dflt
from sys import argv
import pathlib
import json

if len(argv) != 3:
    print("Usage: prune_mtsd_dataset.py <dataset> <orig split>")
    exit(1)

dataset_path = pathlib.Path(argv[1])
orig_split = pathlib.Path(argv[2])

ids = map(str.strip, open(orig_split, 'r').readlines())
for id in ids:
    anno_file = dataset_path / "annotations" / f"{id}.json"
    anno = json.load(open(anno_file, 'r'))
    labels = (obj["label"] for obj in anno["objects"])
    classes = list(filter(None, map(dflt.convert_label, labels)))
    if classes:
        print(id)
