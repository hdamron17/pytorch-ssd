import vision.datasets.mtsd_default_classes as dflt
from sys import argv
import pathlib
import json
import os
from tqdm import tqdm

if len(argv) != 4:
    print("Usage: prune_mtsd_dataset.py <dataset> <orig split> <output>")
    exit(1)

dataset_path = pathlib.Path(argv[1])
orig_split = pathlib.Path(argv[2])
output = pathlib.Path(argv[3])

ids = list(map(str.strip, open(orig_split, 'r').readlines()))
if output.exists() and not input("File exists: '%s'.\nContinue? [y/N]" % output).lower().startswith('y'):
    exit(0)
with open(output, 'w+') as ofile:
    for id in tqdm(ids):
        anno_file = dataset_path / "annotations" / f"{id}.json"
        if not os.path.isfile(anno_file):
            continue
        anno = json.load(open(anno_file, 'r'))
        labels = (obj["label"] for obj in anno["objects"])
        classes = list(filter(None, map(dflt.convert_label, labels)))
        if classes:
            ofile.write(id + '\n')
