import vision.datasets.mtsd_default_classes as dflt
from sys import argv
import pathlib
import json
import os
import numpy as np
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

label_image_indexes = {}
for i, id in enumerate(tqdm(ids)):
    anno_file = dataset_path / "annotations" / f"{id}.json"
    if not os.path.isfile(anno_file):
        continue
    anno = json.load(open(anno_file, 'r'))
    labels = (obj["label"] for obj in anno["objects"])
    for label_id in filter(None, map(dflt.convert_label, labels)):
        label_image_indexes.setdefault(label_id, set()).add(i)

label_stat = {k: len(v) for k, v in label_image_indexes.items()}
print("Stat: %s" % ", ".join("%s: %s" % (k, label_stat[k]) for k in sorted(label_stat.keys())))
min_image_num = min(label_stat.values())
print("Min num: %d" % min_image_num)
sample_image_indexes = set()
for image_indexes in label_image_indexes.values():
    image_indexes = np.array(list(image_indexes))
    sub = np.random.permutation(image_indexes)[:min_image_num]
    sample_image_indexes.update(sub)
sample_data = [ids[i] for i in sample_image_indexes]

with open(output, 'w+') as ofile:
    for id in sample_data:
        ofile.write(id + '\n')
