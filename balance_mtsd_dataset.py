import vision.datasets.mtsd_default_classes as dflt
from sys import argv
import pathlib
import json
import os
import numpy as np
from tqdm import tqdm

if len(argv) < 4:
    print("Usage: prune_mtsd_dataset.py <dataset> <orig split> <output> [-y]")
    exit(1)

dataset_path = pathlib.Path(argv[1])
orig_split = pathlib.Path(argv[2])
output = pathlib.Path(argv[3])

ids = list(map(str.strip, open(orig_split, 'r').readlines()))
if output.exists() and not ("-y" in argv[4:] or input("File exists: '%s'.\nContinue? [y/N]" % output).lower().startswith('y')):
    exit(0)

CATEGORIES = dflt.CATEGORIES
N = len(CATEGORIES)

label_image_indexes = [set() for _ in range(N)]
label_image_counts = {}
label_counts = [0] * N
for i, id in enumerate(tqdm(ids)):
    anno_file = dataset_path / "annotations" / f"{id}.json"
    if not os.path.isfile(anno_file):
        continue
    anno = json.load(open(anno_file, 'r'))
    labels = (obj["label"] for obj in anno["objects"])
    for label_id in filter(lambda x: x is not None, map(dflt.convert_label, labels)):
        label_image_indexes[label_id].add(i)
        label_image_counts.setdefault(i, [0] * N)[label_id] += 1
        label_counts[label_id] += 1

label_stat = [len(stat) for stat in label_image_indexes]
print("Stat %s" % label_stat)
print("Count %s" % label_counts)

# Calculate class overlap
co_counts = np.zeros((N,N))
for x in range(N):
    for y in range(N):
        # [x,y] -> avg count of x in images labelled y
        co_counts[x,y] = sum(label_image_counts[id][x] for id in label_image_indexes[y]) / len(label_image_indexes[y])

select_sizes = np.clip(np.linalg.inv(co_counts) @ np.ones(N), 0.01, None)  # Determine number of each class to get one per class
M = min(np.array(label_stat) / select_sizes)  # Find best scaling factor
alt_select_sizes = (select_sizes * M).astype(np.int32)  # Scale to keep as many as possible
print("Number of images to select from each class: ", alt_select_sizes)

sample_image_indexes = set()
for size, image_indexes in zip(alt_select_sizes, label_image_indexes):
    image_indexes = np.array(list(image_indexes))
    sub = np.random.permutation(image_indexes)[:size]
    sample_image_indexes.update(sub)
sample_data = [ids[i] for i in sample_image_indexes]

new_label_counts = [0] * N
for i in sample_image_indexes:
    for k in range(N):
        if i in label_image_counts:
            new_label_counts[k] += label_image_counts[i][k]
print("New count %s" % new_label_counts)

with open(output, 'w+') as ofile:
    for id in sample_data:
        ofile.write(id + '\n')
