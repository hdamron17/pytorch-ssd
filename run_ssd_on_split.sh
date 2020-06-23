set -e

if [ "$#" -ne 2 ]; then
  echo "Usage: run_ssd_on_split.sh <model pth> <split file>"
  exit 1
fi

model=$1
split=$2

for id in $(cat "$split"); do
  CUDA_VISIBLE_DEVICES="" \
  python run_ssd_example.py \
    mb2-ssd-lite \
    "$model" \
    models/mtsd-model-labels.txt \
    /tmp/mapillary/images/${id}.jpg
  mv run_ssd_example_output.jpg models/output/${id}.jpg
done
