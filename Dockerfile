# Parity check only: run tests and infer.py inside the Kaggle Python image.
# Pin the tag to the image the competition kernels use (see https://github.com/Kaggle/docker-python/releases).
FROM gcr.io/kaggle-gpu-images/python:latest

ENV PYTHONPATH=/kaggle/working/src
WORKDIR /kaggle/working

# Nothing is copied: mount the repository at /kaggle/working and input/ at /kaggle/input, e.g.
#   docker run --rm --gpus all -v "$PWD:/kaggle/working" -v "$PWD/input:/kaggle/input" <image> \
#     python experiments/exp000_baseline/infer.py --model-dir output/exp000_baseline/default/model
