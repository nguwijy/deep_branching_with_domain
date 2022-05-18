cd $(dirname "$0")
cd ../
docker run -p 8888:8888 --gpus all --rm -it -v ${PWD}:/deep_branching_with_domain --ipc=host nvidia-pytorch:latest
