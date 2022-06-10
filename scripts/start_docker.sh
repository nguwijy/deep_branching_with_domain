cd $(dirname "$0")
username=$(grep -r "ARG username" Dockerfile | sed "s/ARG username=//")
cd ../
docker run -p 8888:8888 \
    --gpus all \  # comment this line if you do not have gpu
    --rm -it \
    -v ${PWD}:/home/${username}/deep_branching_with_domain \
    --network host \
    nguwijy/deep_branching
