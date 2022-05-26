cd $(dirname "$0")
cd ../
docker run -p 8888:8888 -v ${PWD}:/root dolfinx/lab
