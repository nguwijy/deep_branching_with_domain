# Bash scripts for deep branching solver

## start_docker.sh
To avoid installing all python dependencies
of deep branching solver on your machine,
it is recommended to use `docker`.
All you have to do is to
[install docker](https://docs.docker.com/engine/installation/)
and run:
```bash
bash start_docker.sh
```
This will direct you to the docker environment,
where you can start the jupyter server using:
```bash
jupyter notebook
```
If you do not have GPU in your machine,
comment the line `--gpus all`
in `start_docker.sh` before running the script.

## build_docker.sh
If you do not build your own container
before running `start_docker.sh`,
you cannot change or create any file
inside the docker environment due to permission issue.
To build the container with proper permission,
make sure you run the following before running `start_docker.sh`:
```bash
bash build_docker.sh
```

## start_dolphinx.sh
This script is to activate the docker environment
used by the notebook `dolphinx.ipynb`,
and can be run via:
```bash
bash start_dolphinx.sh
```

## png_to_gif.sh
This script is to convert png files plotted
for Navier-Stokes equation to gif.
To use this, run:
```bash
bash png_to_gif.sh /path/to/plot/directory
```
The path to plot directory is usually stored inside the `logs` folder.
