# convert png files in the directory to gif
# bash png_to_gif /path/to/plot/directory
cd $1
files=$(ls -t *velocity_color.png)
convert -delay 20 -loop 0 -dispose Background $files velocity_color.gif
files=$(ls -t *velocity_quiver.png)
convert -delay 20 -loop 0 -dispose Background $files velocity_quiver.gif
