# generate html documentation based on the docstring
cd $(dirname "$0")
cd ../
pdoc --force --html -c latex_math=True -o doc branch.branch
