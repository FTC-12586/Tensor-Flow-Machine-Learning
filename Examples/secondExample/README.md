# To create a conda environment

conda env create -f environment.yml

# To activate environment

conda activate tf_env

# Now run scripts eg:

python3 scripts/tensorflow_model.py

# To visualize tflite file:

python3 scripts/visualize.py FreightFrenzy_BC.tflite tflite_viz.html