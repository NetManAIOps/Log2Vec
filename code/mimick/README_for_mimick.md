# Mimick modeling

This directory is dedicated to the Mimick algorithm itself.
Starting with an embedding dictionary and (optionally) a target vocabulary, the tools here will provide you with:
1. A model that can be loaded to perform inference on new words downstream; and
1. (If needed) an embedding dictionary for the target vocabulary.

For help with any specific script in this directory, run it with `--help`. This will also describe the parameters.

## Pipeline

1. `make_dataset.py` to create a training regimen for the model. Only needs to be called once per input embeddings table.
1. `model.py` to train the model, save it, and output embeddings. Default is LSTM, CNN (1 layer) available via `--use-cnn` parameter.
1. If needed, `nearest_vecs.py` and `inter_nearest_vecs.py` can be used for querying the model for nearest vectors in any embeddings dictionary. `inter_` is the interactive version.

