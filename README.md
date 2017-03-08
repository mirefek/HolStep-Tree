# HolStep Tree

This code extends [baseline models for holstep](https://github.com/tensorflow/deepmath/tree/master/holstep_baselines) for by tree RNN. The long-term aim is to build a neural network useful in automated theorem proving. This tree approach improves original development accuracy for step classification from 83% to 88%. We tried it to use it directly on Mizar dataset prepared by Josef Urban for dependency selection, but it was not successful.

The current code provides just step classification but it can be extended for any other tasks including formula generation (decoding). See [TODO](TODO.md) for ideas of extensions.

## Requirements

[Tensorflow 1.0](https://www.tensorflow.org/)

## Use

- Download [modified HolStep dataset](http://atrey.karlin.mff.cuni.cz/~mirecek/holstep/e-hol-ml-dataset.tgz) to the directory with code, unpack and run: python main.py
```
python main.py
```
- Download [Mizar dataset modified by Josef Urban](https://github.com/JUrban/deepmath/tree/master/nnhpdata) as mizar-dataset and run
```
python main.py --data_path mizar-dataset --divide_test_data 0.1  --simple_data
```
- Accuracy is printed on the terminal and also can by viewed using tensorboard:
```
tensorboard --logdir ./logs/ &
chromium-browser http://localhost:6006/
```

## Optional features

- Input dropping
- Word embedding
- Multi layer Tree-RNN in both directions: from nodes to root and from root to nodes
- Word guessing like word2vec for better word embeddings
- Generation of formulas based on the tree structure
- Learning of word embeddings by their definitions
