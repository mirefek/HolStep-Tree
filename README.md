# HolStep Tree

This code extends [baseline models for holstep](https://github.com/tensorflow/deepmath/tree/master/holstep_baselines) by tree RNN.

## Requirements

[Tensorflow 1.0](https://www.tensorflow.org/)

## Use

 - Download [modified HolStep dataset](http://atrey.karlin.mff.cuni.cz/~mirecek/holstep/e-hol-ml-dataset.tgz) to the directory with code, unpack and run: python main.py
 - Download [Mizar dataset modified by Josef Urban](https://github.com/JUrban/deepmath/tree/master/nnhpdata) as mizar-dataset and run: python main.py --data_path mizar-dataset --divide_test_data 0.1  --simple_data
 