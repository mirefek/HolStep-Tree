# HolStep Tree

## Where to start with extension

- Run "python main.py --help" and play with hyperparameters to achieve better accuracy.
  - dev accuracy 89% on the HolStep dataset not reached yet
  - The most stupid algorithm on Mizar dataset which distinguish dependencies just by names
    and does not take the conjecture into acount has accuracy 71.75%.
    The network is not able to defeat it.
- See simple_network.py for a simple example of the network
- See cells.py and implement / use another tree RNN cells
- Try run test_generation.py and modify it to something useful

## Simpler tasks

Feel free to acomplish these tasks (and let me know :-) ).

### Data

- [ ] Mix variables on input
- [ ] Ability to read different data format, for instance [TPTP](http://www.cs.miami.edu/~tptp/)
- [ ] Use definition of constants for learning of their embeddings -- data missing

### Network

- [ ] Ability to log embeddings in char_emb mode
- [ ] More flexibility with tree-RNN inputs (not neccesarily the same dim), split interface and dimension
- [ ] Classic dropout (inside network, tf.nn.dropout)
- [ ] Dependency selection (the 'D' lines) -- big output (negative sampling / hierarchical softmax)

## Ideas

- How on Earth can we use conjectures effectively?? Use of conjectures does not seem to influence
  the development accuracy out of range of statistical significance. Is there a bug?
- Rewrite using Tensorflow Fold:
  - https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/sentiment.ipynb
    - Or not? Is it possible to store data on the tree in Fold?
- batching in formula generation?
- some kind of attention ...
