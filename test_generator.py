import numpy as np
import tensorflow as tf
import test_tree_utils as ttu
from generator import Generator
from tf_tree_utils import TreePlaceholder
import tensorflow.python.debug as tf_debug

import sys
import traceback_utils

class Network:

    def __init__(self, dim, len_lines):

        vocab_size = len(ttu.vocabulary)
        self.len_lines = len_lines

        graph = tf.Graph()
        graph.seed = 42
        self.session = tf.Session(graph = graph)

        with self.session.graph.as_default():

            line_emb = tf.tanh(tf.get_variable(name="line_embeddings", shape=[len_lines, dim]))
            embeddings = tf.tanh(tf.get_variable(name="raw_embeddings", shape=[vocab_size+1, dim]))

            self.preselection = tf.placeholder(tf.int32, [None], name='preselection')
            self.line_indices = tf.placeholder(tf.int32, [None], name='line_indices') # for training
            self.line_index = tf.placeholder(tf.int32, [], name='line_index') # for generation

            self.structure = TreePlaceholder()
            op_symbols = ['*', '/']
            op_symbols = [ttu.reverse_voc[s] for s in op_symbols]
            generator = Generator(dim, op_symbols, embeddings, self.preselection)

            init_state = line_emb[self.line_index]
            init_states = tf.gather(line_emb, self.line_indices)

            (types_loss, self.types_acc), (const_loss, self.const_acc) =\
                generator.train(init_states, self.structure)
            self.training = tf.train.AdamOptimizer().minimize(types_loss+const_loss)
            self.prediction = generator(init_state)

            self.session.run(tf.global_variables_initializer())

        self.session.graph.finalize()

    def train(self, structure, line_indices):

        data = self.structure.feed(structure)
        data.update({self.preselection: ttu.preselection, self.line_indices: line_indices})
        types_acc, const_acc, _ = self.session.run([self.types_acc, self.const_acc, self.training], data)

        return types_acc, const_acc

    def predict(self):

        ext_vocab = ['<unk>']+ttu.vocabulary
        predictions = []
        for i in range(self.len_lines):
            prediction = self.session.run(self.prediction, {self.line_index: i})
            prediction = ' '.join([ext_vocab[w+1] for w in prediction])
            predictions.append(prediction)

        return predictions

if __name__ == "__main__":

    sys.excepthook = traceback_utils.shadow('/usr/')

    lines = []
    lines.append("P * f1 f2\n")
    lines.append("P * / b0 * ! * * c= * * cGSPEC / b1 * b0 * cSETSPEC b2 b1 * b0 / b1 / b2 * * c/\ b2 * * c= b1 b3 f0\n")
    lines.append("P * * * * * f1 f2 f3 f4 f5 f100\n")
    lines.append("P * * c= * * c- * cSUC f0 * cSUC f1 * * c- f0 f1\n")
    lines.append("P * * c= / b0 * f0 b0 f0\n")
    lines.append("P / b0 * b1 b2\n")
    lines.append("P cT\n")

    real_structure = ttu.lines_to_tree_structure(lines)

    print(ttu.vocabulary)
    network = Network(20, len(lines))
    for i in range(401):

        types_acc, const_acc = network.train(real_structure, np.arange(len(lines)))
        if i%20 == 0:
            print("{}: types {},  const {}".format(i, types_acc, const_acc))
            predictions = network.predict()
            for ori, pred in zip(lines, predictions):
                print("Original: {}".format(ori.rstrip()))
                print("Prediction: {}".format(pred))
