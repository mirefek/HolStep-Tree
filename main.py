from network import Network, version
import tensorflow as tf
from data_utils import DataParser
import logging # vestige of holstep_baselines, used by data_parser
import tree_utils as tree
import os
import sys
import numpy as np
import traceback_utils
import argparse
import time

# When used, it may be useful to modify functions draw_batch_of_steps_and_conjectures_in_order,
# so that they return only the first step from every conjecture. Otherwise conjectures could mix
# together and it would not show since they all are the same

def test_consistency(tests_num):
    #line = data_parser.tokenize("P f0")
    #encoder.load_preselection([line])
    #steps = encoder.encode([line])
    #print("simple test: {}".format(line))
    #print("encoded: {}".format(steps))
    #print(network.predict(steps, None, encoder.get_preselection()))
    #print("simple test DONE".format(line))

    index = (0,0)
    batch, index = data_parser.draw_batch('val', tests_num, get_conjectures = args.conjectures, begin_index = index)
    del batch['labels']
    predictions, logits = network.predict(batch)

    isolated_predictions, isolated_logits = [], []
    index = (0,0)
    for i in range(tests_num):
        batch, index = data_parser.draw_batch('val', 1, get_conjectures = args.conjectures, begin_index = index)
        del batch['labels']
        prediction, logit = network.predict(batch)

        isolated_predictions.append(prediction[0])
        isolated_logits.append(logit[0])

    isolated_predictions = np.array(isolated_predictions)
    isolated_logits = np.array(isolated_logits)

    print("In batch:")
    print("  Logits: {}".format(logits))
    print("  Prediction: {}".format(predictions))
    print("Isolated:")
    print("  Logits: {}".format(isolated_logits))
    print("  Prediction: {}".format(isolated_predictions))
    print("Difference:")
    print("  Logits: {}".format(isolated_logits-logits))
    print("  Prediction: {}".format(isolated_predictions-predictions))

logging.root.setLevel(logging.INFO)
sys.excepthook = traceback_utils.shadow('/usr/') # hide entrails of Tensorflow in error messages

cmd_parser = argparse.ArgumentParser(prog='tree-holstep',
                                     description='Run tree RNN network on Holstep dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

cmd_parser.add_argument('--version', action='version', version='%(prog)s '+version)
cmd_parser.add_argument('--quiet', dest='quiet', action='store_true')
cmd_parser.set_defaults(quiet=False)
cmd_parser.add_argument('--consistency_check', dest='consistency_check', action='store_true', help="After first epoch check whether the network is consistent and exit")
cmd_parser.set_defaults(consistency_check=False)
cmd_parser.add_argument('--measure_memory', dest='measure_memory', action='store_true',
                        help = "Measure the amount of memory occupied by parsed dataset and print it to stdout.")
cmd_parser.set_defaults(measure_memory=False)
cmd_parser.add_argument('--measure_time', dest='measure_time', action='store_true',
                        help = "Measure separately the time of (testing / training) (data preparation / execution) and print it to stdout.")
cmd_parser.set_defaults(measure_time=False)
cmd_parser.add_argument("--vocabulary_file", default=None, type=str, help="Vocabulary file name.")
cmd_parser.add_argument("--data_path", default='./e-hol-ml-dataset/', type=str, help="Path to dataset.")
cmd_parser.add_argument('--simple_data', dest='simple_data', action='store_true',
                        help = "Simple data format without names and text lines.")
cmd_parser.set_defaults(simple_data=False)
cmd_parser.add_argument("--divide_test_data", default=None, type=float,
                        help="If data in data_path are not divided into test and train parts, take a given fraction of it as test data.")
cmd_parser.add_argument("--truncate_train_data", default=1, type=float,
                        help="Load only this fraction of training data.")
cmd_parser.add_argument("--truncate_test_data", default=1, type=float,
                        help="Load only this fraction of validation data.")
cmd_parser.add_argument("--log_dir", default='./logs/', type=str, help="Directory for tensorboard logs.")
cmd_parser.add_argument("--no-log_dir", dest='log_dir', action='store_const', const=None)
cmd_parser.add_argument('--conjectures', dest='conjectures', action='store_true', help="Use conjectures (conditioned classification).")
cmd_parser.add_argument('--no-conjectures', dest='conjectures', action='store_false')
cmd_parser.set_defaults(conjectures=True)
cmd_parser.add_argument('--char_emb', dest='char_emb', action='store_true', help="Use character embeddings")
cmd_parser.set_defaults(char_emb=False)
cmd_parser.add_argument('--pooling', dest='pooling', action='store_true', help="Use max pooling on steps.")
cmd_parser.set_defaults(pooling=False)
cmd_parser.add_argument('--extra_layer', dest='extra_layer', action='store_true', help="Use one more down_up layer.")
cmd_parser.set_defaults(extra_layer=False)
cmd_parser.add_argument('--step_as_index', dest='step_as_index', action='store_true', help="Ignore the inner structure of a step.")
cmd_parser.set_defaults(step_as_index=False)
cmd_parser.add_argument('--word2vec', default=None, nargs=2, type=float, help="Word2vec-like encoding, expects two coeficient for type_loss and const_loss respectively.")
cmd_parser.add_argument('--definitions', default=None, nargs=2, type=str, help="First argument determines the filename formed by lines 'd <tokens> <tokens in definitions>', second argument is the training coefficitent")
cmd_parser.add_argument('--known_only', dest='known_only', action='store_true', help="Discard data with unknown tokens.")
cmd_parser.add_argument('--allow_unknown', dest='known_only', action='store_false')
cmd_parser.set_defaults(known_only=False)
cmd_parser.add_argument("--threads", default=4, type=int, help="Number of computing threads.")
cmd_parser.add_argument("--epochs", default=40, type=int, help="Number of epochs.")
cmd_parser.add_argument("--batch_size", default=64, type=int, help="Batch size for training.")
cmd_parser.add_argument("--batches_per_epoch", default=2000, type=int, help="Number of batches in one epoch.")
cmd_parser.add_argument("--test_batch_size", default=128, type=int, help="Batch size for testing.")
cmd_parser.add_argument("--rnn_dim", default=128, type=int, help="Dimension of RNN state.")
#cmd_parser.add_argument("--appl_hidden", default=128, type=int, help="Size of hidden layer in 'applications'.")
cmd_parser.add_argument("--hidden", default=256, type=int, help="Size of the final hidden layer.")
cmd_parser.add_argument("--o_dropout", default=0.5, type=float, help="Input dropout coefficient.")
cmd_parser.add_argument("--i_dropout", default=0.5, type=float, help="Input dropout coefficient.")
cmd_parser.add_argument("--i_dropout_protect", default=0.2, type=float, help="Fraction of vocabulary protected before dropout.")
cmd_parser.add_argument('--log_graph', dest='log_graph', action='store_true', help="Add graph to tensorflow log.")
cmd_parser.set_defaults(log_graph=False)
cmd_parser.add_argument('--log_embeddings', dest='log_embeddings', action='store_true', help="Add embeddings to tensorflow log.")
cmd_parser.set_defaults(log_embeddings=False)
args = cmd_parser.parse_args()

expname = "{}-{}-epochs-{}-dim-{}-{}-dropout-{}-{}".format(['uncond', 'cond'][args.conjectures], version, args.epochs,
                                                              args.rnn_dim, args.hidden,
                                                              args.i_dropout, args.i_dropout_protect)
if args.pooling: expname += "-pooling"
if args.char_emb: expname += "-char_emb"
if args.pooling: expname += "-pooling"
if args.extra_layer: expname += "-extra_layer"
if args.step_as_index: expname += "-step_as_index"
if args.word2vec is not None: expname += "-w2vec-{}-{}".format(args.word2vec[0], args.word2vec[1])
if args.definitions is not None:
    def_fname, def_coef = args.definitions
    def_coef = float(def_coef)
    expname += "-def-{}".format(def_coef)
else: def_fname, def_coef = None, None

if args.measure_memory:
    import psutil

    process = psutil.Process(os.getpid())
    print("Initial memory: {}M".format(process.memory_info().rss / 10**6))

encoder = tree.TokenEncoder(('*', '/'), char_emb=args.char_emb)
data_parser = DataParser(args.data_path, encoder = encoder, voc_filename = args.vocabulary_file,
                         discard_unknown = args.known_only, ignore_deps = True, verbose = not args.quiet,
                         simple_format = args.simple_data,
                         divide_test = args.divide_test_data, truncate_test = args.truncate_test_data, truncate_train = args.truncate_train_data,
                         complete_vocab = args.char_emb, step_as_index = args.step_as_index, def_fname = def_fname
)

if args.measure_memory:
    print("Memory after parsing: {}M".format(process.memory_info().rss / 10**6))

network = Network(logdir = args.log_dir, threads = args.threads, expname = expname)
network.construct(vocab_size = len(data_parser.vocabulary_index), use_conjectures=args.conjectures,
                  dim=args.rnn_dim, hidden_size=args.hidden,
                  extra_layer = args.extra_layer, w2vec = args.word2vec,
                  use_pooling = args.pooling, num_chars = encoder.char_num,
                  max_step_index = data_parser.max_step_index, definitions_coef = def_coef)
if args.log_graph: network.log_graph()
if args.log_embeddings: network.log_vocabulary(data_parser.vocabulary_index)

if args.measure_time:
    training_num = 0       # overall number of processed training steps
    training_prep_time = 0 # sum of training preparation times
    training_time = 0      # sum of training execution times

    testing_num = 0        # overall number of processed testing steps
    testing_prep_time = 0  # sum of test preparation times
    testing_time = 0       # sum of test execution times

if not args.quiet: accum_train_accuracy = 0
for epoch in range(1, args.epochs+1):
    for i in range(args.batches_per_epoch):
        if not args.quiet and i > 0 and i%200 == 0:
            sys.stdout.write('\n')

        if args.measure_time: start_time = time.time()

        # Draw training data
        batch = data_parser.draw_batch('train', args.batch_size, get_conjectures = args.conjectures,
                                       definitions_size = args.batch_size)
        # ... done

        if args.measure_time:
            end_time = time.time()
            training_prep_time += end_time-start_time

            start_time = time.time()

        # Train
        accuracy = network.train(batch, dropout=(args.i_dropout, args.i_dropout_protect, args.o_dropout))
        # ... done

        if args.measure_time:
            end_time = time.time()
            training_time += end_time-start_time
            training_num += 1

        if not args.quiet:
            accum_train_accuracy = 0.01*accuracy + 0.99*accum_train_accuracy

            sys.stdout.write("Epoch {:>4}: Train {:>5}: current {:<9}, accumulated {:<15}".format(epoch, i+1, accuracy, accum_train_accuracy))
            sys.stdout.flush()
            sys.stdout.write('\r')

    if not args.quiet: sys.stdout.write('\n')

    if args.log_embeddings: network.log_embeddings()
    if args.consistency_check:
        test_consistency(10)
        exit()

    index = (0,0)
    sum_accuracy = sum_loss = 0
    processed_test_samples = 0
    if not args.quiet: counter = 1
    while True:
        if args.measure_time: start_time = time.time()

        # Draw testing data  
        batch, index = data_parser.draw_batch('val', args.test_batch_size, get_conjectures = args.conjectures, definitions_size = 0,
                                              begin_index = index)
        # ... done

        if batch['size'] == 0: break # We are on the end of the testing dataset, so nothing left

        if args.measure_time:
            end_time = time.time()
            testing_prep_time += end_time-start_time

            start_time = time.time()

        # Evaluation of current part
        accuracy, loss = network.evaluate(batch)
        # ... done

        if args.measure_time:
            end_time = time.time()
            testing_time += end_time-start_time
            testing_num += 1

        sum_accuracy += accuracy*batch['size']
        sum_loss += loss*batch['size']
        processed_test_samples += batch['size']

        if batch['size'] < args.test_batch_size: break # Just a smaller batch left -> we are on the end of the testing dataset

        if not args.quiet:
            sys.stdout.write("Epoch {:>4}: Test {:>5}: current acc {:<9} loss {:<15}, avg acc {:<15} loss {:<15}".\
                             format(epoch, counter, accuracy, loss, sum_accuracy/processed_test_samples, sum_loss/processed_test_samples))
            counter += 1
            sys.stdout.flush()
            sys.stdout.write('\r')

    if not args.quiet:
        sys.stdout.write("\nDevelopment accuracy: {}, avg. loss: {}\n".format(sum_accuracy/processed_test_samples, sum_loss/processed_test_samples))

    # Log testing summary
    if network.summary_writer:
        dev_summary = tf.Summary(value=[
            tf.Summary.Value(tag="dev/accuracy", simple_value=sum_accuracy/processed_test_samples),
            tf.Summary.Value(tag="dev/loss", simple_value=sum_loss/processed_test_samples),
        ])
        network.summary_writer.add_summary(dev_summary, network.training_step)

if args.measure_time:
    def print_time(sec):
        return time.strftime("%H:%M:%S", time.gmtime(sec))
    print("Time:")
    print("Training preparation: overall {}, avg {} sec".format(print_time(training_prep_time), training_prep_time/training_num))
    print("Testing preparation: overall {}, avg {} sec".format(print_time(testing_prep_time), testing_prep_time/testing_num))
    print("Training: overall {}, avg {} sec".format(print_time(training_time), training_time/training_num))
    print("Testing: overall {}, avg {} sec".format(print_time(testing_time), testing_time/testing_num))
