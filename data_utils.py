"""class DataParser = Utility for reading data from files and creating ordered or random batches from them

When created, DataParser loads the data into memory in following form:
  
  self.vocabulary_index = list of used words (ordered by frequency)
  self.reverse_vocabulary_index = dict {'word': index of 'word' in self.vocabulary_index}
     can be loaded from / saved to a file
     provide 'tokenization' = translating lines into lists of indices into self.vocabulary_index
                                   -1 for unknown words

  self.train_conjectures
  self.val_conjectures
     = lists of conjectures,
    every conjecture = dict
      'name':      title of the conjecture (string)
      'filename':  string like 'e-hol-ml-dataset/train/00042'
      'conj':      tokenized conecture, i.e. wording translated into indices into self.vocabulary_index
      'deps':      can be list of tokenized dependencies, currently not used
      '+', '-':    lists of tokenized positive and negative steps

There are 4 methods for getting data (for user):
  draw_random_batch_of_steps
  draw_random_batch_of_steps_and_conjectures
  draw_batch_of_steps_in_order
  draw_batch_of_steps_and_conjectures_in_order

Keyword arguments: split='train' (default) or 'dev', batch_size = 128 (default)
Return values of random versions:
  draw_random_batch_of_steps -> ([steps, preselection], labels)
  draw_random_batch_of_steps -> ([steps, conjectures, preselection], labels)
steps and conjectures = tree data, for explicit format, see tree_utils.py
preselection = the list of words used in the batch, for explicit format, see tree_utils.py
labels = numpy array of zeros (= unuseful step) and ones (= useful step)

The "in_order" versions can moreover get an argument begin_index, the begin of data is index (0,0).
Then it returns pair (data, end_index), where data are in the format of appropriate "random" version.
If it reaches the end of data, it returns partial or empty result. So it can be in loop like that:

index = (0,0)
while True:
  (input_data, labels), index = draw_batch_of_steps_in_order(begin_index = index, 'val', batch_size)
  if len(labels) == 0: break
  process(input_data, labels)
  if len(labels) < batch_size: break

Methods for parsing (used by constructor)
  build_vocabulary(fnames = list of filenames from which the words are loaded)
  save_vocabulary(filename)
  load_vocabulary(filename)
  parse_file_list(fnames = list of filenames, each file contains one conjecture)
   -> parse_file(filename)
"""

from __future__ import print_function
import os
import sys
import logging
import random

import numpy as np

class DataParser(object):

    # discard_unknown = save only conjectures / steps without unknown words
    # ignore_deps = do not save lists of dependencies -- 'deps' of a conjecture
    def __init__(self, source_dir, encoder, verbose=1, voc_filename=None, discard_unknown = False, ignore_deps = False):
        random.seed(1337)

        self.verbose = verbose
        train_dir = os.path.join(source_dir, 'train')
        val_dir = os.path.join(source_dir, 'test')
        train_fnames = sorted([
            os.path.join(train_dir, fname)
            for fname in os.listdir(train_dir) if fname.isdigit()])

        val_fnames = sorted([
            os.path.join(val_dir, fname)
            for fname in os.listdir(val_dir) if fname.isdigit()])

        if voc_filename and os.path.isfile(voc_filename):
            self.vocabulary_index = self.load_vocabulary(voc_filename)
        else:
            if verbose:
                logging.info('Building vocabulary...')
            self.vocabulary_index = self.build_vocabulary(train_fnames)
            if voc_filename: self.save_vocabulary(voc_filename)

        if verbose:
            logging.info('Found %s unique tokens.', len(self.vocabulary_index))

        self.reverse_vocabulary_index = dict(
            [(self.vocabulary_index[key], key) for key in range(len(self.vocabulary_index))])

        if encoder is None: return

        encoder.set_vocab(self.reverse_vocabulary_index, self.vocabulary_index)
        self.encoder = encoder

        self.discard_unknown = discard_unknown
        self.ignore_deps = ignore_deps

        self.train_conjectures = self.parse_file_list(train_fnames)
        self.val_conjectures = self.parse_file_list(val_fnames)

    def save_vocabulary(self, filename):
        f = open(filename, 'w')
        for token in self.vocabulary_index: print(token, file=f)
        f.close()

    def load_vocabulary(self, filename):
        f = open(filename, 'r')
        vocabulary = f.read().splitlines()
        f.close()
        return vocabulary

    def build_vocabulary(self, fnames):
        vocabulary_freq = dict()
        for fname in fnames:
            f = open(fname)
            for line in f:
                if line[0] == 'P':
                    for token in line.rstrip()[2:].split():
                        if token not in vocabulary_freq:
                            vocabulary_freq[token] = 1
                        else: vocabulary_freq[token] += 1
            f.close()
        vocabulary = sorted([(freq, token) for (token, freq) in vocabulary_freq.items()], reverse=True)

        # By uncommenting these, you log the vocabulary together with frequencies
        #
        #f = open('vocab_freq', 'w')
        #for (freq, token) in vocabulary: print("{} {}".format(freq, token), file=f)
        #f.close()

        return [token for (freq, token) in vocabulary]

    def parse_file_list(self, fnames): # load a list of conjectures into memory
        conjectures = []
        for fname in fnames:
            if self.verbose:
                sys.stdout.write("Loading {}    ".format(fname))
                sys.stdout.flush()
                sys.stdout.write('\r')

            conjecture = self.parse_file(fname)
            if conjecture: conjectures.append(conjecture)

        if self.verbose: sys.stdout.write('\n')

        return conjectures

    def display_stats(self, conjectures): # vestige of holstep_baselines, not maintained, not working
        dep_counts = []
        dep_lengths = []
        conj_lengths = []
        pos_step_counts = []
        pos_step_lengths = []
        neg_step_counts = []
        neg_step_lengths = []

        logging.info('%s conjectures in total.', len(conjectures))
        for key, value in conjectures.items():
            deps = value['deps']
            conj = value['conj']
            pos_steps = value['+']
            neg_steps = value['-']
            dep_counts.append(len(deps))
            if deps:
                dep_lengths.append(np.mean([len(x) for x in deps]))
            conj_lengths.append(len(conj))
            pos_step_counts.append(len(pos_steps))
            if pos_steps:
                pos_step_lengths.append(np.mean([len(x) for x in pos_steps]))
            neg_step_counts.append(len(neg_steps))
            if neg_steps:
                neg_step_lengths.append(np.mean([len(x) for x in neg_steps]))
        logging.info('total number of steps: %s',
                     np.sum(pos_step_counts) + np.sum(neg_step_counts))
        logging.info('mean number of positive steps per conjecture: %s',
                     np.mean(pos_step_counts))
        logging.info('mean number of negative steps per conjecture: %s',
                     np.mean(neg_step_counts))
        logging.info('mean conjecture length: %s', np.mean(conj_lengths))
        logging.info('mean number of dependencies: %s', np.mean(dep_counts))
        logging.info('mean dependency length: %s', np.mean(dep_lengths))
        logging.info('mean number of positive steps: %s',
                     np.mean(pos_step_counts))
        logging.info('mean number of negative steps: %s',
                     np.mean(neg_step_counts))
        logging.info('mean positive step length: %s', np.mean(pos_step_lengths))
        logging.info('mean negative step length: %s', np.mean(neg_step_lengths))
        # TODO: plot histograms

    def parse_file(self, fname): # parse a single file with a single conjecture
        f = open(fname)
        name = f.readline().rstrip()[2:]

        def tokenize(line):
            line = line.rstrip()[2:]
            tokens = [self.reverse_vocabulary_index.get(tokstr, -1) for tokstr in line.split()]
            return tokens

        f.readline() # text line
        prefix_line = f.readline()
        conj = tokenize(prefix_line)
        if self.discard_unknown and min(conj) < 0: return None

        conjecture = {
            'name': name,
            'filename': fname,
            'deps': [],
            '+': [],
            '-': [],
            'conj': conj,
        }
        while 1:
            line = f.readline()
            if not line:
                break
            marker = line[0]
            if marker == 'D':
                text_line = f.readline() # text line
                token_line = f.readline() # prefix line
                if not self.ignore_deps:
                    content = tokenize(token_line)
                    if not (self.discard_unknown and min(content) < 0):
                        conjecture['deps'].append(content)
            elif marker in {'+', '-'}:
                token_line = f.readline()
                content = tokenize(token_line)

                if not (self.discard_unknown and min(content) < 0):
                    conjecture[marker].append(content)

        return conjecture

    def draw_random_batch_of_steps(self, split='train', batch_size=128):
        if split == 'train':
            all_conjectures = self.train_conjectures
        elif split == 'val':
            all_conjectures = self.val_conjectures
        else:
            raise ValueError('`split` must be in {"train", "val"}.')

        labels = np.random.randint(0, 2, size=(batch_size,))
        steps = []
        i = 0
        while len(steps) < batch_size:
            conjecture = random.choice(all_conjectures)
            if labels[i]:
                conjecture_steps = conjecture['+']
            else:
                conjecture_steps = conjecture['-']
            if not conjecture_steps:
                continue
            step = random.choice(conjecture_steps)
            steps.append(step)
            i += 1

        self.encoder.load_preselection(steps)
        return [self.encoder.encode(steps), self.encoder.get_preselection()], labels

    def draw_batch_of_steps_in_order(self, begin_index=(0,0), split='train', batch_size=128):
        conjecture_index, step_index = begin_index

        if split == 'train':
            all_conjectures = self.train_conjectures
        elif split == 'val':
            all_conjectures = self.val_conjectures
        else:
            raise ValueError('`split` must be in {"train", "val"}.')

        labels = []
        steps = []
        while len(steps) < batch_size and conjecture_index < len(all_conjectures):
            conjecture = all_conjectures[conjecture_index]
            conjecture_steps = conjecture['+'] + conjecture['-']
            if len(conjecture_steps) > step_index:
                step_labels = ([1] * len(conjecture['+']) +
                               [0] * len(conjecture['-']))
                remaining = batch_size - len(steps)
                steps += conjecture_steps[step_index: step_index + remaining]
                labels += step_labels[step_index: step_index + remaining]
                step_index += remaining
            else:
                step_index = 0
                conjecture_index += 1

        labels = np.asarray(labels).astype('float32')

        self.encoder.load_preselection(steps)
        return ([self.encoder.encode(steps), self.encoder.get_preselection()], labels), (conjecture_index, step_index)

    def draw_batch_of_steps_and_conjectures_in_order(self, begin_index=(0,0), split='train', batch_size=128):
        conjecture_index, step_index = begin_index

        if split == 'train':
            all_conjectures = self.train_conjectures
        elif split == 'val':
            all_conjectures = self.val_conjectures
        else:
            raise ValueError('`split` must be in {"train", "val"}.')

        labels = []
        conjectures = []
        steps = []
        while len(steps) < batch_size and conjecture_index < len(all_conjectures):
            conjecture = all_conjectures[conjecture_index]
            conjecture_steps = conjecture['+'] + conjecture['-']
            #conjecture_steps = conjecture_steps[:1]
            if len(conjecture_steps) > step_index:
                step_labels = ([1] * len(conjecture['+']) +
                               [0] * len(conjecture['-']))
                #step_labels = step_labels[:1]
                remaining = batch_size - len(steps)
                steps += conjecture_steps[step_index: step_index + remaining]
                added_labels = step_labels[step_index: step_index + remaining]
                labels += added_labels
                conjectures += [conjecture['conj']] * len(added_labels)
                step_index += remaining
            else:
                step_index = 0
                conjecture_index += 1
        labels = np.asarray(labels).astype('float32')

        self.encoder.load_preselection(steps+conjectures)
        return (([self.encoder.encode(steps), self.encoder.encode(conjectures), self.encoder.get_preselection()], labels),
                (conjecture_index, step_index))

    def draw_random_batch_of_steps_and_conjectures(self, split='train',
                                                   batch_size=128):
        if split == 'train':
            all_conjectures = self.train_conjectures
        elif split == 'val':
            all_conjectures = self.val_conjectures
        else:
            raise ValueError('`split` must be in {"train", "val"}.')

        labels = np.random.randint(0, 2, size=(batch_size,))
        conjectures = []
        steps = []
        i = 0
        while len(steps) < batch_size:
            conjecture = random.choice(all_conjectures)
            if labels[i]:
                conjecture_steps = conjecture['+']
            else:
                conjecture_steps = conjecture['-']
            if not conjecture_steps:
                continue
            step = random.choice(conjecture_steps)
            conjectures.append(conjecture['conj'])
            steps.append(step)
            i += 1

        self.encoder.load_preselection(steps+conjectures)
        return [self.encoder.encode(steps), self.encoder.encode(conjectures), self.encoder.get_preselection()], labels

if __name__ == "__main__":
    # when loaded alone, just test that data can be loaded
    parser = DataParser("e-hol-ml-dataset", None, voc_filename='training_vocabulary')
