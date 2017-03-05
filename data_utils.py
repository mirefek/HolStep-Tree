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
    def __init__(self, source_dir, encoder, verbose=1, voc_filename=None,
                 discard_unknown = False, ignore_deps = False, simple_format = False,
                 check_input = False, divide_test = None, truncate_train = 1, truncate_test = 1,
                 complete_vocab = False, step_as_index = False, def_fname = None):
        random.seed(1337)

        self.simple_format = simple_format
        self.verbose = verbose
        self.check_input = check_input

        if divide_test is None:
            train_dir = os.path.join(source_dir, 'train')
            val_dir = os.path.join(source_dir, 'test')
            train_fnames = sorted([
                os.path.join(train_dir, fname)
                for fname in os.listdir(train_dir)])

            val_fnames = sorted([
                os.path.join(val_dir, fname)
                for fname in os.listdir(val_dir)])
        else:
            train_fnames = [
                os.path.join(source_dir, fname)
                for fname in os.listdir(source_dir)]
            random.shuffle(train_fnames)
            val_fnames = sorted(train_fnames[-int(divide_test*len(train_fnames)):])
            train_fnames = sorted(train_fnames[:-len(val_fnames)])

        train_fnames = train_fnames[:int(truncate_train*len(train_fnames))]
        val_fnames = val_fnames[:int(truncate_test*len(val_fnames))]

        if voc_filename and os.path.isfile(voc_filename):
            self.vocabulary_index = self.load_vocabulary(voc_filename)
        else:
            if verbose:
                logging.info('Building vocabulary...')

            vocab_fnames = train_fnames
            if complete_vocab: vocab_fnames = vocab_fnames + val_fnames
            if def_fname: vocab_fnames = vocab_fnames + [def_fname]

            self.vocabulary_index = self.build_vocabulary(vocab_fnames)
            if voc_filename: self.save_vocabulary(voc_filename)

        if verbose:
            logging.info('Found %s unique tokens.', len(self.vocabulary_index))

        self.reverse_vocabulary_index = dict(
            [(self.vocabulary_index[key], key) for key in range(len(self.vocabulary_index))])

        #if encoder is None: return

        if encoder: encoder.set_vocab(self.reverse_vocabulary_index, self.vocabulary_index)
        self.encoder = encoder

        self.discard_unknown = discard_unknown
        self.ignore_deps = ignore_deps

        self.step_as_index = step_as_index
        self.train_conjectures = self.parse_file_list(train_fnames)
        self.val_conjectures = self.parse_file_list(val_fnames)
        if verbose: print("Loaded {} training conjectures, {} validation conjectures.".format(
            len(self.train_conjectures), len(self.val_conjectures)
        ))

        if def_fname: self.definitions = self.parse_definitions(def_fname)

        if step_as_index:
            steps_set = set()
            for conj in self.train_conjectures:
                for step in conj['+']+conj['-']:
                    steps_set.add(step)
            self.max_step_index = len(steps_set)
            steps_set = dict((step,i) for i,step in enumerate(steps_set))
            for conj in self.train_conjectures + self.val_conjectures:
                conj['+'] = [steps_set.get(step, -1) for step in conj['+'] ]
                conj['-'] = [steps_set.get(step, -1) for step in conj['-'] ]

        else: self.max_step_index = None

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
                if self.simple_format or line[0] == 'P' or line[0] == 'd':
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

    def tokenize(self, line):
        line = line.rstrip()[2:]
        tokens = [self.reverse_vocabulary_index.get(tokstr, -1) for tokstr in line.split()]
        if self.check_input:
            try:
                self.encoder([tokens])
            except IOError:
                print("Line: {}".format(line))
                print("File: {}".format(fname))
                raise

        return tokens

    def parse_file(self, fname): # parse a single file with a single conjecture

        f = open(fname)
        line = f.readline()
        name = line.rstrip()[2:]

        if self.simple_format: prefix_line = line
        else:
            f.readline() # text line
            prefix_line = f.readline()

        conj = self.tokenize(prefix_line)
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
                if self.simple_format: prefix_line = line
                else:
                    text_line = f.readline()
                    prefix_line = f.readline()

                if not self.ignore_deps:
                    content = self.tokenize(prefix_line)
                    if not (self.discard_unknown and min(content) < 0):
                        conjecture['deps'].append(content)
            elif marker in {'+', '-'}:
                if self.simple_format: prefix_line = line
                else: prefix_line = f.readline()

                if self.step_as_index: content = prefix_line.rstrip()[2:]
                else: content = self.tokenize(prefix_line)

                if not (self.discard_unknown and min(content) < 0):
                    conjecture[marker].append(content)

        return conjecture

    def parse_definitions(self, fname):

        result = []
        f =  open(fname)
        for line in f:
            sline = line.rstrip()[2:].split()
            tokens = [self.reverse_vocabulary_index[w] for w in sline]
            result.append((tokens[0], tokens[1:]))

        return result

    def draw_batch(self, split, batch_size, get_conjectures = True, only_pos = False, begin_index = None, use_preselection = True, definitions_size = None):

        in_order = (begin_index is not None)

        if split == 'train':
            all_conjectures = self.train_conjectures
        elif split == 'val':
            all_conjectures = self.val_conjectures
        else:
            raise ValueError('`split` must be in {"train", "val"}.')

        # Preparation of steps and conjectures
        
        steps = []
        conjectures = []
        if in_order:
            labels = []
            conjecture_index, step_index = begin_index
            while len(steps) < batch_size and conjecture_index < len(all_conjectures):
                conjecture = all_conjectures[conjecture_index]

                if only_pos: conjecture_steps = conjecture['+']
                else: conjecture_steps = conjecture['+']+conjecture['-']

                if len(conjecture_steps) > step_index:
                    if only_pos: step_labels = [1] * len(conjecture['+'])
                    else: step_labels = [1] * len(conjecture['+']) + [0] * len(conjecture['-'])

                    remaining = batch_size - len(steps)

                    added_labels = step_labels[step_index: step_index + remaining]
                    labels += added_labels

                    steps += conjecture_steps[step_index: step_index + remaining]
                    if get_conjectures: conjectures += [conjecture['conj']] * len(added_labels)

                    step_index += remaining
                else:
                    step_index = 0
                    conjecture_index += 1

            labels = np.asarray(labels)
        else:
            if only_pos: labels = np.ones((batch_size,), int)
            else: labels = np.random.randint(0, 2, size=(batch_size,))

            while len(steps) < batch_size:
                conjecture = random.choice(all_conjectures)
                if labels[len(steps)]:
                    conjecture_steps = conjecture['+']
                else:
                    conjecture_steps = conjecture['-']

                if conjecture_steps:
                    step = random.choice(conjecture_steps)
                    steps.append(step)
                    if get_conjectures: conjectures.append(conjecture['conj'])

        # Preparation of definitions

        if definitions_size is not None:
            definitions = []
            while len(definitions) < definitions_size: definitions.append(random.choice(self.definitions))
            def_tokens = [token for token, definition in definitions]
            definitions = [definition for token, definition in definitions]

        # Preselection -- the used words

        if use_preselection:
            all_data = []
            if definitions_size is not None: all_data = all_data + [def_tokens] + definitions
            if not self.step_as_index: all_data = all_data + steps
            if get_conjectures: all_data = all_data + conjectures
            preselection = self.encoder.load_preselection(all_data)

            if definitions_size is not None:
                def_tokens = [preselection.translation[token] for token in def_tokens]

        else: preselection = None

        # encoding data
        
        if definitions_size is not None:
            definitions = self.encoder(definitions, preselection)
            def_tokens = np.array(def_tokens)
        if get_conjectures:
            conjectures = self.encoder(conjectures, preselection)

        if self.step_as_index: steps = np.array(steps)
        else: steps = self.encoder(steps, preselection)

        # Packing data

        batch = dict()
        batch['steps'] = steps
        if get_conjectures: batch['conjectures'] = conjectures
        if preselection is not None: batch['preselection'] = preselection.data
        if definitions_size is not None:
            batch['def_tokens'] = def_tokens
            batch['definitions'] = definitions
        batch['labels'] = labels
        batch['size'] = len(labels)

        if in_order: return batch, (conjecture_index, step_index)
        else: return batch

    def draw_random_batch_of_steps(self, split='train', batch_size=128, **kwargs):
        batch = self.draw_batch(split, batch_size, get_conjectures = False, **kwargs)
        return [batch['steps'], batch['preselection'], batch['labels']]

    def draw_batch_of_steps_in_order(self, begin_index=(0,0), split='train', batch_size=128):
        batch, index = self.draw_batch(split, batch_size, get_conjectures = False, begin_index = begin_index, **kwargs)
        return [batch['steps'], batch['preselection'], batch['labels']], index

    def draw_batch_of_steps_and_conjectures_in_order(self, begin_index=(0,0), split='train', batch_size=128, **kwargs):
        batch, index = self.draw_batch(split, batch_size, get_conjectures = True, begin_index = begin_index)
        return [batch['steps'], batch['conjectures'], batch['preselection'], batch['labels']], index

    def draw_random_batch_of_steps_and_conjectures(self, split='train', batch_size=128, **kwargs):
        batch = self.draw_batch(split, batch_size, get_conjectures = True, **kwargs)
        return [batch['steps'], batch['conjectures'], batch['preselection'], batch['labels']]

if __name__ == "__main__":
    # when loaded alone, just test that data can be loaded
    parser = DataParser("mizar-dataset", None, simple_format = True, divide_test = 0.1, step_as_index = True)
    print(parser.max_step_index)
