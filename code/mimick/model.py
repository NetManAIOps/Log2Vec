'''
Script for training Mimick model to predict OOV word embeddings based on pre-trained embeddings dictionary.
'''
from __future__ import division
from collections import Counter

import collections
import argparse
import random
import pickle
import logging
import progressbar
import os
import math
import datetime
import codecs
import dynet as dy
import numpy as np

from util import wordify, charseq
from consts import *

__author__ = "Yuval Pinter, 2017"

Instance = collections.namedtuple("Instance", ["chars", "word_emb"])

######################################

class CNNMimick:
    '''
    Implementation details inferred from
    http://dynet.readthedocs.io/en/latest/python_ref.html#convolution-pooling-operations
    Another dynet CNN impl example: https://goo.gl/UFkSwf
    '''

    def __init__(self, c2i, num_conv_layers=DEFAULT_CNN_LAYERS, char_dim=DEFAULT_CHAR_DIM,\
                hidden_dim=DEFAULT_HIDDEN_DIM, window_width=DEFAULT_WINDOW_WIDTH,\
                pooling_maxk=DEFAULT_POOLING_MAXK, w_stride=DEFAULT_STRIDE,\
                word_embedding_dim=DEFAULT_WORD_DIM, file=None):
        self.c2i = c2i
        self.pooling_maxk = pooling_maxk
        self.stride = [1, w_stride]
        self.char_dim = char_dim
        self.hidden_dim = hidden_dim
        self.window_width = window_width
        self.myModel = dy.Model()
        
        ### TODO allow more layers (create list of length num_conv_layers,
        ### don't forget relu and max-pooling after each in predict_emb)
        if num_conv_layers > 1:
            print("Warning: unsupported value passed for number of CNN layers: {}. Using 1 instead."\
                    .format(num_conv_layers))
        
        self.char_lookup = self.myModel.add_lookup_parameters((len(c2i), char_dim), name="ce")
        self.conv = self.myModel.add_parameters((1, window_width, char_dim, hidden_dim), name="conv")
        self.conv_bias = self.myModel.add_parameters((hidden_dim), name="convb")
        
        self.cnn_to_rep_params = self.myModel.add_parameters((word_embedding_dim, hidden_dim * pooling_maxk), name="H")
        self.cnn_to_rep_bias = self.myModel.add_parameters(word_embedding_dim, name="Hb")
        self.mlp_out = self.myModel.add_parameters((word_embedding_dim, word_embedding_dim), name="O")
        self.mlp_out_bias = self.myModel.add_parameters(word_embedding_dim, name="Ob")
        
        if file is not None:
            ### NOTE - dynet 2.0 only supports explicit loading into params, so
            ### dimensionalities all need to be specified in init
            self.myModel.populate(file)
 
    def predict_emb(self, chars):
        dy.renew_cg()

        conv_param = dy.parameter(self.conv)
        conv_param_bias = dy.parameter(self.conv_bias)

        H = dy.parameter(self.cnn_to_rep_params)
        Hb = dy.parameter(self.cnn_to_rep_bias)
        O = dy.parameter(self.mlp_out)
        Ob = dy.parameter(self.mlp_out_bias)
        
        # padding
        pad_char = self.c2i[PADDING_CHAR]
        padding_size = self.window_width // 2 # TODO also consider w_stride?
        char_ids = ([pad_char] * padding_size) + chars + ([pad_char] * padding_size)
        if len(chars) < self.pooling_maxk:
            # allow k-max pooling layer output to transform to affine
            char_ids.extend([pad_char] * (self.pooling_maxk - len(chars)))
        
        embeddings = dy.concatenate_cols([self.char_lookup[cid] for cid in char_ids])
        reshaped_embeddings = dy.reshape(dy.transpose(embeddings), (1, len(char_ids), self.char_dim))
        
        # not using is_valid=False due to maxk-pooling-induced extra padding
        conv_out = dy.conv2d_bias(reshaped_embeddings, conv_param, conv_param_bias, self.stride, is_valid=True)
        
        relu_out = dy.rectify(conv_out)
        
        ### pooling when max_k can only be 1, not sure what other differences may be
        #poolingk = [1, len(chars)]
        #pooling_out = dy.maxpooling2d(relu_out, poolingk, self.stride, is_valid=True)
        #pooling_out_flat = dy.reshape(pooling_out, (self.hidden_dim,))
        
        ### another possible way for pooling is just max_dim(relu_out, d=1)
        
        pooling_out = dy.kmax_pooling(relu_out, self.pooling_maxk, d=1) # d = what dimension to max over
        pooling_out_flat = dy.reshape(pooling_out, (self.hidden_dim * self.pooling_maxk,))

        return O * dy.tanh(H * pooling_out_flat + Hb) + Ob

    def loss(self, observation, target_rep):
        return dy.squared_distance(observation, dy.inputVector(target_rep))

    def set_dropout(self, p):
        ### TODO see if supported/needed
        pass

    def disable_dropout(self):
        ### TODO see if supported/needed
        pass

    def save(self, file_name):
        self.myModel.save(file_name)
        # character mapping saved separately
        pickle.dump(self.c2i, open(file_name[:-4] + '.c2i', "wb"))

    @property
    def model(self):
        return self.myModel
        
######################################

class LSTMMimick:

    def __init__(self, c2i, num_lstm_layers=DEFAULT_LSTM_LAYERS,\
                char_dim=DEFAULT_CHAR_DIM, hidden_dim=DEFAULT_HIDDEN_DIM,\
                word_embedding_dim=DEFAULT_WORD_DIM, file=None):
        self.c2i = c2i
        self.myModel = dy.Model()
        
        # Char LSTM Parameters
        self.char_lookup = self.myModel.add_lookup_parameters((len(c2i), char_dim), name="ce")
        self.char_fwd_lstm = dy.LSTMBuilder(num_lstm_layers, char_dim, hidden_dim, self.myModel)
        self.char_bwd_lstm = dy.LSTMBuilder(num_lstm_layers, char_dim, hidden_dim, self.myModel)

        # Post-LSTM Parameters
        self.lstm_to_rep_params = self.myModel.add_parameters((word_embedding_dim, hidden_dim * 2), name="H")
        self.lstm_to_rep_bias = self.myModel.add_parameters(word_embedding_dim, name="Hb")
        self.mlp_out = self.myModel.add_parameters((word_embedding_dim, word_embedding_dim), name="O")
        self.mlp_out_bias = self.myModel.add_parameters(word_embedding_dim, name="Ob")
        
        if file is not None:
            # read from saved file; see old_load() for dynet 1.0 format
            ### NOTE - dynet 2.0 only supports explicit loading into params, so
            ### dimensionalities all need to be specified in init
            self.myModel.populate(file)

    def predict_emb(self, chars):
        dy.renew_cg()

        finit = self.char_fwd_lstm.initial_state()
        binit = self.char_bwd_lstm.initial_state()

        H = dy.parameter(self.lstm_to_rep_params)
        Hb = dy.parameter(self.lstm_to_rep_bias)
        O = dy.parameter(self.mlp_out)
        Ob = dy.parameter(self.mlp_out_bias)

        pad_char = self.c2i[PADDING_CHAR]
        char_ids = [pad_char] + chars + [pad_char]
        embeddings = [self.char_lookup[cid] for cid in char_ids]

        bi_fwd_out = finit.transduce(embeddings)
        bi_bwd_out = binit.transduce(reversed(embeddings))

        rep = dy.concatenate([bi_fwd_out[-1], bi_bwd_out[-1]])

        return O * dy.tanh(H * rep + Hb) + Ob

    def loss(self, observation, target_rep):
        return dy.squared_distance(observation, dy.inputVector(target_rep))

    def set_dropout(self, p):
        self.char_fwd_lstm.set_dropout(p)
        self.char_bwd_lstm.set_dropout(p)

    def disable_dropout(self):
        self.char_fwd_lstm.disable_dropout()
        self.char_bwd_lstm.disable_dropout()

    def save(self, file_name):
        self.myModel.save(file_name)

        # character mapping saved separately
        pickle.dump(self.c2i, open(file_name[:-4] + '.c2i', "wb"))
        
    def old_save(self, file_name):
        members_to_save = []
        members_to_save.append(self.char_lookup)
        members_to_save.append(self.char_fwd_lstm)
        members_to_save.append(self.char_bwd_lstm)
        members_to_save.append(self.lstm_to_rep_params)
        members_to_save.append(self.lstm_to_rep_bias)
        members_to_save.append(self.mlp_out)
        members_to_save.append(self.mlp_out_bias)
        self.myModel.save(file_name, members_to_save)

        # character mapping saved separately
        pickle.dump(self.c2i, open(file_name[:-4] + '.c2i', "wb"))
            
    def old_load():
        # for this to load in __init__() there's no param init necessary
        model_members = iter(self.myModel.load(file))
        self.char_lookup = model_members.next()
        self.char_fwd_lstm = model_members.next()
        self.char_bwd_lstm = model_members.next()
        self.lstm_to_rep_params = model_members.next()
        self.lstm_to_rep_bias = model_members.next()
        self.mlp_out = model_members.next()
        self.mlp_out_bias = model_members.next()

    @property
    def model(self):
        return self.myModel

def dist(instance, vec):
    we = instance.word_emb
    if options.cosine:
        return 1.0 - (we.dot(vec) / (np.linalg.norm(we) * np.linalg.norm(vec)))
    return np.linalg.norm(we - vec)

if __name__ == "__main__":

    # ===-----------------------------------------------------------------------===
    # Argument parsing
    # ===-----------------------------------------------------------------------===
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help=".pkl file to use")
    parser.add_argument("--vocab", required=True, nargs="*", help="total vocab to output")
    parser.add_argument("--output", help="file with all embeddings")
    parser.add_argument("--model-out", default="model.bin", help="file with model parameters")
    parser.add_argument("--lang", default="en", help="language (optional, appears in log dir name)")
    parser.add_argument("--char-dim", type=int, default=DEFAULT_CHAR_DIM, help="dimension for character embeddings (default = {})".format(DEFAULT_CHAR_DIM))
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM, help="dimension for LSTM layers (default = {})".format(DEFAULT_HIDDEN_DIM))
    ### LSTM ###
    parser.add_argument("--num-lstm-layers", type=int, default=DEFAULT_LSTM_LAYERS, help="Number of LSTM layers (default = {})".format(DEFAULT_LSTM_LAYERS))
    ### CNN ###
    parser.add_argument("--use-cnn", dest="cnn", action="store_true", help="if toggled, train CNN and not LSTM")
    parser.add_argument("--num-conv-layers", type=int, default=DEFAULT_CNN_LAYERS, help="Number of CNN layers (default = 1, more not currently supported)")
    parser.add_argument("--window-width", type=int, default=DEFAULT_WINDOW_WIDTH, help="Width of CNN layers (default = 3)")
    parser.add_argument("--pooling-maxk", type=int, default=DEFAULT_POOLING_MAXK, help="K for K-max pooling (default = 1)")
    parser.add_argument("--stride", default=DEFAULT_STRIDE, dest="w_stride", help="'Width' stride for CNN layers (default = 1)")
    ### END ###
    parser.add_argument("--early-stopping", action="store_true", help="early stops if dev loss hasn't improved in last {} epochs".format(EARLY_STOPPING_CONST))
    parser.add_argument("--all-from-mimick", action="store_true", help="if toggled, vectors in original training set are overriden by Mimick-generated vectors")
    parser.add_argument("--normalized-targets", action="store_true", help="if toggled, train on normalized vectors from set")
    parser.add_argument("--dropout", default=-1, type=float, help="amount of dropout to apply to LSTM part of graph")
    parser.add_argument("--num-epochs", default=10, type=int, help="Number of full passes through training set (default = 10)")
    parser.add_argument("--learning-rate", default=0.01, type=float, help="Initial learning rate")
    parser.add_argument("--cosine", action="store_true", help="Use cosine as diff measure")
    parser.add_argument("--log-dir", dest="log_dir", help="Directory where to write logs / serialized models")
    parser.add_argument("--dynet-mem", help="Ignore this outside argument")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--log-to-stdout", dest="log_to_stdout", action="store_true", help="Log to STDOUT")
    options = parser.parse_args()

    # Set up logging
    log_dir = "embedding_train_mimick-{}-{}{}".format(datetime.datetime.now().strftime('%y%m%d%H%M%S'),\
                                                      options.lang, '-DEBUG' if options.debug else '')
    if options.log_dir is not None:
        if not os.path.exists(options.log_dir):
            os.mkdir(options.log_dir)
        log_dir = os.path.join(options.log_dir, log_dir)
    os.mkdir(log_dir)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if options.log_to_stdout:
        logging.basicConfig(level=logging.INFO)
    else:
        formatter = logging.Formatter('%(message)s')
        handler = logging.FileHandler(log_dir + '/log.txt', "w", 'utf-8')
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    root_logger.info("Training dataset: {}".format(options.dataset))
    root_logger.info("Output vocabulary: {}".format(options.vocab[0] if len(options.vocab) == 1 else options.vocab))
    root_logger.info("Vectors output location: {}".format(options.output))
    root_logger.info("Model output location: {}".format(options.model_out))
    if options.all_from_mimick:
        root_logger.info("All output vectors to be written from Mimick inference.")
    if options.early_stopping:
        root_logger.info("Implemented early stopping, epoch count = {}".format(EARLY_STOPPING_CONST))
    
    root_logger.info("\nModel Architecture: {}".format('CNN' if options.cnn else 'LSTM'))
    if options.cnn:
        root_logger.info("Layers: {}".format(options.num_conv_layers))
        root_logger.info("Window width: {}".format(options.window_width))
        root_logger.info("Pooling Max-K: {}".format(options.pooling_maxk))
        root_logger.info("Stride: {}".format(options.w_stride))
    else:
        root_logger.info("Layers: {}".format(options.num_lstm_layers))
    root_logger.info("Character embedding dimension: {}".format(options.char_dim))
    root_logger.info("Hidden dimension: {}".format(options.hidden_dim))
    
    # Load training set
    dataset = pickle.load(open(options.dataset, "rb"))
    c2i = dataset["c2i"]
    i2c = { i: c for c, i in c2i.items() } # inverse map
    training_instances = dataset["training_instances"]
    test_instances = dataset["test_instances"]
    populate_test_insts_from_vocab = len(test_instances) == 0
    emb_dim = len(training_instances[0].word_emb)

    # Load words to write
    vocab_words = {}
    if populate_test_insts_from_vocab:
        train_words = [wordify(w, i2c) for w in training_instances]
    for filename in options.vocab:
        with codecs.open(filename, "r", "utf-8") as vocab_file:
            for vw in vocab_file.readlines():
                vw = vw.strip()
                if vw in vocab_words: continue
                vocab_words[vw] = np.zeros(emb_dim)
                if populate_test_insts_from_vocab and vw not in train_words:
                    test_instances.append(Instance(charseq(vw, c2i), np.zeros(emb_dim)))
    
    if populate_test_insts_from_vocab:
        # need to update i2c if saw new characters
        i2c = { i: c for c, i in c2i.items() }

    if not options.cnn:
        model = LSTMMimick(c2i, options.num_lstm_layers, options.char_dim, options.hidden_dim, emb_dim)
    else:
        model = CNNMimick(c2i, options.num_conv_layers, options.char_dim, options.hidden_dim,\
                options.window_width, options.pooling_maxk, options.w_stride, emb_dim)
    
    trainer = dy.MomentumSGDTrainer(model.model, options.learning_rate, 0.9)
    root_logger.info("Training Algorithm: {}".format(type(trainer)))

    root_logger.info("Number training instances: {}".format(len(training_instances)))

    # Create dev set
    random.shuffle(training_instances)
    dev_cutoff = int(99 * len(training_instances) / 100)
    dev_instances = training_instances[dev_cutoff:]
    training_instances = training_instances[:dev_cutoff]

    if options.debug:
        train_instances = training_instances[:int(len(training_instances)/20)]
        dev_instances = dev_instances[:int(len(dev_instances)/20)]
    else:
        train_instances = training_instances

    if options.normalized_targets:
        train_instances = [Instance(ins.chars, ins.word_emb/np.linalg.norm(ins.word_emb)) for ins in train_instances]
        dev_instances = [Instance(ins.chars, ins.word_emb/np.linalg.norm(ins.word_emb)) for ins in dev_instances]

    epcs = int(options.num_epochs)
    dev_losses = []

    # Shuffle set, divide into cross-folds each epoch
    for epoch in range(epcs):
        bar = progressbar.ProgressBar()

        train_loss = 0.0
        train_correct = Counter()
        train_total = Counter()

        if options.dropout > 0:
            model.set_dropout(options.dropout)

        for instance in bar(train_instances):
            if len(instance.chars) <= 0: continue
            obs_emb = model.predict_emb(instance.chars)
            loss_expr = model.loss(obs_emb, instance.word_emb)
            loss = loss_expr.scalar_value()

            # Bail if loss is NaN
            if math.isnan(loss):
                assert False, "NaN occured"

            train_loss += loss

            # Do backward pass and update parameters
            loss_expr.backward()
            trainer.update()

        root_logger.info("\n")
        root_logger.info("Epoch {} complete".format(epoch + 1))
        # here used to be a learning rate update, no longer supported in dynet 2.0
        print(trainer.status())

        # Evaluate dev data
        model.disable_dropout()
        dev_loss = 0.0

        bar = progressbar.ProgressBar()
        for instance in bar(dev_instances):
            if len(instance.chars) <= 0: continue
            obs_emb = model.predict_emb(instance.chars)
            dev_loss += model.loss(obs_emb, instance.word_emb).scalar_value()

        root_logger.info("Train Loss: {}".format(train_loss))
        root_logger.info("Dev Loss: {}".format(dev_loss))
        
        dev_losses.append(dev_loss)
        if options.early_stopping and dev_loss == min(dev_losses):
            # save model
            root_logger.info("Saving epoch model")
            model.save(options.model_out)
        
        if epoch >= 5 and options.early_stopping and np.argmin(dev_losses[-EARLY_STOPPING_CONST:]) == 0:
            root_logger.info("Early stopping after {} epochs. Reloading best model".format(epoch + 1))
            model.model.populate(options.model_out)
            
            # recompute dev loss to make sure
            dev_loss = 0.0
            bar = progressbar.ProgressBar()
            for instance in bar(dev_instances):
                if len(instance.chars) <= 0: continue
                obs_emb = model.predict_emb(instance.chars)
                dev_loss += model.loss(obs_emb, instance.word_emb).scalar_value()
            root_logger.info("recomputed dev loss: {}".format(dev_loss))
            break

    if not options.early_stopping:
        # save model (with e-s it's already saved)
        model.save(options.model_out)
    
    # populate vocab_words and compute dataset statistics
    pretrained_vec_norms = 0.0
    inferred_vec_norms = 0.0
    for instance in train_instances + dev_instances:
        word = wordify(instance, i2c)
        if word in vocab_words:
            pretrained_vec_norms += np.linalg.norm(instance.word_emb)
            if options.all_from_mimick:
                # infer and populate
                obs_emb = model.predict_emb(instance.chars)
                vocab_words[word] = np.array(obs_emb.value())
                inferred_vec_norms += np.linalg.norm(vocab_words[word])
            else:
                # populate using vocab embeddings
                vocab_words[word] = instance.word_emb
    
    root_logger.info("\n")
    root_logger.info("Average norm for pre-trained in vocab: {}".format(pretrained_vec_norms / len(vocab_words)))
    
    # Infer for test set
    showcase_size = 5
    top_to_show = 10
    showcase = [] # sample for similarity sanity check
    for idx, instance in enumerate(test_instances):
        word = wordify(instance, i2c)
        obs_emb = model.predict_emb(instance.chars)
        vocab_words[word] = np.array(obs_emb.value())
        inferred_vec_norms += np.linalg.norm(vocab_words[word])

        if options.debug:
            # reservoir sampling
            if idx < showcase_size:
                showcase.append(word)
            else:
                rand = random.randint(0,idx-1)
                if rand < showcase_size:
                    showcase[rand] = word

    inferred_denom = len(vocab_words) if options.all_from_mimick else len(test_instances)
    root_logger.info("Average norm for trained: {}".format(inferred_vec_norms / inferred_denom))

    if options.debug:
        similar_words = {}
        for w in showcase:
            vec = vocab_words[w]
            top_k = [(wordify(instance, i2c),d) for instance,d in sorted([(inst, dist(inst, vec)) for inst in training_instances], key=lambda x: x[1])[:top_to_show]]
            print(w, [(i,d) for i,d in top_k])
            similar_words[w] = top_k


    # write all
    if options.output is not None:
        with codecs.open(options.output, "wb", "utf-8") as writer:
            writer.write("{} {}\n".format(len(vocab_words), emb_dim))
            for vw, emb in vocab_words.items():
                writer.write(vw + " ")
                for i in emb:
                    writer.write("{:.6f} ".format(i))
                writer.write("\n")
