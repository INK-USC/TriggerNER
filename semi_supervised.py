"""semi_supervised.py: semi_supervised learning with triggers (self training)

using 20% of the train data w/ triggers (already in trigger_20.txt file in each dataset),
and rest 80% of train data as unlabeled dataset.

Written in 2020 by Dong-Ho Lee.
"""
from model.soft_matcher import *
from model.soft_inferencer import *
from model.soft_inferencer_naive import SoftSequenceNaive
from config import Reader, Config, ContextEmb
from config.utils import load_bert_vec, get_optimizer, lr_decay
from config.eval import evaluate_batch_insts
from util import remove_duplicates
from typing import List
from tqdm import tqdm
from common import Sentence, Instance
import argparse, os, time, random
import numpy as np

def parse_arguments(parser):
    ###Training Hyperparameters
    parser.add_argument('--device', type=str, default="cpu", choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2','cuda:3', 'cuda:4', 'cuda:5', 'cuda:6'],
                        help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--digit2zero', action="store_true", default=True,
                        help="convert the number to 0, make it true is better")
    parser.add_argument('--dataset', type=str, default="CONLL")
    parser.add_argument('--embedding_file', type=str, default="dataset/glove.6B.100d.txt",
                        help="we will using random embeddings if file do not exist")
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=10, help="default batch size is 10 (works well)")
    parser.add_argument('--num_epochs', type=int, default=10, help="Usually we set to 10.")
    parser.add_argument('--num_epochs_soft', type=int, default=20, help="Usually we set to 20.")
    parser.add_argument('--train_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--dev_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--test_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--trig_optimizer', type=str, default="adam")

    ##model hyperparameter
    parser.add_argument('--model_folder', type=str, default="english_model", help="The name to save the model files")
    parser.add_argument('--hidden_dim', type=int, default=200, help="hidden size of the LSTM")
    parser.add_argument('--use_crf_layer', type=int, default=1, help="1 is for using crf layer, 0 for not using CRF layer", choices=[0,1])
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout for embedding")
    parser.add_argument('--use_char_rnn', type=int, default=1, choices=[0, 1], help="use character-level lstm, 0 or 1")
    parser.add_argument('--context_emb', type=str, default="none", choices=["none", "elmo", "bert"], help="contextual word embedding")
    parser.add_argument('--ds_setting', nargs='+', help="+ hard / soft matching") # soft, hard
    parser.add_argument('--percentage', type=int, default=100, help="how much percentage of training dataset to use")
    parser.add_argument('--unlabeled_percentage', type=float, default=0.8, help="how much percentage of training dataset to be used for unlabeld data")

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args

def main():
    parser = argparse.ArgumentParser()
    opt = parse_arguments(parser)
    conf = Config(opt)
    reader = Reader(conf.digit2zero)
    dataset, max_length, label_length = reader.read_trigger_txt(conf.trigger_file, -1)

    reader.merge_labels(dataset)

    trains = reader.read_txt(conf.train_all_file, conf.train_num)
    devs = reader.read_txt(conf.dev_file, conf.dev_num)
    tests = reader.read_txt(conf.test_file, conf.test_num)
    print(len(dataset))
    if conf.context_emb == ContextEmb.bert:
        print('Loading the BERT vectors for all datasets.')
        conf.context_emb_size = load_bert_vec(conf.trigger_file + "." + conf.context_emb.name + ".vec", dataset)

    # setting for data
    conf.use_iobes(trains)
    conf.use_iobes(dataset)
    conf.use_iobes(devs)
    conf.use_iobes(tests)

    conf.optimizer = opt.trig_optimizer
    conf.build_label_idx(dataset)
    conf.build_word_idx(trains, devs, tests)
    conf.build_emb_table()
    conf.map_insts_ids(dataset)
    conf.map_insts_ids(trains)
    conf.map_insts_ids(devs)
    conf.map_insts_ids(tests)

    dataset = reader.trigger_percentage(dataset, conf.percentage)
    encoder = SoftMatcher(conf, label_length)
    trainer = SoftMatcherTrainer(encoder, conf, devs, tests)

    # matching module training
    random.shuffle(dataset)
    trainer.train_model(conf.num_epochs_soft, dataset)
    logits, predicted, triggers = trainer.get_triggervec(dataset)
    # all the trigger vectors, trigger type, string name of the trigger
    triggers_remove = remove_duplicates(logits, predicted, triggers, dataset)

    numbers = int(len(trains) * (1 - opt.unlabeled_percentage))
    print("number of train instances : ", numbers)
    initial_trains = trains[:numbers]
    unlabeled_x = trains[numbers:]

    for data in unlabeled_x:
        data.output_ids = None

    # sequence labeling module self-training
    random.shuffle(dataset)
    inference = SoftSequence(conf, encoder)
    sequence_trainer = SoftSequenceTrainer(inference, conf, devs, tests, triggers_remove)
    sequence_trainer.self_training(conf.num_epochs, dataset, unlabeled_x)

if __name__ == "__main__":
    main()














