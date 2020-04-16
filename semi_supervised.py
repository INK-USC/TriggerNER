"""semi_supervised.py: semi_supervised learning with triggers (self training)

using 20% of the train data w/ triggers (already in trigger.txt file in each dataset),
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
    parser.add_argument('--device', type=str, default="cpu", choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2','cuda:3', 'cuda:4', 'cuda:5'],
                        help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--digit2zero', action="store_true", default=True,
                        help="convert the number to 0, make it true is better")
    parser.add_argument('--dataset', type=str, default="BC5CDR")
    parser.add_argument('--embedding_file', type=str, default="dataset/glove.6B.100d.txt",
                        help="we will using random embeddings if file do not exist")
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--learning_rate', type=float, default=0.01)  ##only for sgd now
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=10, help="default batch size is 10 (works well)")
    parser.add_argument('--num_epochs', type=int, default=10, help="Usually we set to 10.")
    parser.add_argument('--num_epochs_soft', type=int, default=10, help="Usually we set to 10.")
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

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args

def train_procedure(model, config: Config, epoch: int, train_insts: List[Instance], dev_insts: List[Instance], test_insts: List[Instance], devscore=None, testscore=None):
    optimizer = get_optimizer(config, model)
    random.shuffle(train_insts)
    batched_data = batching_list_instances(config, train_insts, is_soft=False, is_naive=True)
    dev_batches = batching_list_instances(config, dev_insts)
    test_batches = batching_list_instances(config, test_insts)

    if devscore is None:
        best_dev = [-1, 0]
    else:
        best_dev = devscore

    if testscore is None:
        best_test = [-1, 0]
    else:
        best_test = testscore

    for i in range(1, epoch + 1):
        epoch_loss = 0
        start_time = time.time()
        model.zero_grad()
        if config.optimizer.lower() == "sgd":
            optimizer = lr_decay(config, optimizer, i)
        for index in tqdm(np.random.permutation(len(batched_data))):
            model.train()
            loss = model(*batched_data[index][0:5], batched_data[index][-3])
            epoch_loss += loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
            model.zero_grad()

        end_time = time.time()
        print("Epoch %d: %.5f, Time is %.2fs" % (i, epoch_loss, end_time - start_time), flush=True)
        model.eval()
        dev_metrics = evaluate_model(config, model, dev_batches, "dev", dev_insts)
        test_metrics = evaluate_model(config, model, test_batches, "test", test_insts)
        if dev_metrics[2] > best_dev[0]:
            print("saving the best model...")
            best_dev[0] = dev_metrics[2]
            best_dev[1] = i
            best_test[0] = test_metrics[2]
            best_test[1] = i
        model.zero_grad()

    return model, best_dev, best_test


def train_model(current_model, config: Config, epoch: int, train_insts: List[Instance], dev_insts: List[Instance], test_insts: List[Instance], devscore=None, testscore=None):
    if current_model is None:
        model = SoftSequenceNaive(config)
    else:
        model = current_model
    train_procedure(model, config, epoch, train_insts, dev_insts, test_insts, devscore, testscore)

    return model


def evaluate_model(config: Config, model: SoftSequenceNaive, batch_insts_ids, name: str, insts: List[Instance]):
    ## evaluation
    metrics = np.asarray([0, 0, 0], dtype=int)
    batch_id = 0
    batch_size = config.batch_size
    for batch in batch_insts_ids:
        one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
        batch_max_scores, batch_max_ids = model.decode(*batch[0:5], None)
        metrics += evaluate_batch_insts(one_batch_insts, batch_max_ids, batch[6], batch[1], config.idx2labels, config.use_crf_layer)
        batch_id += 1
    p, total_predict, total_entity = metrics[0], metrics[1], metrics[2]
    precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
    recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
    fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    print("[%s set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (name, precision, recall, fscore), flush=True)
    return [precision, recall, fscore]


def main():
    parser = argparse.ArgumentParser()
    opt = parse_arguments(parser)
    conf = Config(opt)
    reader = Reader(conf.digit2zero)
    dataset, max_length, label_length = reader.read_trigger_txt(conf.trigger_file, -1)

    reader.merge_labels(dataset)

    trains = reader.read_txt(conf.train_file, conf.dev_num)
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

    encoder = SoftMatcher(conf, label_length)
    trainer = SoftMatcherTrainer(encoder, conf, devs, tests)

    # matching module training
    random.shuffle(dataset)
    trainer.train_model(10, dataset)
    logits, predicted, triggers = trainer.get_triggervec(dataset)
    triggers_remove = remove_duplicates(logits, predicted, triggers, dataset)

    numbers = int(len(trains) * 0.2)
    print("number of train instances : ", numbers)
    initial_trains = trains[:numbers]
    unlabeled_x = trains[numbers:]

    for data in unlabeled_x:
        data.output_ids = None

    # sequence labeling module self-training
    random.shuffle(dataset)
    inference = SoftSequence(conf, encoder)
    sequence_trainer = SoftSequenceTrainer(inference, conf, devs, tests, triggers_remove)
    sequence_trainer.self_training(20, dataset, unlabeled_x)

if __name__ == "__main__":
    main()














