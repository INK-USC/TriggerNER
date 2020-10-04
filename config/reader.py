#
# @author: Allan
# edited by Dong-Ho Lee
#

from tqdm import tqdm
from common import Sentence, Instance
from typing import List
import re
from copy import deepcopy
import random
random.seed(1337)

class Reader:

    def __init__(self, digit2zero:bool=True):
        """
        Read the dataset into Instance
        :param digit2zero: convert the digits into 0, which is a common practice for LSTM-CRF.
        """
        self.digit2zero = digit2zero
        self.vocab = set()

    def read_txt(self, file: str, number: int = -1) -> List[Instance]:
        print("Reading file: " + file)
        insts = []
        with open(file, 'r', encoding='utf-8') as f:
            words = []
            labels = []
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                if line == "":
                    insts.append(Instance(Sentence(words), labels))
                    words = []
                    labels = []
                    if len(insts) == number:
                        break
                    continue
                word, label = line.split()
                if self.digit2zero:
                    word = re.sub('\d', '0', word) # replace digit with 0.
                words.append(word)
                self.vocab.add(word)
                labels.append(label)
        print("number of sentences: {}".format(len(insts)))
        return insts

    def read_trigger_txt(self, file: str, percentage, number: int = -1) -> List[Instance]:
        label_vocab = dict()
        print("Reading file: " + file)
        insts = []
        max_length = 0
        with open(file, 'r', encoding='utf-8') as f:
            words = []
            labels = []
            word_index = 0
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                if line == "":
                    # check the sequence of index to find entity which consists of multiple words
                    entity_dict = dict()
                    for ent in labels:
                        if ent[0].startswith("B-") or ent[0].startswith("I-") or ent[0].startswith("T-"):
                            if ent[0].split("-")[1] not in entity_dict:
                                entity_dict[ent[0].split("-")[1]] = [[words[ent[1]], ent[1]]]
                            else:
                                entity_dict[ent[0].split("-")[1]].append([words[ent[1]], ent[1]])

                    # entity word, index, type
                    trigger_positions = []
                    trigger_keys = []
                    for key in entity_dict:
                        if key in ['0','1','2','3','4','5','6','7','8','9']:
                            trigger_positions.append([i[1] for i in entity_dict[key]])
                            trigger_keys.append(" ".join(i[0] for i in entity_dict[key]))
                        else:
                            if key not in label_vocab:
                                label_vocab[key] = len(label_vocab)
                            trigger_label = label_vocab[key]

                    final_labels = []
                    for label in labels:
                        if label[0].startswith("T"):
                            final_labels.append("O")
                        else:
                            final_labels.append(label[0])

                    for trigger_position, trigger_key in zip(trigger_positions, trigger_keys):
                        insts.append(Instance(Sentence(words), final_labels, None, trigger_label, trigger_position, trigger_key))

                    #insts.append(Instance(Sentence(words), labels, None, trigger_label, trigger_positions))
                    if len(words) > max_length:
                        max_length = len(words)
                    words = []
                    labels = []
                    word_index = 0
                    if len(insts) == number:
                        break
                    continue
                word, label = line.split()

                if self.digit2zero:
                    word = re.sub('\d', '0', word) # replace digit with 0.
                words.append(word)
                self.vocab.add(word)
                labels.append([label, word_index])
                word_index += 1

        print("number of sentences: {}".format(len(insts)))
        return insts, max_length, len(label_vocab)


    def merge_labels(self, dataset):
        inst_dictionary = dict()

        for inst in dataset:
            key = " ".join(word for word in inst.input.words)
            if key not in inst_dictionary:
                inst_dictionary[key] = [inst]
            else:
                inst_dictionary[key].append(inst)

        for key in inst_dictionary:
            candidate_labels = []
            for inst in inst_dictionary[key]:
                candidate_labels.append(inst.output)

            final_labels = []
            for j in range(len(candidate_labels[0])):
                is_entity = False
                for i in range(len(candidate_labels)):
                    if candidate_labels[i][j] != 'O':
                        final_labels.append(candidate_labels[i][j])
                        is_entity = True
                        break
                if is_entity == False:
                    final_labels.append('O')

            for inst in inst_dictionary[key]:
                inst.output = final_labels

    def trigger_percentage(self, dataset, percentage):
        inst_dictionary = dict()

        for inst in dataset:
            key = " ".join(word for word in inst.input.words)
            if key not in inst_dictionary:
                inst_dictionary[key] = [inst]
            else:
                inst_dictionary[key].append(inst)

        numbers = int(len(inst_dictionary) * percentage / 100)
        new_inst_keys = list(inst_dictionary.keys())

        random.shuffle(new_inst_keys)
        new_inst_keys = new_inst_keys[:numbers]

        new_inst = []
        for key in new_inst_keys:
            new_inst.extend(inst_dictionary[key])

        return new_inst