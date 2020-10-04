from config import Reader, Config
import argparse
import random

# TODO: deletion
def trigger_percentage(dataset, percentage, original):
    inst_dictionary = dict()
    inst_copy_dictionary = dict()
    original_dictionary = dict()

    for inst in original:
        key = " ".join(word for word in inst.input.words)
        if key not in original_dictionary:
            original_dictionary[key] = [inst]
        else:
            original_dictionary[key].append(inst)

    for inst in dataset:
        key = " ".join(word for word in inst.input.words)
        if key not in inst_dictionary:
            inst_dictionary[key] = [inst]
            inst_copy_dictionary[key] = [inst]
        else:
            inst_dictionary[key].append(inst)
            inst_copy_dictionary[key].append(inst)

    for key in inst_copy_dictionary:
        candidate_labels = []
        for inst in inst_copy_dictionary[key]:
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

        for inst in inst_copy_dictionary[key]:
            inst.output = final_labels

    for key in inst_copy_dictionary:
        count = 0
        for label in inst_copy_dictionary[key][0].output:
            if label.split('-')[0] == 'B' or label.split('-')[0] == 'S':
                count += 1
        inst_copy_dictionary[key] = count

    sort_key_dictionary = {k: v for k, v in sorted(inst_copy_dictionary.items(), key=lambda item: item[1], reverse=True)}
    numbers = int(len(sort_key_dictionary) * percentage / 100)
    sort_key_list = list(sort_key_dictionary.keys())[:numbers]

    new_inst = []
    for key in sort_key_list:
        new_inst.extend(original_dictionary[key])

    text_file = open('trigger_17.txt', 'w')
    for inst in new_inst[::-1]:
        for token, label in zip(inst.input.words, inst.output):
            text_file.write("%s\t%s" % (token, label))
            text_file.write("\n")
        text_file.write("\n")
    text_file.close()


reader = Reader(True)
dataset = 'BC5CDR'
trigger_file = "dataset/" + dataset + "/trigger_20.txt"
original_dataset = reader.read_txt(trigger_file)
dataset, max_length, label_length = reader.read_trigger_txt(trigger_file, -1)
trigger_percentage(dataset, 85, original_dataset)