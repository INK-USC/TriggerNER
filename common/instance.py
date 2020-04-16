#
# @author: Allan
# edited by Dong-Ho Lee
#
from common.sentence import  Sentence
from typing import List

class Instance:
    """
    This class is the basic Instance for a datasample
    """

    def __init__(self, input: Sentence, output: List[str] = None, embedding = None, trigger_label = None, trigger_positions = None, trigger_key = None) -> None:
        """
        Constructor for the instance.
        :param input: sentence containing the words
        :param output: a list of labels
        """
        self.input = input
        self.output = output
        self.elmo_vec = embedding #used for loading the ELMo vector.
        self.word_ids = None
        self.char_ids = None
        self.output_ids = None
        self.is_prediction = None
        self.trigger_label = trigger_label
        self.trigger_positions = trigger_positions
        self.trigger_key = trigger_key
        self.trigger_vec = None
        self.trigger_matching_label = None

    def __len__(self):
        return len(self.input)
