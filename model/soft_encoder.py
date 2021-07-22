"""soft_encoder.py: Encoding sentence with LSTM.
It encodes sentence with Bi-LSTM.
After encoding, it uses all tokens for sentence, and extract some parts for trigger.

Written in 2020 by Dong-Ho Lee.
"""

from config import ContextEmb
from model.charbilstm import CharBiLSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch

class SoftEncoder(nn.Module):
    def __init__(self, config, encoder = None):
        super(SoftEncoder, self).__init__()
        self.config = config
        self.device = config.device
        self.use_char = config.use_char_rnn
        self.context_emb = config.context_emb
        self.input_size = config.embedding_dim

        if self.context_emb != ContextEmb.none:
            self.input_size += config.context_emb_size
        if self.use_char:
            self.char_feature = CharBiLSTM(config)
            self.input_size += config.charlstm_hidden_dim

        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(config.word_embedding), freeze=False).to(self.device)
        self.word_drop = nn.Dropout(config.dropout).to(self.device)
        self.lstm = nn.LSTM(self.input_size, config.hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True).to(self.device)

        if encoder is not None:
            if self.use_char:
                self.char_feature = encoder.char_feature
            self.word_embedding = encoder.word_embedding
            self.word_drop = encoder.word_drop
            self.lstm = encoder.lstm


    def forward(self, word_seq_tensor: torch.Tensor,
                word_seq_lens: torch.Tensor,
                batch_context_emb: torch.Tensor,
                char_inputs: torch.Tensor,
                char_seq_lens: torch.Tensor,
                trigger_position):

        """
        Get sentence and trigger encodings by Bi-LSTM
        :param word_seq_tensor:
        :param word_seq_lens:
        :param batch_context_emb:
        :param char_inputs:
        :param char_seq_lens:
        :param trigger_position: trigger positions in sentence (e.g. [1,4,5])
        :return:
        """

        # lstm_encoding
        word_emb = self.word_embedding(word_seq_tensor)
        if self.context_emb != ContextEmb.none:
            word_emb = torch.cat([word_emb, batch_context_emb.to(self.device)], 2)
        if self.use_char:
            char_features = self.char_feature(char_inputs, char_seq_lens)
            word_emb = torch.cat([word_emb, char_features], 2)
        word_rep = self.word_drop(word_emb)
        sorted_seq_len, permIdx = word_seq_lens.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = word_rep[permIdx]
        packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len.cpu(), True)
        output, _ = self.lstm(packed_words, None)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = output[recover_idx]
        sentence_mask = (word_seq_tensor != torch.tensor(0)).float()

        # trigger part extraction
        if trigger_position is not None:
            max_length = 0
            output_e_list = []
            output_list = [output[i, :, :] for i in range(0, word_rep.size(0))]
            for output_l, trigger_p in zip(output_list, trigger_position):
                output_e = torch.stack([output_l[p, :] for p in trigger_p])
                output_e_list.append(output_e)
                if max_length < output_e.size(0):
                    max_length = output_e.size(0)

            trigger_vec = []
            trigger_mask = []
            for output_e in output_e_list:
                trigger_vec.append(torch.cat([output_e, output_e.new_zeros(max_length - output_e.size(0), self.config.hidden_dim)], 0))
                t_ms = []
                for i in range(output_e.size(0)):
                    t_ms.append(True)
                for i in range(output_e.size(0), max_length):
                    t_ms.append(False)
                t_ms = torch.tensor(t_ms)
                trigger_mask.append(t_ms)
            trigger_vec = torch.stack(trigger_vec)
            trigger_mask = torch.stack(trigger_mask).float()
        else:
            trigger_vec = None
            trigger_mask = None

        return output, sentence_mask, trigger_vec, trigger_mask
