"""soft_attention.py: Structured Self-attention layer.
It creates trigger representations, and sentence representation with negative sampling.
The reason for negative sampling is to learn contrastive loss.

Written in 2020 by Dong-Ho Lee.
"""
import torch
import torch.nn as nn
import random


class SoftAttention(nn.Module):
    def __init__(self, config):
        super(SoftAttention, self).__init__()
        self.config = config
        self.device = config.device

        self.linear = nn.Linear(config.hidden_dim, config.hidden_dim // 2).to(self.device)
        self.hops = config.hidden_dim // 8
        self.ws1 = nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4, bias=False).to(self.device)
        self.ws2 = nn.Linear(config.hidden_dim // 4, self.hops, bias=False).to(self.device)
        self.tanh = nn.Tanh().to(self.device)

    def attention(self, lstm_output, mask):
        """
        Calculate structured self attention.
        :param lstm_output:
        :param mask:
        :return:
        """
        lstm_output = self.linear(lstm_output)

        size = lstm_output.size()
        compressed_reps = lstm_output.contiguous().view(-1, size[2])
        hbar = self.tanh(self.ws1(compressed_reps))  # (batch_size x seq_len) * attn_size
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # batch_size * seq_len * hops
        alphas = torch.transpose(alphas, 1, 2).contiguous().to(self.device)  # batch_size * hops * seq_len
        multi_mask = [mask.unsqueeze(1) for i in range(self.hops)]
        multi_mask = torch.cat(multi_mask, 1).to(self.device)

        penalized_alphas = alphas + -1e7 * (1 - multi_mask)
        alphas = torch.softmax(penalized_alphas.view(-1, size[1]),dim=-1).to(self.device)  # (batch_size x hops) * seq_len
        alphas = alphas.view(size[0], self.hops, size[1])  # batch_size * hops * seq_len

        lstm_output = torch.bmm(alphas, lstm_output).to(self.device)  # batch_size * hops * hidden_size
        lstm_output = lstm_output.mean(1)
        return lstm_output

    def forward(self, sentence_vec, sentence_mask, trigger_vec, trigger_mask):
        """
        Get attention for sentence and trigger. For sentence, generate negative samples
        :param sentence_vec:
        :param sentence_mask:
        :param trigger_vec:
        :param trigger_mask:
        :return:
        """
        sent_rep = self.attention(sentence_vec, sentence_mask)
        trig_rep = self.attention(trigger_vec, trigger_mask)

        # generating negative samples
        trigger_vec_cat = torch.cat([trig_rep, trig_rep], dim=0)
        random.seed(1000)
        row_idxs = list(range(sent_rep.shape[0]))
        random.shuffle(row_idxs)
        sentence_vec_rand = sent_rep[torch.tensor(row_idxs), :]
        sentence_vec_cat = torch.cat([sent_rep, sentence_vec_rand], dim=0)

        return trig_rep, sentence_vec_cat, trigger_vec_cat