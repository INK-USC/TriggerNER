"""soft_matcher.py: Joint training between trigger encoder and trigger matcher
It jointly trains the trigger encoder and trigger matcher

trigger encoder -> classification of trigger representation from encoder / attention
trigger matcher -> contrastive loss of sentence representation and trigger representation from encoder / attention

Written in 2020 by Dong-Ho Lee.
"""

from config import ContextEmb, batching_list_instances
from config.utils import get_optimizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.soft_encoder import SoftEncoder
from model.soft_attention import SoftAttention
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np

class ContrastiveLoss(nn.Module):
    def __init__(self, margin, device):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9
        self.device = device

    def forward(self, output1, output2, target, size_average=True):
        target = target.to(self.device)
        distances = (output2 - output1).pow(2).sum(1).to(self.device)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class SoftMatcher(nn.Module):
    def __init__(self, config, num_classes):
        super(SoftMatcher, self).__init__()
        self.config = config
        self.device = config.device

        self.encoder = SoftEncoder(self.config)
        self.attention = SoftAttention(self.config)

        # final calssification layers
        self.trigger_type_layer = nn.Linear(config.hidden_dim // 2, num_classes).to(self.device)

    def forward(self, word_seq_tensor: torch.Tensor,
                word_seq_lens: torch.Tensor,
                batch_context_emb: torch.Tensor,
                char_inputs: torch.Tensor,
                char_seq_lens: torch.Tensor,
                trigger_position):

        output, sentence_mask, trigger_vec, trigger_mask = \
            self.encoder(word_seq_tensor, word_seq_lens, batch_context_emb, char_inputs, char_seq_lens, trigger_position)
        trig_rep, sentence_vec_cat, trigger_vec_cat = self.attention(output, sentence_mask, trigger_vec, trigger_mask)
        final_trigger_type = self.trigger_type_layer(trig_rep)
        return trig_rep, F.log_softmax(final_trigger_type, dim=1), sentence_vec_cat, trigger_vec_cat


class SoftMatcherTrainer(object):
    def __init__(self, model, config, dev, test):
        self.model = model
        self.config = config
        self.device = config.device
        self.input_size = config.embedding_dim
        self.context_emb = config.context_emb
        self.use_char = config.use_char_rnn
        if self.context_emb != ContextEmb.none:
            self.input_size += config.context_emb_size
        if self.use_char:
            self.input_size += config.charlstm_hidden_dim
        self.contrastive_loss = ContrastiveLoss(1.0, self.device)
        self.dev = dev
        self.test = test

    def train_model(self, num_epochs, train_data):
        batched_data = batching_list_instances(self.config, train_data)
        self.optimizer = get_optimizer(self.config, self.model, 'adam')
        criterion = nn.NLLLoss()
        for epoch in range(num_epochs):
            epoch_loss = 0
            self.model.zero_grad()
            for index in tqdm(np.random.permutation(len(batched_data))):
                self.model.train()
                trig_rep, trig_type_probas, match_trig, match_sent = self.model(*batched_data[index][0:5], batched_data[index][-2])
                trigger_loss = criterion(trig_type_probas, batched_data[index][-1])
                soft_matching_loss = self.contrastive_loss(match_trig, match_sent, torch.stack([torch.tensor(1)]*trig_rep.size(0) + [torch.tensor(0)]*trig_rep.size(0)))
                loss = trigger_loss + soft_matching_loss
                epoch_loss = epoch_loss + loss.data
                loss.backward(retain_graph=True)
                self.optimizer.step()
                self.model.zero_grad()
            print(epoch_loss)
            self.test_model(train_data)
            self.model.zero_grad()

        return self.model


    def test_model(self, test_data):
        batched_data = batching_list_instances(self.config, test_data)
        self.model.eval()
        predicted_list = []
        target_list = []
        match_target_list = []
        matched_list = []
        for index in tqdm(np.random.permutation(len(batched_data))):
            trig_rep, trig_type_probas, match_trig, match_sent = self.model(*batched_data[index][0:5], batched_data[index][-2])
            trig_type_value, trig_type_predicted = torch.max(trig_type_probas, 1)
            target = batched_data[index][-1]
            target_list.extend(target.tolist())
            predicted_list.extend(trig_type_predicted.tolist())

            match_target_list.extend([torch.tensor(1)]*trig_rep.size(0) + [torch.tensor(0)]*trig_rep.size(0))
            distances = (match_trig - match_sent).pow(2).sum(1)
            distances = torch.sqrt(distances)
            matched_list.extend((distances < 1.0).long().tolist())

        print("trigger classification accuracy ", accuracy_score(predicted_list, target_list))
        print("soft matching accuracy ", accuracy_score(matched_list, match_target_list))


    def get_triggervec(self, data):
        batched_data = batching_list_instances(self.config, data)
        self.model.eval()
        logits_list = []
        predicted_list = []
        trigger_list = []
        for index in tqdm(range(len(batched_data))):
            trig_rep, trig_type_probas, match_trig, match_sent = self.model(*batched_data[index][0:5], batched_data[index][-2])
            trig_type_value, trig_type_predicted = torch.max(trig_type_probas, 1)
            ne_batch_insts = data[index * self.config.batch_size:(index + 1) * self.config.batch_size]
            for idx in range(len(trig_rep)):
                ne_batch_insts[idx].trigger_vec = trig_rep[idx]
            logits_list.extend(trig_rep)
            predicted_list.extend(trig_type_predicted)
            word_seq = batched_data[index][0]
            trigger_positions = batched_data[index][-2]

            for ws, tp in zip(word_seq, trigger_positions):
                trigger_list.append(" ".join(self.config.idx2word[ws[index]] for index in tp))

        return logits_list, predicted_list, trigger_list
