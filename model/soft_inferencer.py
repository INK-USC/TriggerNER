"""soft_inferencer.py: Inference on Unlabeled Sentences
compute the similarities between the self-attended sentence representations and the trigger representations.
using the most suitable triggers as additional inputs to inferencer. (CRF)

Written in 2020 by Dong-Ho Lee.
"""
from config import ContextEmb, batching_list_instances
from config.eval import evaluate_batch_insts
from config.utils import get_optimizer
from model.linear_crf_inferencer import LinearCRF
from model.soft_encoder import SoftEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np

class SoftSequence(nn.Module):
    def __init__(self, config, softmatcher, encoder=None, print_info=True):
        super(SoftSequence, self).__init__()
        self.config = config
        self.device = config.device
        self.encoder = SoftEncoder(self.config)
        if encoder is not None:
            self.encoder = encoder

        self.softmatch_encoder = softmatcher.encoder
        self.softmatch_attention = softmatcher.attention
        self.label_size = config.label_size
        self.inferencer = LinearCRF(config, print_info=print_info)
        self.hidden2tag = nn.Linear(config.hidden_dim * 2, self.label_size).to(self.device)

        self.w1 = nn.Linear(config.hidden_dim, config.hidden_dim // 2).to(self.device)
        self.w2 = nn.Linear(config.hidden_dim // 2, config.hidden_dim // 2).to(self.device)
        self.attn1 = nn.Linear(config.hidden_dim // 2, 1).to(self.device)
        self.attn2 = nn.Linear(config.hidden_dim + config.hidden_dim // 2, 1).to(self.device)
        self.attn3 = nn.Linear(config.hidden_dim // 2, 1).to(self.device)

        self.applying = Variable(torch.randn(config.hidden_dim, config.hidden_dim // 2), requires_grad=True).to(self.device)
        self.tanh = nn.Tanh().to(self.device)
        self.perturb = nn.Dropout(config.dropout).to(self.device)


    def forward(self, word_seq_tensor: torch.Tensor,
                word_seq_lens: torch.Tensor,
                batch_context_emb: torch.Tensor,
                char_inputs: torch.Tensor,
                char_seq_lens: torch.Tensor,
                trigger_position, tags):

        batch_size = word_seq_tensor.size(0)
        max_sent_len = word_seq_tensor.size(1)

        output, sentence_mask, trigger_vec, trigger_mask = \
            self.encoder(word_seq_tensor, word_seq_lens, batch_context_emb, char_inputs, char_seq_lens,
                         trigger_position)

        if trigger_vec is not None:
            trig_rep, sentence_vec_cat, trigger_vec_cat = self.softmatch_attention(output, sentence_mask, trigger_vec, trigger_mask)

            # attention
            weights = []
            for i in range(len(output)):
                trig_applied = self.tanh(self.w1(output[i].unsqueeze(0)) + self.w2(trig_rep[i].unsqueeze(0).unsqueeze(0)))
                x = self.attn1(trig_applied) #63,1
                x = torch.mul(x.squeeze(0), sentence_mask[i].unsqueeze(1))
                x[x==0] = float('-inf')
                weights.append(x)
            normalized_weights = F.softmax(torch.stack(weights), 1)
            attn_applied1 = torch.mul(normalized_weights.repeat(1,1,output.size(2)), output)
        else:
            weights = []
            for i in range(len(output)):
                trig_applied = self.tanh(
                    self.w1(output[i].unsqueeze(0)) + self.w1(output[i].unsqueeze(0)))
                x = self.attn1(trig_applied)  # 63,1
                x = torch.mul(x.squeeze(0), sentence_mask[i].unsqueeze(1))
                x[x == 0] = float('-inf')
                weights.append(x)
            normalized_weights = F.softmax(torch.stack(weights), 1)
            attn_applied1 = torch.mul(normalized_weights.repeat(1, 1, output.size(2)), output)

        output = torch.cat([output, attn_applied1], dim=2)
        lstm_scores = self.hidden2tag(output)
        maskTemp = torch.arange(1, max_sent_len + 1, dtype=torch.long).view(1, max_sent_len).expand(batch_size,
                                                                                                    max_sent_len).to(
            self.device)
        mask = torch.le(maskTemp, word_seq_lens.view(batch_size, 1).expand(batch_size, max_sent_len)).to(self.device)

        if self.inferencer is not None:
            unlabeled_score, labeled_score = self.inferencer(lstm_scores, word_seq_lens, tags, mask)
            sequence_loss = unlabeled_score - labeled_score
        else:
            sequence_loss = self.compute_nll_loss(lstm_scores, tags, mask, word_seq_lens)

        return sequence_loss


    def decode(self, word_seq_tensor: torch.Tensor,
               word_seq_lens: torch.Tensor,
               batch_context_emb: torch.Tensor,
               char_inputs: torch.Tensor,
               char_seq_lens: torch.Tensor,
               trig_rep):

        output, sentence_mask, _, _ = \
            self.encoder(word_seq_tensor, word_seq_lens, batch_context_emb, char_inputs, char_seq_lens, None)

        soft_output, soft_sentence_mask, _, _ = \
            self.softmatch_encoder(word_seq_tensor, word_seq_lens, batch_context_emb, char_inputs, char_seq_lens, None)
        soft_sent_rep = self.softmatch_attention.attention(soft_output, soft_sentence_mask)

        trig_vec = trig_rep[0]
        trig_key = trig_rep[1]

        n = soft_sent_rep.size(0)
        m = trig_vec.size(0)
        d = soft_sent_rep.size(1)

        soft_sent_rep_dist = soft_sent_rep.unsqueeze(1).expand(n, m, d)
        trig_vec_dist = trig_vec.unsqueeze(0).expand(n, m, d)

        dist = torch.pow(soft_sent_rep_dist-trig_vec_dist, 2).sum(2).sqrt()
        dvalue, dindices = torch.min(dist, dim=1)

        trigger_list = []
        for i in dindices.tolist():
            trigger_list.append(trig_vec[i])
        trig_rep = torch.stack(trigger_list)

        # attention
        weights = []
        for i in range(len(output)):
            trig_applied = self.tanh(self.w1(output[i].unsqueeze(0)) + self.w2(trig_rep[i].unsqueeze(0).unsqueeze(0)))
            x = self.attn1(trig_applied)
            x = torch.mul(x.squeeze(0), sentence_mask[i].unsqueeze(1))
            x[x == 0] = float('-inf')
            weights.append(x)
        normalized_weights = F.softmax(torch.stack(weights), 1)
        attn_applied1 = torch.mul(normalized_weights.repeat(1, 1, output.size(2)), output)


        output = torch.cat([output, attn_applied1], dim=2)

        lstm_scores = self.hidden2tag(output)
        bestScores, decodeIdx = self.inferencer.decode(lstm_scores, word_seq_lens, None)

        return bestScores, decodeIdx


class SoftSequenceTrainer(object):
    def __init__(self, model, config, dev, test, triggers):
        self.model = model
        self.config = config
        self.device = config.device
        self.input_size = config.embedding_dim
        self.context_emb = config.context_emb
        self.use_char = config.use_char_rnn
        self.triggers = triggers
        if self.context_emb != ContextEmb.none:
            self.input_size += config.context_emb_size
        if self.use_char:
            self.input_size += config.charlstm_hidden_dim
        self.dev = dev
        self.test = test


    def train_model(self, num_epochs, train_data, eval):
        batched_data = batching_list_instances(self.config, train_data)
        self.optimizer = get_optimizer(self.config, self.model, 'sgd')
        for epoch in range(num_epochs):
            epoch_loss = 0
            self.model.zero_grad()
            for index in tqdm(np.random.permutation(len(batched_data))):
                self.model.train()
                sequence_loss = self.model(*batched_data[index][0:5], batched_data[index][-2], batched_data[index][-3])
                loss = sequence_loss
                epoch_loss = epoch_loss + loss.data
                loss.backward(retain_graph=True)
                self.optimizer.step()
                self.model.zero_grad()
            print(epoch_loss)
            if eval:
                self.model.eval()
                dev_batches = batching_list_instances(self.config, self.dev)
                test_batches = batching_list_instances(self.config, self.test)
                dev_metrics = self.evaluate_model(dev_batches, "dev", self.dev, self.triggers)
                test_metrics = self.evaluate_model(test_batches, "test", self.test, self.triggers)
                self.model.zero_grad()
        return self.model


    def self_training(self, num_epochs, train_data, unlabeled_data):
        self.optimizer = get_optimizer(self.config, self.model, 'sgd')
        merged_data = train_data
        unlabels = unlabeled_data
        for epoch in range(num_epochs):
            batched_data = batching_list_instances(self.config, merged_data)
            epoch_loss = 0
            self.model.zero_grad()
            for index in tqdm(np.random.permutation(len(batched_data))):
                self.model.train()
                sequence_loss = self.model(*batched_data[index][0:5], batched_data[index][-2], batched_data[index][-3])
                loss = sequence_loss
                epoch_loss = epoch_loss + loss.data
                loss.backward(retain_graph=True)
                self.optimizer.step()
                self.model.zero_grad()
            print(epoch_loss)

            self.model.eval()
            dev_batches = batching_list_instances(self.config, self.dev)
            test_batches = batching_list_instances(self.config, self.test)
            dev_metrics = self.evaluate_model(dev_batches, "dev", self.dev, self.triggers)
            test_metrics = self.evaluate_model(test_batches, "test", self.test, self.triggers)
            self.model.zero_grad()

            weaklabel, unlabel = self.weak_label_selftrain(unlabels, self.triggers)
            merged_data = merged_data + weaklabel
            unlabels = unlabel
            print(len(merged_data), len(weaklabel), len(unlabels))
        return self.model


    def evaluate_model(self, batch_insts_ids, name: str, insts, triggers):
        ## evaluation
        metrics = np.asarray([0, 0, 0], dtype=int)
        batch_id = 0
        batch_size = self.config.batch_size
        for batch in batch_insts_ids:
            one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
            batch_max_scores, batch_max_ids = self.model.decode(*batch[0:5], triggers)
            metrics += evaluate_batch_insts(one_batch_insts, batch_max_ids, batch[6], batch[1], self.config.idx2labels,
                                            self.config.use_crf_layer)
            batch_id += 1
        p, total_predict, total_entity = metrics[0], metrics[1], metrics[2]
        precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
        recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
        fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
        print("[%s set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (name, precision, recall, fscore), flush=True)
        return [precision, recall, fscore]


    def weakly_labeling(self, batch_insts_ids, insts, triggers):
        batch_id = 0
        batch_size = self.config.batch_size
        matched = []
        matched_scores = []
        matched_sentences = 0
        unlabeled = []
        for batch in batch_insts_ids:
            one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
            batch_max_scores, batch_max_ids = self.model.decode(*batch[0:5], triggers)
            match_indices = (torch.sum(batch_max_ids, dim=1) > 0).nonzero().squeeze(1).tolist()
            unmatch_indices = (torch.sum(batch_max_ids, dim=1) == 0).nonzero().squeeze(1).tolist()
            matched_sentences += len(match_indices)
            word_seq_lens = batch[1].cpu().numpy()
            for idx in match_indices:
                length = word_seq_lens[idx]
                prediction = batch_max_ids[idx][:length].tolist()
                prediction = prediction[::-1]
                is_match = False
                for pred in prediction:
                    if pred != self.config.label2idx['O']:
                        one_batch_insts[idx].output_ids = prediction
                        one_batch_insts[idx].trigger_label = -1
                        one_batch_insts[idx].trigger_positions = [i for i in range(0, len(prediction))]
                        matched.append(one_batch_insts[idx])
                        matched_scores.append(batch_max_scores[idx])
                        is_match = True
                        break
                if is_match == False:
                    unlabeled.append(one_batch_insts[idx])
            for idx in unmatch_indices:
                unlabeled.append(one_batch_insts[idx])
            batch_id += 1
        return matched, unlabeled, matched_scores


    def weak_label_selftrain(self, unlabeled_data, triggers):
        batched_data = batching_list_instances(self.config, unlabeled_data, is_soft=False, is_naive=True)
        weakly_labeled, unlabeled, confidence = self.weakly_labeling(batched_data, unlabeled_data, triggers)

        confidence_order = [i[0] for i in sorted(enumerate(confidence), key=lambda x: x[1])]
        threshold = int(len(confidence_order) * 0.01)
        high_confidence = confidence_order[:threshold]
        low_confidence = confidence_order[threshold:]

        final_weakly_labeled = [weakly_labeled[i] for i in high_confidence]
        unlabeled = unlabeled + [weakly_labeled[i] for i in low_confidence]

        return final_weakly_labeled, unlabeled