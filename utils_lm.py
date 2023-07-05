import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
import copy
import string


    
class SentenceLoader(Dataset):
    '''return the id before the first gender word'''
    def __init__(self, files, tokenizer, embeddings, gender_w, return_bias_w=False, max_len=128):
        self.lines = []
        self.eos_id = tokenizer(tokenizer.eos_token)['input_ids'][0]
        for file in files:
            with open(file, 'r') as f:
                for line in  f.readlines():
                    self.lines.append(line[:-1])
        self.data = []
        for line in self.lines:
#             sent, tks_gender, last_tk_id = line.split(' || ')
            # tks_gender = tks.split(' ')
            sent = line
            tks_gender = []
            sent_ngender = []
            meet_gender = False
            for tk_o in sent.split(' '):
#                 tk = tk_o.translate(str.maketrans('', '', string.punctuation))
                tk = tk_o
                if tk in gender_w:
                    meet_gender = True
                    tks_gender.append(tk)
                if not meet_gender:
                    sent_ngender.append(tk_o)
            sent_ngender = ' '.join(sent_ngender)
            tks_gender = ' '.join(tks_gender)
            last_tk_id = len( sent.split(' ') ) - 1
#             last_tk_id = int(last_tk_id)
            tokenized = tokenizer(sent)
            if len(tokenized['input_ids']) >= max_len:
                continue
            input_ids, attention_mask = tokenized['input_ids'], tokenized['attention_mask']
            # input_ids.append(self.eos_id)
            # attention_mask.append(1)
            seq_len = len(input_ids)
            pad_len = max_len - len(input_ids)
            input_ids.extend([self.eos_id]*pad_len)
            attention_mask.extend([0]*pad_len)

            # sample to clip
            sent_tk = sent.split(' ')
            assert last_tk_id < len(sent_tk)
            if not return_bias_w:
                bias_word_id = random.randint(last_tk_id, len(sent_tk)-1)
            else:
                bias_word_id = last_tk_id
            sent_clip = ' '.join( sent.split(' ')[:bias_word_id+1] )
            bias_tk_id = len( tokenizer(sent_clip)['input_ids'] ) - 1
            
            ngender_id = len( tokenizer(sent_ngender)['input_ids'] ) - 1
            assert ngender_id < bias_tk_id

            tks_gender_id = tokenizer(tks_gender)['input_ids']
            tks_gender_embs = embeddings(torch.tensor(tks_gender_id))
            tks_gender_emb = tks_gender_embs.mean(0).detach()
            self.data.append( {'bias_tk_id': bias_tk_id, 'input_ids': input_ids, 'attention_mask': attention_mask, 'bias_emb': tks_gender_emb, 'seq_len': seq_len, 'label_ids': torch.tensor(input_ids), 'ngender_id': ngender_id} )
        print(len(self.lines) - len(self.data))



    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        d = self.data[item]
        # return d['input_ids'], d['attention_mask'], d['bias_emb'], d['seq_len'], d['rand_tk_id']
        return d

    
    
    
    
    
    
    

class SentenceLoader_LM(Dataset):
    def __init__(self, files, tokenizer, max_len=128):
        self.lines = []
        self.eos_id = tokenizer(tokenizer.eos_token)['input_ids'][0]
        for file in files:
            with open(file, 'r') as f:
                for line in  f.readlines():
                    self.lines.append(line[:-1])
        self.data = []
        for line in self.lines:
#             sent, tks_gender, last_tk_id = line.split(' || ')
            # tks_gender = tks.split(' ')
            sent = line
            tokenized = tokenizer(sent)
            if len(tokenized['input_ids']) >= max_len:
                continue
            input_ids, attention_mask = tokenized['input_ids'], tokenized['attention_mask']
            # input_ids.append(self.eos_id)
            # attention_mask.append(1)
            seq_len = len(input_ids)
            pad_len = max_len - len(input_ids)
            input_ids.extend([self.eos_id]*pad_len)
            attention_mask.extend([0]*pad_len)

            self.data.append( {'input_ids': input_ids, 'attention_mask': attention_mask, 'label_ids': torch.tensor(input_ids)} )
        print(len(self.lines) - len(self.data))



    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        d = self.data[item]
        # return d['input_ids'], d['attention_mask'], d['bias_emb'], d['seq_len'], d['rand_tk_id']
        return d




def get_select_mask(mask_size, bias_tk_ids, seq_lens=None, scale=1):
    batch_size, max_len = mask_size
    mask = torch.zeros(mask_size)
    device = bias_tk_ids.device
    if seq_lens is None:
        mask[torch.tensor(range(batch_size), device=device), bias_tk_ids] = 1
    else:
        assert all( torch.lt(bias_tk_ids, seq_lens) )
        mask_all = copy.deepcopy(mask)
        for b in range(batch_size):
            mask_all[b, bias_tk_ids[b]:seq_lens[b]] = 1
        # indices = torch.argwhere(mask_all)
        indices = mask_all.nonzero()
        num_sel = min(len(indices), batch_size*scale)
        p_sel = torch.randperm(len(indices))[:num_sel].to(device)
        indices_sel = indices[p_sel]
        mask[indices_sel[:,0], indices_sel[:,1]] = 1
    return mask





def r(logits, logits_o, labels, ids_o, t=4):
   
    mask = torch.zeros(logits.size()[:2])
    mask_l = torch.zeros(logits.size()[:2])
    device = ids_o.device
    ids = copy.deepcopy(ids_o)
    ids[ids==0] = 1
    mask_l[torch.tensor(range(logits.size(0)), device=device), ids] = 1
#     if not all(ids>0):
#         breakpoint()
    mask[torch.tensor(range(logits.size(0)), device=device), ids-1] = 1
    
    probs = torch.nn.functional.softmax(logits / t, dim=-1)
#     breakpoint()
    probs_s = probs[mask.bool(),:]
    labels_s = labels[mask_l.bool()]
    ps = probs_s[torch.tensor(range(probs_s.size(0)), device=device), labels_s]
    assert len(ps) == logits.size(0)
    
    probs_o = torch.nn.functional.softmax(logits_o / t, dim=-1)
    probs_o_s = probs_o[mask.bool(),:]
    labels_o_s = labels[mask_l.bool()]
    ps_o = probs_o_s[torch.tensor(range(probs_o_s.size(0)), device=device), labels_o_s]
    assert len(ps_o) == logits_o.size(0)
    
    w = ps / (ps_o + 1e-7)
    
    return w / w.sum()




