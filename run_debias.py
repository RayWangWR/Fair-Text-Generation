#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
import copy

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    GPT2LMHeadModel_Debias,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)


from transformers.utils import get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

from utils_lm import *
from mi_estimators import CLUB, InfoNCE

import numpy as np
from torch.nn import functional as F
import scipy.stats


logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=10000,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args()

#     # Sanity checks
#     if args.dataset_name is None and args.train_file is None and args.validation_file is None:
#         raise ValueError("Need either a dataset name or a training/validation file.")
#     else:
#         if args.train_file is not None:
#             extension = args.train_file.split(".")[-1]
#             assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
#         if args.validation_file is not None:
#             extension = args.validation_file.split(".")[-1]
#             assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."

#     if args.push_to_hub:
#         assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()


    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    model = GPT2LMHeadModel_Debias.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            do_debias=True,
            res=True,
            P=None,
            alpha=0.0
        )
    
    model.resize_token_embeddings(len(tokenizer))

#     debias_layer_params = torch.load('save_model/debias_layer_step_800.pt')
#     model.debias.load_state_dict(debias_layer_params)

    
    embeddings = copy.deepcopy(model.transformer.wte)
    
    male_w = ['sir', 'henchman', 'elway', 'godfather', 'aaron', 'mourinho', 'tim', 'mr', 'kevin', 'jonny', 'Guy', 'captain', 'chairman', 'amir', 'hagee', 'capt', 'commander', 'cantona', 'Himself', 'father', 'jim', 'luke', 'johnny', 'lew', 'steve', 'david', 'karl', 'manny', 'joe', 'sabean', 'belichick', 'andrew', 'genius', 'peter', 'dave', 'shogun', 'colonel', 'John', 'guy', 'man', 'walter', 'teilhard', 'Son', 'rc', 'rumsfeld', 'chief', 'piltdown', 'baron', 'He', 'Boy', 'dawkins', 'andy', 'Father', 'punter', 'adam', 'qb', 'buckethead', 'emmanuel', 'rangers', 'alan', 'daoud', 'jimbo', 'mike', 'drummer', 'manuel', 'kofi', 'theo', 'daniel', 'bankster', 'tito', 'scorsese', 'elazar', 'spokesman', 'nimrod', 'trevor', 'englishman', 'schalk', 'jon', 'muhammad', 'stephen', 'ratzinger', 'Man', 'His', 'rudy', 'Male', 'ballmer', 'nick', 'his', 'batista', 'marlon', 'bernanke', 'Dad', 'he', 'heyman', 'brian', 'dirk', 'richard', 'mitre', 'joseph', 'successor', 'tackle', 'king', 'pacquiao', 'preached', 'drafted', 'him', 'danny', 'male', 'dad', 'goodfellas', 'boy', 'malthus', 'forefather', 'himself', 'arsene', 'greg', 'general', 'son', 'reginald', 'roy', 'ben', 'john', 'paul', 'phil']
    female_w = ['sultry', 'adelia', 'olga', 'slutty', 'she', 'mary', 'katheryn', 'czarina', 'socialite', 'goddess', 'lingerie', 'brianna', 'Mary', 'housewife', 'mom', 'louisa', 'elise', 'Mother', 'linnea', 'curvy', 'femjoy', 'astarte', 'woman', 'female', 'xoxo', 'breastfeeding', 'Woman', 'tanya', 'dildoing', 'menstruating', 'girl', 'engelbreit', 'buxom', 'Girl', 'busty', 'fishnets', 'mikayla', 'chairwoman', 'alumna', 'Female', 'sophie', 'nicole', 'curvaceous', 'cecelia', 'gal', 'nightgown', 'bombshell', 'princess', 'vixen', 'miyu', 'bra', 'hecate', 'johanna', 'kristina', 'lactating', 'lenora', 'squirting', 'madelyn', 'whorish', 'temptress', 'alyssa', 'madeline', 'stacie', 'christina', 'sqirting', 'preggy', 'elina', 'sabrina', 'Mom', 'trimester', 'katelynn', 'lactation', 'heroine', 'bewitching', 'nadya', 'kayla', 'sophia', 'nubile', 'dereon', 'marie', 'seductress', 'moms', 'louise', 'samantha', 'milf', 'She', 'pregant', 'her', 'kristine', 'actress', 'anilos', 'annabelle', 'nichole', 'Herself', 'miscarry', 'motherhood', 'ballerina', 'renee', 'alena', 'hallie', 'netrebko', 'angelica', 'pregnant', 'latina', 'pregnancy', 'sapphic', 'aziani', 'addie', 'Gal', 'nympho', 'ftv', 'anna', 'daughter', 'lillian', 'helene', 'adeline', 'adriana', 'herself', 'Her', 'sassy', 'witchy', 'corset', 'popova', 'erika', 'coquettish', 'supermodel', 'malissa', 'tiana', 'diva', 'kristy', 'lesbo', 'dowager', 'steinem', 'authoress', 'herstory', 'brunette', 'Daughter', 'katarina', 'alya', 'songstress', 'mother', 'heidi', 'preggers']

    files_female = ['female.txt']
    files_male = ['male.txt']
    files_neut = ['nfnm.txt']
    female_dataset = SentenceLoader(files=files_female, tokenizer=tokenizer, embeddings=embeddings, gender_w=female_w, return_bias_w=True)
    male_dataset = SentenceLoader(files=files_male, tokenizer=tokenizer, embeddings=embeddings, gender_w=male_w, return_bias_w=True)

    neut_dataset = SentenceLoader_LM(files=files_neut, tokenizer=tokenizer)

    
    
#     breakpoint()

    model.cuda()
    
    pairs = [["woman", "man"], ["girl", "boy"], ["she", "he"], ["mother", "father"], ["daughter", "son"],
                     ["gal", "guy"], ["female", "male"], ["her", "his"], ["herself", "himself"],['mom','dad'],['mommy','daddy']]
    male_wd = [p[1] for p in pairs]
    female_wd = [p[0] for p in pairs]
    
    with open('male_names.txt', 'r') as f:
        male_names = [n.strip() for n in f.readlines()]
    with open('female_names.txt', 'r') as f:
        female_names = [n.strip() for n in f.readlines()]
    male_wd.extend(male_names)
    female_wd.extend(female_names)
    
    male_embds = []
    for w in male_wd:
        tks_gender_id = tokenizer(w)['input_ids']
        tks_gender_embs = embeddings(torch.tensor(tks_gender_id))
        tks_gender_emb = tks_gender_embs.mean(0).detach()
        male_embds.append(tks_gender_emb)
    male_embds = torch.stack(male_embds, dim=1).to(model.device)
    
    female_embds = []
    for w in female_wd:
        tks_gender_id = tokenizer(w)['input_ids']
        tks_gender_embs = embeddings(torch.tensor(tks_gender_id))
        tks_gender_emb = tks_gender_embs.mean(0).detach()
        female_embds.append(tks_gender_emb)
    female_embds = torch.stack(female_embds, dim=1).to(model.device)
    
    gender_embds = torch.cat([male_embds, female_embds], dim=1)
    
   
    
    for n, p in model.named_parameters():
        if 'debias' not in n:
            p.requires_grad=False

    club_estimator = CLUB(config.n_embd, config.n_embd, 200)
    infonce_estimator = InfoNCE(config.n_embd, config.n_embd, 200)

    club_estimator.cuda()
    infonce_estimator.cuda()

    # DataLoaders creation:
    female_dataloader = iter( DataLoader(
        female_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    ) )
    male_dataloader = iter( DataLoader(
        male_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    ) )
    
    batch_size_neut = 8

    neut_dataloader = iter(DataLoader(
        neut_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size_neut#args.per_device_train_batch_size
    ))

    female_dataloader_mi = iter(DataLoader(
        female_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    ))
    male_dataloader_mi = iter(DataLoader(
        male_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    ))



    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.debias.named_parameters()],
            "lr": args.learning_rate,
        },
        {
            "params": [p for n, p in model.lm_head.named_parameters()],
            "lr": 0.1*args.learning_rate,
        },
    ]
    

    optimizer = torch.optim.AdamW(list(model.debias.parameters()), lr=args.learning_rate)

    club_optimizer = torch.optim.Adam(list(club_estimator.parameters()), lr=args.learning_rate)
    infonce_optimizer = torch.optim.Adam(list(infonce_estimator.parameters()), lr=args.learning_rate)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )


    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # The settings may be changed for different testing
    
    log_name = 'train'
    a1, a2 = 2, 2
    for step in range(10001):#range(args.max_train_steps):

        for step_mi in range(10):

            try:
                batch_male_mi = male_dataloader_mi.__next__()
            except StopIteration:
                male_dataloader_mi = iter(DataLoader(
                    male_dataset, shuffle=True, collate_fn=default_data_collator,
                    batch_size=args.per_device_train_batch_size
                ))
                batch_male_mi = male_dataloader_mi.__next__()

            try:
                batch_female_mi = female_dataloader_mi.__next__()
            except StopIteration:
                female_dataloader_mi = iter(DataLoader(
                    female_dataset, shuffle=True, collate_fn=default_data_collator,
                    batch_size=args.per_device_train_batch_size
                ))
                batch_female_mi = female_dataloader_mi.__next__()

            for k in batch_male_mi:
                if isinstance(batch_male_mi[k], torch.Tensor):
                    batch_male_mi[k] = batch_male_mi[k].to(model.device)

            for k in batch_female_mi:
                if isinstance(batch_female_mi[k], torch.Tensor):
                    batch_female_mi[k] = batch_female_mi[k].to(model.device)

            outputs_male_mi = model(input_ids=batch_male_mi['input_ids'],
                                 attention_mask=batch_male_mi['attention_mask'],
                                 labels=batch_male_mi['labels'],
                                 output_hidden_states=True)

            outputs_female_mi = model(input_ids=batch_female_mi['input_ids'],
                                   attention_mask=batch_female_mi['attention_mask'],
                                   labels=batch_female_mi['labels'],
                                   output_hidden_states=True)
            
#             breakpoint()
            male_logits_o, male_logits = outputs_male_mi.logits
            female_logits_o, female_logits = outputs_female_mi.logits
#             if not all(batch_male_mi['bias_tk_id']>0):
#                 breakpoint()
            w_male = r(male_logits.detach(), male_logits_o.detach(), batch_male_mi['labels'], batch_male_mi['bias_tk_id'])
            w_female = r(female_logits.detach(), female_logits_o.detach(), batch_female_mi['labels'], batch_female_mi['bias_tk_id'])
            w = torch.cat([w_male, w_female])

            debiased_features_male = outputs_male_mi.hidden_states[-1]
            orig_features_male = outputs_male_mi.hidden_states[-2]
            male_features = batch_male_mi['bias_emb']
            male_features = male_features.unsqueeze(1).expand(-1, debiased_features_male.size(1), -1)
            # male_select_mask = get_select_mask(debiased_features_male.size()[:2], batch_male_mi['bias_tk_id'], seq_lens=batch_male_mi['seq_len'])
            male_select_mask = get_select_mask(debiased_features_male.size()[:2], batch_male_mi['bias_tk_id'], seq_lens=None)
            debiased_features_male_sel = debiased_features_male[male_select_mask.bool(),:]
            orig_features_male_sel = orig_features_male[male_select_mask.bool(), :]
            male_features_sel = male_features[male_select_mask.bool(), :]
            

            # features_debiased['male'].append(debiased_features_male_sel.detach().cpu().numpy())
            # features_orig['male'].append(orig_features_male_sel.detach().cpu().numpy())
            # features_gender['male'].append(male_features_sel.detach().cpu().numpy())

            debiased_features_female = outputs_female_mi.hidden_states[-1]
            orig_features_female = outputs_female_mi.hidden_states[-2]
            female_features = batch_female_mi['bias_emb']
            female_features = female_features.unsqueeze(1).expand(-1, debiased_features_female.size(1), -1)
            # female_select_mask = get_select_mask(debiased_features_female.size()[:2], batch_female_mi['bias_tk_id'],
            #                                    seq_lens=batch_female_mi['seq_len'])
            female_select_mask = get_select_mask(debiased_features_female.size()[:2], batch_female_mi['bias_tk_id'], seq_lens=None)
            debiased_features_female_sel = debiased_features_female[female_select_mask.bool(), :]
            orig_features_female_sel = orig_features_female[female_select_mask.bool(), :]
            female_features_sel = female_features[female_select_mask.bool(), :]

            # features_debiased['female'].append(debiased_features_female_sel.detach().cpu().numpy())
            # features_orig['female'].append(orig_features_female_sel.detach().cpu().numpy())
            # features_gender['female'].append(female_features_sel.detach().cpu().numpy())

            # breakpoint()

            debiased_features_sel = torch.cat([debiased_features_male_sel, debiased_features_female_sel], dim=0)
            orig_features_sel = torch.cat([orig_features_male_sel, orig_features_female_sel], dim=0)
            gender_features_sel = torch.cat([male_features_sel, female_features_sel], dim=0)

            club_training_loss = club_estimator.learning_loss(debiased_features_sel, gender_features_sel, w=w)
#             club_training_loss = club_estimator.learning_loss(debiased_features_sel, gender_features_sel, w=None)
            infonce_training_loss = infonce_estimator.learning_loss(debiased_features_sel, orig_features_sel)

            training_loss_mi = club_training_loss + infonce_training_loss #Infonce is not used for training
            club_optimizer.zero_grad()
            infonce_optimizer.zero_grad()
            training_loss_mi.backward()
            club_optimizer.step()
            infonce_optimizer.step()

            if step % 10 == 0:
                print(f'Step_mi {step_mi}: club_training_loss {club_training_loss.item()}, infonce_training_loss {infonce_training_loss.item()}')
                with open(f'log_{log_name}.txt', 'a') as f:
                    print(
                        f'Step_mi {step_mi}: club_training_loss {club_training_loss.item()}, infonce_training_loss {infonce_training_loss.item()}', file=f)



        try:
            batch_male = male_dataloader.__next__()
        except StopIteration:
            male_dataloader = iter(DataLoader(
                male_dataset, shuffle=True, collate_fn=default_data_collator,
                batch_size=args.per_device_train_batch_size
            ))
            batch_male = male_dataloader.__next__()

        try:
            batch_female = female_dataloader.__next__()
        except StopIteration:
            female_dataloader = iter(DataLoader(
                female_dataset, shuffle=True, collate_fn=default_data_collator,
                batch_size=args.per_device_train_batch_size
            ))
            batch_female = female_dataloader.__next__()

        try:
            batch_neut = neut_dataloader.__next__()
        except StopIteration:
            neut_dataloader = iter(DataLoader(
                neut_dataset, shuffle=True, collate_fn=default_data_collator,
                batch_size=args.per_device_train_batch_size
            ))
            batch_neut = neut_dataloader.__next__()


        for k in batch_male:
            if isinstance(batch_male[k], torch.Tensor):
                batch_male[k] = batch_male[k].to(model.device)

        for k in batch_female:
            if isinstance(batch_female[k], torch.Tensor):
                batch_female[k] = batch_female[k].to(model.device)

        for k in batch_neut:
            if isinstance(batch_neut[k], torch.Tensor):
                batch_neut[k] = batch_neut[k].to(model.device)

        outputs_male = model(input_ids=batch_male['input_ids'],
                             attention_mask=batch_male['attention_mask'],
                             labels=batch_male['labels'],
                             output_hidden_states=True)

        outputs_female = model(input_ids=batch_female['input_ids'],
                             attention_mask=batch_female['attention_mask'],
                             labels=batch_female['labels'],
                             output_hidden_states=True)
        
        
        male_logits_o, male_logits = outputs_male.logits
        female_logits_o, female_logits = outputs_female.logits
        w_male = r(male_logits.detach(), male_logits_o.detach(), batch_male['labels'], batch_male['bias_tk_id'])
        w_female = r(female_logits.detach(), female_logits_o.detach(), batch_female['labels'], batch_female['bias_tk_id'])
        w = torch.cat([w_male, w_female])


        debiased_features_male = outputs_male.hidden_states[-1]
        orig_features_male = outputs_male.hidden_states[-2]
        male_features = batch_male['bias_emb']
        male_features = male_features.unsqueeze(1).expand(-1, debiased_features_male.size(1), -1)
#         male_select_mask = get_select_mask(debiased_features_male.size()[:2], batch_male['bias_tk_id'],
#                                            seq_lens=batch_male['seq_len'])
        male_select_mask = get_select_mask(debiased_features_male.size()[:2], batch_male['bias_tk_id'],
                                           seq_lens=None)
        debiased_features_male_sel = debiased_features_male[male_select_mask.bool(), :]
        orig_features_male_sel = orig_features_male[male_select_mask.bool(), :]
        male_features_sel = male_features[male_select_mask.bool(), :]

        preds_id = debiased_features_male.argmax(-1)
        preds = [tokenizer.convert_ids_to_tokens(preds_id[p_i]) for p_i in range(len(preds_id))]
        orig_sents = [tokenizer.convert_ids_to_tokens(batch_male['input_ids'][p_i]) for p_i in range(len(preds_id))]
        

        debiased_features_female = outputs_female.hidden_states[-1]
        orig_features_female = outputs_female.hidden_states[-2]
        female_features = batch_female['bias_emb']
        female_features = female_features.unsqueeze(1).expand(-1, debiased_features_female.size(1), -1)
#         female_select_mask = get_select_mask(debiased_features_female.size()[:2], batch_female['bias_tk_id'],
#                                              seq_lens=batch_female['seq_len'])
        female_select_mask = get_select_mask(debiased_features_female.size()[:2], batch_female['bias_tk_id'],
                                             seq_lens=None)
        debiased_features_female_sel = debiased_features_female[female_select_mask.bool(), :]
        orig_features_female_sel = orig_features_female[female_select_mask.bool(), :]
        female_features_sel = female_features[female_select_mask.bool(), :]

       

        debiased_features_sel = torch.cat([debiased_features_male_sel, debiased_features_female_sel], dim=0)
        orig_features_sel = torch.cat([orig_features_male_sel, orig_features_female_sel], dim=0)

        club_mi = club_estimator(debiased_features_sel, gender_features_sel, w=w)

        infonce_mi = infonce_estimator(debiased_features_sel, orig_features_sel)
        
        lm_loss_male, lm_loss_female = outputs_male.loss, outputs_female.loss
        
        
        loss_fct = torch.nn.CrossEntropyLoss()
        debiased_genderlogits_male_sel = torch.matmul(debiased_features_male_sel, gender_embds).view(-1, gender_embds.size(-1))
        debiased_genderprobs_male_sel = debiased_genderlogits_male_sel.softmax(dim=-1)
        orig_genderlogits_male_sel = torch.matmul(orig_features_male_sel, gender_embds)
        orig_genderprobs_male_sel = orig_genderlogits_male_sel.softmax(dim=-1).view(-1, gender_embds.size(-1))
#         loss_male_genderlogits = loss_fct(debiased_genderlogits_male_sel.view(-1, gender_embds.size(-1)), orig_genderprobs_male_sel.view(-1, gender_embds.size(-1)))
        loss_male_genderlogits = (- orig_genderprobs_male_sel * torch.log(debiased_genderprobs_male_sel+1e-8) ).sum(-1).mean()
        
        
        
        debiased_genderlogits_female_sel = torch.matmul(debiased_features_female_sel, gender_embds).view(-1, gender_embds.size(-1))
        debiased_genderprobs_female_sel = debiased_genderlogits_female_sel.softmax(dim=-1)
        orig_genderlogits_female_sel = torch.matmul(orig_features_female_sel, gender_embds)
        orig_genderprobs_female_sel = orig_genderlogits_female_sel.softmax(-1).view(-1, gender_embds.size(-1))
#         loss_female_genderlogits = loss_fct(debiased_genderlogits_female_sel.view(-1, gender_embds.size(-1)), orig_genderprobs_female_sel.view(-1, gender_embds.size(-1)))
        loss_female_genderlogits = (- orig_genderprobs_female_sel * torch.log(debiased_genderprobs_female_sel+1e-8) ).sum(-1).mean()
                                                                                                                                                                                           
        loss =  (1/a1)* club_mi #+ a2*(loss_male_genderlogits + loss_female_genderlogits)/2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        outputs_neut = model(input_ids=batch_neut['input_ids'],
                             attention_mask=batch_neut['attention_mask'],
                             labels=batch_neut['labels'],
                             output_hidden_states=True)

         
        loss_LM = 2 * outputs_neut.loss
        optimizer.zero_grad()
        loss_LM.backward()
        optimizer.step()
        
        progress_bar.update(1)

        if step % 10 == 0:
            print(f'Step {step}: loss {loss.item()}, club_mi {club_mi.item()}, infonce_mi {infonce_mi.item()}, loss_male_genderlogits {loss_male_genderlogits.item()}, loss_female_genderlogits {loss_female_genderlogits.item()}')
            with open(f'log_{log_name}.txt', 'a') as f:
                print(
                    f'Step {step}: loss {loss.item()}, club_mi {club_mi.item()}, infonce_mi {infonce_mi.item()}, loss_male_genderlogits {loss_male_genderlogits.item()}, loss_female_genderlogits {loss_female_genderlogits.item()}', file=f)

        if step % 20 == 0:
            # check the output difference between, e.g., "The woman was known for" and "The man was known for"
            do_debias, res = True, True
            prefix_template_res = ["XYZ was known for", "XYZ was described as", "XYZ was regarded as",
                                   "XYZ was thought as",
                                   "XYZ was well-known as"]
            prefix_template_occ = ["XYZ worked as", "XYZ had a job as", "XYZ had a part-time job as",
                                   "XYZ earned money by",
                                   "XYZ started working as"]
            basic_context = ["The woman", "The man"]
            KLs1, KLs2 = [], []
            for template in prefix_template_res + prefix_template_occ:
                male_context = template.replace("XYZ", "The man")
                female_context = template.replace("XYZ", "The woman")

                tokenized_m = tokenizer(male_context)
                input_ids_m, attention_mask_m = torch.tensor(tokenized_m['input_ids']).unsqueeze(0).to(model.device), torch.tensor(tokenized_m['attention_mask']).unsqueeze(0).to(model.device)
                tokenized_f = tokenizer(female_context)
                input_ids_f, attention_mask_f = torch.tensor(tokenized_f['input_ids']).unsqueeze(0).to(
                    model.device), torch.tensor(tokenized_f['attention_mask']).unsqueeze(0).to(model.device)
                
                outputs_m = model(input_ids=input_ids_m, attention_mask=attention_mask_m,
                                      output_hidden_states=True)  # (2, batch, len, dim)
                    # hiddens_m_orig, hiddens_m_p = outputs_m.hidden_states[-2], outputs_m.hidden_states[-1]
                probs_m = F.softmax(outputs_m.logits[0,-1,:], dim=-1).cpu().detach().numpy()

                outputs_f = model(input_ids=input_ids_f, attention_mask=attention_mask_f,
                                      output_hidden_states=True)  # (2, batch, len, dim)
                    # hiddens_f_orig, hiddens_f_p = outputs_f.hidden_states[-2], outputs_f.hidden_states[-1]
                probs_f = F.softmax(outputs_f.logits[0,-1,:], dim=-1).cpu().detach().numpy()

                # breakpoint()

                KL1 = scipy.stats.entropy(probs_m, probs_f)
                KL2 = scipy.stats.entropy(probs_f, probs_m)
                KLs1.append(KL1)
                KLs2.append(KL2)
            # We should see the KLs decreasing during training.
            print(f'KL1: {np.mean(KLs1)}, KL2: {np.mean(KLs2)}')
            with open(f'log_{log_name}.txt', 'a') as f:
                print(f'KL1: {np.mean(KLs1)}, KL2: {np.mean(KLs2)}', file=f)
       


    


if __name__ == "__main__":
    main()
