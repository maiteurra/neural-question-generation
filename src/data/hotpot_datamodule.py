from transformers import BartTokenizer, BertTokenizer
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import datasets
import logging
import random
import torch
import hydra
import tqdm
import os

log = logging.getLogger('hotpot_datamodule')


class HotpotDataModule(pl.LightningDataModule):
    def __init__(self, conf):
        super().__init__()
        # configuration
        self.conf = conf
        # dataset
        self.dataset = None
        # dataset paths
        if len(conf.dataset.preprocessing.path) > 0:
            self.dataset_path = os.path.join(os.path.split(hydra.utils.get_original_cwd())[0], 'datasets', conf.dataset.preprocessing.path)
        else:
            self.dataset_path = os.path.join(os.path.split(hydra.utils.get_original_cwd())[0], 'datasets', conf.dataset.name) + ('combined' if self.conf.dataset.preprocessing.combine else '')
        self.dataset_setup_path = self.dataset_path + '_' + conf.model.name + ('_context' if self.conf.dataset.setup.context else '')
        # tokenizers
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    def prepare_data(self):
        if self.conf.dataset.setup.from_disk:  # do not load preprocessed dataset
            dataset = None
        elif self.conf.dataset.preprocessing.from_disk and os.path.exists(self.dataset_path):  # load preprocessed dataset
            log.info(f'Loading preprocessed dataset from {self.dataset_path}')
            dataset = datasets.load_from_disk(self.dataset_path)
            # import IPython
            # IPython.embed()
            # exit(1)
        else:  # preprocess dataset
            log.info(f'Loading dataset {self.conf.dataset.name} with splits {list(self.conf.dataset.preprocessing.splits)} (combine: {self.conf.dataset.preprocessing.combine})')
            if self.conf.dataset.preprocessing.combine:
                # get raw dataset combined
                dataset_raw = datasets.load_dataset(self.conf.dataset.name, name='distractor', split='+'.join(self.conf.dataset.preprocessing.splits))
                # split into train-val-test
                dataset_raw = dataset_raw.train_test_split(test_size=self.conf.dataset.preprocessing.test_split, shuffle=self.conf.dataset.preprocessing.shuffle)
                dataset_raw['train'] = dataset_raw['train'].train_test_split(test_size=self.conf.dataset.preprocessing.val_split / (1 - self.conf.dataset.preprocessing.test_split), shuffle=self.conf.dataset.preprocessing.shuffle)
                dataset = datasets.DatasetDict({'train': dataset_raw['train']['train'], 'val': dataset_raw['train']['test'], 'test': dataset_raw['test']})
            else:
                # get raw dataset
                dataset_train, dataset_test = datasets.load_dataset(self.conf.dataset.name, name='distractor', split=list(self.conf.dataset.preprocessing.splits))
                # split into train-val-test
                dataset_train = dataset_train.train_test_split(test_size=self.conf.dataset.preprocessing.val_split, shuffle=self.conf.dataset.preprocessing.shuffle)
                dataset = datasets.DatasetDict({'train': dataset_train['train'], 'val': dataset_train['test'], 'test': dataset_test})

            # preprocess dataset
            log.info('Preprocessing dataset')
            dataset = dataset.map(self._preprocess_dataset)

            # save dataset
            log.info(f'Saving dataset to: {self.dataset_path}')
            dataset.save_to_disk(self.dataset_path)

        self.dataset = dataset

    def setup(self, stage=None):
        if stage != 'test' or (stage == 'test' and (not self.conf.training.train or self.conf.model.name == 'bert_clf+bart')):
            if self.conf.dataset.setup.from_disk and os.path.exists(self.dataset_setup_path):  # load setup dataset
                # get setup dataset
                log.info(f'Loading set up dataset from {self.dataset_setup_path}')
                self.dataset = datasets.load_from_disk(self.dataset_setup_path)
            elif self.dataset is not None:  # preprocessed dataset loaded --> setup preprocessed dataset
                log.info('Setting up dataset')
                # filter question type and difficulty
                if self.conf.dataset.setup.question_types != ['comparison', 'bridge']:
                    log.info(f'Filtering dataset by question type: {", ".join(self.conf.dataset.setup.question_types)}')
                    self.dataset = self.dataset.filter(lambda x: x['type'] in self.conf.dataset.setup.question_types)
                if self.conf.dataset.setup.difficulty_levels != ['easy', 'medium', 'hard']:
                    log.info(f'Filtering dataset by question difficulty: {", ".join(self.conf.dataset.setup.difficulty_levels)}')
                    self.dataset = self.dataset.filter(lambda x: x['level'] in self.conf.dataset.setup.difficulty_levels)

                # tokenize and format input depending on model
                log.info(f'Tokenizing for model: {self.conf.model.name}')
                if self.conf.model.name == 'bart':  # supporting facts or context for bart (generation)
                    self.dataset = self.dataset.map(function=self._tokenize_bart)
                    self.dataset.set_format(type='torch', columns=['input_ids', 'input_attention_mask', 'target_ids'])
                elif self.conf.model.name == 'bart_multi':  # supporting facts or context for bart with multi objective (classification + generation)
                    self.dataset = self.dataset.map(function=self._tokenize_bart_multi)
                    self.dataset = self.dataset.filter(lambda x: sum(x['segment_idx_mask']) > 0)  # filter examples with no sf
                    self.dataset.set_format('torch', columns=['input_ids', 'input_attention_mask', 'segment_idx', 'segment_idx_mask', 'target_labels', 'target_ids'])

                elif self.conf.model.name == 'bert_clf':  # context sentences for bert (classification)
                    self._tokenize_bert()
                    self.dataset.set_format(type='torch', columns=['input_ids', 'input_attention_mask', 'target_labels'])

                elif self.conf.model.name == 'bert_sum':  # context for bert sum (classification)
                    self.dataset = self.dataset.map(function=self._tokenize_bert_sum)
                    self.dataset = self.dataset.filter(lambda x: sum(x['segment_idx_mask']) > 0)  # filter examples with no sf
                    self.dataset.set_format(type='torch', columns=['input_ids', 'input_attention_mask', 'segment_ids', 'segment_idx', 'segment_idx_mask', 'target_labels'])

                elif self.conf.model.name == 'bert_clf+bart':  # context for bert + target for bart (generation)
                    self._tokenize_bert_bart()
                    self.dataset.set_format(type='torch', columns=['classifier_input_ids', 'classifier_input_attention_mask', 'sentences', 'answer', 'generator_target_ids'])

                # save set up dataset
                if self.conf.dataset.setup.save_to_disk:
                    log.info(f'Saving set up dataset to: {self.dataset_setup_path}')
                    self.dataset.save_to_disk(self.dataset_setup_path)

            # calculate class count and weights
            if self.conf.model.name == 'bert_clf':
                self._class_weights()

    def transfer_batch_to_device(self, batch, device):
        for key in batch.keys():
            if isinstance(batch[key], list):  # transfer each tensor in list to device
                batch[key] = [tensor.to(device) for tensor in batch[key]]
            else:  # transfer tensor to device
                batch[key] = batch[key].to(device)
        return batch

    def train_dataloader(self):
        if self.conf.model.name == 'bert_clf':  # weighted sampling
            sampler = torch.utils.data.WeightedRandomSampler(self.samples_weight, num_samples=int(self.class_sample_count[1] * 0.8), replacement=False)
            dataloader = DataLoader(self.dataset['train'], batch_size=self.conf.training.batch_size, drop_last=True, sampler=sampler)
            return dataloader
        else:
            # indices = torch.randint(len(self.dataset['train']),  (100,))
            # return DataLoader(self.dataset['train'].select(indices), batch_size=self.conf.training.batch_size, drop_last=True, shuffle=True)
            return DataLoader(self.dataset['train'], batch_size=self.conf.training.batch_size, drop_last=True, shuffle=True)

    def val_dataloader(self):
        if self.conf.model.name == 'bert_clf':
            indices = torch.randint(len(self.dataset['val']), (int(self.class_sample_count[1] * 0.1),))
            return DataLoader(self.dataset['val'].select(indices), batch_size=self.conf.training.batch_size, drop_last=False, shuffle=False)
        else:
            # indices = torch.randint(len(self.dataset['val']), (100,))
            # return DataLoader(self.dataset['val'].select(indices), batch_size=self.conf.training.batch_size, drop_last=False, shuffle=False)
            return DataLoader(self.dataset['val'], batch_size=self.conf.training.batch_size, drop_last=False, shuffle=False)

    def test_dataloader(self):
        # import IPython
        # IPython.embed()
        # exit(1)
        if self.conf.model.name == 'bert_clf':
            indices = torch.randint(len(self.dataset['test']), (int(self.class_sample_count[1] * 0.1),))
            return DataLoader(self.dataset['test'].select(indices), batch_size=self.conf.training.batch_size, drop_last=False, shuffle=False)
        else:
            # indices = torch.randint(len(self.dataset['test']), (100,))
            # return DataLoader(self.dataset['test'].select(indices), batch_size=self.conf.training.batch_size, drop_last=False, shuffle=False)
            return DataLoader(self.dataset['test'], batch_size=self.conf.training.batch_size, drop_last=False, shuffle=False)

    @staticmethod
    def _preprocess_dataset(x):
        # get supporting facts sentences and labels
        x['context_supporting_facts'] = []
        x['supporting_facts_labels'] = [[0 for _ in paragraph] for paragraph in x['context']['sentences']]
        for sent_id, title in zip(x['supporting_facts']['sent_id'], x['supporting_facts']['title']):
            paragraph_idx = x['context']['title'].index(title) if title in x['context']['title'] else -1
            if paragraph_idx != -1 and sent_id < len(x['context']['sentences'][paragraph_idx]):
                x['context_supporting_facts'].append(x['context']['sentences'][paragraph_idx][sent_id].strip())
                x['supporting_facts_labels'][paragraph_idx][sent_id] = 1
        return x

    def _tokenize_bart(self, x):
        # tokenize concatenated (supporting facts or context) + answer
        answer = x['answer'] if self.conf.dataset.setup.answer is True else None
        if self.conf.dataset.setup.context:  # context
            sentences = [s for par in x['context']['sentences'] for s in par]
            text = ' '.join(sentences)
            input_dict = self.bart_tokenizer(text=text, text_pair=answer, add_special_tokens=True, max_length=self.conf.dataset.setup.max_length_input_context, padding="max_length", truncation=True)
        else:  # supporting facts
            if len(self.conf.dataset.preprocessing.path) > 0:
                text = x['context_supporting_facts']
            else:
                text = ' '.join(x['context_supporting_facts'])
            input_dict = self.bart_tokenizer(text=text, text_pair=answer, add_special_tokens=True, max_length=self.conf.dataset.setup.max_length_input_sf, padding="max_length", truncation=True)
        x['input_ids'], x['input_attention_mask'] = input_dict['input_ids'], input_dict['attention_mask']

        # tokenize target question
        x['target_ids'] = self.bart_tokenizer.encode(x['question'], max_length=self.conf.dataset.setup.max_length_target, padding="max_length", truncation=True)
        return x

    def _tokenize_bart_multi(self, x):
        # join sentences, add [CLS] and [SEP], tokenize, and get attention mask
        sentences = [s for par in x['context']['sentences'] for s in par]
        context = ' {} {} '.format(self.bart_tokenizer.sep_token, self.bart_tokenizer.cls_token).join(sentences)
        tokenized_context = [self.bart_tokenizer.cls_token] + self.bart_tokenizer.tokenize(context) + [self.bart_tokenizer.sep_token]
        tokenized_answer = self.bart_tokenizer.tokenize(x['answer'])
        len1 = len(tokenized_context)
        tokenized_context = tokenized_context[:(self.conf.dataset.setup.max_length_input_context - len(tokenized_answer) - (2 if len1 + len(tokenized_context) + 1 > self.conf.dataset.setup.max_length_input_context else 1))]
        tokenized_context += [self.bart_tokenizer.sep_token] + ([self.bart_tokenizer.sep_token] if len1 != len(tokenized_context) else []) + tokenized_answer
        input_attention_mask = [1] * len(tokenized_context) + [0] * (self.conf.dataset.setup.max_length_input_context - len(tokenized_context))
        tokenized_context = tokenized_context + [self.bart_tokenizer.pad_token] * (self.conf.dataset.setup.max_length_input_context - len(tokenized_context))
        input_ids = self.bart_tokenizer.convert_tokens_to_ids(tokenized_context)

        # get segment idx
        segment_idx = [i for i, token in enumerate(tokenized_context) if token == self.bart_tokenizer.cls_token]

        # sentence labels
        target_labels = [x['supporting_facts_labels'][par_idx][s_idx] for par_idx, par in enumerate(x['context']['sentences']) for s_idx, s in enumerate(par)]
        target_labels = target_labels[:len(segment_idx)]
        target_labels += [0] * (60 - len(target_labels))

        # segment idx and mask
        segment_idx_mask = [1] * len(segment_idx) + [0] * (60 - len(segment_idx))
        segment_idx = segment_idx + [len(input_ids) - 1] * (60 - len(segment_idx))
        idx_zero = [idx for idx, label in enumerate(target_labels) if label == 0 and segment_idx_mask[idx] == 1]
        if len(idx_zero) > sum(target_labels):
            idx_ignore_zero = random.sample(idx_zero, k=len(idx_zero) - sum(target_labels))
            for idx in idx_ignore_zero:
                segment_idx_mask[idx] = 0

        x['input_ids'] = input_ids
        x['input_attention_mask'] = input_attention_mask
        x['segment_idx'] = segment_idx
        x['segment_idx_mask'] = segment_idx_mask
        x['target_labels'] = target_labels
        x['target_ids'] = self.bart_tokenizer.encode(x['question'], max_length=self.conf.dataset.setup.max_length_target, padding="max_length", truncation=True)
        return x

    def _tokenize_bert(self):
        # collect all sentences, tokenize and get label
        for key in self.dataset.keys():
            dataset = {'input_ids': [], 'input_attention_mask': [], 'target_labels': []}
            for x in tqdm.tqdm(self.dataset[key]):
                for paragraph_idx, paragraph in enumerate(x['context']['sentences']):
                    for sentence_idx, sentence in enumerate(paragraph):
                        input_dict = self.bert_tokenizer(text=sentence, text_pair=x['answer'], add_special_tokens=True, max_length=self.conf.dataset.setup.max_length_input_sentence, padding="max_length", truncation=True)
                        dataset['input_ids'].append(input_dict['input_ids'])
                        dataset['input_attention_mask'].append(input_dict['attention_mask'])
                        dataset['target_labels'].append(x['supporting_facts_labels'][paragraph_idx][sentence_idx])
            self.dataset[key] = datasets.Dataset.from_dict(dataset)

    def _tokenize_bert_sum(self, x):
        # join sentences, add [CLS] and [SEP], tokenize, and get attention mask
        sentences = [s for par in x['context']['sentences'] for s in par]
        context = ' {} {} '.format(self.bert_tokenizer.sep_token, self.bert_tokenizer.cls_token).join(sentences)
        tokenized_context = [self.bert_tokenizer.cls_token] + self.bert_tokenizer.tokenize(context) + [self.bert_tokenizer.sep_token]
        tokenized_context = tokenized_context[:self.conf.dataset.setup.max_length_input_context]
        input_attention_mask = [1] * len(tokenized_context) + [0] * (self.conf.dataset.setup.max_length_input_context - len(tokenized_context))
        tokenized_context = tokenized_context + [self.bert_tokenizer.pad_token] * (self.conf.dataset.setup.max_length_input_context - len(tokenized_context))
        input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokenized_context)

        # get segment ids
        segment_idx = [i for i, token in enumerate(tokenized_context) if token == self.bert_tokenizer.cls_token]
        segments = [segment_idx[i + 1] - segment_idx[i] for i in range(len(segment_idx) - 1)]
        segment_ids = []
        for i, s in enumerate(segments):
            segment_ids += s * [0] if i % 2 == 0 else s * [1]
        segment_ids += [1 - segment_ids[-1]] * (self.conf.dataset.setup.max_length_input_context - len(segment_ids))

        # sentence labels
        target_labels = [x['supporting_facts_labels'][par_idx][s_idx] for par_idx, par in enumerate(x['context']['sentences']) for s_idx, s in enumerate(par)]
        target_labels = target_labels[:len(segment_idx)]
        target_labels += [0] * (60 - len(target_labels))

        # segment idx and mask
        segment_idx_mask = [1] * len(segment_idx) + [0] * (60 - len(segment_idx))
        segment_idx = segment_idx + [len(segment_ids)-1] * (60 - len(segment_idx))
        idx_zero = [idx for idx, label in enumerate(target_labels) if label == 0 and segment_idx_mask[idx] == 1]
        if len(idx_zero) > sum(target_labels):
            idx_ignore_zero = random.sample(idx_zero, k=len(idx_zero) - sum(target_labels))
            for idx in idx_ignore_zero:
                segment_idx_mask[idx] = 0

        x['input_ids'] = input_ids
        x['input_attention_mask'] = input_attention_mask
        x['segment_ids'] = segment_ids
        x['segment_idx'] = segment_idx
        x['segment_idx_mask'] = segment_idx_mask
        x['target_labels'] = target_labels
        return x

    def _tokenize_bert_bart(self):
        # tokenize context and add padding
        for key in self.dataset.keys():
            dataset = {'classifier_input_ids': [], 'classifier_input_attention_mask': [], 'sentences': [], 'answer': [], 'generator_target_ids': []}
            for x in tqdm.tqdm(self.dataset[key]):
                sentences = {'classifier_input_ids': [], 'sentences': [], 'classifier_input_attention_mask': []}
                # get sentences and tokenize
                for paragraph_idx, paragraph in enumerate(x['context']['sentences']):
                    for sentence_idx, sentence in enumerate(paragraph):
                        sentences['sentences'].append(self.bart_tokenizer(text=sentence, add_special_tokens=True, max_length=self.conf.dataset.setup.max_length_input_sentence, padding="max_length", truncation=True)['input_ids'])
                        input_dict = self.bert_tokenizer(text=sentence, text_pair=x['answer'], add_special_tokens=True, max_length=self.conf.dataset.setup.max_length_input_sentence, padding="max_length", truncation=True)
                        sentences['classifier_input_ids'].append(input_dict['input_ids'])
                        sentences['classifier_input_attention_mask'].append(input_dict['attention_mask'])
                        if len(sentences['classifier_input_ids']) >= self.conf.dataset.setup.max_sentence_context:
                            break
                    if len(sentences['classifier_input_ids']) >= self.conf.dataset.setup.max_sentence_context:
                        break

                # context padding
                while len(sentences['classifier_input_ids']) < self.conf.dataset.setup.max_sentence_context:
                    sentences['sentences'].append([self.bart_tokenizer.pad_token_id] * self.conf.dataset.setup.max_length_input_sentence)
                    sentences['classifier_input_ids'].append([self.bert_tokenizer.pad_token_id] * self.conf.dataset.setup.max_length_input_sentence)
                    sentences['classifier_input_attention_mask'].append([0] * self.conf.dataset.setup.max_length_input_sentence)

                # add to dataset
                dataset['classifier_input_ids'].append(sentences['classifier_input_ids'])
                dataset['classifier_input_attention_mask'].append(sentences['classifier_input_attention_mask'])
                dataset['sentences'].append(sentences['sentences'])
                dataset['answer'].append(self.bart_tokenizer(text=x['answer'], add_special_tokens=True, max_length=15, padding="max_length", truncation=True)['input_ids'])
                dataset['generator_target_ids'].append(self.bart_tokenizer.encode(x['question'], max_length=self.conf.dataset.setup.max_length_target, padding="max_length", truncation=True))
            self.dataset[key] = datasets.Dataset.from_dict(dataset)

    def _class_weights(self):
        # class count
        target = np.array(self.dataset['train']['target_labels'])
        self.class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
        # weight for each class
        self.class_weight = 1. / self.class_sample_count
        # weight for each sample
        self.samples_weight = torch.from_numpy(np.array([self.class_weight[t] for t in target])).double()
