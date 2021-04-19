from transformers import BertTokenizer, BartTokenizer
import pytorch_lightning as pl
import numpy as np
import datasets
import random
import torch
import hydra
import os

from models import BertClf


class BertDataset(pl.LightningModule):
    def __init__(self, conf=None):
        super().__init__()
        # save conf, accessible in self.hparams.conf
        self.save_hyperparameters()
        # pretrained models
        self.sentence_classifier = BertClf.load_from_checkpoint(checkpoint_path=os.path.join(os.path.split(hydra.utils.get_original_cwd())[0], 'outputs', conf.model.classifier_path))
        # tokenizers
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        # classification threshold
        self.threshold = conf.model.threshold
        # dataset
        self.dataset = datasets.DatasetDict({'train': [], 'val': [], 'test': []})
        self._dataset = {'train': self._init_dataset(), 'val': self._init_dataset(), 'test': self._init_dataset()}
        self.dataset_path = os.path.join(os.path.split(hydra.utils.get_original_cwd())[0], 'datasets', 'supporting_facts') + (f'_augmented' if self.hparams.conf.model.data_augmentation else '') + f'_{self.threshold}'

    def forward(self, x, **kwargs):
        return None

    def configure_optimizers(self):
        return None

    def training_step(self, batch, batch_idx):
        self._add_predicted_supporting_facts(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self._add_predicted_supporting_facts(batch, mode='val')

    def test_step(self, batch, batch_idx):
        self._add_predicted_supporting_facts(batch, mode='test')

    def _add_predicted_supporting_facts(self, batch, mode='train'):
        src_ids, src_mask, src_text, answer, tgt_ids = batch['classifier_input_ids'], batch['classifier_input_attention_mask'], batch['sentences'], batch['answer'], batch['generator_target_ids']
        src_ids, src_mask, src_text = torch.stack(src_ids, dim=1), torch.stack(src_mask, dim=1), torch.stack(src_text, dim=1)
        batch_dim, par_dim, sent_dim = src_ids.shape
        src_ids_flat, src_mask_flat = src_ids.view(batch_dim * par_dim, sent_dim), src_mask.view(batch_dim * par_dim, sent_dim)

        logits, = self.sentence_classifier.bert(src_ids_flat, attention_mask=src_mask_flat)
        probs = torch.nn.functional.softmax(logits, dim=1)[:, 1]

        if self.hparams.conf.model.data_augmentation:
            self._data_augmentation(probs, src_text, answer, tgt_ids, mode)
        else:
            self._supporting_fact_selection(probs, src_ids, src_text, answer, tgt_ids, mode)

    def _supporting_fact_selection(self, probs, src_ids, src_text, answer, tgt_ids, mode):
        batch_dim, par_dim, sent_dim = src_ids.shape
        predictions = (probs > self.threshold).view(batch_dim, par_dim)

        # get sf
        sf_idx = torch.nonzero(predictions, as_tuple=True)
        supporting_facts = src_text[sf_idx[0], sf_idx[1], :]

        # decode
        supporting_facts_text = [''] * batch_dim
        supporting_facts_count = [0] * batch_dim
        for idx, sf in zip(sf_idx[0], supporting_facts):
            sf = self.bart_tokenizer.decode(sf, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            supporting_facts_text[idx.item()] += (' ' if len(supporting_facts_text[idx.item()]) > 0 else '') + sf
            supporting_facts_count[idx.item()] += 1
        answers = [self.bart_tokenizer.decode(ans, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ans in answer]
        questions = [self.bart_tokenizer.decode(q, skip_special_tokens=True, clean_up_tokenization_spaces=True) for q in tgt_ids]

        self._dataset[mode]['supporting_facts_count'].extend(supporting_facts_count)
        self._dataset[mode]['context_supporting_facts'].extend(supporting_facts_text)
        self._dataset[mode]['answer'].extend(answers)
        self._dataset[mode]['question'].extend(questions)

    def _data_augmentation(self, probs, src_text, answer, tgt_ids, mode):
        batch_dim, par_dim, sent_dim = src_text.shape
        assert batch_dim == 1

        probs = probs.view(batch_dim, par_dim)

        answers = [self.bart_tokenizer.decode(ans, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ans in answer]
        questions = [self.bart_tokenizer.decode(q, skip_special_tokens=True, clean_up_tokenization_spaces=True) for q in tgt_ids]
        for _ in range(4):
            number_supporting_facts = min(random.choices([2, 3], weights=[95, 5], k=1)[0], probs.shape[1])
            idx = np.random.choice(list(range(probs.shape[1])), size=number_supporting_facts, replace=False, p=torch.nn.functional.softmax(probs * 10, dim=1).detach().cpu().numpy()[0])
            sample, _ = torch.sort(torch.from_numpy(idx), dim=0)

            supporting_facts_text = ''
            batch_index = torch.tensor([0] * sample.shape[0])
            supporting_facts = src_text[batch_index, sample.long(), :]
            for sf in supporting_facts:
                sf = self.bart_tokenizer.decode(sf, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
                supporting_facts_text += (' ' if len(supporting_facts_text) > 0 else '') + sf

            # add
            self._dataset[mode]['supporting_facts_count'].append(number_supporting_facts)
            self._dataset[mode]['context_supporting_facts'].append(supporting_facts_text)
            self._dataset[mode]['answer'].extend(answers)
            self._dataset[mode]['question'].extend(questions)

    def training_epoch_end(self, outputs):
        self.dataset['train'] = datasets.Dataset.from_dict(self._dataset['train'])

    def validation_epoch_end(self, outputs):
        self.dataset['val'] = datasets.Dataset.from_dict(self._dataset['val'])

    def test_epoch_end(self, outputs):
        self.dataset['test'] = datasets.Dataset.from_dict(self._dataset['test'])
        self.dataset = self.dataset.filter(lambda x: len(x['context_supporting_facts']) > 0)
        self.dataset.save_to_disk(self.dataset_path)

    @staticmethod
    def _init_dataset():
        return {
            'supporting_facts_count': [],
            'context_supporting_facts': [],
            'answer': [],
            'question': [],
        }
