from transformers import BertTokenizer, BartTokenizer
import pytorch_lightning as pl
import hydra
import torch
import os

from utils import shift_tokens_right, update_files
from models import BertClf, BartQG


class BertClfBartQG(pl.LightningModule):
    def __init__(self, conf=None):
        super().__init__()
        # save conf, accessible in self.hparams.conf
        self.save_hyperparameters()
        # pretrained models
        self.sentence_classifier = BertClf.load_from_checkpoint(checkpoint_path=os.path.join(os.path.split(hydra.utils.get_original_cwd())[0], 'outputs', conf.model.classifier_path))
        self.question_generator = BartQG.load_from_checkpoint(checkpoint_path=os.path.join(os.path.split(hydra.utils.get_original_cwd())[0], 'outputs', conf.model.generator_path))
        # tokenizers
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        # classification threshold
        self.threshold = 0.947
        # loss
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=self.bart_tokenizer.pad_token_id)

    def forward(self, x, **kwargs):
        return None

    # optimizer
    def configure_optimizers(self):
        # params = list(self.sentence_classifier.parameters()) + list(self.question_generator.parameters())
        params = self.question_generator.parameters()
        return torch.optim.Adam(params, lr=self.hparams.conf.training.lr)

    # TRAIN
    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss}

    def _get_loss(self, batch):
        src_ids, src_mask, src_text, answer, tgt_ids = batch['classifier_input_ids'], batch['classifier_input_attention_mask'], batch['sentences'], batch['answer'], batch['generator_target_ids']
        supporting_facts_text = [''] * src_ids[0].shape[0]

        src_ids, src_mask, src_text = torch.stack(src_ids, dim=1), torch.stack(src_mask, dim=1), torch.stack(src_text, dim=1)
        batch_dim, par_dim, sent_dim = src_ids.shape
        src_ids_flat, src_mask_flat = src_ids.view(batch_dim * par_dim, sent_dim), src_mask.view(batch_dim * par_dim, sent_dim)

        logits, = self.sentence_classifier.bert(src_ids_flat, attention_mask=src_mask_flat)
        probs = torch.nn.functional.softmax(logits, dim=1)[:, 1]
        predictions = (probs > self.threshold).view(batch_dim, par_dim)

        # get sf
        sf_idx = torch.nonzero(predictions, as_tuple=True)
        supporting_facts = src_text[sf_idx[0], sf_idx[1], :]

        # assuming batch=1
        supporting_facts = supporting_facts.view(-1)
        input_ids = torch.cat((supporting_facts[supporting_facts > 3], torch.tensor([self.bart_tokenizer.sep_token_id]).to('cuda'), answer[answer > 3]), 0)[:self.hparams.conf.dataset.setup.max_length_input_sf]
        input_ids = input_ids.unsqueeze(0).long()
        # input_mask = torch.cat((torch.ones_like(input_ids).to('cuda'), torch.zeros((self.hparams.conf.dataset.setup.max_length_input_sf - input_ids.shape[0])).to('cuda')), 0)
        # input_ids = torch.cat((input_ids, torch.tensor([self.bart_tokenizer.pad_token_id] * (self.hparams.conf.dataset.setup.max_length_input_sf - input_ids.shape[0])).to('cuda')))
        # input_ids, input_mask = input_ids.unsqueeze(0).long(), input_mask.unsqueeze(0).long()

        # import IPython
        # IPython.embed()

        # decode
        # for idx, sf in zip(sf_idx[0], supporting_facts):
        #     sf = self.bart_tokenizer.decode(sf, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        #     supporting_facts_text[idx.item()] += (' ' if len(supporting_facts_text[idx.item()]) > 0 else '') + sf
        # answers = [self.bart_tokenizer.decode(ans, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ans in answer]
        #
        # # encode
        # sf_input = [self.bart_tokenizer(text=supporting_facts_text[idx], text_pair=answers[idx], add_special_tokens=True, max_length=self.hparams.conf.dataset.setup.max_length_input_sf, padding="max_length", truncation=True) for idx in range(len(answers))]
        # sf_input_ids = [torch.tensor(input_dict['input_ids']) for input_dict in sf_input]
        # sf_input_attentions = [torch.tensor(input_dict['attention_mask']) for input_dict in sf_input]
        # sf_input_ids, sf_input_attentions = torch.stack(sf_input_ids).to(src_ids.device), torch.stack(sf_input_attentions).to(src_ids.device)

        # get output
        decoder_input_ids = shift_tokens_right(tgt_ids, self.bart_tokenizer.pad_token_id)
        outputs = self.question_generator(input_ids, decoder_input_ids=decoder_input_ids, use_cache=False)
        logits = outputs[0]
        # get loss
        loss = self.loss(logits.view(-1, logits.shape[-1]), tgt_ids.view(-1))
        return loss

    # TEST
    def test_step(self, batch, batch_idx):
        src_ids, src_mask, tgt_ids = batch['classifier_input_ids'], batch['classifier_input_attention_mask'], batch['generator_target_ids']
        supporting_facts = [''] * src_ids[0].shape[0]

        # classify sentences to select supporting facts
        for sentences, masks in zip(src_ids, src_mask):
            logits, = self.sentence_classifier.bert(sentences, attention_mask=masks)
            probs = torch.nn.functional.softmax(logits, dim=1)[:, 1]
            predictions = probs >= self.threshold

            # decode
            for idx, sentence in enumerate(sentences):
                if predictions[idx]:
                    sentence = self.bert_tokenizer.decode(sentence, skip_special_tokens=False, clean_up_tokenization_spaces=True)
                    sentence = sentence[sentence.find('[CLS]') + len('[CLS]'):sentence.find('[SEP]')]
                    supporting_facts[idx] += sentence

        # get context answer
        answers = []
        for sentence in src_ids[0]:
            sentence = self.bert_tokenizer.decode(sentence, skip_special_tokens=False, clean_up_tokenization_spaces=True)
            sentence = sentence[sentence.find('[SEP]') + len('[SEP]'):sentence.find('[SEP]', sentence.find('[SEP]') + len('[SEP]'), len(sentence))]
            answers.append(sentence.strip())

        # encode supporting facts
        sf_ids = []
        sf_mask = []
        for idx, sf in enumerate(supporting_facts):
            input_dict = self.bart_tokenizer(text=sf, text_pair=answers[idx], add_special_tokens=True, max_length=100, padding="max_length", truncation=True)
            ids, masks = input_dict['input_ids'], input_dict['attention_mask']
            sf_ids.append(ids)
            sf_mask.append(masks)
        sf_ids, sf_mask = torch.tensor(sf_ids).to('cuda'), torch.tensor(sf_mask).to('cuda')

        # generate questions with selected supporting facts
        generated_ids = self.question_generator.bart.generate(
            input_ids=sf_ids,
            attention_mask=sf_mask,
            decoder_start_token_id=self.bart_tokenizer.pad_token_id,
            num_beams=self.hparams.conf.testing.num_beams,
            max_length=self.hparams.conf.testing.max_length_generation
        )

        predictions, targets = self._decode(generated_ids, tgt_ids)
        update_files(self.hparams.conf, predictions, targets, supporting_facts)

    def _decode(self, generated_ids, tgt_ids):
        predictions = [self.bart_tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) + "\n" for w in generated_ids]
        targets = [self.bart_tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) + "\n" for w in tgt_ids]
        return predictions, targets

    # progress bar
    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict
