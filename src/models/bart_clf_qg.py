from transformers import BartForConditionalGeneration, BartTokenizer
import pytorch_lightning as pl
import torch

from utils import shift_tokens_right, update_files


class BartClfQG(pl.LightningModule):
    def __init__(self, conf=None):
        super().__init__()
        # save conf, accessible in self.hparams.conf
        self.save_hyperparameters()
        # model and tokenizer
        self.bart = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        self.linear1 = torch.nn.Linear(self.bart.config.hidden_size, 1)
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        # loss
        self.loss_classifier = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.loss_generation = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        # change embeddings to enable longer input sequences
        pos_embeddings = BartLearnedPositionalEmbedding(self.hparams.conf.dataset.setup.max_length_input_context, self.bart.config.hidden_size, padding_idx=1)
        pos_embeddings.weight.data[:1026] = self.bart.model.encoder.embed_positions.weight.data
        pos_embeddings.weight.data[1026:] = self.bart.model.encoder.embed_positions.weight.data[-1][None, :].repeat(self.hparams.conf.dataset.setup.max_length_input_context - 1026 + 2, 1)
        self.bart.model.encoder.embed_positions = pos_embeddings

    def forward(self, x, **kwargs):
        return self.bart(x, **kwargs)

    # optimizer
    def configure_optimizers(self):
        params_clf = list(self.bart.model.shared.parameters()) + list(self.bart.model.encoder.parameters())
        params_gen = self.bart.parameters()
        return torch.optim.Adam(params_clf, lr=self.hparams.conf.training.lr), torch.optim.Adam(params_gen, lr=self.hparams.conf.training.lr)
        # return torch.optim.Adam(params_gen, lr=self.hparams.conf.training.lr)

    # TRAIN
    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:  # classifier
            loss = self._get_loss_clf(batch)
            self.log('train_loss_clf', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return {'loss': loss}
        elif optimizer_idx == 1:  # generator
            loss = self._get_loss_qg(batch)
            self.log('train_loss_qg', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss_classification = self._get_loss_clf(batch)
        loss_generation = self._get_loss_qg(batch)
        self.log('val_loss_clf', loss_classification, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss_qg', loss_generation, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss_generation}

    def _get_loss_clf(self, batch):
        # get model input and target
        src_ids, src_mask, seg_idx, seg_idx_mask, tgt_labels, tgt_ids = batch['input_ids'], batch['input_attention_mask'], batch['segment_idx'], batch['segment_idx_mask'], batch['target_labels'], batch['target_ids']
        decoder_input_ids = shift_tokens_right(tgt_ids, self.tokenizer.pad_token_id)

        # get encoder output
        outputs = self.bart(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, output_hidden_states=True, return_dict=True, use_cache=False)
        encoder_output = outputs['encoder_last_hidden_state']

        # select sentence representation embeddings
        seg_idx = seg_idx.unsqueeze(dim=2).repeat(1, 1, encoder_output.shape[-1])
        sent_embeddings = encoder_output.gather(dim=1, index=seg_idx)
        # filter mask
        mask_idx = torch.nonzero(seg_idx_mask, as_tuple=True)
        sent_embeddings = sent_embeddings[mask_idx[0], mask_idx[1], :]
        tgt_labels = tgt_labels[mask_idx[0], mask_idx[1]]

        # classifier
        logits = self.linear1(sent_embeddings)
        logits = logits.squeeze().float()

        # loss
        tgt_labels = tgt_labels.float()
        loss = self.loss_classifier(logits, tgt_labels)
        loss = loss / torch.sum(seg_idx_mask)
        if loss.isnan():
            print('Nan')
            import IPython
            IPython.embed()
        return loss

    def _get_loss_qg(self, batch):
        # get model input and target
        src_ids, src_mask, tgt_ids = batch['input_ids'], batch['input_attention_mask'], batch['target_ids']
        decoder_input_ids = shift_tokens_right(tgt_ids, self.tokenizer.pad_token_id)

        # get output
        outputs = self.bart(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
        logits = outputs[0]
        # get loss
        loss = self.loss_generation(logits.view(-1, logits.shape[-1]), tgt_ids.view(-1))
        return loss

    # TEST
    def test_step(self, batch, batch_idx):
        src_ids, src_mask, tgt_ids = batch['input_ids'], batch['input_attention_mask'], batch['target_ids']
        generated_ids = self.bart.generate(
            input_ids=src_ids,
            attention_mask=src_mask,
            decoder_start_token_id=self.tokenizer.pad_token_id,
            num_beams=self.hparams.conf.testing.num_beams,
            max_length=self.hparams.conf.testing.max_length_generation,
        )

        predictions, targets, inputs = self._decode(generated_ids, tgt_ids, src_ids)
        update_files(self.hparams.conf, predictions, targets, inputs)

    def _decode(self, generated_ids, tgt_ids, src_ids):
        predictions = [self.tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) + "\n" for w in generated_ids]
        targets = [self.tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) + "\n" for w in tgt_ids]
        inputs = [self.tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) + "\n" for w in src_ids]
        return predictions, targets, inputs

    # progress bar
    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict


class BartLearnedPositionalEmbedding(torch.nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        assert padding_idx is not None, "`padding_idx` should not be None, but of type int"
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models dont have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape.shape[:2]  # changed
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions + self.offset)
