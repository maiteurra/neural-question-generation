from transformers import BartForConditionalGeneration, BartTokenizer
import pytorch_lightning as pl
import torch

from utils import shift_tokens_right, update_files


class BartQG(pl.LightningModule):
    def __init__(self, conf=None):
        super().__init__()
        # save conf, accessible in self.hparams.conf
        self.save_hyperparameters()
        # model and tokenizer
        self.bart = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        # loss
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        # change embeddings to enable longer input sequences
        if self.hparams.conf.dataset.setup.context:
            pos_embeddings = BartLearnedPositionalEmbedding(self.hparams.conf.dataset.setup.max_length_input_context, self.bart.config.hidden_size, padding_idx=1)
            pos_embeddings.weight.data[:1026] = self.bart.model.encoder.embed_positions.weight.data
            pos_embeddings.weight.data[1026:] = self.bart.model.encoder.embed_positions.weight.data[-1][None, :].repeat(self.hparams.conf.dataset.setup.max_length_input_context - 1026 + 2, 1)
            self.bart.model.encoder.embed_positions = pos_embeddings

    def forward(self, x, **kwargs):
        return self.bart(x, **kwargs)

    # optimizer
    def configure_optimizers(self):
        return torch.optim.Adam(self.bart.parameters(), lr=self.hparams.conf.training.lr)

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
        # get model input and target
        src_ids, src_mask, tgt_ids = batch['input_ids'], batch['input_attention_mask'], batch['target_ids']
        decoder_input_ids = shift_tokens_right(tgt_ids, self.tokenizer.pad_token_id)

        # get output
        outputs = self.bart(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)  # output_hidden_states=True, return_dict=True
        logits = outputs[0]
        # get loss
        loss = self.loss(logits.view(-1, logits.shape[-1]), tgt_ids.view(-1))
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
