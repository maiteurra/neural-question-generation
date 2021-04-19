from transformers import BertModel, BertConfig
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import torch.nn as nn
import pickle
import torch
import wandb

from utils import F1, Precision, Recall


class BertSum(pl.LightningModule):
    def __init__(self, conf=None):
        super().__init__()
        # save conf, accessible in self.hparams.conf
        self.save_hyperparameters()

        # MODEL
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # change hidden layers from 12 to 10 (memory limit)
        bert_config = BertConfig(self.bert.config.vocab_size, num_hidden_layers=10)
        self.bert = BertModel(bert_config)
        # change embeddings to enable longer input sequences
        pos_embeddings = nn.Embedding(self.hparams.conf.dataset.setup.max_length_input_context, self.bert.config.hidden_size)
        pos_embeddings.weight.data[:512] = self.bert.embeddings.position_embeddings.weight.data
        pos_embeddings.weight.data[512:] = self.bert.embeddings.position_embeddings.weight.data[-1][None, :].repeat(self.hparams.conf.dataset.setup.max_length_input_context - 512, 1)
        self.bert.embeddings.position_embeddings = pos_embeddings
        # classification layers
        self.linear1 = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        # TODO: model to encode answer (and fix)

        # metrics
        self.evaluation_metrics = torch.nn.ModuleDict({
            'train_metrics': self._init_metrics(mode='train'),
            'val_metrics': self._init_metrics(mode='val'),
            'test_metrics': self._init_metrics(mode='test'), })
        # loss
        self.loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, x, **kwargs):
        return self.bert(x, **kwargs)

    # optimizer
    def configure_optimizers(self):
        params = list(self.bert.parameters()) + list(self.linear1.parameters())
        return torch.optim.Adam(params, lr=self.hparams.conf.training.lr)

    # TRAIN
    def training_step(self, batch, batch_idx):
        loss, metrics = self._get_loss(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        metrics = self._log_metrics(metrics, mode='train')
        return {'loss': loss, **metrics}

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._get_loss(batch, mode='val')
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        metrics = self._log_metrics(metrics, mode='val')
        return {'val_loss': loss, **metrics}

    def _get_loss(self, batch, mode='train'):
        src_ids, src_mask, seg_ids, seg_idx, seg_idx_mask, tgt_labels = batch['input_ids'], batch['input_attention_mask'], batch['segment_ids'], batch['segment_idx'], batch['segment_idx_mask'], batch['target_labels']
        position_ids = torch.tensor(list(range(self.hparams.conf.dataset.setup.max_length_input_context))).to('cuda')

        # bert
        last_hidden_state, pooler_output = self.bert(src_ids, attention_mask=src_mask, token_type_ids=seg_ids, position_ids=position_ids)

        # select sentence representation embeddings
        seg_idx = seg_idx.unsqueeze(dim=2).repeat(1, 1, last_hidden_state.shape[-1])
        sent_embeddings = last_hidden_state.gather(dim=1, index=seg_idx)
        # filter mask
        mask_idx = torch.nonzero(seg_idx_mask, as_tuple=True)
        sent_embeddings = sent_embeddings[mask_idx[0], mask_idx[1], :]
        tgt_labels = tgt_labels[mask_idx[0], mask_idx[1]]

        # classifier
        logits = self.linear1(sent_embeddings)
        logits = logits.squeeze().float()

        # loss
        tgt_labels = tgt_labels.float()
        loss = self.loss(logits, tgt_labels)
        loss = loss / torch.sum(seg_idx_mask)

        # metrics
        preds = self.sigmoid(logits)
        preds = torch.stack([1 - preds, preds], dim=1)
        metrics = self._get_metrics(preds, tgt_labels, mode)

        return loss, metrics

    def validation_epoch_end(self, outputs):
        metrics = self._compute_metrics(mode='val')
        self._log_precision_recall_curve(metrics)
        self._log_confusion_matrix(metrics)

    # TEST
    def test_step(self, batch, batch_idx):
        loss, metrics = self._get_loss(batch, mode='test')
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        metrics = self._log_metrics(metrics, mode='test')
        return {'test_loss': loss, **metrics}

    def test_epoch_end(self, outputs):
        metrics = self._compute_metrics(mode='test')
        self._log_confusion_matrix(metrics)

    # progress bar
    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict

    # METRICS
    def _log_metrics(self, metrics, mode):
        metrics = {key + (f'_{mode}' if mode != 'train' else ''): value for key, value in metrics.items() if key != 'confusion_matrix' and key != 'precision_recall_curve'}
        for k, m in metrics.items():
            self.log(k, m, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return metrics

    def _log_precision_recall_curve(self, metrics):
        precision, recall, thresholds = metrics['precision_recall_curve']
        plt.plot(recall.cpu().numpy(), precision.cpu().numpy(), 'ro')
        plt.xlabel('recall')
        plt.ylabel('precision')
        self.logger.experiment.log({f'precision_recall_curve_{self.current_epoch}': wandb.Image(plt)})
        plt.clf()
        f1 = 2 * (precision * recall) / (precision + recall)
        data = {
            'precision': precision.cpu().numpy().tolist(),
            'recall': recall.cpu().numpy().tolist(),
            'f1': f1.cpu().numpy().tolist(),
            'thresholds': thresholds.cpu().numpy().tolist(),
            'argmax': torch.argmax(f1).cpu().numpy().tolist()
        }

        with open(f'precision_recall_{self.current_epoch}', 'wb') as file:
            pickle.dump(data, file)

    def _log_confusion_matrix(self, metrics):
        confusion_matrix = metrics['confusion_matrix']
        heatmap = sns.heatmap(confusion_matrix.cpu().numpy(), annot=True, fmt='g')
        figure = heatmap.get_figure()
        self.logger.experiment.log({f'confusion_matrix_{self.current_epoch}': wandb.Image(figure)})
        plt.clf()

    def _get_metrics(self, prediction, target, mode='train'):
        metrics = {}
        for name, metric in self.evaluation_metrics[mode + '_metrics'].items():
            metrics[name] = metric(prediction, target)
        return metrics

    def _compute_metrics(self, mode='train'):
        metrics = {}
        for name, metric in self.evaluation_metrics[mode + '_metrics'].items():
            metrics[name] = metric.compute()
        return metrics

    @staticmethod
    def _init_metrics(mode):
        metrics = torch.nn.ModuleDict({
            'accuracy': pl.metrics.Accuracy(),
            'f1': F1(),
            'precision': Precision(),
            'recall': Recall(),
        })
        if mode != 'train':
            metrics['confusion_matrix'] = pl.metrics.ConfusionMatrix(num_classes=2)
            metrics['precision_recall_curve'] = pl.metrics.PrecisionRecallCurve(pos_label=1)

        return metrics
