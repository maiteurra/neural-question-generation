from transformers import BertForSequenceClassification
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import pickle
import torch
import wandb

from utils import Precision, Recall, F1


class BertClf(pl.LightningModule):
    def __init__(self, conf=None):
        super().__init__()
        # save conf, accessible in self.hparams.conf
        self.save_hyperparameters()
        # model
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, output_attentions=False, output_hidden_states=False)
        # metrics
        self.evaluation_metrics = torch.nn.ModuleDict({
            'train_metrics': self._init_metrics(mode='train'),
            'val_metrics': self._init_metrics(mode='val'),
            'test_metrics': self._init_metrics(mode='test'), })
        # loss
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, **kwargs):
        return self.bert(x, **kwargs)

    # optimizer
    def configure_optimizers(self):
        return torch.optim.Adam(self.bert.parameters(), lr=self.hparams.conf.training.lr)

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
        src_ids, src_mask, tgt_labels = batch['input_ids'], batch['input_attention_mask'], batch['target_labels']
        logits, = self(src_ids, attention_mask=src_mask)[:2]

        loss = self.loss(logits, tgt_labels)
        metrics = self._get_metrics(logits, tgt_labels, mode)
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
            if name == 'precision_recall_curve':
                prob = torch.nn.functional.softmax(prediction, dim=1)
                index = torch.tensor([1]).to(prob.device)
                prob = prob.index_select(dim=1, index=index).squeeze()
                metrics[name] = metric(prob, target)
            else:
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
