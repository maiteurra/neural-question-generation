from typing import Tuple

from pytorch_lightning.metrics import Metric
import torch

METRIC_EPS = 1e-6


def _input_format_classification_one_hot(preds: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert preds and target tensors into one hot spare label tensors
    Args:
        preds: tensor with probabilities/logits or multilabel tensor
        target: tensor with ground true labels
    """
    if not (preds.ndim == target.ndim + 1):
        raise ValueError("preds and target must have one additional dimension for preds")

    if preds.ndim == target.ndim + 1:
        # multi class probabilities
        preds = torch.argmax(preds, dim=1)
        target = target.long()

    return preds, target


class Precision(Metric):
    def __init__(self, dist_sync_on_step=False):
        super(Precision, self).__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state('true_positives', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('predicted_positives', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = _input_format_classification_one_hot(preds=preds, target=target)

        # multiply because we are counting (1, 1) pair for true positives
        self.true_positives += torch.sum(preds * target)
        self.predicted_positives += torch.sum(preds)

    def compute(self):
        return self.true_positives / (self.predicted_positives + METRIC_EPS)


class Recall(Metric):
    def __init__(self, dist_sync_on_step=False):
        super(Recall, self).__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state('true_positives', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('actual_positives', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = _input_format_classification_one_hot(preds=preds, target=target)
        # multiply because we are counting (1, 1) pair for true positives
        self.true_positives += torch.sum(preds * target)
        self.actual_positives += torch.sum(target)

    def compute(self):
        return self.true_positives / (self.actual_positives + METRIC_EPS)


class F1(Metric):
    def __init__(self, dist_sync_on_step=False):
        super(F1, self).__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state('true_positives', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('predicted_positives', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('actual_positives', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = _input_format_classification_one_hot(preds=preds, target=target)
        # multiply because we are counting (1, 1) pair for true positives
        self.true_positives += torch.sum(preds * target)
        self.predicted_positives += torch.sum(preds)
        self.actual_positives += torch.sum(target)

    def compute(self):
        precision = self.true_positives / (self.predicted_positives + METRIC_EPS)
        recall = self.true_positives / (self.actual_positives + METRIC_EPS)
        return 2 * (precision * recall) / (precision + recall + METRIC_EPS)


if __name__ == '__main__':
    target = torch.tensor([0, 1, 1, 1, 0])
    preds = torch.tensor([[0.4949, 0.5051],
                          [0.5000, 0.5000],
                          [0.3985, 0.6015],
                          [0.4902, 0.5098],
                          [0.5000, 0.5000]])

    precision = Precision()
    recall = Recall()
    f1 = F1()

    print('precision:', precision(preds, target))
    print('recall:', recall(preds, target))
    print('f1:', f1(preds, target))
