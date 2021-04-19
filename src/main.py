from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
import hydra
import torch
import os

from models import BartQG, BartClfQG, BertClf, BertClfBartQG, BertSum, BertDataset
from data import HotpotDataModule
from utils import evaluation_metrics


@hydra.main(config_name="config", config_path="../config")
def main(conf):
    # set seed
    pl.seed_everything(conf.seed)

    # load datamodule
    data = HotpotDataModule(conf=conf)

    # load model
    model_name = conf.model.name + ('_dataset' if conf.model.dataset else '')
    if conf.training.train:
        if conf.training.from_checkpoint:
            model = models[model_name].load_from_checkpoint(checkpoint_path=os.path.join(os.path.split(hydra.utils.get_original_cwd())[0], 'outputs', conf.training.from_checkpoint))
        else:
            model = models[model_name](conf=conf)
    else:
        model = models[model_name].load_from_checkpoint(checkpoint_path=os.path.join(os.path.split(hydra.utils.get_original_cwd())[0], 'outputs', conf.testing.model_path))

    # TRAINER
    callbacks = []

    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=conf.training.model_checkpoint.dirpath,
        filename=conf.training.model_checkpoint.filename,
        monitor=conf.training.model_checkpoint.monitor,
        save_last=conf.training.model_checkpoint.save_last,
        save_top_k=conf.training.model_checkpoint.save_top_k
    )
    callbacks.append(checkpoint_callback)

    # early stop callback
    if conf.training.early_stopping.early_stop:
        early_stop_callback = EarlyStopping(
            monitor=conf.training.early_stopping.monitor,
            patience=conf.training.early_stopping.patience,
            mode=conf.training.early_stopping.mode,
        )
        callbacks.append(early_stop_callback)

    # logger
    wandb_logger = WandbLogger(name=model_name, project='neural-question-generation')

    # trainer
    trainer = pl.Trainer(
        accumulate_grad_batches=conf.training.grad_cum,
        callbacks=callbacks,
        default_root_dir='.',
        deterministic=True,
        fast_dev_run=conf.debug,
        flush_logs_every_n_steps=10,
        gpus=(1 if torch.cuda.is_available() else 0),
        logger=wandb_logger,
        log_every_n_steps=100,
        max_epochs=conf.training.max_epochs,
        num_sanity_val_steps=0,
        reload_dataloaders_every_epoch=True,
        # val_check_interval=0.05,
    )

    # TODO: tune

    # train
    if conf.training.train:
        trainer.fit(model=model, datamodule=data)

    # test
    if conf.testing.test:
        trainer.test(model=model, datamodule=data)
        if model_name != 'bert_clf' and model_name != 'bert_sum' and model_name != 'bert_clf+bart_dataset':
            results = evaluation_metrics(conf)
            wandb_logger.log_metrics(results)


models = {
    'bart': BartQG,
    'bart_multi': BartClfQG,
    'bert_clf': BertClf,
    'bert_clf+bart': BertClfBartQG,
    'bert_clf+bart_dataset': BertDataset,
    'bert_sum': BertSum,
}

if __name__ == "__main__":
    main()


