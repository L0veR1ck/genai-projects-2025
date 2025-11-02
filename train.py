import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.utils.init_utils import set_random_seed, setup_saving_and_logging


warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="persongen_train_lora")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)

    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    device = torch.device(config.trainer.device)

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device, logger)

    # build model architecture, then print to console
    model = instantiate(config.model, device=device)
    model.prepare_for_training()

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)

    metrics = []
    for metric_name in config.inference_metrics:
        metric_config = config.metrics[metric_name]
        metrics.append(instantiate(metric_config, name=metric_name, device=device))

    # build optimizer, learning rate scheduler
    trainable_params = model.get_trainable_params(config)
    optimizer = instantiate(config.optimizer, params=trainable_params)

    for i, group in enumerate(optimizer.param_groups):
        logger.info(f"Param group <{group['name']}>:")
        logger.info(f"  learning rate: {group['lr']}")
        logger.info(f"  weight decay:  {group['weight_decay']}")
        logger.info(f"  betas:  {group['betas']}")
        logger.info(f"  eps:  {group['eps']}")

        # list the names or number of params
        logger.info(f"  num params:    {len(group['params'])}")

    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)


    pipeline = instantiate(
        config.pipeline,
        model=model,
        device=device
    )

    trainer = instantiate(
        config.trainer,
        model=model,
        pipe=pipeline,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        global_config=config,
        device=device,
        dataloaders=dataloaders,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        _recursive_=False
    )

    trainer.train()


if __name__ == "__main__":
    main()
