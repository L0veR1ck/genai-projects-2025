import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.utils.init_utils import set_random_seed, setup_saving_and_logging_inference


warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="persongen_inference_lora")
def main(config):
    """
    Main script for inference. Instantiates the model, pipeline,
    metrics, logger, writer, and dataloaders. Runs Inferencer to
    generate and evaluate images.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    project_config = OmegaConf.to_container(config)
    
    logger = setup_saving_and_logging_inference(config)
    writer = instantiate(config.writer, logger, project_config)

    device = torch.device(config.inferencer.device)

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device, logger)

    # build model architecture, then print to console
    model = instantiate(config.model, device=device)
    model.prepare_for_training()


    metrics = []
    for metric_name in config.inference_metrics:
        metric_config = config.metrics[metric_name]
        metrics.append(instantiate(metric_config, name=metric_name, device=device))

    
    pipeline = instantiate(
        config.pipeline,
        model=model,
        device=device
    )
        
    inferencer = instantiate(
        config.inferencer,
        model=model,
        pipe=pipeline,
        metrics=metrics,
        global_config=config,
        device=device,
        dataloaders=dataloaders,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        _recursive_=False
    )

    inferencer.inference()


if __name__ == "__main__":
    main()
