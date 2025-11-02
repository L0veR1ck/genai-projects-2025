import torch
import json
import os
from pathlib import Path

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
            
class BaseInferencer(BaseTrainer):
    def __init__(
        self,
        model,
        pipe,
        metrics,
        global_config,
        device,
        dataloaders,
        logger,
        writer,
        batch_transforms,
        # inferencer args
        epoch_len,
        epochs_to_infer,
        ckpt_dir,
        exp_save_dir,
        seed,
    ):  
        self.is_train = True

        self.config = global_config
        self.device = device

        self.logger = logger

        self.model = model
        self.pipe = pipe
        self.batch_transforms = batch_transforms
        self.writer = writer
        self.epoch_len = epoch_len

        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        # define metrics
        self.metrics = metrics
        self.evaluation_metrics = MetricTracker()
        
        self.epochs_to_infer = epochs_to_infer
        self.ckpt_dir = ckpt_dir
        self.exp_save_dir = exp_save_dir
        self.seed = seed       
    
    def inference(self):
        """
        Full inference logic

        """
        for epoch in self.epochs_to_infer:
            self._last_epoch = epoch
            result = self._inference_epoch(epoch)

            # save logged information into logs dict
            logs = {"epoch": epoch}
            logs.update(result)

            # print logged information to the screen
            for key, value in logs.items():
                self.logger.info(f"    {key:15s}: {value}")
                    
    
    def _inference_epoch(self, epoch):
        """
        Inference logic for an particular echeckpoint.

        Args:
            epoch (int): number of current checkpoint.
        Returns:
            logs (dict): logs that contain the average loss and metric in
                this epoch.
        """
        logs = {}
        self.is_train = False

        self.writer.set_step(epoch  * self.epoch_len)
        self.writer.add_scalar("general/epoch", epoch)
            
        if epoch != 0:
            ckpt_pth = Path(self.ckpt_dir) / f"checkpoint-epoch{epoch}.pth"
            self._from_pretrained(ckpt_pth)
        
        for part, dataloader in self.evaluation_dataloaders.items():
            self.images_storage = []
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            self.save_results(epoch, part)
            logs.update(**{f"{part}/{name}": value for name, value in val_logs.items()})

        return logs
    
    def store_batch(self, images, prompt):
        """
        Stora batch of images and prompts to process them lates

        Args:
            images (list[PIL.Image]): batch of generated images.
            prompt (str): prompt of generated images
        """
        self.images_storage.append((prompt, images))

    def save_results(self, epoch, part):
        """
        Process gathered data (save images, metrics and etc.)

        Args:
            epoch (int): number of current checkpoint.
            part (str): partition to evaluate on
        """
        
        output_dir = Path(self.exp_save_dir) / f"checkpoint_{epoch}/{part}"
        os.makedirs(output_dir, exist_ok=True)
        metrics_dict = {
            "prompts": []
        }
        
        for prompt, images_batch in self.images_storage:
            metrics_dict["prompts"].append(prompt)
            batch_dir = output_dir/ prompt.replace(" ", "_")
            os.makedirs(batch_dir, exist_ok=True)
            
            for i, image in enumerate(images_batch):
                imgae_pth = batch_dir / f"{i}.jpg"
                image.save(imgae_pth)
        metrics_dict.update(self.evaluation_metrics._data)
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics_dict, f)

    

class LoraInferencer(BaseInferencer):
    @torch.no_grad()
    def process_evaluation_batch(self, batch, eval_metrics):
        # Set model to eval mode
        self.model.unet.eval()
        
        # Generate images via pipeline
        generated_images = self.pipe(
            prompt=batch["prompt"],
            negative_prompt=self.config.validation_args.negative_prompt,
            num_images_per_prompt=self.config.validation_args.num_images_per_prompt,
            num_inference_steps=self.config.validation_args.num_inference_steps,
            guidance_scale=self.config.validation_args.guidance_scale,
            height=self.config.validation_args.height,
            width=self.config.validation_args.width,
        ).images
        
        batch['generated'] = generated_images
        
        for metric in self.metrics:
            metric_result = metric(**batch)
            for k, v in metric_result.items():
                eval_metrics.update(k, v)

        self.store_batch(generated_images, batch["prompt"])
                
        return batch
    
    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # For inference, we can skip batch logging as we save images to disk
        pass
