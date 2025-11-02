import torch
from diffusers import StableDiffusionXLPipeline


class SDXLPipeline:
    @staticmethod
    def from_pretrained(model, pretrained_model_name_or_path, torch_dtype, use_safetensors=True, variant=None, *args, **kwargs):
        """
        Create SDXL pipeline from pretrained model.
        
        Args:
            model: DiffusionLora model instance
            pretrained_model_name_or_path: Path or name of pretrained model
            torch_dtype: Data type for the pipeline
            use_safetensors: Whether to use safetensors
            variant: Model variant (e.g. 'fp16')
            
        Returns:
            StableDiffusionXLPipeline: Configured pipeline
        """
        # Convert string dtype to torch dtype
        if torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        # Create pipeline with model's components
        pipeline = StableDiffusionXLPipeline(
            vae=model.vae,
            text_encoder=model.text_encoder,
            text_encoder_2=model.text_encoder_2,
            tokenizer=model.tokenizer,
            tokenizer_2=model.tokenizer_2,
            unet=model.unet,
            scheduler=model.noise_scheduler,
        )
        
        # Move to device
        pipeline = pipeline.to(model.device)
        
        # Set to eval mode
        pipeline.vae.eval()
        pipeline.text_encoder.eval()
        pipeline.text_encoder_2.eval()
        pipeline.unet.eval()
        
        return pipeline
        