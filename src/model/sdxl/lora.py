import torch
from torch import nn
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from peft import LoraConfig, get_peft_model


class DiffusionLora(nn.Module):
    def __init__(self, pretrained_model_name_or_path, rank, lora_modules, init_lora_weights, weight_dtype, device, target_size=1024):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.target_size = target_size
        self.device = device
        self.lora_rank = rank
        self.lora_modules = lora_modules
        self.init_lora_weights = init_lora_weights
        
        # Determine weight dtype
        if weight_dtype == "fp16":
            self.weight_dtype = torch.float16
        elif weight_dtype == "bf16":
            self.weight_dtype = torch.bfloat16
        else:
            self.weight_dtype = torch.float32
        
        # Load tokenizers
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer_2",
        )
        
        # Load text encoders
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
        )
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder_2",
        )
        
        # Load noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="scheduler",
        )
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
        )
        
        # Load UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
        )
        
    def prepare_for_training(self):
        """
        Prepare model for training by freezing non-trainable components
        and adding LoRA adapters to the UNet.
        """
        # Freeze VAE and text encoders
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        
        # Freeze UNet initially
        self.unet.requires_grad_(False)
        
        # Move modules to device and dtype
        self.vae.to(self.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.device, dtype=self.weight_dtype)
        self.text_encoder_2.to(self.device, dtype=self.weight_dtype)
        self.unet.to(self.device, dtype=self.weight_dtype)
        
        # Set VAE to eval mode
        self.vae.eval()
        self.text_encoder.eval()
        self.text_encoder_2.eval()
        
        # Configure LoRA for UNet
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank,
            init_lora_weights=self.init_lora_weights,
            target_modules=self.lora_modules,
        )
        
        # Add LoRA adapters to UNet
        self.unet = get_peft_model(self.unet, lora_config)
        
        # Print trainable parameters info
        self.unet.print_trainable_parameters()
        
    def get_trainable_params(self, config):
        """
        Get trainable parameters for optimizer.
        
        Args:
            config: Hydra config with lr_for_unet
            
        Returns:
            list: List of parameter groups for optimizer
        """
        trainable_params = [
            {
                'params': self.unet.parameters(), 
                'lr': config.lr_for_unet, 
                'name': 'unet_lora'
            },
        ]
        return trainable_params
    
    def get_state_dict(self):
        """
        Return state dict of the trainable model (LoRA weights only).
        
        Returns:
            dict: State dict containing LoRA weights
        """
        return self.unet.state_dict()
    
    def load_state_dict_(self, state_dict):
        """
        Load state dict to the model.
        
        Args:
            state_dict (dict): State dict to load
        """
        self.unet.load_state_dict(state_dict, strict=False)
    
    def _encode_prompt(self, prompt):
        """
        Encode prompt using SDXL's dual text encoders.
        
        Args:
            prompt (list[str]): List of prompts
            
        Returns:
            tuple: (prompt_embeds, pooled_prompt_embeds)
        """
        batch_size = len(prompt)
        
        # Tokenize with first tokenizer
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        # Encode with first text encoder
        prompt_embeds_1 = self.text_encoder(
            text_input_ids,
            output_hidden_states=True,
        )
        pooled_prompt_embeds_1 = prompt_embeds_1[0]
        prompt_embeds_1 = prompt_embeds_1.hidden_states[-2]
        
        # Tokenize with second tokenizer
        text_inputs_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs_2.input_ids.to(self.device)
        
        # Encode with second text encoder
        prompt_embeds_2 = self.text_encoder_2(
            text_input_ids_2,
            output_hidden_states=True,
        )
        pooled_prompt_embeds_2 = prompt_embeds_2[0]
        prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]
        
        # Concatenate embeddings from both encoders
        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
        
        return prompt_embeds, pooled_prompt_embeds_2
    
    def forward(self, pixel_values, prompt, do_cfg=False, original_sizes=None, crop_top_lefts=None, *args, **kwargs):
        """
        Forward pass for training.
        
        Args:
            pixel_values (torch.Tensor): Input images of size (bs, 3, H, W)
            prompt (list[str]): List of text prompts
            do_cfg (bool): Whether to perform classifier-free guidance
            original_sizes (list): Original sizes for SDXL conditioning
            crop_top_lefts (list): Crop coordinates for SDXL conditioning
            
        Returns:
            dict: Dictionary containing 'model_pred' and 'target'
        """
        batch_size = pixel_values.shape[0]
        
        # Encode images to latent space
        latents = self.vae.encode(pixel_values.to(dtype=self.weight_dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        
        # Sample noise
        noise = torch.randn_like(latents)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device
        ).long()
        
        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Encode prompts
        prompt_embeds, pooled_prompt_embeds = self._encode_prompt(prompt)
        
        # Prepare added conditioning for SDXL
        # Default values if not provided
        if original_sizes is None:
            original_sizes = [[self.target_size, self.target_size]] * batch_size
        if crop_top_lefts is None:
            crop_top_lefts = [[0, 0]] * batch_size
        
        # Convert to tensors if needed
        if not isinstance(original_sizes, torch.Tensor):
            original_sizes = torch.tensor(original_sizes, dtype=torch.long, device=self.device)
        else:
            original_sizes = original_sizes.to(device=self.device, dtype=torch.long)
            
        if not isinstance(crop_top_lefts, torch.Tensor):
            crop_top_lefts = torch.tensor(crop_top_lefts, dtype=torch.long, device=self.device)
        else:
            crop_top_lefts = crop_top_lefts.to(device=self.device, dtype=torch.long)
            
        target_sizes = torch.tensor([[self.target_size, self.target_size]] * batch_size, dtype=torch.long, device=self.device)
        
        # Prepare time ids (SDXL specific)
        add_time_ids = torch.cat([original_sizes, crop_top_lefts, target_sizes], dim=-1)
        add_time_ids = add_time_ids.to(dtype=self.weight_dtype)
        
        # Prepare added conditioning
        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids,
        }
        
        # Handle classifier-free guidance
        if do_cfg:
            # Create unconditional embeddings
            uncond_prompt = [""] * batch_size
            uncond_prompt_embeds, uncond_pooled_prompt_embeds = self._encode_prompt(uncond_prompt)
            
            # Concatenate conditional and unconditional
            prompt_embeds = torch.cat([uncond_prompt_embeds, prompt_embeds])
            noisy_latents = torch.cat([noisy_latents, noisy_latents])
            timesteps = torch.cat([timesteps, timesteps])
            
            # Double the added conditioning
            added_cond_kwargs = {
                "text_embeds": torch.cat([uncond_pooled_prompt_embeds, pooled_prompt_embeds]),
                "time_ids": torch.cat([add_time_ids, add_time_ids]),
            }
        
        # Predict noise with UNet
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
        ).sample
        
        # Get target (what we're trying to predict)
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        
        # Handle CFG case
        if do_cfg:
            target = torch.cat([target, target])
        
        return {
            'model_pred': model_pred,
            'target': target,
        }
