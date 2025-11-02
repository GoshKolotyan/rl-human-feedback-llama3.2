import torch

class SFTConfigs:
    def __init__(self):
        # Model settings
        self.model_id = "meta-llama/Llama-3.2-3B"
        self.torch_dtype = torch.float16
        self.device_map = "auto"
        
        # LoRA settings (optimized for 12GB GPU)
        self.lora_r = 8  # Reduced from 16 to 8 (fewer trainable params)
        self.lora_alpha = 16  # Keep 2x rank ratio
        self.lora_dropout = 0.05
        self.target_modules = ["q_proj", "v_proj"]  # Only 2 modules instead of 4
        
        # Training hyperparameters
        self.learning_rate = 2e-4
        self.num_train_epochs = 2
        self.per_device_train_batch_size = 1  # Reduced from 4 to 1 (OOM fix)
        self.gradient_accumulation_steps = 16  # Increased to maintain effective batch=16
        self.max_seq_length = 512
        
        # Optimization
        self.warmup_steps = 100
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0
        self.optim = "paged_adamw_8bit"
        self.lr_scheduler_type = "cosine"
        
        # Memory optimization
        self.fp16 = True
        self.gradient_checkpointing = True
        
        # Paths and saving
        self.output_dir = "./checkpoints/sft_model"
        self.logging_dir = "./logs/sft"
        self.save_strategy = "epoch"
        self.save_total_limit = 2
        
        # Logging and evaluation
        self.logging_steps = 10
        self.eval_steps = 500  # Evaluate less frequently to save time
        self.evaluation_strategy = "steps"  # Re-enabled with optimizations
        self.per_device_eval_batch_size = 1  # Small batch for eval


        # Dataset path 

        self.train_path = "Dataset/train"
        self.eval_path = "Dataset/test"  # Fixed: was "eval" but directory is "test"