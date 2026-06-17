import os
import torch

try:
    from peft import get_peft_model, LoraConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class Config:
    def __init__(self):
        # --- core ---
        self.model_type = 'base'
        self.img_size = 512

        # Root holding the DINOv3 ImageNet-pretrained weights.
        # Override with the env var DINOV3_WEIGHTS_DIR.
        pretrained_dir = os.environ.get('DINOV3_WEIGHTS_DIR', 'pretrained')

        self.MODEL_MAP = {
            'small': {
                'name': 'dinov3_vits16',
                'checkpoint': os.path.join(pretrained_dir,
                                           'dinov3_vits16_pretrain_lvd1689m-08c60483.pth'),
                'embed_dim': 384,
                'depth': 12,
            },
            'base': {
                'name': 'dinov3_vitb16',
                'checkpoint': os.path.join(pretrained_dir,
                                           'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'),
                'embed_dim': 768,
                'depth': 12,
            },
            'large': {
                'name': 'dinov3_vitl16',
                'checkpoint': os.path.join(pretrained_dir,
                                           'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'),
                'embed_dim': 1024,
                'depth': 24,
            },
        }

        current = self.MODEL_MAP[self.model_type]
        self.backbone_name   = current['name']
        self.checkpoint_path = current['checkpoint']
        self.embed_dim       = current['embed_dim']
        self.depth           = current['depth']
        self.out_indices     = [int(self.depth * (i + 1) / 4) - 1 for i in range(4)]

        # --- paths ---
        # repo_dir: local clone of the official DINOv3 repo (used by torch.hub.load).
        # dataset_root: root of the prepared dataset (see README "Data structure").
        # Both can be overridden via environment variables.
        self.repo_dir      = os.environ.get('DINOV3_REPO_DIR', 'dinov3')
        self.dataset_root  = os.environ.get('DATASET_ROOT', 'data/Colon')
        dataset_name       = os.path.basename(self.dataset_root.rstrip('/'))
        self.work_dir      = f'work_dirs/{self.model_type}_{dataset_name}'

        # --- pipeline flags ---
        self.debug        = False
        self.run_stage1   = True
        self.run_stage2   = True
        self.use_amp      = False
        self.max_grad_norm = 1.0

        # --- class info (overwritten after label scan) ---
        self.num_classes     = 2
        self.label_mapping   = {0: 0, 1: 1}
        self.original_labels = [0, 1]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed   = 42

        # --- stage 1 (2.5D) ---
        self.s1_batch_size = 96 if self.img_size == 224 else 24
        self.s1_lr         = 1e-5
        self.s1_epochs     = 2 if self.debug else 50
        self.s1_patch_size = 16

        # --- stage 2 (3D) ---
        self.s2_batch_size  = 4 if self.img_size == 224 else 1
        self.s2_lr          = 1e-4
        self.s2_epochs      = 2 if self.debug else 50
        self.s2_clip_depth  = 16
        self.s2_infer_window = 16

        # --- LoRA ---
        self.use_lora      = PEFT_AVAILABLE
        self.lora_r        = 16
        self.lora_alpha    = 16
        self.lora_dropout  = 0.1
        self.lora_target   = ['qkv', 'proj', 'fc1', 'fc2']

        self.min_lr        = 1e-6
        self.voxel_spacing = (1.0, 1.0, 1.0)

    def update_model(self, model_type):
        """Switch backbone size and refresh derived fields."""
        self.model_type   = model_type
        current           = self.MODEL_MAP[model_type]
        self.backbone_name   = current['name']
        self.checkpoint_path = current['checkpoint']
        self.embed_dim       = current['embed_dim']
        self.depth           = current['depth']
        self.out_indices     = [int(self.depth * (i + 1) / 4) - 1 for i in range(4)]
        self.s1_batch_size   = 96 if self.img_size == 224 else 24
        self.s1_epochs       = 2 if self.debug else 50
        self.s2_epochs       = 2 if self.debug else 50
        print(f'>>> Config updated: model={model_type}, embed_dim={self.embed_dim}, out_indices={self.out_indices}')
