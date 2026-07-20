import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class nnInteractiveMetric(nn.Module):
    """
    A 3D perceptual-distance metric backed by local nnInteractive weights.

    All encoder stages are exposed through ``target_layers`` and spatial
    distance maps can be restored to the input resolution.
    """

    def __init__(self, weights_dir, requires_grad=True, device=None, nnunet_base=None):
        super(nnInteractiveMetric, self).__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._configure_nnunet(nnunet_base)
        self.num_interaction_channels = 7

        # Load and freeze the local nnInteractive backbone.
        self.backbone = self._load_local_backbone(weights_dir)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Probe the backbone to discover stage channel counts.
        self.channels_dict = self._probe_architecture()
        self.num_total_stages = len(self.channels_dict)

        # Training uses every encoder stage by default.
        self.target_layers = list(range(self.num_total_stages))

        # Store the input shape for optional spatial-map upsampling.
        self._current_target_shape = None

        # Build one learnable channel-weight projection per stage.
        self.lin_layers = nn.ModuleDict()
        for layer in self.target_layers:
            C = self.channels_dict[layer]
            conv = nn.Conv3d(C, 1, kernel_size=1, bias=False)
            nn.init.constant_(conv.weight, 1.0)
            if not requires_grad:
                for p in conv.parameters():
                    p.requires_grad = False
            self.lin_layers[str(layer)] = conv

        # ================================
        # Register persistent feature-extraction hooks.
        # ================================
        self._feature_dict = {}
        self._layer_id_map = {}
        self._hook_handles = []
        for i, stage in enumerate(self.backbone.encoder.stages):
            self._layer_id_map[stage] = i
            handle = stage.register_forward_hook(self._extract_hook)
            self._hook_handles.append(handle)

    @staticmethod
    def _configure_nnunet(nnunet_base):
        """Configure nnU-Net paths without embedding machine-specific locations."""
        if nnunet_base is None:
            return
        base_path = Path(nnunet_base).expanduser().resolve()
        os.environ["nnUNet_raw"] = str(base_path / "raw")
        os.environ["nnUNet_preprocessed"] = str(base_path / "preprocessed")
        os.environ["nnUNet_results"] = str(base_path / "results")

    def _load_local_backbone(self, weights_dir):
        weights_dir = Path(weights_dir).expanduser()
        if not weights_dir.exists():
            raise FileNotFoundError(f"Local weights directory not found: {weights_dir}")
        try:
            from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
        except ImportError as exc:
            raise ImportError(
                "nnInteractiveMetric requires nnInteractive. "
                "Install it following the official nnInteractive instructions."
            ) from exc
        print(f"Loading nnInteractive model from {weights_dir}")
        session = nnInteractiveInferenceSession(device=self.device, use_torch_compile=False)
        session.initialize_from_trained_model_folder(str(weights_dir))
        print("nnInteractive model loaded successfully")
        return session.network

    def _probe_architecture(self):
        print("Probing nnInteractive encoder stages")
        channels_dict = {}
        temp_hooks = []
        self._hook_features = []

        def probe_hook(module, input, output):
            feat = output[0] if isinstance(output, (list, tuple)) else output
            self._hook_features.append(feat)

        for stage in self.backbone.encoder.stages:
            temp_hooks.append(stage.register_forward_hook(probe_hook))
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, 1 + self.num_interaction_channels, 64, 64, 64, device=self.device)
                self.backbone(dummy)
        finally:
            for handle in temp_hooks:
                handle.remove()
        for i, feat in enumerate(self._hook_features):
            channels_dict[i] = feat.shape[1]
        self._hook_features = []
        print(f"Discovered {len(channels_dict)} encoder stages")
        return channels_dict

    def _prepare_input(self, x):
        if x.ndim != 5 or x.shape[1] != 1:
            raise ValueError("Expected input shape (batch, 1, depth, height, width)")
        B, _, D, H, W = x.shape
        prompts = torch.zeros((B, self.num_interaction_channels, D, H, W), dtype=x.dtype, device=x.device)
        return torch.cat([x, prompts], dim=1)

    def _extract_hook(self, module, input, output):
        feat = output[0] if isinstance(output, (list, tuple)) else output
        layer_id = self._layer_id_map[module]
        self._feature_dict[layer_id] = feat

    def forward(self, x):
        self._current_target_shape = x.shape[2:]
        x_in = self._prepare_input(x)
        self._feature_dict = {}
        self.backbone(x_in)

        features = [self._feature_dict[i] for i in self.target_layers]
        return features

    def train(self, mode=True):
        """Keep the frozen nnInteractive backbone in evaluation mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def calc_distance(self, feat0, feat1, layer, return_spatial=False):
        f0_norm = F.normalize(feat0, p=2, dim=1, eps=1e-6)
        f1_norm = F.normalize(feat1, p=2, dim=1, eps=1e-6)
        diff_sq = (f0_norm - f1_norm) ** 2

        weighted = self.lin_layers[str(layer)](diff_sq)
        dist = torch.mean(weighted, dim=[2, 3, 4])
        scalar_dist = torch.mean(dist)

        if return_spatial:
            if self._current_target_shape is None:
                raise RuntimeError("Input shape is unavailable; call forward(x) before requesting a spatial map")
            spatial_map = F.interpolate(weighted, size=self._current_target_shape, mode='trilinear',
                                        align_corners=False)
            return scalar_dist, spatial_map
        return scalar_dist


# PEP 8-friendly public name while retaining compatibility with existing code.
NNInteractiveMetric = nnInteractiveMetric
