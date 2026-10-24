import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import configure_nnunet
configure_nnunet()
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

class nnInteractiveMetric(nn.Module):

    def __init__(self, weights_dir, requires_grad=True, device='cuda:0'):
        super(nnInteractiveMetric, self).__init__()
        self.device = torch.device(device)
        self.num_interaction_channels = 7
        self.backbone = self._load_local_backbone(weights_dir)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.channels_dict = self._probe_architecture()
        self.num_total_stages = len(self.channels_dict)
        self.target_layers = list(range(self.num_total_stages))
        self._current_target_shape = None
        self.lin_layers = nn.ModuleDict()
        for layer in self.target_layers:
            C = self.channels_dict[layer]
            conv = nn.Conv3d(C, 1, kernel_size=1, bias=False)
            nn.init.constant_(conv.weight, 1.0)
            if not requires_grad:
                for p in conv.parameters():
                    p.requires_grad = False
            self.lin_layers[str(layer)] = conv
        self._feature_dict = {}
        self._layer_id_map = {}
        self._hook_handles = []
        for i, stage in enumerate(self.backbone.encoder.stages):
            self._layer_id_map[stage] = i
            handle = stage.register_forward_hook(self._extract_hook)
            self._hook_handles.append(handle)

    def _load_local_backbone(self, weights_dir):
        if not os.path.isdir(weights_dir):
            raise FileNotFoundError(f'Local nnInteractive model directory not found: {weights_dir}')
        print(f'Loading local nnInteractive model: {weights_dir}')
        session = nnInteractiveInferenceSession(device=self.device, use_torch_compile=False)
        session.initialize_from_trained_model_folder(weights_dir)
        print('Local nnInteractive model loaded.')
        return session.network

    def _probe_architecture(self):
        print('Inspecting encoder feature channels...')
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
                dummy = torch.zeros(1, 1 + self.num_interaction_channels, 64, 64, 64).to(self.device)
                self.backbone(dummy)
            for i, feat in enumerate(self._hook_features):
                channels_dict[i] = feat.shape[1]
        finally:
            for h in temp_hooks:
                h.remove()
            self._hook_features = []
        print(f'Encoder inspection complete: {len(channels_dict)} stages.')
        return channels_dict

    def _prepare_input(self, x):
        if x.ndim != 5 or x.shape[1] != 1 or any(size < 2 for size in x.shape[2:]):
            raise ValueError('Expected a tensor shaped (B, 1, D, H, W) with spatial dimensions >= 2.')
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

    def calc_distance(self, feat0, feat1, layer, return_spatial=False):
        f0_norm = F.normalize(feat0, p=2, dim=1, eps=1e-06)
        f1_norm = F.normalize(feat1, p=2, dim=1, eps=1e-06)
        diff_sq = (f0_norm - f1_norm) ** 2
        weighted = self.lin_layers[str(layer)](diff_sq)
        dist = torch.mean(weighted, dim=[2, 3, 4])
        scalar_dist = torch.mean(dist)
        if return_spatial:
            if self._current_target_shape is None:
                raise RuntimeError('Call forward(x) before requesting a spatial map.')
            spatial_map = F.interpolate(weighted, size=self._current_target_shape, mode='trilinear', align_corners=False)
            return (scalar_dist, spatial_map)
        return scalar_dist

    def spatial_difference(self, fixed, warped):
        if fixed.shape != warped.shape:
            raise ValueError("Fixed and warped images must have identical shapes.")
        fixed_features = self(fixed)
        warped_features = self(warped)
        spatial_map = torch.zeros_like(fixed[:, :1])
        for index, layer in enumerate(self.target_layers):
            f0 = F.normalize(fixed_features[index], p=2, dim=1, eps=1e-6)
            f1 = F.normalize(warped_features[index], p=2, dim=1, eps=1e-6)
            weighted = self.lin_layers[str(layer)]((f0 - f1) ** 2)
            spatial_map += F.interpolate(
                weighted, size=fixed.shape[2:], mode="trilinear", align_corners=False
            )
        return spatial_map / len(self.target_layers)
