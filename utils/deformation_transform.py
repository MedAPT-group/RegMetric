import torch
import torch.nn.functional as F
import math

def cubic_bspline_kernel(device):
    k = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32, device=device) / 16.0
    return k

def bspline_upsample_3d(control_field, target_size):
    B, C, Dc, Hc, Wc = control_field.shape
    D, H, W = target_size
    device = control_field.device
    field = F.interpolate(control_field, size=target_size, mode='trilinear', align_corners=True)
    kernel_1d = cubic_bspline_kernel(device)
    kernel_d = kernel_1d.view(1, 1, 5, 1, 1).repeat(C, 1, 1, 1, 1)
    kernel_h = kernel_1d.view(1, 1, 1, 5, 1).repeat(C, 1, 1, 1, 1)
    kernel_w = kernel_1d.view(1, 1, 1, 1, 5).repeat(C, 1, 1, 1, 1)
    field = F.conv3d(field, kernel_d, padding=(2, 0, 0), groups=C)
    field = F.conv3d(field, kernel_h, padding=(0, 2, 0), groups=C)
    field = F.conv3d(field, kernel_w, padding=(0, 0, 2), groups=C)
    return field

def apply_bspline_deform(image, magnitude=1, control_spacing=16, return_flow=False):
    B, C, D, H, W = image.shape
    device = image.device
    if min(D, H, W) < 2 or control_spacing <= 0:
        raise ValueError('Spatial dimensions must be >= 2 and control_spacing must be positive.')
    if magnitude <= 0:
        if return_flow:
            return (image, torch.zeros((B, 3, D, H, W), device=device))
        return image
    Dc = max(2, D // control_spacing)
    Hc = max(2, H // control_spacing)
    Wc = max(2, W // control_spacing)
    control_field = torch.randn(B, 3, Dc, Hc, Wc, device=device) * magnitude
    flow = bspline_upsample_3d(control_field, (D, H, W))
    d = torch.linspace(-1, 1, D, device=device)
    h = torch.linspace(-1, 1, H, device=device)
    w = torch.linspace(-1, 1, W, device=device)
    grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing='ij')
    base_grid = torch.stack([grid_w, grid_h, grid_d], dim=-1)
    base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)
    flow_norm = torch.zeros_like(base_grid)
    flow_norm[..., 0] = flow[:, 0] / ((W - 1) / 2)
    flow_norm[..., 1] = flow[:, 1] / ((H - 1) / 2)
    flow_norm[..., 2] = flow[:, 2] / ((D - 1) / 2)
    final_grid = base_grid + flow_norm
    warped = F.grid_sample(image, final_grid, mode='bilinear', padding_mode='border', align_corners=True)
    if return_flow:
        return (warped, flow)
    return warped

def apply_rigid_transform(image, transform_type, translate_voxels=0.0, rotate_angle=0.0):
    B, C, D, H, W = image.shape
    device = image.device
    theta = torch.zeros((B, 3, 4), device=device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 2, 2] = 1.0
    if transform_type == 'translation':
        rand_dir = torch.randn(B, 3, device=device)
        rand_dir = F.normalize(rand_dir, p=2, dim=1)
        norm_scale_x = translate_voxels * 2.0 / max(W - 1, 1)
        norm_scale_y = translate_voxels * 2.0 / max(H - 1, 1)
        norm_scale_z = translate_voxels * 2.0 / max(D - 1, 1)
        theta[:, 0, 3] = rand_dir[:, 0] * norm_scale_x
        theta[:, 1, 3] = rand_dir[:, 1] * norm_scale_y
        theta[:, 2, 3] = rand_dir[:, 2] * norm_scale_z
    elif transform_type == 'rotation':
        rad = math.radians(rotate_angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        axis_choice = torch.randint(0, 3, (B,), device=device)
        R = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
        is_x = axis_choice == 0
        is_y = axis_choice == 1
        is_z = axis_choice == 2
        if is_x.any():
            R[is_x, 1, 1] = cos_a
            R[is_x, 1, 2] = -sin_a
            R[is_x, 2, 1] = sin_a
            R[is_x, 2, 2] = cos_a
        if is_y.any():
            R[is_y, 0, 0] = cos_a
            R[is_y, 0, 2] = sin_a
            R[is_y, 2, 0] = -sin_a
            R[is_y, 2, 2] = cos_a
        if is_z.any():
            R[is_z, 0, 0] = cos_a
            R[is_z, 0, 1] = -sin_a
            R[is_z, 1, 0] = sin_a
            R[is_z, 1, 1] = cos_a
        theta[:, :3, :3] = R
    else:
        raise ValueError("transform_type must be 'translation' or 'rotation'.")
    grid = F.affine_grid(theta, image.size(), align_corners=True)
    warped_image = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped_image

def random_deform(image, geo_mag=1.0):
    return apply_bspline_deform(image, magnitude=geo_mag, control_spacing=16)
