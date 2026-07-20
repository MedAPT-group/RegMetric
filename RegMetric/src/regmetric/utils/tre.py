import torch
import itertools


def prod_n(lst):
    if not lst:
        raise ValueError("Cannot multiply an empty list")
    prod = lst[0].clone()
    for p in lst[1:]:
        prod = prod * p
    return prod


def sub2ind(siz, subs, **kwargs):
    if len(siz) != len(subs):
        raise ValueError(f"Inconsistent dimensions: size has {len(siz)}, indices have {len(subs)}")

    ndx = subs[-1].clone()
    stride = 1
    for dimension in range(len(siz) - 2, -1, -1):
        stride *= int(siz[dimension + 1])
        ndx = ndx + subs[dimension] * stride

    return ndx


def interpn(vol, loc, interp_method='linear'):
    if isinstance(loc, (list, tuple)):
        loc = torch.stack(loc, -1)

    nb_dims = loc.shape[-1]

    if nb_dims != len(vol.shape[:-1]):
        raise ValueError(
            f"Coordinate dimension {nb_dims} does not match volume dimension {len(vol.shape[:-1])}"
        )

    if nb_dims > len(vol.shape):
        raise ValueError(f"Coordinate dimension {nb_dims} exceeds volume dimension {len(vol.shape)}")

    if len(vol.shape) == nb_dims:
        vol = torch.unsqueeze(vol, -1)

    loc = loc.to(device=vol.device, dtype=torch.float32)

    if isinstance(vol.shape, (torch.Size,)):
        volshape = list(vol.shape)
    else:
        volshape = vol.shape

    if interp_method == "linear":
        loc0 = torch.floor(loc)

        max_loc = [d - 1 for d in list(vol.shape)]

        clipped_loc = [torch.clamp(loc[..., d], 0, max_loc[d]) for d in range(nb_dims)]
        loc0lst = [torch.clamp(loc0[..., d], 0, max_loc[d]) for d in range(nb_dims)]

        loc1 = [torch.clamp(loc0lst[d] + 1, 0, max_loc[d]) for d in range(nb_dims)]
        locs = [[f.to(dtype=torch.long) for f in loc0lst], [f.to(dtype=torch.long) for f in loc1]]

        diff_loc1 = [loc1[d] - clipped_loc[d] for d in range(nb_dims)]
        diff_loc0 = [1 - d for d in diff_loc1]
        weights_loc = [diff_loc1, diff_loc0]

        cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
        interp_vol = 0

        for c in cube_pts:
            subs = [locs[c[d]][d] for d in range(nb_dims)]

            idx = sub2ind(vol.shape[:-1], subs)
            idx = torch.as_tensor(idx, dtype=torch.long, device=vol.device)
            vol_val = torch.reshape(vol, (-1, volshape[-1]))[idx]

            wts_lst = [weights_loc[c[d]][d] for d in range(nb_dims)]
            wt = prod_n(wts_lst)

            # Keep interpolation weights on the same device as the volume values.
            wt = torch.unsqueeze(wt, -1).to(vol.device)

            interp_vol += wt * vol_val

    else:
        if interp_method != "nearest":
            raise ValueError("interp_method must be 'linear' or 'nearest'")
        roundloc = torch.round(loc).to(dtype=torch.long)
        max_loc = [int(d) - 1 for d in vol.shape[:-1]]
        roundloc = [torch.clamp(roundloc[..., d], 0, max_loc[d]) for d in range(nb_dims)]

        idx = sub2ind(vol.shape[:-1], roundloc)
        idx = idx.to(device=vol.device, dtype=torch.long)
        interp_vol = torch.reshape(vol, (-1, vol.shape[-1]))[idx]

    return interp_vol


def point_spatial_transformer(x, sdt_vol_resize=1):

    surface_points, trf = x
    trf = trf * sdt_vol_resize
    surface_pts_D = surface_points.shape[-1]
    trf_D = trf.shape[-1]
    assert surface_pts_D in [trf_D, trf_D + 1]

    if surface_pts_D == trf_D + 1:
        li_surface_pts = torch.unsqueeze(surface_points[..., -1], -1)
        surface_points = surface_points[..., :-1]

    diff = interpn(trf, surface_points)
    ret = surface_points + diff

    if surface_pts_D == trf_D + 1:
        ret = torch.cat((ret, li_surface_pts), -1)
    return ret
