import torch
from pytorch3d.ops.knn import knn_points

def smpl_lbsweight_top_k(
        lbs_weights, 
        points, 
        template_points, 
        K=6, 
    ):
    '''ref: https://github.com/JanaldoChen/Anim-NeRF
    Args:  
    '''
    bz, np, _ = points.shape
    with torch.no_grad():
        results = knn_points(points, template_points, K=K)
        dists, idxs = results.dists, results.idx
    neighbs_dist = dists
    neighbs = idxs
    weight_std = 0.1
    weight_std2 = 2. * weight_std ** 2
    xyz_neighbs_lbs_weight = lbs_weights[neighbs] # (bs, n_rays*K, k_neigh, 24)
    xyz_neighbs_weight_conf = torch.exp(
        -torch.sum(
            torch.abs(xyz_neighbs_lbs_weight - xyz_neighbs_lbs_weight[..., 0:1, :]), dim=-1
        )/weight_std2) # (bs, n_rays*K, k_neigh)
    xyz_neighbs_weight_conf = torch.gt(xyz_neighbs_weight_conf, 0.9).float()
    xyz_neighbs_weight = torch.exp(-neighbs_dist) # (bs, n_rays*K, k_neigh)
    xyz_neighbs_weight *= xyz_neighbs_weight_conf
    xyz_neighbs_weight = xyz_neighbs_weight / xyz_neighbs_weight.sum(-1, keepdim=True) # (bs, n_rays*K, k_neigh)

    # xyz_neighbs_transform = batch_index_select(verts_transform, neighbs) # (bs, n_rays*K, k_neigh, 4, 4)
    xyz_neighbs_lbs_weight = torch.sum(xyz_neighbs_weight.unsqueeze(-1) * xyz_neighbs_lbs_weight, dim=2) # (bs, n_rays*K, 4, 4)
    xyz_dist = torch.sum(xyz_neighbs_weight * neighbs_dist, dim=2, keepdim=True) # (bs, n_rays*K, 1)

    return xyz_dist, xyz_neighbs_lbs_weight