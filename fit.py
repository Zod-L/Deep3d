import torch
from scipy.io import loadmat
from mesh import load_obj_mesh, save_obj_mesh, save_obj_mesh_with_rgb
import torch.nn.functional as F
import numpy as np
import cv2
import os



#-------------------------------------------------
bfm_name = "BFM/fuse_model_front_tex_align.mat"
indir = "checkpoints/pretrain_model/results/im/epoch_20_000000/"
outdir = "./fit_out/"
if not os.path.exists(outdir):
    os.mkdir(outdir)
#----------------------------------------





step = 20000
device = torch.device("cuda", 0)


bfm_model = loadmat(bfm_name)


real_hape = torch.from_numpy(bfm_model["mean_cartoon_shape"]).to(device=device).permute(1, 0)
cartoon_shape =  torch.from_numpy(bfm_model["meanshape"]).to(device=device).permute(1, 0)
meanshape = (real_hape + cartoon_shape) / 2
meanshape = meanshape.reshape([-1, 3])
meanshape = meanshape - meanshape.mean(axis=0, keepdims=True)
meanshape = meanshape.reshape([-1, 1])

#meantex = torch.from_numpy((bfm_model["mean_cartoon_texture"] + bfm_model["meantex"]) / 2).to(device=device).permute(1, 0)
# meantex = meantex.reshape(-1, 3)
# meantex[bfm_model["eyebrow_mask"]] = meantex[29604, :]
# save_obj_mesh_with_rgb("mean.obj", meanshape.reshape(-1, 3), bfm_model["tri"]-1, bfm_model["meantex"].reshape(-1, 3))
#_, _, meantex = load_obj_mesh("BFM/mean.obj")
#meantex = torch.from_numpy(meantex.reshape(1, -1)).to(device=device).permute(1, 0)
#meantex =  torch.from_numpy(bfm_model["mean_cartoon_texture"].reshape(1, -1)).to(device=device).permute(1, 0)
id_cartoon = torch.tensor(bfm_model["idBase"], dtype=torch.float64, device=device)[:, 80:]
#tex_cartoon =  torch.from_numpy(bfm_model["texBase"]).to(device=device)[:, 80:]


front_idx = loadmat("BFM/select_vertex_id.mat")["select_id"][:, 0] - 1


fnames = [fname for fname in os.listdir(indir) if fname.endswith(".obj")]
for fname in fnames:
    print(fname)
    target_vert, _, target_tex = load_obj_mesh(indir + fname)
    coeff = torch.tensor(loadmat(indir + fname.split(".")[-2])["id"], dtype=torch.float64, device=device).permute(1, 0)


    target_vert = torch.from_numpy(target_vert.reshape(-1, 1)).to(device=device)
    #target_tex = torch.from_numpy(target_tex.reshape(-1, 1)).to(device=device)
    target_vert -= meanshape
    #target_tex -= meantex

    coeff_id = torch.randn(size=(id_cartoon.shape[1], 1), dtype=torch.float64).to(device).requires_grad_(True)
    #coeff_tex = torch.randn(size=(tex_cartoon.shape[1], 1), dtype=torch.float64).to(device).requires_grad_(True)
    # cv2.imwrite("text_coeff.png", torch.clamp(coeff_tex, 0, 1).repeat(1, 160).detach().cpu().numpy() * 255)
    # cv2.imwrite("shape_coeff.png", torch.clamp(coeff_id, 0, 1).repeat(1, 160).detach().cpu().numpy()  * 255)

    opt = torch.optim.Adam([coeff_id])
    for i in range(step):
        vert = id_cartoon @ coeff_id
        diff = F.mse_loss(vert, target_vert)
        reg = torch.norm(coeff_id)
        loss = diff #+ 0.001 * reg
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 1000 == 0:
            print(f"L2 difference: {diff} regulation: {reg}")


    # del opt
    # opt = torch.optim.Adam([coeff_tex])
    # for i in range(step*5):
    #     color = tex_cartoon @ coeff_tex
    #     diff = F.mse_loss(color, target_tex)
    #     reg = torch.norm(coeff_tex)
    #     loss = diff #+ 0.001 * reg
    #     opt.zero_grad()
    #     loss.backward()
    #     opt.step()
    #     if i % 1000 == 0:
    #         print(f"L2 difference: {diff} regulation: {reg}")


    res_vert = (id_cartoon @ coeff_id).detach() + meanshape
    #res_color = (tex_cartoon @ coeff_tex).detach() + meantex
    res_vert = res_vert.cpu().numpy()
    #res_color = np.clip(res_color.cpu().numpy(), 0, 1)
    #save_obj_mesh_with_rgb("res.obj", res_vert.reshape(-1, 3), bfm_model["tri"]-1, res_color.reshape(-1, 3))
    save_obj_mesh(outdir + fname, res_vert.reshape(-1, 3), bfm_model["tri"]-1)




