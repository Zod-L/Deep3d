import torch
from scipy.io import loadmat, savemat
from mesh import load_obj_mesh, save_obj_mesh_with_rgb, save_obj_mesh
import torch.nn.functional as F
import numpy as np
import torch
import cv2
import os
import shutil
from render.Renderer_DECA import SRenderY, set_rasterizer
import render.Renderer_DECA_util as util
from torchvision.utils import save_image
from util.nvdiffrast import MeshRenderer


bfm_model = loadmat("BFM/01_MorphableModel.mat")
save_obj_mesh_with_rgb("./mean.obj", bfm_model["shapeMU"].reshape(-1, 3),  bfm_model["tl"]-1, bfm_model["texMU"].reshape(-1, 3))

# for k,v in bfm_model.items():
#     try:
#         print(f"{k} : {v.shape}")
#     except:
#         pass

# # print(bfm_model["eyebrow_mask"].dtype)
# real_shape = bfm_model["meanshape"].reshape(-1, 3)
# real_tex = bfm_model["meantex"].reshape(-1, 3)
# toon_shape = bfm_model["mean_cartoon_shape"].reshape(-1, 3) 
# toon_tex = bfm_model["mean_cartoon_texture"].reshape(-1, 3) 
# fuse_shape = (real_shape + toon_shape) / 2
# fuse_tex = (real_tex + toon_tex) / 2
# tri = bfm_model["tri"] - 1
# save_obj_mesh_with_rgb("real.obj", real_shape, tri, real_tex)
# save_obj_mesh_with_rgb("toon.obj", toon_shape, tri, toon_tex)
# save_obj_mesh_with_rgb("fuse.obj", fuse_shape, tri, fuse_tex)




# _,_,real_mask = load_obj_mesh("real_mask.obj")
# real_idx = np.all(real_mask == [1,0,0], axis=-1)


device = torch.device("cuda", 0)
template = "uv_mesh/Remesh_temp.obj"
fov = 2 * np.arctan(256 / 1015) * 180 / np.pi
renderer = MeshRenderer(
    rasterize_fov=fov, znear=5, zfar=15, rasterize_size=int(2 * 256), use_opengl=True
)
verts, tri, tex = load_obj_mesh("mean_with_neck.obj")




verts = torch.FloatTensor(verts).unsqueeze(0).to(device)
tri = torch.FloatTensor(tri).to(device)
tex = torch.FloatTensor(tex).unsqueeze(0).to(device)
verts[..., -1] = 6.5 - verts[..., -1]
pred_mask, _,pred_face = renderer(verts, tri, feat=tex)
print(pred_face.max())
save_image(pred_face, "./mean.png")


