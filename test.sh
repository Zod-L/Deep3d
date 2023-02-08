CUDA_VISIBLE_DEVICES=0 python test.py --name=pretrain_model \
--epoch=20 --tex_from_im=False --bfm_model=BFM_model_front.mat \
--datalist=./datalist/real --root=