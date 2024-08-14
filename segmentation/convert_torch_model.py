# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv

import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes
import cv2
import os.path as osp
import os
import numpy as np
import torch 

from mmdeploy.apis import torch2onnx
# from mmdeploy.utils import load_config

import time


def set_parser():
    parser = ArgumentParser()
    # parser.add_argument('test_img', help='Test image')
    parser.add_argument('model_cfg', help='Model config file')
    parser.add_argument('deploy_cfg', help='Deploy config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--img_wid', type=int, default=512, help='Image width')
    parser.add_argument('--img_hei', type=int, default=512, help='Image height')
    parser.add_argument('--work_dir', type=str, default="work", help='working directory')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--palette',default='ade20k',choices=['ade20k', 'cityscapes', 'cocostuff'], help='Color palette used for segmentation map')

    return parser

def main():
    
    parser = set_parser()
    args = parser.parse_args()
    

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.model_cfg, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    checkpoint_path = args.checkpoint
    if (not 'CLASSES' in checkpoint.get('meta', {})) or (not 'PALETTE' in checkpoint.get('meta', {})):
        checkpoint['meta'] = {}
        checkpoint['meta']['CLASSES'] = get_classes(args.palette)
        checkpoint['meta']['PALETTE'] = get_palette(args.palette)

        checkpoint_path = os.path.join(args.work_dir,"tmp_model.pth")
        with open(checkpoint_path, 'wb') as f:
            torch.save(checkpoint, f)
            f.flush()
        
        checkpoint = load_checkpoint(model, os.path.join(args.work_dir,"tmp_model.pth"), map_location='cpu')
        print(checkpoint.keys())

    test_img = np.random.randint(0, 255, (args.img_hei, args.img_wid, 3), dtype=np.uint8)
    timg_path = os.path.join(args.work_dir, "test.jpg")
    cv2.imwrite(timg_path, test_img)

    # deploy_cfg, model_cfg = load_config(args.deploy_cfg, args.model_cfg)


    # convert_to_onnx
    # torch2onnx(img: Any, 
    #             work_dir: str, 
    #             save_file: str, 
    #             deploy_cfg: Union[str, mmengine.config.config.Config], 
    #             model_cfg: Union[str, mmengine.config.config.Config], 
    #             model_checkpoint: Union[str, NoneType] = None, 
    #             device: str = 'cuda:0')
    # print(type(model))
    # print(help(torch2onnx)))
    torch2onnx(timg_path, args.work_dir, "model.onnx", args.deploy_cfg, 
               args.model_cfg, checkpoint_path, device=args.device)



if __name__ == '__main__':
    main()