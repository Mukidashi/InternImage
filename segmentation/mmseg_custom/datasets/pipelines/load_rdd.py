# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import cv2

from mmseg.datasets.builder import PIPELINES



def convert_color_map_into_one_hot(img, limg):
    road_color = [[254,255,189], [203,192,255], [83,50,250], [102,255,102], [245,61,184]]
    damage_color =[[200,25,255], [150,192,255], [255, 100, 0]]

    if limg.shape[0] != img.shape[0] or limg.shape[1] != img.shape[1]:
        limg = cv2.resize(limg,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_NEAREST)

    otensor1 = np.zeros((img.shape[0],img.shape[1],6))
    otensor2 = np.zeros((img.shape[0],img.shape[1],4))
    for i in range(len(road_color)):
        cmask = np.all(img == road_color[i], axis=-1)
        otensor1[cmask,i+1] = 255

    for i in range(len(damage_color)):
        cmask = np.all(img == damage_color[i], axis=-1)
        otensor2[cmask,i+1] = 255
        otensor1[cmask,1] = 255

    for i in range(len(road_color)):
        cmask = np.all(limg == road_color[i], axis=-1)
        otensor1[cmask,i+1] = 255
        otensor1[cmask,:(i+1)] = 0

    bg_mask = np.all(otensor1==0,axis=2)
    otensor1[bg_mask,0] = 255

    bg_mask = np.all(otensor2==0,axis=2)
    otensor2[bg_mask,0] = 255


    return np.concatenate((otensor1,otensor2),axis=2)



@PIPELINES.register_module()
class LoadRddAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        apath1 = results['ann_info']['seg_map']
        apath2 = results['ann_info']['seg_map2']
        aimg = cv2.imread(apath1)
        limg = cv2.imread(apath2)

        otensor = convert_color_map_into_one_hot(aimg,limg)
        results['gt_semantic_seg'] = otensor.astype(np.uint8)
        results['seg_fields'].append('gt_semantic_seg')

        # if self.file_client is None:
        #     self.file_client = mmcv.FileClient(**self.file_client_args)

        # if results.get('seg_prefix', None) is not None:
        #     filename = osp.join(results['seg_prefix'],
        #                         results['ann_info']['seg_map'])
        # else:
        #     filename = results['ann_info']['seg_map']
        # img_bytes = self.file_client.get(filename)
        # gt_semantic_seg = mmcv.imfrombytes(
        #     img_bytes, flag='unchanged',
        #     backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # # reduce zero_label
        # if self.reduce_zero_label:
        #     # avoid using underflow conversion
        #     gt_semantic_seg[gt_semantic_seg == 0] = 255
        #     gt_semantic_seg = gt_semantic_seg - 1
        #     gt_semantic_seg[gt_semantic_seg == 254] = 255
        # # modify if custom classes
        # if results.get('label_map', None) is not None:
        #     # Add deep copy to solve bug of repeatedly
        #     # replace `gt_semantic_seg`, which is reported in
        #     # https://github.com/open-mmlab/mmsegmentation/pull/1445/
        #     gt_semantic_seg_copy = gt_semantic_seg.copy()
        #     for old_id, new_id in results['label_map'].items():
        #         gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        # results['gt_semantic_seg'] = gt_semantic_seg
        # results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str