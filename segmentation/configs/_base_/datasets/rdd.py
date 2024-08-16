# dataset settings
dataset_type = 'RddDataset'
data_root = '/home/k-morioka/Data/UrbanX/rdd/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadRddAnnotations'),
    # dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='Resize', img_scale=(1024,512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='RddFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize',img_scale=(512,512)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
    # dict(
    #     type='MultiScaleFlipAug',
    #     img_scale=(2048, 512),
    #     # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    #     flip=False,
    #     transforms=[
    #         dict(type='Resize', keep_ratio=True),
    #         dict(type='RandomFlip'),
    #         dict(type='Normalize', **img_norm_cfg),
    #         dict(type='ImageToTensor', keys=['img']),
    #         dict(type='Collect', keys=['img']),
    #     ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        data_list="/home/k-morioka/Data/UrbanX/rdd/2024_fps/2024-fps_train_with-lane.list",
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        data_list="/home/k-morioka/Data/UrbanX/rdd/2024_fps/2024-fps_valid_with-lane.list",
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        data_list="/home/k-morioka/Data/UrbanX/rdd/2022-11-02/2022-11-02_test_with-lane.list",
        pipeline=test_pipeline))
