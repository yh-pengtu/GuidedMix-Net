_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

find_unused_parameters = True
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='GuidedMix_EncoderDecoder',
    pretrained='open-mmlab://resnet101_v1c',
    lambda_factor=0.1,
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='ASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    components=dict(out_modules=['decode_head.conv_seg', 'decode_head.aspp_modules']),
    kl_loss=dict(
        type='FocalLoss',
        use_sigmoid=True,
        gamma=0.5,
        reduction='mean',
        loss_weight=10.0),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# dataset settings
dataset_type = 'CityscapesDataset'
data_root = 'data/Cityscapes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (769, 769)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2049, 1025), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2049, 1025),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='SemiDataset',
        dataset=dict(
            type='Semi'+dataset_type,
            data_root=data_root,
            img_dir='leftImg8bit/train',
            ann_dir='gtFine/train',
            split='./3sseg_datasplits/cityscapes_splits/372_train_supervised.txt',
            pipeline=train_pipeline),
        unsup_dataset=dict(
            type='Semi'+dataset_type,
            data_root=data_root,
            img_dir='leftImg8bit/train',
            ann_dir='gtFine/train',
            split='./3sseg_datasplits/cityscapes_splits/372_train_unsupervised.txt',
            pipeline=train_pipeline)),
    val=dict(
        type='Semi'+dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        split=None,
        pipeline=test_pipeline),
    test=dict(
        type='Semi'+dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        split=None,
        pipeline=test_pipeline))

# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# # runtime settings
runner = dict(type='IterBasedRunner', max_iters=40000)