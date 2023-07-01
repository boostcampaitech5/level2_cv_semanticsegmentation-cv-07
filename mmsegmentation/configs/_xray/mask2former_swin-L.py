default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
tta_model = dict(type='SegTTAModel')
dataset_type = 'XRayDataset'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadXRayAnnotations'),
    dict(type='Resize', scale=(512, 512)),
    dict(type='TransposeAnnotations'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512)),
    dict(type='LoadXRayAnnotations'),
    dict(type='TransposeAnnotations'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='XRayDataset',
        is_train=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadXRayAnnotations'),
            dict(type='Resize', scale=(512, 512)),
            dict(type='TransposeAnnotations'),
            dict(type='PackSegInputs')
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='XRayDataset',
        is_train=False,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(512, 512)),
            dict(type='LoadXRayAnnotations'),
            dict(type='TransposeAnnotations'),
            dict(type='PackSegInputs')
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='XRayDataset',
        is_train=False,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(512, 512)),
            dict(type='LoadXRayAnnotations'),
            dict(type='TransposeAnnotations'),
            dict(type='PackSegInputs')
        ]))
val_evaluator = dict(type='DiceMetric')
test_evaluator = dict(type='DiceMetric')
pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'
# crop_size = (640, 640)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(640, 640))
num_classes = 29
depths = [2, 2, 18, 2]
model = dict(
    type='EncoderDecoderWithoutArgmax',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(640, 640)),
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        frozen_stages=-1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'
        )),
    decode_head=dict(
        type='Mask2FormerHead',
        in_channels=[192, 384, 768, 1536],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_classes=29,
        num_queries=100,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type='mmdet.MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=True,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True))),
                init_cfg=None),
            positional_encoding=dict(num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                cross_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True)),
            init_cfg=None),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 0.1
            ]),
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(
                        type='mmdet.CrossEntropyLossCost',
                        weight=5.0,
                        use_sigmoid=True),
                    dict(
                        type='mmdet.DiceCost',
                        weight=5.0,
                        pred_act=True,
                        eps=1.0)
                ]),
            sampler=dict(type='mmdet.MaskPseudoSampler'))),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = dict({
    'backbone':
    dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'absolute_pos_embed':
    dict(lr_mult=0.1, decay_mult=0.0),
    'relative_position_bias_table':
    dict(lr_mult=0.1, decay_mult=0.0),
    'query_embed':
    dict(lr_mult=1.0, decay_mult=0.0),
    'query_feat':
    dict(lr_mult=1.0, decay_mult=0.0),
    'level_embed':
    dict(lr_mult=1.0, decay_mult=0.0),
    'backbone.stages.0.blocks.0.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.0.blocks.1.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.1.blocks.0.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.1.blocks.1.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.0.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.1.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.2.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.3.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.4.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.5.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.6.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.7.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.8.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.9.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.10.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.11.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.12.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.13.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.14.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.15.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.16.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.17.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.3.blocks.0.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.3.blocks.1.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.0.downsample.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.1.downsample.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.downsample.norm':
    dict(lr_mult=0.1, decay_mult=0.0)
})
optimizer = dict(
    type='AdamW', lr=0.0001, weight_decay=0.05, eps=1e-08, betas=(0.9, 0.999))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-08,
        betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys=dict({
            'backbone':
            dict(lr_mult=0.1, decay_mult=1.0),
            'backbone.patch_embed.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'absolute_pos_embed':
            dict(lr_mult=0.1, decay_mult=0.0),
            'relative_position_bias_table':
            dict(lr_mult=0.1, decay_mult=0.0),
            'query_embed':
            dict(lr_mult=1.0, decay_mult=0.0),
            'query_feat':
            dict(lr_mult=1.0, decay_mult=0.0),
            'level_embed':
            dict(lr_mult=1.0, decay_mult=0.0),
            'backbone.stages.0.blocks.0.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.0.blocks.1.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.1.blocks.0.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.1.blocks.1.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.0.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.1.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.2.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.3.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.4.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.5.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.6.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.7.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.8.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.9.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.10.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.11.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.12.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.13.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.14.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.15.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.16.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.17.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.3.blocks.0.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.3.blocks.1.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.0.downsample.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.1.downsample.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.downsample.norm':
            dict(lr_mult=0.1, decay_mult=0.0)
        }),
        norm_decay_mult=0.0))
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=160000,
        by_epoch=False)
]
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=5000,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
auto_scale_lr = dict(enable=False, base_batch_size=16)
launcher = 'none'
work_dir = './work_dirs/mask2former_swin-l-in22k'
