_base_ = ['../../_base_/default_runtime.py']

# model settings
num_frames = 8
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='UniFormerV2',
        input_resolution=256,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        t_size=num_frames,
        dw_reduction=1.5,
        backbone_drop_path_rate=0.,
        temporal_downsample=False,
        no_lmhra=True,
        double_lmhra=True,
        return_list=[8, 9, 10, 11],
        n_layers=4,
        n_dim=768,
        n_head=12,
        mlp_factor=4.,
        drop_path_rate=0.,
        mlp_dropout=[0.5, 0.5, 0.5, 0.5],
        clip_pretrained=False,
        pretrained=None),
    cls_head=dict(
        type='UniFormerHead',
        dropout_ratio=0.5,
        num_classes=2,
        in_channels=768),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))