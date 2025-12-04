model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='SwinTransformer3D',
        arch='base',
        pretrained=None,
        pretrained2d=None,
        patch_size=(2, 4, 4),
        window_size=(8, 7, 7),
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True),
    cls_head=dict(
        type='I3DHead',
        in_channels=1024,
        num_classes=3,
        spatial_type='avg',
        dropout_ratio=0.5),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
