# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='MViT', 
        arch='base',
        temporal_size=32,
        drop_path_rate=0.3
    ),
    cls_head=dict(
        type='MViTHead',
        in_channels=768,
        num_classes=2,
        label_smooth_eps=0.1),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
