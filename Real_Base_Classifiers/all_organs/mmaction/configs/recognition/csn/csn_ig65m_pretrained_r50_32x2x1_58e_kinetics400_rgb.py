_base_ = [
    '../../_base_/models/ircsn_r152.py', '../../_base_/default_runtime.py'
]

# model settings
model = dict(
    backbone=dict(
        depth=50,
        norm_eval=False,
        pretrained=  # noqa: E251
        None  # noqa: E501
    ))