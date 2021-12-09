checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type = 'WandbLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '../pretrain_weights/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth'
resume_from = None
workflow = [('train', 1)]
