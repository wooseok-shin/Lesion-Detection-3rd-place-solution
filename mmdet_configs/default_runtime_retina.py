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
load_from = '../pretrain_weights/retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth'
resume_from = None
workflow = [('train', 1)]
