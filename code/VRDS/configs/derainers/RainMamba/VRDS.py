# RainMamba
model = dict(
    type='DrainNet',
    generator=dict(
        type='RainMamba',
        num_features=128,
        feat_pretrained=
        'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'
    ),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean')
)

train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0)
train_dataset_type = 'SRFolderMultipleGTDataset'
val_dataset_type = 'SRFolderMultipleGTDataset'

demo_pipeline = [
    dict(type='GenerateSegmentIndices', start_idx=0, interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path', 'key'])
]

data = dict(
    workers_per_gpu=6,
    train_dataloader=dict(samples_per_gpu=2, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=800,
        dataset=dict(
            type='SRFolderMultipleGTDataset',
            lq_folder='../data/VRDS/lq',
            gt_folder='../data/VRDS/gt',
            pipeline=[
                dict(
                    type='GenerateSegmentIndices',
                    interval_list=[1],
                    filename_tmpl='{08d}.png',
                    start_idx=0),
                dict(
                    type='LoadImageFromFileList',
                    io_backend='disk',
                    key='lq',
                    channel_order='rgb'),
                dict(
                    type='LoadImageFromFileList',
                    io_backend='disk',
                    key='gt',
                    channel_order='rgb'),
                dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
                dict(type='PairedRandomCrop', gt_patch_size=256),
                dict(
                    type='Flip',
                    keys=['lq', 'gt'],
                    flip_ratio=0.5,
                    direction='horizontal'),
                dict(
                    type='Flip',
                    keys=['lq', 'gt'],
                    flip_ratio=0.5,
                    direction='vertical'),
                dict(
                    type='RandomTransposeHW',
                    keys=['lq', 'gt'],
                    transpose_ratio=0.5),
                dict(type='FramesToTensor', keys=['lq', 'gt']),
                dict(
                    type='Collect',
                    keys=['lq', 'gt'],
                    meta_keys=['lq_path', 'gt_path'])
            ],
            scale=1,
            num_input_frames=5,
            test_mode=False)),
    test=dict(
        type='SRFolderMultipleGTDataset',
        lq_folder="../data/VRDS/test/lq",
        gt_folder="../data/VRDS/test/gt",
        pipeline=[
            dict(
                type='GenerateSegmentIndices',
                start_idx=0,
                interval_list=[1],
                filename_tmpl='{:08d}.png'),
            dict(
                type='LoadImageFromFileList',
                io_backend='disk',
                key='lq',
                channel_order='rgb'),
            dict(
                type='LoadImageFromFileList',
                io_backend='disk',
                key='gt',
                channel_order='rgb'),
            dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
            dict(type='FramesToTensor', keys=['lq', 'gt']),
            dict(
                type='Collect',
                keys=['lq', 'gt'],
                meta_keys=['lq_path', 'gt_path', 'key'])
        ],
        scale=1,
        test_mode=True))
optimizers = dict(generator=dict(type='Adam', lr=0.0004, betas=(0.9, 0.999)))
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=1e-09,
    by_epoch=False)
total_iters = 300000
checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
log_config = dict(
    interval=1000, hooks=[dict(type='TextLoggerHook', by_epoch=False), dict(type='TensorboardLoggerHook', log_dir='../experiment/logs/VRDS/', by_epoch=False)])
visual_config = None
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '../experiment/VRDS/'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
gpus = 1
