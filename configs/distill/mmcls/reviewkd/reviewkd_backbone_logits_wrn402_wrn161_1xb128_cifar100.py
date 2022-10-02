_base_ = [
    'mmcls::_base_/datasets/cifar100_bs16.py',
    'mmcls::_base_/schedules/cifar10_bs128.py',
    'mmcls::_base_/default_runtime.py'
]
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005)
lr_config = dict(policy='step', step=[60, 120, 160], gamma=0.2)
train_dataloader = dict(batch_size=128, num_workers=10, pin_memory=True)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1,max_keep_ckpts=2))
model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    data_preprocessor=dict(
        type='ImgDataPreprocessor',
        # RGB format normalization parameters
        mean=[129.304, 124.070, 112.434],
        std=[68.170, 65.392, 70.418],
        # convert image from BGR to RGB
        bgr_to_rgb=False),
    architecture=dict(
        _scope_='mmcls',
        type='ImageClassifier',
        backbone=dict(
            _scope_='mmrazor',
            type='WideResNet',
            depth=16,
            num_stages=3,
            widen_factor=1,
        ),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=100,
            in_channels=64,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5),
        )),
    teacher=dict(
        _scope_='mmcls',
        type='ImageClassifier',
        backbone=dict(
            _scope_='mmrazor',
            type='WideResNet',
            depth=40,
            num_stages=3,
            widen_factor=2,
        ),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=100,
            in_channels=128,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5),
        )),
    teacher_ckpt=  # noqa: E251
    distiller=dict(
        type='ReviewKDDistiller',
        student_recorders=dict(
            bb_s1=dict(type='ModuleOutputs', source='backbone.layer1'),
            bb_s2=dict(type='ModuleOutputs', source='backbone.layer2'),
            bb_s3=dict(type='ModuleOutputs', source='backbone.layer3'),
            bb_s4=dict(type='ModuleOutputs', source='neck'),
        ),
        teacher_recorders=dict(
            bb_s1=dict(type='ModuleOutputs', source='backbone.layer1'),
            bb_s2=dict(type='ModuleOutputs', source='backbone.layer2'),
            bb_s3=dict(type='ModuleOutputs', source='backbone.layer3'),
            bb_s4=dict(type='ModuleOutputs', source='neck'),
        ),
        distill_losses=dict(
            loss_s4=dict(type='HCLLoss', loss_weight=5),
            loss_s3=dict(type='HCLLoss', loss_weight=5),
            loss_s2=dict(type='HCLLoss', loss_weight=5),
            loss_s1=dict(type='HCLLoss', loss_weight=5),
            ),
        connectors=dict(
             loss_s4_sfeat=dict(
                type='ABFConnector',
                in_channel=64,
                mid_channel=64,
                out_channel=128,
            ),
            loss_s3_sfeat=dict(
                type='ABFConnector',
                in_channel=64,
                mid_channel=64,
                out_channel=128,
                residual='bb_s4',
            ),
            loss_s2_sfeat=dict(
                type='ABFConnector',
                in_channel=32,
                mid_channel=64,
                out_channel=64,
                residual='bb_s3',
            ),
            loss_s1_sfeat=dict(
                type='ABFConnector',
                in_channel=16,
                mid_channel=64,
                out_channel=32,
                residual='bb_s2',
            ),
        ),
        loss_forward_mappings=dict(
            loss_s3=dict(
                s_feature=dict(
                    from_student=True,
                    recorder='bb_s3',
                    connector='loss_s3_sfeat'),
                t_feature=dict(from_student=False, recorder='bb_s3')),
            loss_s2=dict(
                s_feature=dict(
                    from_student=True,
                    recorder='bb_s2',
                    connector='loss_s2_sfeat'),
                t_feature=dict(from_student=False, recorder='bb_s2')),
            loss_s1=dict(
                s_feature=dict(
                    from_student=True,
                    recorder='bb_s1',
                    connector='loss_s1_sfeat'),
                t_feature=dict(from_student=False, recorder='bb_s1')),
        ),
    ))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
