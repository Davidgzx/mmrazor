_base_ = [
    'mmcls::_base_/datasets/cifar10_bs16.py',
    'mmcls::_base_/schedules/cifar10_bs128.py',
    'mmcls::_base_/default_runtime.py'
]
train_dataloader = dict(batch_size=128, num_workers=6)
model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    data_preprocessor=dict(
        type='ImgDataPreprocessor',
        # RGB format normalization parameters
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        # convert image from BGR to RGB
        bgr_to_rgb=True),
    architecture=dict(
        _scope_='mmcls',
        type='ImageClassifier',
        backbone=dict(
            _scope_='mmrazor',
            type='WideResNet',
            depth=16,
            num_stages=3,
            widen_factor=2,
        ),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=10,
            in_channels=128,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5),
        )),
    teacher=dict(
        _scope_='mmcls',
        type='ImageClassifier',
        backbone=dict(
            _scope_='mmrazor',
            type='WideResNet',
            depth=28,
            num_stages=3,
            widen_factor=4,
        ),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=10,
            in_channels=256,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5),
        )),
    teacher_ckpt=  # noqa: E251
    'https://download.openmmlab.com/mmrazor/v1/wide_resnet/wrn28_4_b16x8_cifar10_20220831_173536-d6f8725c.pth',  # noqa: E501
    distiller=dict(
        type='ReviewKDDistiller',
        student_recorders=dict(
            bb_s1=dict(type='ModuleOutputs', source='backbone.layer2.0.relu1'),
            bb_s2=dict(type='ModuleOutputs', source='backbone.layer3.0.relu1'),
            bb_s3=dict(type='ModuleOutputs', source='backbone.relu'),
        ),
        teacher_recorders=dict(
            bb_s1=dict(type='ModuleOutputs', source='backbone.layer2.0.relu1'),
            bb_s2=dict(type='ModuleOutputs', source='backbone.layer3.0.relu1'),
            bb_s3=dict(type='ModuleOutputs', source='backbone.relu'),
        ),
        distill_losses=dict(
            loss_s3=dict(type='HCLLoss', loss_weight=1),
            loss_s2=dict(type='HCLLoss', loss_weight=0.5),
            loss_s1=dict(type='HCLLoss', loss_weight=0.25)),
        connectors=dict(
            loss_s3_sfeat=dict(
                type='ABFConnector',
                in_channel=128,
                mid_channel=128,
                out_channel=256,
            ),
            loss_s2_sfeat=dict(
                type='ABFConnector',
                in_channel=64,
                mid_channel=128,
                out_channel=128,
                residual='bb_s3',
            ),
            loss_s1_sfeat=dict(
                type='ABFConnector',
                in_channel=32,
                mid_channel=128,
                out_channel=64,
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
