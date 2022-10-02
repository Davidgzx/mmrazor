_base_ = [
    'mmcls::_base_/datasets/cifar100_bs16.py',
    'mmcls::_base_/schedules/cifar10_bs128.py',
    'mmcls::_base_/default_runtime.py'
]
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005)
lr_config = dict(policy='step', step=[60, 120, 160], gamma=0.2)
train_dataloader = dict(batch_size=128, num_workers=12, pin_memory=True)
val_dataloader = dict(batch_size=128, num_workers=12, pin_memory=True)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1,max_keep_ckpts=2))
teacher_ckpt = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar100_20210528-67b58a1b.pth'  # noqa: E501
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
        type='mmcls.ImageClassifier',
        backbone=dict(
            type='mmcls.ResNet_CIFAR',
            depth=18,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='mmcls.GlobalAveragePooling'),
        head=dict(
            type='mmcls.LinearClsHead',
            num_classes=100,
            in_channels=512,
            loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0))),
    teacher=dict(
        cfg_path='mmcls::resnet/resnet50_8xb16_cifar100.py', pretrained=True),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ReviewKDDistiller',
        student_recorders=dict(
            bb_s4=dict(type='ModuleOutputs', source='backbone.layer4.1'),
            bb_s3=dict(type='ModuleOutputs', source='backbone.layer3.1'),
            bb_s2=dict(type='ModuleOutputs', source='backbone.layer2.1'),
            bb_s1=dict(type='ModuleOutputs', source='backbone.layer1.1'),
        ),
        teacher_recorders=dict(
            bb_s4=dict(type='ModuleOutputs', source='backbone.layer4.2'),
            bb_s3=dict(type='ModuleOutputs', source='backbone.layer3.5'),
            bb_s2=dict(type='ModuleOutputs', source='backbone.layer2.3'),
            bb_s1=dict(type='ModuleOutputs', source='backbone.layer1.2'),
        ),
        distill_losses=dict(
            loss_s4=dict(type='HCLLoss', loss_weight=2),
            loss_s3=dict(type='HCLLoss', loss_weight=2),
            loss_s2=dict(type='HCLLoss', loss_weight=2),
            loss_s1=dict(type='HCLLoss', loss_weight=2),
            ),
        connectors=dict(
            loss_s4_sfeat=dict(
                type='ABFConnector',
                in_channel=512,
                mid_channel=512,
                out_channel=2048,
            ),
            loss_s3_sfeat=dict(
                type='ABFConnector',
                in_channel=256,
                mid_channel=512,
                out_channel=1024,
                residual='bb_s4',
            ),
            loss_s2_sfeat=dict(
                type='ABFConnector',
                in_channel=128,
                mid_channel=512,
                out_channel=512,
                residual='bb_s3',
            ),
            loss_s1_sfeat=dict(
                type='ABFConnector',
                in_channel=64,
                mid_channel=512,
                out_channel=256,
                residual='bb_s2',
            ),
        ),
        loss_forward_mappings=dict(
            loss_s4=dict(
                s_feature=dict(
                    from_student=True,
                    recorder='bb_s4',
                    connector='loss_s4_sfeat'),
                t_feature=dict(from_student=False, recorder='bb_s4')),
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
resume = True
resume_from = 'work_dirs/reviewkd_backbone_logits_resnet50_resnet18_1xb128_cifar100/last_checkpoint'