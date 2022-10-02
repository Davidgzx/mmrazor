_base_ = [
    'mmcls::_base_/datasets/cifar100_bs16.py',
    '../../../_base_/vanilla_models/wrn16_2_cifar10.py',
    'mmcls::_base_/schedules/cifar10_bs128.py',
    'mmcls::_base_/default_runtime.py',
]
model = dict(head=dict(num_classes=100))
train_dataloader = dict(
    batch_size=128, num_workers=0, persistent_workers=False)
val_dataloader = dict(batch_size=128, num_workers=0, persistent_workers=False)
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005)
lr_config = dict(policy='step', step=[60, 120, 160], gamma=0.2)
test_evaluator = dict(topk=(1, 5))
