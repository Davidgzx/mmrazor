_base_ = ['wrn16-w2_b1x128_cifar100.py']
model = dict(
    backbone=dict(depth=40, widen_factor=2), head=dict(in_channels=128))
