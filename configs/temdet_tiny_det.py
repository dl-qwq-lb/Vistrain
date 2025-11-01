_base_ = 'mmdet::rtmdet/rtmdet_tiny_8xb32-300e_coco.py'

# 1. 类别数
num_classes = 1
model = dict(
    bbox_head=dict(
        num_classes=num_classes))

# 2. 数据路径
data_root = 'Vistrain/data/VisDrone2019-DET-'
train_ann = 'annotations/instances_train.json'
val_ann   = 'annotations/instances_val.json'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 640), keep_ratio=False),
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')]

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=train_ann,
        data_prefix=dict(img='train/images'),
        metainfo=dict(classes=('object',)),
        filter_cfg=dict(filter_empty_gt=True, min_size=1),  
        pipeline=train_pipeline))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=val_ann,
        data_prefix=dict(img='val/images'),
        metainfo=dict(classes=('object',)),
        # 验证集通常不应过滤空 GT，以便完整评估
        # 如果一定要过滤，请确保不会导致验证集为空
        pipeline=test_pipeline))
test_dataloader = val_dataloader

# 3. 训练 36 epoch（可改）
max_epochs = 18
train_cfg = dict(max_epochs=max_epochs)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-4, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', eta_min=1e-6, by_epoch=True, begin=1, end=max_epochs)]

# 4. 评估器：改为使用当前数据集的标注文件，覆盖 base 的 COCO2017 默认路径
val_evaluator = dict(
    type='CocoMetric',
    ann_file=f"{data_root}/{val_ann}",
    metric='bbox')
test_evaluator = val_evaluator

# 5. 可视化配置：添加 TensorBoard 后端
vis_backends = [
    dict(_scope_='mmdet', type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')  # 添加 TensorBoard 后端
]
visualizer = dict(
    _scope_='mmdet',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),  # 在 visualizer 中也添加 TensorBoard
    ])