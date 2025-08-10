accumulative_counts = 2
batch_size = 2
betas = (
    0.9,
    0.999,
)
custom_hooks = [
    dict(
        tokenizer=dict(
            pretrained_model_name_or_path=
            '/home/luoyx/InternVL/InternVL-main_test',
            trust_remote_code=True,
            type='transformers.AutoTokenizer.from_pretrained'),
        type='xtuner.engine.hooks.DatasetInfoHook'),
]
data_path = [
    '/home/luoyx/finetune/data/new11_6_without_multi/all_data/_0.json',
]
data_root = '/home/luoyx/finetune/data/new11_6_without_multi/all_data/'
dataloader_num_workers = 4
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=1500,
        max_keep_ckpts=1,
        save_optimizer=False,
        type='mmengine.hooks.CheckpointHook'),
    logger=dict(
        interval=10,
        log_metric_by_epoch=False,
        type='mmengine.hooks.LoggerHook'),
    param_scheduler=dict(type='mmengine.hooks.ParamSchedulerHook'),
    sampler_seed=dict(type='mmengine.hooks.DistSamplerSeedHook'),
    timer=dict(type='mmengine.hooks.IterTimerHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
image_folder = [
    '/home/luoyx/finetune/data/new11_6_without_multi/all_data/',
]
launcher = 'pytorch'
llava_dataset = dict(
    data_paths=[
        '/home/luoyx/finetune/data/new11_6_without_multi/all_data/_0.json',
    ],
    image_folders=[
        '/home/luoyx/finetune/data/new11_6_without_multi/all_data/',
    ],
    max_length=8192,
    model_path='/home/luoyx/InternVL/InternVL-main_test',
    repeat_times=[
        1,
    ],
    template='xtuner.utils.PROMPT_TEMPLATE.internlm2_chat',
    type='xtuner.dataset.InternVL_V1_5_Dataset')
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
lr = 5e-05
max_epochs = 1
max_length = 8192
max_norm = 1
model = dict(
    freeze_llm=True,
    freeze_visual_encoder=True,
    llm_lora=dict(
        lora_alpha=256,
        lora_dropout=0.05,
        r=128,
        target_modules=None,
        task_type='CAUSAL_LM',
        type='peft.LoraConfig'),
    model_path='/home/luoyx/InternVL/InternVL-main_test',
    type='xtuner.model.InternVL_V1_5')
optim_type = 'torch.optim.AdamW'
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        lr=5e-05,
        type='torch.optim.AdamW',
        weight_decay=0.01),
    type='DeepSpeedOptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=0.03,
        start_factor=1e-05,
        type='mmengine.optim.LinearLR'),
    dict(
        begin=0.03,
        by_epoch=True,
        convert_to_iter_based=True,
        end=1,
        eta_min=0.0,
        type='mmengine.optim.CosineAnnealingLR'),
]
path = '/home/luoyx/InternVL/InternVL-main_test'
prompt_template = 'xtuner.utils.PROMPT_TEMPLATE.internlm2_chat'
randomness = dict(deterministic=False, seed=None)
repeat_times = [
    1,
]
resume = False
runner_type = 'FlexibleRunner'
save_steps = 1500
save_total_limit = 1
strategy = dict(
    config=dict(
        bf16=dict(enabled=True),
        fp16=dict(enabled=False, initial_scale_power=16),
        gradient_accumulation_steps='auto',
        gradient_clipping='auto',
        train_micro_batch_size_per_gpu='auto',
        zero_allow_untested_optimizer=True,
        zero_force_ds_cpu_optimizer=False,
        zero_optimization=dict(overlap_comm=True, stage=1)),
    exclude_frozen_parameters=True,
    gradient_accumulation_steps=2,
    gradient_clipping=1,
    sequence_parallel_size=1,
    train_micro_batch_size_per_gpu=2,
    type='xtuner.engine.DeepSpeedStrategy')
tokenizer = dict(
    pretrained_model_name_or_path='/home/luoyx/InternVL/InternVL-main_test',
    trust_remote_code=True,
    type='transformers.AutoTokenizer.from_pretrained')
train_cfg = dict(max_epochs=1, type='xtuner.engine.runner.TrainLoop')
train_dataloader = dict(
    batch_size=2,
    collate_fn=dict(type='xtuner.dataset.collate_fns.default_collate_fn'),
    dataset=dict(
        data_paths=[
            '/home/luoyx/finetune/data/new11_6_without_multi/all_data/_0.json',
        ],
        image_folders=[
            '/home/luoyx/finetune/data/new11_6_without_multi/all_data/',
        ],
        max_length=8192,
        model_path='/home/luoyx/InternVL/InternVL-main_test',
        repeat_times=[
            1,
        ],
        template='xtuner.utils.PROMPT_TEMPLATE.internlm2_chat',
        type='xtuner.dataset.InternVL_V1_5_Dataset'),
    num_workers=4,
    sampler=dict(
        length_property='modality_length',
        per_device_batch_size=4,
        type='xtuner.dataset.samplers.LengthGroupedSampler'))
visualizer = None
warmup_ratio = 0.03
weight_decay = 0.01
work_dir = './logs/e_it'
