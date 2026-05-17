# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

_CHECKPOINT_ROOT = os.path.expanduser(
    os.environ.get("SAPIENS_CHECKPOINT_ROOT", "~/sapiens2_host")
)
_DATA_ROOT = os.path.expanduser(os.environ.get("DATA_ROOT", "~/sapiens_data"))

warmup_iters = 500

## 32 nodes, 8 gpus: 256 gpus. bs: 8, global bs: 2048.
# ~10M samples / 2048 = 4850 iters/epoch
num_iters = 4850 * 3

# ------------------------------------------------------------------------------
vis_every_iters = 100
log_every_iters = 10
save_every_iters = 1000
val_every_iters = 1000

# # # # debug
# vis_every_iters = 1
# log_every_iters = 1
# val_every_iters = 2
# save_every_iters = 1000

load_from = None
resume = False

# ------------------------------------------------------------------
model_name = "sapiens2_1b"
embed_dim = 1536
num_layers = 40
num_heads = 24

layer_decay_rate = 0.9
pretrained_checkpoint = f"{_CHECKPOINT_ROOT}/pretrain/sapiens2_1b_pretrain.safetensors"
##-----------------------------------------------------------------
image_size = (1024, 768)  ## height x width
patch_size = 16

# ------------------------------------------------------------------
use_fsdp = True
# use_fsdp = False

use_compile = True
# use_compile = False

## DDP config
if use_fsdp is False:
    accelerator_cfg = dict(
        type="DDP",
        log_with="tensorboard",
        # find_unused_parameters=True,
        gradient_accumulation_steps=1,  # only accumulation=1 is supported. Otherwise, the LR scheduler will be off.
        max_interval=num_iters,
        # mixed_precision="bf16",  # Options: ‘no’,‘fp16’,‘bf16’ or ‘fp8’.
        step_scheduler_with_optimizer=False,  ## schedule independent of n_gpus
    )

else:
    accelerator_cfg = dict(
        type="FSDP",
        log_with="tensorboard",
        gradient_accumulation_steps=1,  # only accumulation=1 is supported. Otherwise, the LR scheduler will be off.
        max_interval=num_iters,
        mixed_precision="bf16",  # Options: ‘no’,‘fp16’,‘bf16’ or ‘fp8’.
        step_scheduler_with_optimizer=False,
        fsdp_cfg=dict(
            fsdp_version=2,  # DTensor-based engine
            state_dict_type="SHARDED_STATE_DICT",  # SHARDED_STATE_DICT | FULL_STATE_DICT
            # state_dict_type="FULL_STATE_DICT",  # TODO: resume from this is not working
            mixed_precision=dict(
                param_dtype="bf16",
                reduce_dtype="bf16",
            ),
            cpu_ram_efficient_loading=False,
        ),
    )

if use_compile:
    accelerator_cfg["compile_cfg"] = dict(
        backend="inductor",
        mode="default",  # Options: "default", "reduce-overhead", "max-autotune"
        fullgraph=False,
        dynamic=False,
    )

# ------------------------------------------------------------------
randomness = dict(seed=0, deterministic=False, diff_rank_seed=True)
logger = dict(
    type="Logger",
    log_interval=log_every_iters,
)
checkpoint = dict(
    type="Checkpointer",
    save_interval=save_every_iters,
)

visualizer = dict(
    type="MattingVisualizer",
    vis_interval=vis_every_iters,
    vis_max_samples=4,
    vis_image_width=384,
    vis_image_height=512,
)


##-----------------------------------------------------------------

train_pipeline = [
    dict(
        type="MattingRandomJPEGCompression",
        prob=0.5,
    ),
    dict(
        type="MattingCropAlphaBBox",
        prob=0.5,
        padding_ratio=(0.0, 0.1),
    ),
    dict(
        type="MattingRandomResize",
        base_height=image_size[0],
        base_width=image_size[1],
        ratio_range=(0.4, 2.0),
        keep_ratio=True,
    ),
    dict(type="MattingRandomCrop", crop_size=image_size),
    dict(
        type="MattingRandomRotate",
        prob=0.5,
        degree=60,
        pad_val=(255, 255, 255),
        seg_pad_val=0,
    ),  ## the black pixels are set as background
    dict(
        type="MattingRandomFlip",
        prob=0.5,
    ),
    dict(
        type="MattingResize",
        height=image_size[0],
        width=image_size[1],
        keep_ratio=False,
    ),
    dict(type="MattingPhotoMetricDistortion"),
    dict(type="MattingPackInputs"),
]

gss_train_pipeline = [
    dict(
        type="MattingRandomBackground",
        prob=1.0,
        background_images_root=[
            f"{_DATA_ROOT}/matting/backgrounds",
        ],
    ),
    dict(
        type="MattingRandomJPEGCompression",
        prob=0.5,
    ),
    dict(
        type="MattingCropAlphaBBox",
        prob=0.5,
        padding_ratio=(0.0, 0.1),
    ),
    dict(
        type="MattingRandomResize",
        base_height=image_size[0],
        base_width=image_size[1],
        ratio_range=(0.4, 2.0),
        keep_ratio=True,
    ),
    dict(type="MattingRandomCrop", crop_size=image_size),
    dict(
        type="MattingRandomRotate",
        prob=0.5,
        degree=60,
        pad_val=(255, 255, 255),
        seg_pad_val=0,
    ),  ## the black pixels are set as background
    dict(
        type="MattingRandomFlip",
        prob=0.5,
    ),
    dict(
        type="MattingResize",
        height=image_size[0],
        width=image_size[1],
        keep_ratio=False,
    ),
    dict(type="MattingPhotoMetricDistortion"),
    dict(type="MattingPackInputs"),
]

test_pipeline = [
    dict(
        type="MattingResize",
        height=image_size[0],
        width=image_size[1],
        keep_ratio=False,
        test_mode=True,
    ),
    dict(type="MattingPackInputs", test_mode=True),
]

##------------------------------------------------------------------------
dataset_gss_train = dict(
    type="MattingGSSDataset",
    data_root=f"{_DATA_ROOT}/matting/gss",
    ann_file=f"{_DATA_ROOT}/matting/gss/fgr_pha.txt",
    pipeline=gss_train_pipeline,
)
dataset_p3m_train = dict(
    type="MattingBaseDataset",
    data_root=f"{_DATA_ROOT}/matting/p3m",
    ann_file=f"{_DATA_ROOT}/matting/p3m.json",
    pipeline=train_pipeline,
)
dataset_capturegen4d = dict(
    type="MattingBaseDataset",
    data_root=f"{_DATA_ROOT}/matting/capturegen4d",
    ann_file=f"{_DATA_ROOT}/matting/capturegen4d.json",
    pipeline=train_pipeline,
)
dataset_metasim = dict(
    type="MattingBaseDataset",
    data_root=f"{_DATA_ROOT}/matting/metasim",
    ann_file=f"{_DATA_ROOT}/matting/metasim.json",
    pipeline=train_pipeline,
)

train_datasets = [
    dataset_gss_train,
    dataset_p3m_train,
    dataset_capturegen4d,
    dataset_metasim,
]
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    shuffle=True,
    dataset=dict(
        type="CombinedDataset",
        datasets=train_datasets,
    ),
)

val_dataloader = None

val_cfg = None

data_preprocessor = dict(
    type="ImagePreprocessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,  ## convert from bgr to rgb for pretrained models
)

##-----------------------------------------------------------------
model = dict(
    type="MattingEstimator",
    backbone=dict(
        type="Sapiens2",
        arch=model_name,
        img_size=image_size,
        patch_size=patch_size,
        final_norm=True,
        use_tokenizer=False,
        with_cls_token=True,
        out_type="featmap",
        init_cfg=dict(type="Pretrained", checkpoint=pretrained_checkpoint),
    ),
    decode_head=dict(
        type="MattingHead",
        in_channels=embed_dim,
        upsample_channels=[768, 512, 256, 128],  ## 1K resolution
        conv_out_channels=[64, 32, 16],
        conv_kernel_sizes=[3, 3, 3],
        out_channels=4,
        loss_decode=[
            dict(type="MattingL1Loss", loss_weight=1.0),
            dict(type="MattingGradLoss", loss_weight=1.0),
            dict(type="MattingLaplacianLoss", loss_weight=1.0),
        ],
    ),
)


##-----------------------------------------------------------------
optimizer = dict(
    type="AdamW",
    lr=5e-4,
    betas=(0.9, 0.999),
    weight_decay=0.1,
    paramwise_cfg=dict(
        num_layers=num_layers,
        layer_decay_rate=layer_decay_rate,
    ),
    fused=True,
)

scheduler = dict(
    type="SequentialLR",
    milestones=[warmup_iters],
    schedulers=[
        dict(type="LinearLR", start_factor=1e-3, total_iters=warmup_iters),
        dict(
            type="PolynomialLR",
            total_iters=num_iters - warmup_iters,
            power=1.0,
        ),
    ],
)

clip_grad = dict(mode="norm", max_norm=4.0, norm_type=2.0)
