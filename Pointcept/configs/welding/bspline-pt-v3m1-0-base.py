_base_ = ["../_base_/default_runtime.py"]

save_path = "exp/welding/bspline-pt-v3m1-0-base"

# misc custom setting
seed = 42
batch_size = 8  # bs: total bs in all gpus
num_worker = 1
# Mixing two point clouds into one. (ref: https://arxiv.org/pdf/2110.02210)
mix_prob = 0.0
empty_cache = False
enable_amp = True
num_spl_coef = 16

# model settings
model = dict(
    type="DefaultBsplineCoefPredictor",
    spl_coef_channels=3,    # x, y, z
    num_spl_coef=num_spl_coef,
    backbone_embed_dim=512,
    backbone=dict(
        type="PT-v3m1",
        in_channels=2,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=True, # No decoder if True, else use decoder.
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        dict(type="MSELoss", loss_weight=1.0),
    ],
)

# scheduler settings
epoch = 400
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.0001],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0001)]

# dataset settings
dataset_type = "WeldingDataset"
data_root = "data/welding"
# Keys to predict
spline_coef_keys = ["spl_c"]
# Keys to make batch i.e. N x M x ... -> B x N x M x ... (Pointcept collate_fn do concat data in default).
batch_keys = ["edge", "edge_ds", "spl_t", "spl_c", "spl_k", "spl_coef"]

data = dict(
    num_spl_coef=num_spl_coef,
    spline_coef_keys=spline_coef_keys,
    train=dict(
        type=dataset_type,
        spline_coef_keys=spline_coef_keys,
        batch_keys=batch_keys,
        split="train",
        data_root=data_root,
        transform=[
            # dict(type="CenterShift", apply_z=True),
            # dict(
            #     type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            # ),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            # dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            # dict(type="RandomFlip", p=0.5),
            # dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            # dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            # dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            # dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            # Grid down sampling.
            # dict(type="RemoveNonSeg"),
            dict(
                type="GridSample",
                grid_size=0.002,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=102400, mode="random"),
            # dict(type="CenterShift", apply_z=False),
            # dict(type="NormalizeColor"),
            # dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                # These keys are only remained in dataloader.
                keys=("coord", "grid_coord", "obj_segment", "edge", "edge_ds", "spl_t", "spl_c", "spl_k", "spl_coef"),
                # Keys used for point features, "feat"
                feat_keys=("obj_segment_onehot",),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        spline_coef_keys=spline_coef_keys,
        batch_keys=batch_keys,
        split="val",
        data_root=data_root,
        transform=[
            # dict(type="CenterShift", apply_z=True),
            # dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            # dict(type="RemoveNonSeg"),
            dict(
                type="GridSample",
                grid_size=0.002,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            # dict(type="CenterShift", apply_z=False),
            # dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "obj_segment", "edge", "edge_ds", "spl_t", "spl_c", "spl_k", "spl_coef"),
                feat_keys=("obj_segment_onehot",),
            ),
        ],
        test_mode=False,
    ),
    # TODO: Configure properly
    test=dict(
        type=dataset_type,
        spline_coef_keys=spline_coef_keys,
        batch_keys=batch_keys,
        split="test",
        data_root=data_root,
        transform=[
            # dict(type="RemoveNonSeg"),
            dict(
                type="GridSample",
                grid_size=0.002,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color", "obj_segment", "edge", "edge_ds", "spl_t", "spl_c", "spl_k", "spl_coef"),
                feat_keys=("obj_segment_onehot",),
            ),
        ],
        # Disable test mode due to all coverage point cloud in test mode.
        test_mode=False,
    ),
)

# hooks
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="BsplineEvaluator", knots_from=["uniform", "dataset"], eval_u_size=501),
    dict(type="CheckpointSaver", save_freq=None),
]

# tester
test = dict(type="BsplineTester", knots_from=["uniform", "dataset"], eval_u_size=501, vis=True)
