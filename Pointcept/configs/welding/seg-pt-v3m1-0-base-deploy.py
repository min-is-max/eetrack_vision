# configs/welding/seg-pt-v3m1-0-base-deploy.py
_base_ = ["../_base_/default_runtime.py"]

save_path = "exp/welding/deploy-run"
seed = 42
batch_size = 8
num_worker = 1
mix_prob = 0.0
empty_cache = False
enable_amp = True
num_spl_coef = 16
use_normal = False

# --- 모델은 그대로 ---
model = dict(
    type="DefaultSegmentorV2",
    num_classes=2,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=5 if use_normal else 2,
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
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", weight=[0.1, 1.0], loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# 스케줄러(안 써도 되지만, test 스크립트가 cfg를 읽으므로 남겨둠)
epoch = 1
eval_epoch = 1
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

# --- 데이터 루트: 배포 전처리 산출물 ---
dataset_type = "WeldingDataset"
data_root = "data/welding"

spline_coef_keys = ["spl_c"]
batch_keys = ["edge", "edge_ds", "spl_t", "spl_c", "spl_k", "spl_coef"]
collect_keys = ["coord", "grid_coord", "segment", "obj_segment"]

data = dict(
    num_classes=2,
    ignore_index=-1,
    names=["none", "edge"],
    # train/val은 안 쓰지만, 참조 오류 피하려고 최소 정의
    train=dict(
        type=dataset_type,
        spline_coef_keys=spline_coef_keys,
        batch_keys=batch_keys,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="RemoveNonSeg", key="obj_segment"),
            dict(type="GridSample", grid_size=0.002, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="ToTensor"),
            dict(type="Collect", keys=collect_keys, feat_keys=("obj_segment_onehot",)),
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
            dict(type="RemoveNonSeg", key="obj_segment"),
            dict(type="GridSample", grid_size=0.002, hash_type="fnv", mode="train",
                 return_grid_coord=True, return_inverse=True),
            dict(type="ToTensor"),
            dict(type="Collect", keys=collect_keys, feat_keys=("obj_segment_onehot",)),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        spline_coef_keys=spline_coef_keys,
        batch_keys=batch_keys + ["visible_edge"],
        split="test",
        data_root=data_root,
        transform=[
            dict(type="RemoveNonSeg", key="obj_segment"),
            dict(type="GridSample", grid_size=0.002, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="ToTensor"),
            dict(type="Collect", keys=collect_keys + ["color", "visible_edge"],
                 feat_keys=("obj_segment_onehot",)),
        ],
        test_mode=False,
    ),
)

# --- 평가/로그 훅 단순화: 메트릭 훅 제거 (배포 미리보기 용)
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="InformationWriter"),
]

# --- 커스텀 테스터 지정
# 예: @TESTERS.register_module(name="DeployPreviewTester")
test = dict(type="DeployPreviewTester", verbose=True)

# 선택) W&B 끄기
enable_wandb = False
