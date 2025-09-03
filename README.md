# welding_line_detection

## Installation

* Clone this repository

```bash
git clone git@github.com:mydulee06/welding_line_detection.git
git submodule update --init --recursive
```

### With Docker

* Follow [here](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/deployment/docker.html).
* Clone this repo without submodule update.

* To use Pointcept with docker, use .docker/Dockerfile

```bash
docker build -t <DOCKERHUB_USERNAME>/weld-perc:0.0.1 .docker/
# If you want to push the image to docker hub,
# docker push <DOCKERHUB_USERNAME>/weld-perc:0.0.1
```

### Without Docker

* Follow [here](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/pip_installation.html).

### Install additional dependencies

```bash
pip install -r requirements.txt
```

### Install Grounded-SAM2

* NOTE: not tested in IsaacLab docker.

```bash
cd checkpoints
bash download_ckpts.sh

cd gdino_checkpoints
bash download_ckpts.sh

pip install -e .

pip install --no-build-isolation -e grounding_dino
```

### Install FoundationStereo

* All you need to run FoundationStereo is installing `requirements.txt`.

### Install Pointcept

* Install pointops

```bash
cd Pointcept/libs/pointops
# usual
python setup.py install
# docker & multi GPU arch
TORCH_CUDA_ARCH_LIST="ARCH LIST" python  setup.py install
# e.g. 7.5: RTX 3000; 8.0: a100 More available in: https://developer.nvidia.com/cuda-gpus
TORCH_CUDA_ARCH_LIST="7.5 8.0" python  setup.py install
```

* For more detail, refer to `Pointcept/README.md`.

## Data generation

* Generate welding object mesh.

```bash
python data_gen/models/generator_ABP.py # ABP models
# To generate specimen, first download the obj file in here: https://drive.google.com/file/d/1dg7QSZhqNZDjM3NZjApDWsCi53OElC9C/view?usp=sharing
python data_gen/models/generator_specimen.py --single_specimen_obj_path data_gen/assets/weld_objects/meshes/specimen/single.obj --count 100
```

* Convert mesh to usd.
1. Set converter_context.merge_all_meshes to False by `converter_context.merge_all_meshes = False` in `IsaacLab/source/isaaclab/isaaclab/sim/converters/mesh_converter.py` to give semantics to each part.

2. Convert multiple meshes to usd.

```bash
python data_gen convert_meshes_to_usd.py data_gen/assets/weld_objects/meshes/specimen data_gen/assets/weld_objects/usd/specimen --mass 1.0
```

* Render RGB images using IsaacLab with random object usds in root.

```bash
python data_gen/build_synthetic_isaacsim.py --object_usd_root data_gen/assets/weld_objects/usd/specimen --headless --path_tracing
```

* Visualize rendered data.

```bash
python data_gen/visualize_data.py data_gen/output/ABP_1/0
```

### Segmentation with Grounded-SAM2

* Run video segmentation

```bash
python script/gsa2_video_tracking.py data_gen/output/ABP_1/0/left/color --text_prompt "pillar. bottom plate." --mask_dir data_gen/output/ABP_1/0/left/mask --result_dir data_gen/output/ABP_1/0/left/result
```

### Evaluate Grounded-SAM2 with GT mask

```bash
python script/gsa2_image_segmentation.py data_gen/output/specimen/0/left/color --mask_dir script/output/test1 --result_dir script/output/test1 --plate_mask_gt_dir data_gen/output/specimen/0/left/mask_gt/plate --pillar_mask_gt_dir data_gen/output/specimen/0/left/mask_gt/pillar
```

### Depth estimation with FoundationStereo

* Run depth estimation

```bash
python script/depth_estimating.py data_gen/output/ABP_1/0/left/color data_gen/output/ABP_1/0/right/color data_gen/output/ABP_1/0/left/intrinsic
```

## Line detection

```bash
python script/line_detecting.py data_gen/output/ABP_1/0
```

## Pointcept

### Data pre-processing for Pointcept

```bash
python data_gen/preprocess_poincept.py data_gen/output/specimen
```

### Training

* Training Point Transformer v3

```bash
cd Pointcept
PYTHONPATH=./ python tools/train.py --config-file configs/welding/seg-pt-v3m1-0-base.py --num-gpus 1 [--options num_worker=8 enable_wandb=False save_path=exp/welding/seg-pt-v3m1-0-base]
```

* Testing Point Transformer v3

```bash
cd Pointcept
PYTHONPATH=./ python tools/test.py --config-file configs/welding/seg-pt-v3m1-0-base.py --num-gpus 1 --options save_path=exp/welding/seg-pt-v3m1-0-base weight=exp/welding/seg-pt-v3m1-0-base/model/model_last.pth [test.verbose=Ture/False]
```
