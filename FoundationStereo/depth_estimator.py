import os
from omegaconf import OmegaConf
import torch
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *

FS_PATH = os.path.dirname(__file__)


class DepthEstimator:
    def __init__(
        self,
        ckpt_dir="{FS_PATH}/pretrained_models/23-51-11/model_best_bp2.pth",
    ):
        ckpt_dir = ckpt_dir.format(FS_PATH=FS_PATH)
        self._init_model(ckpt_dir)


    def _init_model(self, ckpt_dir):
        cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
        if 'vit_size' not in cfg:
            cfg['vit_size'] = 'vitl'
        args = OmegaConf.create(cfg)

        self.model = FoundationStereo(args)

        ckpt = torch.load(ckpt_dir)
        logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
        self.model.load_state_dict(ckpt['model'])

        self.model.cuda()
        self.model.eval()


    def predict(
        self,
        left_image,
        right_image,
        K,
        baseline=0.063,
        scale=1.0,
        hiera=False,
        valid_iters=32,
        remove_invisible=True,
    ):
        assert scale<=1, "scale must be <=1"
        left_image = cv2.resize(left_image, fx=scale, fy=scale, dsize=None)
        right_image = cv2.resize(right_image, fx=scale, fy=scale, dsize=None)
        H,W = left_image.shape[:2]

        left_image = torch.as_tensor(left_image).cuda().float()[None].permute(0,3,1,2)
        right_image = torch.as_tensor(right_image).cuda().float()[None].permute(0,3,1,2)
        padder = InputPadder(left_image.shape, divis_by=32, force_square=False)
        left_image, right_image = padder.pad(left_image, right_image)

        with torch.cuda.amp.autocast(True):
            if not hiera:
                disp = self.model.forward(left_image, right_image, iters=valid_iters, test_mode=True)
            else:
                disp = self.model.run_hierachical(left_image, right_image, iters=valid_iters, test_mode=True, small_ratio=0.5)
        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H,W)
        
        if remove_invisible:
            yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
            us_right = xx-disp
            invalid = us_right<0
            disp[invalid] = np.inf

        K[:2] *= scale
        depth = K[0,0]*baseline/disp

        return depth
        