import json
#import trt_pose.coco
import numpy as np
import argparse
import os
import time
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths_demo
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger
from core.inference import demo_preds_function
from core.inference import gaussian_modulation_torch
from utils.transforms import flip_back
import models
#from save_pose import SaveObjects, save_kp2d_to_json

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args

keypoints: {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"}
skeleton = [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
skeleton_array = np.array(skeleton)-1
args = parse_args()
update_config(cfg, args)

logger, final_output_dir, tb_log_dir = create_logger(
    cfg, args.cfg, 'valid')

#logger.info(pprint.pformat(args))
#logger.info(cfg)5

JSON_OUTPUT_DIR = './output_demo/camera_demo_hrnet'

'''
initialize hrnet w32 128x96 model
'''

model_path = './mymodels/model_best.pth'
cudnn.benchmark = cfg.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
    cfg, is_train=False
)
logger.info('=> loading model from {}'.format(model_path))
model.load_state_dict(torch.load(model_path), strict=False)
logger.info('model already loaded!')


# print(model.stage2[0].branches[0][-1].bn2)
# print(model.stage2[0].branches[1][-1].bn2)

# print(model.stage3[0].branches[0][-1].bn2)
# print(model.stage3[0].branches[1][-1].bn2)
# print(model.stage3[0].branches[2][-1].bn2)
# print(model.stage3[0].branches[2][-1])

# print(model.stage4[0].branches[0][-1].bn2)
# print(model.stage4[0].branches[1][-1].bn2)
# print(model.stage4[0].branches[2][-1].bn2)
# print(model.stage4[0].branches[3][-1].bn2)
# print(model.stage4[0].branches[3][-1])
print(model.stage3_cfg)

# print(model)