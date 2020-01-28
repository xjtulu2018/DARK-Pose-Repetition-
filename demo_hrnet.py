import json
#import trt_pose.coco
import numpy as np
import argparse
import os
import time
import pprint
import cv2

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
    
    parser.add_argument('--testModelPath',
                        help='test model path',
                        type=str,
                        default='./models/pytorch/pose_coco/pose_hrnet_w48_384x288.pth')
    parser.add_argument('--testMode',
                        help='test mode',
                        type=str,
                        choices=['video', 'camera'],
                        default='video')
    parser.add_argument('--videoPath',
                        help='video path',
                        type=str,
                        default='./demo/kunkun_cut.mp4')
    parser.add_argument('--outputPath',
                        help='output path',
                        type=str,
                        default='./demo/outpus')
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



'''
initialize hrnet w32 128x96 model
'''

model_path = args.testModelPath  # './mymodels/model_best.pth'
cudnn.benchmark = cfg.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
    cfg, is_train=False
)
logger.info('=> loading model from {}'.format(model_path))
model.load_state_dict(torch.load(model_path), strict=False)
logger.info('model already loaded!')

'''
capture and crop photos
'''
#input size
WIDTH=cfg.MODEL.IMAGE_SIZE[0]
HEIGHT=cfg.MODEL.IMAGE_SIZE[1]

if args.testMode == 'camera':
    cap = cv2.VideoCapture(0)
elif args.testMode == 'video':
    cap = cv2.VideoCapture(args.videoPath)
else:
    raise NotImplementedError('only video or camera input supported')

frameSize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#cv2.imwrite('./test.jpg', cap.read()[1])
if frameSize[0] * HEIGHT > frameSize[1] * WIDTH:
    max_expand = frameSize[1] / 4
    oh_l = 0; oh_h = oh_l + max_expand * 4
    ow_l = (frameSize[0] - max_expand * 3) // 2; ow_h = ow_l + max_expand * 3
else:
    max_expand = frameSize[0] / 3
    oh_l = (frameSize[1] - max_expand * 4) // 2; oh_h = oh_l + max_expand * 4
    ow_l = 0; ow_h = ow_l + max_expand * 3

def preprocess(image):
    img_crop = image[int(oh_l):int(oh_h), int(ow_l):int(ow_h)]
    #cv2.imwrite('./test1.jpg', img_crop)
    frameWidth = img_crop.shape[1]
    frameHeight = img_crop.shape[0]
    image_resize = cv2.resize(img_crop, (WIDTH, HEIGHT))
    return image_resize, img_crop

def preprocess_direct_resize(image):
    image_resize = cv2.resize(image, (WIDTH, HEIGHT))
    return image_resize, image
  

def get_img_np_nchw(image):
    image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image_cv = cv2.resize(image_cv, (128, 96))
    miu = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = np.array(image_cv, dtype=float) / 255.0
    r = (img_np[:, :, 0] - miu[0]) / std[0]
    g = (img_np[:, :, 1] - miu[1]) / std[1]
    b = (img_np[:, :, 2] - miu[2]) / std[2]
    img_np_t = np.array([r, g, b])
    image_np_nchw = np.expand_dims(img_np_t, axis=0)
    return image_np_nchw

def save_kp2d_to_json(kp_2d, dump_dir):
    json_out = {"pose_keypoints_2d": kp_2d.tolist()}
    with open(dump_dir, 'w', encoding='utf-8') as json_file:
        json.dump(json_out, json_file)

'''
model test demo
'''
flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
sigma = cfg.MODEL.SIGMA
num_joints=17

JSON_OUTPUT_DIR = os.path.join(args.outputPath, 'json')
OUTPUT_VIDEO_DIR = os.path.join(args.outputPath, 'video') # './demo_out/video/output_demo.mp4'
if not os.path.exists(JSON_OUTPUT_DIR):
    os.makedirs(JSON_OUTPUT_DIR)
if not os.path.exists(OUTPUT_VIDEO_DIR):
    os.makedirs(OUTPUT_VIDEO_DIR)
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_VIDEO_DIR, 'output_demo.mp4')
#Video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_vid = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, int(20), (360,480))

with torch.no_grad():
    model.cuda()
    model.eval()
    infer_times = []
    counter = 0
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv2.destroyAllWindows()
            break
        if frameSize[1] / frameSize[0] == 480 / 360:
            img_crop, img_crop_origin = preprocess_direct_resize(frame)
            print('direct resize')
        else:
            img_crop, img_crop_origin = preprocess(frame)
            print('crop and resize')
        img_np_nchw = get_img_np_nchw(img_crop).astype(dtype=np.float32)
        input_for_torch = torch.from_numpy(img_np_nchw).cuda()
        output = model(input_for_torch)
        if cfg.TEST.FLIP_TEST:
            # this part is ugly, because pytorch has not supported negative index
            # input_flipped = model(input[:, :, :, ::-1])
            input_flipped = np.flip(input_for_torch.cpu().numpy(), 3).copy()
            input_flipped = torch.from_numpy(input_flipped).cuda()
            output_flipped = model(input_flipped)

            output_flipped = flip_back(output_flipped.cpu().numpy(), flip_pairs)
            output_flipped = torch.from_numpy(output_flipped.copy()).cuda()
            if cfg.TEST.SHIFT_HEATMAP:
                output_flipped[:, :, :, 1:] = output_flipped.clone()[:, :, :, 0:-1]
            output = (output + output_flipped) * 0.5
        #print(output.shape)
        output_DM=gaussian_modulation_torch(output, sigma)
        preds, maxvals = demo_preds_function(cfg, output.clone().cpu().numpy(), output_DM.clone().cpu().numpy())
        preds = preds * 4 * (oh_h - oh_l) / HEIGHT # 128
        preds = preds.reshape([17, -1])
        maxvals = maxvals.reshape(-1)
        vis = maxvals > 0.3
        #print(preds, maxvals)
        #out_vid.write(img_crop_origin)
        save_kp2d_to_json(preds, os.path.join(JSON_OUTPUT_DIR, 'frame_{:05d}.json'.format(counter)))
        print(preds)
        for i in range(17):
            if vis[i]:
                cv2.circle(img_crop_origin, (preds[i][0],preds[i][1]), 2, (0, 0, 255), 3)
        for i in skeleton_array:
            if vis[i[0]] and vis[i[1]]:
                cv2.line(img_crop_origin, (preds[i[0]][0],preds[i[0]][1]), (preds[i[1]][0],preds[i[1]][1]), (0, 0, 255), 2)
    #cv2.imwrite('./test1.jpg', img_crop_origin)
        out_vid.write(img_crop_origin)
        if (counter > 0):
            infer_time = time.time() - t_start
            print('Inference Time:{}'.format(infer_time))
            infer_times.append(infer_time)
        counter += 1
        t_start = time.time()
        img_show = img_crop_origin

        # if img_show is None:
        #     print("Cant Load Image")
        # else:
        #     cv2.imshow("Image", img_show)
        # cv2.imshow('HRNetw32', img_show)
        # key = cv2.waitKey(1)  # ms
        # if key == 27:  # esc
        #     cv2.destroyAllWindows()
        #     break
        #     print('Average Inference Time:{}'.format(np.mean(infer_time)))
