# ------------------------------------------------------------------------------
# Created by Sierkinhane(sierkinhane@163.com)
# ------------------------------------------------------------------------------

import os
import pprint
import argparse

import torch
import torch.backends.cudnn as cudnn
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.models.tab import get_face_alignment_net
from lib.utils import utils
from lib.models.mtcnn import MTCNN
from lib.utils.transforms import crop
from lib.core.evaluation import decode_preds
from lib.utils.imutils import demo_plot
from easydict import EasyDict as edict
import yaml
import cv2
from PIL import Image
import glob
import numpy as np
import time

def parse_args():

    parser = argparse.ArgumentParser(description='Demo for TAB')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--stem', default='vgg', help='experiments', type=str)
    parser.add_argument('--type', default='images', help='demo', type=str)
    parser.add_argument('--file_path', help='image path', default='images/',
                        type=str)
    parser.add_argument('--best_model', default='', help='the best checkpoint for demo', type=str)
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)

    return args, config

def demo_image(config, tab, mtcnn, file_path, device):

    img = np.array(Image.open(file_path).convert('RGB'), dtype=np.float32)
    img_draw = img.copy()

    # detect faces
    boxes, probs = mtcnn.detect(img, landmarks=False)
    if boxes is None:
        return
    # print(boxes)
    boxes = utils.sacle_boxes(boxes)

    tab.eval()
    with torch.no_grad():
        for i, box in enumerate(boxes):

            center_w, center_h = (box[2] + box[0]) / 2, (box[3] + box[1]) / 2
            scale = np.maximum(box[2] - box[0], box[3] - box[1]) / 200.
            scale = torch.Tensor([scale])
            center = torch.Tensor([center_w, center_h])

            face = crop(img, center, scale, [256, 256], rot=0)
            scale = scale.unsqueeze(0)
            center = center.unsqueeze(0)
            face = face.astype(np.float32)
            face_copy = face.copy()
            face = (face / 255.0 - config.DATASET.MEAN) / config.DATASET.STD
            face = face.transpose([2, 0, 1])
            face = face.astype(np.float32)
            face = torch.from_numpy(face).unsqueeze(0).to(device)
            output = tab(face)
            score_map = output.data.cpu()

            preds = decode_preds(config, score_map, center, scale, [64, 64])

            img_draw = demo_plot(config, img_draw, box, preds.cpu().numpy().squeeze(0), color=(255, 0, 255), rec=False, line=True)

    if not os.path.exists('output/demo'):
        os.mkdir('output/demo')

    img_draw = img_draw[...,::-1]
    cv2.imwrite('output/demo/{}_.png'.format(file_path.split('/')[-1].split('.')[0]), face_copy[...,::-1])
    cv2.imwrite('output/demo/{}.png'.format(file_path.split('/')[-1].split('.')[0]), img_draw)

def demo_video(config, tab, mtcnn, file_path, device):

    cap = cv2.VideoCapture(file_path)

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.
    if int(major_ver) < 3:
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    total_frames = int(cap.get(7))

    if not os.path.exists('output/video'):
        os.mkdir('output/video')
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.

    out = cv2.VideoWriter('output/video/{}.avi'.format(file_path.split('/')[-1].split('.')[0]),
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             fps, (frame_width, frame_height))

    # check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # smooth
    # boxes_smoother = utils.Smoother()
    # landmarks_smoother = utils.Smoother()

    # read util video is completed
    count=0
    tab.eval()
    with torch.no_grad():
        while (cap.isOpened()):
            count+=1
            s = time.time()
            # capture frame-by-frame
            ret, frame_inp = cap.read()

            if ret == True:

                frame_draw = frame_inp.copy()
                boxes, probs = mtcnn.detect(frame_inp, landmarks=False)
                if boxes is None:
                    continue

                # boxes = utils.sacle_boxes(boxes)
                frame_inp = (frame_inp - config.DATASET.MEAN) / config.DATASET.STD

                boxes = utils.sacle_boxes(boxes)
                # boxes_smoother.smooth(boxes)
                scales = torch.zeros((len(boxes), 1))
                centers = torch.zeros((len(boxes), 2))
                faces = torch.zeros((len(boxes), 3, 256, 256))
                for i, box, in enumerate(boxes):
                    center_w, center_h = (box[2] + box[0]) / 2, (box[3] + box[1]) / 2
                    scale = np.maximum(box[2] - box[0], box[3] - box[1]) / 200.
                    scale = torch.Tensor([scale])
                    center = torch.Tensor([center_w, center_h])
                    face = crop(frame_inp, center, scale, [256, 256], rot=0)
                    scales[i] = scale
                    centers[i] = center
                    face = face.astype(np.float32)
                    face_copy = face.copy()
                    face = (face / 255.0 - config.DATASET.MEAN) / config.DATASET.STD
                    face = face.transpose([2, 0, 1])
                    face = face.astype(np.float32)
                    faces[i] = torch.from_numpy(face)

                faces = faces.to(device)
                output = tab(faces)
                score_map = output.data.cpu()
                preds = decode_preds(config, score_map, centers, scales, [64, 64])
                # landmarks_smoother.smooth(preds.cpu().numpy())
                for i in range(len(boxes)):
                    frame_draw = demo_plot(config, frame_draw, boxes[i], preds[i].cpu().numpy(),
                                     rec=False, line=False)

                # cv2.imwrite('output/video/jasonwang.png', face_copy)
                out.write(frame_draw)
                # cv2.imshow('frame', frame_draw)
                # press Q on keyboard to exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
            e = time.time()
            if count % 100 == 0:
                print('processed [{}/{}] frames, FPS: {}.'.format(count, total_frames, int(1/(e-s))))


    cap.release()
    cv2.destroyAllWindows()

def demo_camera(config, tab, mtcnn, file_path, device):
    pass

def main():

    args, config = parse_args()

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    tab = get_face_alignment_net(config, stem=args.stem, scbe=False, test=True)

    state_dict = torch.load(args.best_model, map_location='cpu')
    if isinstance(state_dict, dict):
        print("nme", state_dict['best_nme'])
        tab.load_state_dict(state_dict['state_dict'])
    else:
        tab.load_state_dict(state_dict['state_dict'])

    if torch.cuda.is_available:
        device = torch.device('cuda:{}'.format(config.GPUID))
    else:
        device = torch.device('cpu:0')

    tab = tab.to(device)
    # construct mtcnn
    mtcnn = MTCNN(thresholds=[0.6, 0.7, 0.85], keep_all=True, device=device)

    if args.type == 'image':
        demo_image(config, tab, mtcnn, args.file_path, device)
    elif args.type == 'video':
        demo_video(config, tab, mtcnn, args.file_path, device)
    elif args.type == 'camera':
        demo_camera(config, tab, mtcnn, args.file_path, device)
    elif args.type == 'images':
        paths = glob.glob(os.path.join(args.file_path, '*.*g'))
        for path in paths:
            demo_image(config, tab, mtcnn, path, device)

if __name__ == "__main__":

    main()




