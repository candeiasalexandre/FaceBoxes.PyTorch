from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg
from layers.functions.prior_box import PriorBox
from utils.nms_wrapper import nms
#from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.faceboxes import FaceBoxes
from utils.box_utils import decode
from utils.timer import Timer
from face_landmarks import MenpoDlibLandmarkDetector

class FaceBoxesDetector():

    def __init__(self, model_path, cpu, confidence_threshold=0.05, top_k=5000, nms_threshold=0.3, keep_top_k=750, resize_scale=1):
        """[summary]
        
        Arguments:
            model_path {String} -- [description]
            cpu {Bool} -- [description]
        
        Keyword Arguments:
            confidence_threshold {float} -- [description] (default: {0.05})
            top_k {int} -- [description] (default: {5000})
            nms_threshold {float} -- [description] (default: {0.3})
            keep_top_k {int} -- [description] (default: {750})
            resize_scale {int} -- [description] (default: {1})
        """
        torch.set_grad_enabled(False)
        self.face_detector = FaceBoxes(phase='test', size=None, num_classes=2)
        self.face_detector = self._load_model(self.face_detector, model_path, cpu)
        self.face_detector.eval()
        print('Finished loading model!')
        cudnn.benchmark = True
        self.device = torch.device("cpu" if cpu else "cuda")
        self.face_detector = self.face_detector.to(self.device)

        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.nms_threshold = nms_threshold
        self.keep_top_k = keep_top_k
        self.resize_scale = resize_scale
        self.cpu = cpu
    
    def _check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        #print('Missing keys:{}'.format(len(missing_keys)))
        #print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        #print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    def _remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common   prefix 'module.' '''
        #print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def _load_model(self, model, pretrained_path, load_to_cpu):
        print('Loading pretrained model from {}'.format(pretrained_path))
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self._remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self._remove_prefix(pretrained_dict, 'module.')
        self._check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    def detect(self, image):
        """[summary]
        
        Arguments:
            image {[type]} -- [description]
        
        Returns:
            boxes {list of tupple containing 4 floats} (x_min, y_min, x_max, y_max)
            scores {list of floats}
            valid {Bool}
        """
        img = np.float32(image)
        if self.resize_scale !=1:
            img = cv2.resize(img, None, None, fx=self.resize_scale, fy=self.resize_scale, interpolation=cv2.INTER_LINEAR)
        
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)
        
        #forward pass
        loc, conf = self.face_detector(img)

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / self.resize_scale
        boxes = boxes.cpu().numpy()
        scores = conf.data.cpu().numpy()[:, 1]

         # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        #keep = py_cpu_nms(dets, args.nms_threshold)
        keep = nms(dets, self.nms_threshold, force_cpu=self.cpu)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]

        boxes = []
        scores = []
        for k in range(dets.shape[0]):
            boxes.append((dets[k, 0], dets[k, 1], dets[k, 2],  dets[k, 3]))
            scores.append(dets[k, 4])
        
        if len(boxes) > 0:
            return boxes, scores, True
        else:
            return None, None, False


if __name__ == "__main__":
    deteta_caras = FaceBoxesDetector("/home/candeiasalexandre/code/FaceBoxes.PyTorch/weights/FaceBoxes.pth", True)
    landmark_detector = MenpoDlibLandmarkDetector("/home/candeiasalexandre/code/FaceBoxes.PyTorch/weights/dlib_pre_trained/shape_predictor_68_face_landmarks.dat")

    image_path = "/home/candeiasalexandre/code/FaceBoxes.PyTorch/data/foto_eu.jpg"
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    boxes, scores, valid = deteta_caras.detect(img)
    

    img_rect = cv2.rectangle(img, (boxes[0][0], boxes[0][1]), (boxes[0][2], boxes[0][3]), (255,0,0))
    landmarks = landmark_detector.detect(img, boxes[0])

    for landmark in landmarks:
        img_rect = cv2.circle(img_rect, (int(landmark[0]),int(landmark[1])), 2, (255,0,0))

    cv2.imshow('image', img_rect)
    cv2.waitKey(0)
    cv2.destroyAllWindows()