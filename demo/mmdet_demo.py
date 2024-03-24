import torch
import mmcv
import cv2
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector

# setting up the example from: https://mmdetection.readthedocs.io/en/latest/get_started.html#verify-the-installation
config_file = 'config/rtmdet_tiny_8xb32-300e_coco.py'
checkpoint_file = 'config/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
img = 'demo/demo.jpg'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
result = inference_detector(model, img)

# Visualize the result
img = cv2.imread(img)
img_show = img.copy()

def get_crop(img, bbox):
    return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

# Draw bounding boxes on the image
for i, bbox in enumerate(result.pred_instances.bboxes[:5]):
    bbox_int = bbox.int().numpy()
    cv2.rectangle(img_show, (bbox_int[0], bbox_int[1]), (bbox_int[2], bbox_int[3]), (0, 255, 0), 2)
    crop = get_crop(img, bbox_int)
    cv2.imwrite(f"demo/mmdet_out/demo_out_{i}.jpg", crop)
cv2.imwrite("demo/mmdet_out/demo_out.jpg", img_show)
