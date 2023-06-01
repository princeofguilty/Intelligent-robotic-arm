import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

import socket
import time

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from threading import Thread
import threading


print("[info] starting server for matlab")
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server.bind(("192.168.1.3", 8000))
server.bind(("172.20.10.13", 8000))
print("[info] server for matlab started")
# BGR frame
frame = None
# load_frame = False
# load_frame = threading.Event()


def update(cap):
    global frame
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while(True):
        # load_frame.wait()
        while not cap.isOpened():
            print("[info] capture device closed! Retrying ...")
            time.sleep(1)
            cap = cv2.VideoCapture(opt.source)
        
        # while cap.grab():
        #     pass

        # tmp = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, tmp)

        _, frame = cap.read()
        cv2.imshow("frames", frame)
        cv2.waitKey(1)
        # load_frame.clear()

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)
        


def detect(save_img=False):
    global frame
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    trace = False
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    available_classes = [None]*4

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    cap = cv2.VideoCapture(source)
    while not cap.isOpened():
        print("[info] retrying camera connection")
        time.sleep(1)
        cap = cv2.VideoCapture(source)
    
    print("[info] Camera connection established successfuly")
    # assert cap.isOpened(), f'Failed to open {source}'
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = cap.get(cv2.CAP_PROP_FPS) % 100
    # _, img = cap.read()
    
    thread = Thread(target=update, args=([cap]), daemon=True)
    thread.start()
    
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]


    print("[info] waiting for matlab to connect")
    server.listen(1)
    conn, addr = server.accept()
    # conn.send(bytes(str("hello world!").encode('utf-8')))

    while True:
        # load_frame.set()
        while frame is None:
            pass


        frame_backup = frame.copy()
        img = letterbox(frame_backup, 640, stride=stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)


        # img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=False)
    
        available_classes_conf = [0]*4

        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                classes_used = set()

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh)  # label format
                    print("cls :", end="")
                    # print(line)
                    # print(xywh)
                    print(cls)
                    # conn.send(bytes(str(('%g ' * len(line)).rstrip() % line + '\n').encode('utf-8')))
                    # print(('%g ' * len(line)).rstrip() % line + '\n')

                    # conversion for backward compatability for matlab work 
                    cls_no = int(cls)
                    if cls_no == 0:
                        continue
                    else:
                        cls_no -= 1
                    
                    if(conf > available_classes_conf[cls_no]):
                        available_classes[cls_no] = str(('%g ' * len(line)).rstrip() % (cls_no, *xywh) + '\n')
                        available_classes_conf[cls_no] = conf

                        classes_used.add(cls_no)
                    # if view_img:  # Add bbox to image
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, frame_backup, label=label, color=colors[int(cls)], line_thickness=3)
                    cv2.imshow("out", frame_backup)
                    cv2.waitKey(1)

                if(len(classes_used) >=3 and classes_used.__contains__(2)):
                    for i in available_classes[:3]:
                        try:
                            conn.send(bytes(str(i).encode('utf-8')))
                            print("sent: "+i, end="")
                        except:
                            print("[info] waiting for matlab to re-connect")
                            server.listen(1)
                            conn, addr = server.accept()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default='\\\\192.168.1.6\\ahmed\\yolov7\\runs\\train\\exp8\\weights\\best.pt', help='model.pt path(s)')
    parser.add_argument('--weights', nargs='+', type=str, default='best_o2.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='http://172.20.10.14:4747/video', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--source', type=str, default='http://192.168.1.8:4747/video', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    # parser.add_argument('--no-freeze', action='store_true', help='don`t freeze')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')


    opt = parser.parse_args()
    print(opt)

    try:
        detect()
    except Exception as e:
        print(e)
        try:
            print("\n\n[info] try server shutdown on keyboard interrupt")
            server.shutdown(socket.SHUT_RDWR)
            server.close()
        except:
            pass

