from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import win32gui,win32ui,win32con,win32api
from PIL import Image
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from PIL import ImageGrab
from matplotlib.backends.backend_agg import FigureCanvasAgg
STOP = 48
FORWARD = 49
BACKWARD = 50
LEFTROTATE = 51
RIGHTROTATE = 52

def get_windows(windowsname,command):
    handle = win32gui.FindWindow(None,windowsname)
    win32gui.SetForegroundWindow(handle)
    win32gui.PostMessage(handle, 0, 0, 0)
    win32api.keybd_event(command, 0, 0, 0)
    win32api.keybd_event(13, 0, 0, 0)

def dodge():
    start = time.time()
    while time.time() - start < 0.2:
        get_windows("COM3", LEFTROTATE)
        print("Left")


    while time.time() - start < 0.2:
        get_windows("COM3", RIGHTROTATE)
        print("Right")

    while time.time() - start < 0.2:
        get_windows("COM3", FORWARD)
        print("Forward")


def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"

    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))


    return 0.5 * cos + 0.5 if norm else cos


def action(label,x1,x2):
    if label == "person":
        objectCenter = (x1 + x2)/2
        print("objectcenter",objectCenter)
        flag = 0
        start = time.time()
        while time.time() - start < 0.1:
            if objectCenter < 200 and flag == 0:

                get_windows("COM3", LEFTROTATE)
                flag = 1
                print("Left")

            elif objectCenter > 300 and flag == 0:

                get_windows("COM3", RIGHTROTATE)
                flag = 1
                print("Right")
            elif objectCenter < 300 and objectCenter > 200 and flag == 0:

                get_windows("COM3", FORWARD)
                flag = 1
                print("Forward")

def draw_map(horizonCount,horizonLength,rotate,similarity):
    horizonTriger = 0
    verticalCount = 0

    if similarity <= 0.995:
        while horizonTriger < 3:
            flag = 0
            start = time.time()
            while time.time() - start < 0.1 and flag == 0:
                get_windows("COM3", FORWARD)
                flag = 1
                horizonCount += 1
                horizonTriger += 1
                print("Forward",horizonCount)
    # Turn left the first time
    elif similarity > 0.995 and rotate == 0:
        while verticalCount < 9:
            flag = 0
            start = time.time()
            while time.time() - start < 0.2:
                # print(time.time() - start)
                if flag == 0:
                    get_windows("COM3", LEFTROTATE)
                    flag = 1
                    verticalCount += 1
                    print("Left")
        # Reset HorizonCount
        horizonLength.append(horizonCount)
        horizonCount = 0
        # Forward
        get_windows("COM3", FORWARD)
        # Turn left the second time
        verticalCount = 0
        rotate = 1
        while verticalCount < 9:
            flag = 0
            start = time.time()
            while time.time() - start < 0.2:
                # print(time.time() - start)
                if flag == 0:
                    get_windows("COM3", LEFTROTATE)
                    flag = 1
                    verticalCount += 1
                    print("Left")

    elif similarity > 0.995 and rotate == 1:
        while verticalCount < 9:
            flag = 0
            start = time.time()
            while time.time() - start < 0.2:
                # print(time.time() - start)
                if flag == 0:
                    get_windows("COM3", RIGHTROTATE)
                    flag = 1
                    verticalCount += 1
                    print("Right")
        # Reset HorizonCount
        horizonLength.append(horizonCount)
        horizonCount = 0
        # Forward
        get_windows("COM3", FORWARD)
        # Turn left the second time
        verticalCount = 0
        rotate = 0
        while verticalCount < 9:
            flag = 0
            start = time.time()
            while time.time() - start < 0.2:
                if flag == 0:
                    get_windows("COM3", RIGHTROTATE)
                    flag = 1
                    verticalCount += 1
                    print("Right")

    return horizonCount, rotate




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    # img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()



    # 参数说明
    # 第一个参数 开始截图的x坐标
    # 第二个参数 开始截图的y坐标
    # 第三个参数 结束截图的x坐标
    # 第四个参数 结束截图的y坐标
    screen_bbox = (0, 100, 416, 750)
    cnt = 0
    horizonCount = 0
    rotate = 0
    cv2.namedWindow('image', 0)
    pre_scene = []
    horizonLength = []
    time.sleep(10)
    while True:
        try:
            start_time = time.time()
            img_detections = []
            im = ImageGrab.grab(screen_bbox)
            im = im.resize((416,416))
            similarity = 1
            input_imgs = transforms.ToTensor()(np.array(im)).unsqueeze(0).cuda()
            with torch.no_grad():
                detections, cur_scene = model(input_imgs)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

            if len(pre_scene) == 0:
                pre_scene = cur_scene
            else:
                # print("pre_scene[0]",pre_scene[0])
                similarity = cosine_similarity(pre_scene[0].cpu().numpy(),cur_scene[0].cpu().numpy())
                pre_scene = cur_scene

            print("similarity", similarity)
            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - start_time)
            # prev_time = current_time
            print("\t+ Inference Time: %s" % (inference_time))

            # break
            img_detections.extend(detections)

            # Bounding-box colors
            cmap = plt.get_cmap("tab20b")
            colors = [cmap(i) for i in np.linspace(0, 1, 20)]

            print("\nSaving images:")
            # Iterate through images and save plot of detections
            for detections in img_detections:

                # Create plot
                img = np.array(im)
                plt.figure()
                fig, ax = plt.subplots(1)
                ax.imshow(img)

                # Draw bounding boxes and labels of detections
                if detections is not None:
                    # Rescale boxes to original image
                    detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
                    unique_labels = detections[:, -1].cpu().unique()
                    n_cls_preds = len(unique_labels)
                    bbox_colors = random.sample(colors, n_cls_preds)
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                        box_w = x2 - x1
                        box_h = y2 - y1

                        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                        # Create a Rectangle patch
                        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                        # Add the bbox to the plot
                        ax.add_patch(bbox)

                        action(classes[int(cls_pred)], x1, x2)

                        plt.text(
                            x1,
                            y1,
                            s=classes[int(cls_pred)],
                            color="white",
                            verticalalignment="top",
                            bbox={"color": color, "pad": 0},
                        )

                # Save generated image with detections
                plt.axis("off")
                plt.gca().xaxis.set_major_locator(NullLocator())
                plt.gca().yaxis.set_major_locator(NullLocator())


                canvas = FigureCanvasAgg(plt.gcf())
                canvas.draw()
                img = np.array(canvas.renderer.buffer_rgba())
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                cv2.imshow('image', img)

                k = cv2.waitKey(1)

                if k == ord('s'):
                    plt.close('all')
                    break
                    cv2.destroyAllWindows()

                cv2.imwrite(f"./output/followpeople/{cnt}.png", img)
                plt.close('all')
                cnt += 1

        except:
            print("img error.")








