from __future__ import division
from models import *
from utils.datasets import *
import os
import time
import datetime
import argparse
import win32gui,win32ui,win32con,win32api
import cv2
import torch
import torchvision.transforms as transforms
from PIL import ImageGrab
from flask import Flask, render_template, request, make_response
import json
import math

app = Flask(__name__)

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

def calculate_angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    pi = 3.1415
    vector_prod = dx1 * dx2 + dy1 * dy2
    length_prod = math.sqrt(pow(dx1, 2) + pow(dy1, 2)) * math.sqrt(pow(dx2, 2) + pow(dy2, 2))
    cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)
    return (math.acos(cos) / pi) * 180


def parse_route(data):
    distances = []
    angles = []

    for i in range(1,len(data)):
        distance = math.sqrt(math.pow(int(data[i]['x']) - int(data[i - 1]['x']),2) + math.pow(int(data[i]['y']) - int(data[i - 1]['y']),2))
        distances.append(distance)

    for i in range(1,len(data)-1):
        AB = [int(data[i - 1]['x']),int(data[i - 1]['y']),int(data[i]['x']),int(data[i]['y'])]
        BC = [int(data[i]['x']),int(data[i]['y']),int(data[i + 1]['x']),int(data[i + 1]['y'])]
        angle = calculate_angle(AB,BC)
        angles.append(angle)


    distances = np.array(distances) // 25
    angles = np.array(angles) // 15

    distances = list(distances)
    angles = list(angles)
    return distances, angles


@app.route('/command',methods=["GET", "POST"])
def command():
    id = request.args.get("id")

    if id == "1":
        get_windows("COM3", LEFTROTATE)
    elif id == "2":
        get_windows("COM3", RIGHTROTATE)
    elif id == "3":
        get_windows("COM3", FORWARD)
    elif id == "4":
        get_windows("COM3", BACKWARD)
    print(id)
    return id


@app.route('/route',methods=["GET", "POST"])
def route():
    data = json.loads(request.get_data(as_text=True))
    distances, angles = parse_route(data)
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

    print("\nPerforming object detection:")

    route_action(model,distances, angles)
    return "0"

def route_action(model,distances, angles):
    print("distance", distances)
    print("angles", angles)
    screen_bbox = (0, 100, 416, 750)
    cnt = 0

    cv2.namedWindow('image', 0)
    pre_scene = []

    while len(distances) > 0:
        distance_count = 0


        while distance_count < distances[1]:
            flag = 0
            start = time.time()
            while time.time() - start < 0.2:
                if flag == 0:
                    get_windows("COM3", FORWARD)

                    distance_count += 1
                    get_windows("COM3", FORWARD)
                    flag = 1
                    distance_count += 1
            start_time = time.time()
            im = ImageGrab.grab(screen_bbox)
            im = im.resize((416, 416))
            similarity = 0
            input_imgs = transforms.ToTensor()(np.array(im)).unsqueeze(0).cuda()
            with torch.no_grad():
                detections, cur_scene = model(input_imgs)

            if len(pre_scene) == 0:
                pre_scene = cur_scene
            else:
                similarity = cosine_similarity(pre_scene[0].cpu().numpy(), cur_scene[0].cpu().numpy())
                pre_scene = cur_scene

            print("similarity", similarity)
            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - start_time)
            cnt += 1
            print("\t+ Inference Time: %s" % (inference_time))
            if similarity > 0.997:
                verticalCount = 0
                while verticalCount < 12:
                    flag = 0
                    start = time.time()
                    while time.time() - start < 0.2:
                        if flag == 0:
                            get_windows("COM3", LEFTROTATE)
                            flag = 1
                            verticalCount += 1
                            print("Left")
                time.sleep(1.5)
                # Forward
                get_windows("COM3", FORWARD)
                time.sleep(1)
                get_windows("COM3", FORWARD)
                time.sleep(1)
                # Turn left the second time
                verticalCount = 0
                while verticalCount < 7:
                    flag = 0
                    start = time.time()
                    while time.time() - start < 0.2:
                        if flag == 0:
                            get_windows("COM3", RIGHTROTATE)
                            flag = 1
                            verticalCount += 1
                            print("Right")

            time.sleep(1.5)
        del distances[0]
        del distances[1]


def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    # zero_list = [0] * len(x)
    # if x == zero_list or y == zero_list:
    #     return float(1) if x == y else float(0)

    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))


    return 0.5 * cos + 0.5 if norm else cos



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
                # print(time.time() - start)
                if flag == 0:
                    get_windows("COM3", RIGHTROTATE)
                    flag = 1
                    verticalCount += 1
                    print("Right")

    return horizonCount, rotate




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1122)
