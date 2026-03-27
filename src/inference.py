import argparse
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from src import config
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
import os

def get_args():
    parser = argparse.ArgumentParser(description="inference mode faster cnn")
    parser.add_argument("--image_path","-i",type=str,default=config.image_test,help="path to test image")
    parser.add_argument("--video_path","-v",type=str,default=None,help="path to test video")
    parser.add_argument("--checkpoint","-c",type=str,default=config.checkpoint_best)
    parser.add_argument("--conf_threshold",type=float,default=0.5)
    args = parser.parse_args()
    return args


def predict_frame(frame,model,device,categories,conf_threshold):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = image / 255.0

    # normalize giống train
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = (image - mean) / std

    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).float().to(device)
    image = [image]

    with torch.no_grad():
        output = model(image)[0]
        boxes = output["boxes"]
        labels = output["labels"]
        scores = output["scores"]
        for box, label, score in zip(boxes, labels, scores):
            if score >= conf_threshold:
                xmin, ymin, xmax, ymax = box
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 3)
                category = categories[label.item()-1]
                cv2.putText(frame, category, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                            cv2.LINE_AA)
        return  frame

def predict_image(image_path, model, device, categories, conf_threshold):
    frame = cv2.imread(image_path)
    result = predict_frame(frame, model, device, categories, conf_threshold)
    os.makedirs(config.dir_results, exist_ok=True)
    path_result = os.path.join(config.dir_results,"prediction.jpg")
    cv2.imwrite(path_result, result)
    cv2.imshow("Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def predict_video(video_path,model,device,categories, conf_threshold):
    # đọc video
    cap = cv2.VideoCapture(video_path)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    os.makedirs(config.dir_results, exist_ok=True)
    path_result = os.path.join(config.dir_results, "result.mp4")
    out = cv2.VideoWriter(path_result, cv2.VideoWriter_fourcc(*"mp4v"), int(cap.get(cv2.CAP_PROP_FPS)),
                          (width, height))
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        frame = predict_frame(frame, model, device, categories, conf_threshold)
        out.write(frame)

    # giải phóng bộ nhớ
    cap.release()
    out.release()


def deploy(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    categories =  config.categories

    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_channels, num_classes=len(config.categories)+1)
    checkpoint = torch.load(args.checkpoint,map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    model.eval()
    if not args.image_path is None:
        predict_image(args.image_path,model,device,categories,args.conf_threshold)

    if not args.video_path is None:
        predict_video(args.video_path,model,device,categories,args.conf_threshold)



if __name__ == '__main__':
    args = get_args()
    deploy(args)