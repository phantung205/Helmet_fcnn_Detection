import os
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset,DataLoader
import cv2
import torch


class HelmetDataset(Dataset):
    def __init__(self,root,transform=None):
        self.img_dir = os.path.join(root,"images")
        self.label_dir = os.path.join(root,"labels")

        self.transform = transform

        # save a list of valid files
        self.files = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        file = self.files[item]

        # read image
        img_path = os.path.join(self.img_dir,file)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        # read labels
        label_name = file.replace(".png", ".xml")
        xml_path = os.path.join(self.label_dir,label_name)
        boxes,labels = self.parse_xml(xml_path)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        if self.transform:
            image = self.transform(image)

        return image, target

    def parse_xml(self,xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes, labels = [], []

        for obj in root.findall("object"):
            name = obj.find("name").text

            label = 0 if name == "Without Helmet" else 1

            b = obj.find("bndbox")
            xmin = int(b.find("xmin").text)
            ymin = int(b.find("ymin").text)
            xmax = int(b.find("xmax").text)
            ymax = int(b.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
        return boxes,labels






