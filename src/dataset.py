import os
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torch


class HelmetDataset(Dataset):
    def __init__(self,root,is_train=True,transform=None):
        if is_train:
            mode = "train"
        else:
            mode = "val"

        root_mode = os.path.join(root,mode)

        self.img_dir = os.path.join(root_mode,"images")
        self.label_dir = os.path.join(root_mode,"labels")

        self.transform = transform

        # save a list of valid files
        self.files = []
        for f in os.listdir(self.img_dir):
            if not f.endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(self.img_dir, f)

            try:
                img = Image.open(img_path)
                img.verify()
            except:
                continue

            xml_path = os.path.join(
                self.label_dir,
                os.path.splitext(f)[0] + ".xml"
            )

            if not os.path.exists(xml_path):
                continue

            boxes, _ = self.parse_xml(xml_path)
            if len(boxes) == 0:
                continue

            self.files.append(f)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        file = self.files[item]

        # read image
        img_path = os.path.join(self.img_dir,file)
        image = Image.open(img_path).convert("RGB")

        # read labels
        label_name = os.path.splitext(file)[0] + ".xml"
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

            label = 1 if name == "Without Helmet" else 2

            b = obj.find("bndbox")
            xmin = int(b.find("xmin").text)
            ymin = int(b.find("ymin").text)
            xmax = int(b.find("xmax").text)
            ymax = int(b.find("ymax").text)

            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)
        return boxes,labels






