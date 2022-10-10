from torch.utils.data import Dataset
import csv
import torch
import os
from PIL import Image
from torchvision import transforms

"""
label_file 구조
@ image_idx, class_idx, roi_x, roi_y, roi_width, roi_height, image_file_name
>> image_idx        : 이미지 파일별 고유 인덱스 번호 (0~이미지수-1)
>> class_idx        : 랜드마크별 고유 인덱스 번호 (0~랜드마크수-1)
>> roi_x0           : 랜드마크 roi 좌단 좌표
>> roi_y0           : 랜드마크 roi 상단 좌표
>> roi_x1           : 랜드마크 roi 우단 좌표
>> roi_y1           : 랜드마크 roi 하단 좌표
>> image_file_name  : 이미지 파일명
"""

class LandmarkDetectionDataset(Dataset):
    def __init__(self, data_dir, label_file_path, num_classes):
        if not label_file_path == None:
            csvfile = open(label_file_path, newline='', encoding='utf-8')
            csvread = csv.reader(csvfile, delimiter=',')
            self.labels = list(csvread)
            csvfile.close()
        else:
            self.labels = None
        self.data_list = []
        for (dirpath, dirnames, filenames) in os.walk(data_dir):
            self.data_list.extend(filenames)
            break

        self.data_list.sort()
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.num_classes = num_classes

    def __len__(self):
        if self.labels == None:
            return len(self.data_list)
        else:
            return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
                idx = idx.tolist()

        if self.labels == None:
            image_path = os.path.join(self.data_dir, self.data_list[idx])
            image = Image.open(image_path).convert('RGB')

            if not self.transform == None:
                image = self.transform(image)
            else:
                transform = transforms.Compose([transforms.ToTensor()])
                image = transform(image)

            image_id = str(int(self.data_list[idx].split("_")[0]))

            return (image, {}, image_id)
        else:
            image_path = os.path.join(self.data_dir, self.labels[idx][6])
            image = Image.open(image_path).convert('RGB')
            w, h = image.size

            if not self.transform == None:
                image = self.transform(image)
            else:
                transform = transforms.Compose([transforms.ToTensor()])
                image = transform(image)

            image_id = self.labels[idx][0]
            category_id = int(self.labels[idx][1])

            box_x0 = float(self.labels[idx][2])
            box_y0 = float(self.labels[idx][3])
            box_x1 = float(self.labels[idx][4])
            box_y1 = float(self.labels[idx][5])

            if box_x0 > box_x1:
                box_x0, box_x1 = box_x1, box_x0

            if box_y0 > box_y1:
                box_y0, box_y1 = box_y1, box_y0

            if box_x0 < 0.0:
                box_x0 = 0.0
            if box_y0 < 0.0:
                box_y0 = 0.0
            if box_x1 >= float(w):
                box_x1 = float(w) - 1.0
            if box_y1 >= float(h):
                box_y1 = float(h) - 1.0

            box = [box_x0, box_y0, box_x1, box_y1]
            if box_x0 >= box_x1 or box_y0 >= box_y1:
                raise Exception("invalid size %f %f %f %f" % (box_x0, box_y0, box_x1, box_y1))

            target = {}
            target['boxes'] = torch.tensor([box])
            target['labels'] = torch.tensor([category_id])

            return (image, target, image_id)
