
"""
Car Plate Detection using Faster R-CNN (ResNet50-FPN)

This script:
1. Loads COCO-format dataset
2. Applies data augmentation
3. Fine-tunes Faster R-CNN
4. Evaluates using AP@0.5
"""

import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou


DATASET_ROOT = "Car Plate Detection"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# DATASET
class CarPlateDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms

        with open(os.path.join(root, "_annotations.coco.json")) as f:
            coco = json.load(f)

        self.imgs = {img["id"]: img for img in coco["images"]}

        self.anns = {}
        for ann in coco["annotations"]:
            self.anns.setdefault(ann["image_id"], []).append(ann)

        cat_ids = sorted({ann["category_id"] for ann in coco["annotations"]})
        self.cat_map = {cid: i + 1 for i, cid in enumerate(cat_ids)}

        self.ids = list(self.imgs.keys())

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        meta = self.imgs[img_id]

        img_path = os.path.join(self.root, "images", meta["file_name"])
        img = Image.open(img_path).convert("RGB")

        anns = self.anns.get(img_id, [])

        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_map[ann["category_id"]])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)


# TRANSFORMS
class ToTensor:
    def __call__(self, image, target):
        return F.to_tensor(image), target

class RandomFlip:
    def __call__(self, image, target):
        if torch.rand(1) > 0.5:
            image = F.hflip(image)
        return image, target

def get_transform(train):
    t = [ToTensor()]
    if train:
        t.append(RandomFlip())
    return lambda img, tgt: (
        t[0](img, tgt) if len(t) == 1 else t[1](*t[0](img, tgt))
    )


# MODEL
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# TRAINING
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0

    for images, targets in loader:
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# EVALUATION (AP@0.5)
@torch.no_grad()
def compute_ap50(model, loader):
    model.eval()
    preds = []
    gts = []
    image_id = 0

    for images, targets in loader:
        images = [img.to(DEVICE) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            gt_boxes = tgt["boxes"]
            gts.append(gt_boxes)

            pred_boxes = out["boxes"].cpu()
            pred_scores = out["scores"].cpu()

            for i in range(len(pred_boxes)):
                preds.append((image_id, pred_scores[i], pred_boxes[i]))

            image_id += 1

    if len(gts) == 0:
        return 0.0

    preds.sort(key=lambda x: x[1], reverse=True)

    tp, fp = [], []
    matched = [False] * len(gts)

    for img_id, score, box in preds:
        gt_boxes = gts[img_id]

        if len(gt_boxes) == 0:
            fp.append(1)
            tp.append(0)
            continue

        ious = box_iou(box.unsqueeze(0), gt_boxes).squeeze(0)
        max_iou = ious.max().item()

        if max_iou > 0.5:
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    recall = tp / (len(gts) + 1e-6)
    precision = tp / (tp + fp + 1e-6)

    return np.mean(precision)


# MAIN
def main():
    dataset = CarPlateDataset(DATASET_ROOT, get_transform(train=True))
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    num_classes = len(dataset.cat_map) + 1
    model = get_model(num_classes).to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    print("Training started...")
    for epoch in range(5):
        loss = train_one_epoch(model, loader, optimizer)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    ap = compute_ap50(model, loader)
    print(f"Final AP@0.5: {ap:.4f}")

if __name__ == "__main__":
    main()