"""

@author:  AIcandy 
@website: aicandy.vn

"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from aicandy_model_src_bpvbytql.aicandy_model_densenet_ekktouop import DenseNet
import json
import os


# python aicandy_densenet_test_beyugdam.py --image_path ../image_test.jpg --model_path aicandy_model_out_ddmalncc/aicandy_model_pth_silsegko.pth

def load_labels(label_path):
    with open(label_path, 'r') as f:
        labels = {int(line.split(": ")[0]): line.split(": ")[1].strip() for line in f}
    print('labels: ',labels)
    return labels


def predict(image_path, model_path, label_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = load_labels(label_path)
    num_classes = len(labels)

    model = DenseNet(num_blocks=[6, 12, 24, 16], growth_rate=12, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    predicted_class = predicted.item()
    return labels.get(predicted_class, "Unknown")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='AIcandy.vn')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--label_path', type=str, default='label.txt', help='Path to the label file')

    args = parser.parse_args()

    result = predict(args.image_path, args.model_path, args.label_path)
    print(f'Predicted class: {result}')
