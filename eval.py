import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from models.alexnet import AlexNet
    
def predict(images, model_path="best_model.pth"):
    # ----- Load model -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = AlexNet(num_classes=2)

    # ----- Load pretrained AlexNet and modify -----
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    # ----- Apply transform and stack into batch tensor -----
    batch_tensor = torch.stack([transform(img) for img in images]).to(device)

    # ----- Predict -----
    with torch.no_grad():
        output = model(batch_tensor)
        _, predicted = torch.max(output, 1)
        return predicted

def main():
    if len(sys.argv) < 2:
        print("Usage: python eval.py <image_path>")
        return
    
    # ----- Load and preprocess image -----
    image_path = sys.argv[1]
    image = Image.open(image_path).convert("RGB")
    predicted = predict([image])
    for pred in predicted:
        label = "Glasses" if pred == 0 else "No Glasses"
        print(f"Predicted: {label}")

if __name__ == "__main__":
    main()