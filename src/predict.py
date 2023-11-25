"""Module to classify a specified image"""
import torch
from PIL import Image
from models.model import resnet18
from config import config_predict
from config import config_train
from common.utils import get_normalization_transform

if __name__ == "__main__":
    # load the trained model
    model = resnet18(num_classes=config_train.NUM_CLASSES)
    model.load_state_dict(
        torch.load(config_predict.MODEL_PATH, map_location=torch.device("cpu"))
    )
    model.eval()

    data_transform = get_normalization_transform()

    # Load and preprocess the image
    image = Image.open(config_predict.IMAGE_PATH).convert("RGB")
    image_tensor = data_transform(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)

    # Get the predicted class index
    predicted_class_idx = torch.argmax(output).item()
    print(f"Predicted class index: {predicted_class_idx}")

    # Show image
    image.show()
