"""Module to classify a specified image"""
import torch
from PIL import Image
from models.model import resnet18, ResNet
from common.utils import load_yaml_config
from common import datakeywords as dk
from DL.classification_dl import preprocess_image

cfg_train = load_yaml_config("src/config/config_train.yaml")
cfg_predict = load_yaml_config("src/config/config_predict.yaml")


def model_load()->ResNet:
    """Load the trained model based on train and predict configs"""
    net = resnet18(num_classes=cfg_train[dk.NUM_CLASSES_KEY])
    net.load_state_dict(
        torch.load(cfg_predict[dk.MODEL_PATH_KEY], map_location=torch.device("cpu"))
    )
    return net


def load_preprocess_image()->torch.Tensor:
    """Load and preprocess the image based on predict config"""
    im = Image.open(cfg_predict[dk.IMAGE_PATH_KEY]).convert("RGB")
    return preprocess_image(im,to_predict=True)


def perform_inference(im_tensor: torch.Tensor, model: ResNet)->int:
    """Perform inference on an image tensor using the specified model"""
    # Perform inference
    with torch.no_grad():
        output = model(im_tensor)

    # Get the predicted class index
    predicted_class_idx = int(torch.argmax(output).item())
    return predicted_class_idx


def main():
    """The main function of predict.py"""

    # Load and preprocess the image
    image_tensor = load_preprocess_image()

    # Load the trained model
    model = model_load()
    model.eval()

    # Perform inference on the image
    predicted_class_idx = perform_inference(image_tensor, model)

    # Show real image and predicted class index
    print(f"Predicted class index: {predicted_class_idx}")
    image = Image.open(cfg_predict[dk.IMAGE_PATH_KEY]).convert("RGB")
    image.show()


if __name__ == "__main__":
    main()
