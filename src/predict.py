import torch
from PIL import Image
from models.model import resnet18 
from config import config as config_predict
from common.utils import get_normalization_transform

# Load the trained model
model_path = "output/mnist_resnet.pt"

model = resnet18(num_classes=config_predict.num_classes) 
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()


data_transform=get_normalization_transform()

# Load and preprocess the image
image_path = "data/t10k-images-idx3-ubyte_dir/image_1003.png"
image = Image.open(image_path).convert("RGB")
image_tensor = data_transform(image).unsqueeze(0)

# Perform inference
with torch.no_grad():
    output = model(image_tensor)

# Get the predicted class index
predicted_class_idx = torch.argmax(output).item()
print(f"Predicted class index: {predicted_class_idx}")

# Show image
image.show()
