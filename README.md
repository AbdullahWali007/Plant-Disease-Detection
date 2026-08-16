```markdown
# Plant Disease Detection using PyTorch

This repository contains a Jupyter Notebook for training an image classification model to detect plant diseases using the PlantVillage dataset. The implementation utilizes transfer learning with a pre-trained ResNet-18 architecture via PyTorch.

## Requirements

Ensure you have Python installed, then install the required dependencies:

```bash
pip install torch torchvision matplotlib numpy pillow

```

## Dataset Preparation

The code expects the PlantVillage dataset to be organized in a standard PyTorch `ImageFolder` format. Extract your dataset and ensure the directory structure looks like this:

```text
dataset/
├── train/
│   ├── Apple___Apple_scab/
│   ├── Apple___Black_rot/
│   └── ... (other classes)
└── val/
    ├── Apple___Apple_scab/
    └── ...

```

## Training the Model

1. Open the Jupyter Notebook.
2. Update the `DATA_DIR` variable in Cell 2 to point to your dataset directory.
3. Run all cells sequentially.
4. Upon completion, the notebook will save the trained model weights to the working directory as `plant_disease_model.pt`.

---

## How to Use the Saved Model (Inference)

To use your trained `plant_disease_model.pt` file to predict the disease of a new, unseen image, you must recreate the model architecture, load the saved weights, and apply the exact same image transformations used during validation.

You can use the following Python script to run predictions:

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Configuration
MODEL_PATH = 'plant_disease_model.pt'
IMAGE_PATH = 'path/to/your/new_leaf_image.jpg'

# IMPORTANT: You must define the exact list of class names generated during training.
# These match the folder names in your dataset/train directory in alphabetical order.
CLASS_NAMES = [
    'Apple___Apple_scab', 
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    # ... add all other classes here
]
NUM_CLASSES = len(CLASS_NAMES)

# 2. Setup Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 3. Initialize Model Architecture
# We do not need the default weights here since we are loading our own
model = models.resnet18(weights=None) 
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

# 4. Load Saved Weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval() # Set model to evaluation mode

# 5. Define Image Transformations
# These must match the validation transforms used in the training notebook
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# 6. Prediction Function
def predict_image(image_path):
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image)
    
    # Add batch dimension (C, H, W) -> (1, C, H, W)
    input_batch = input_tensor.unsqueeze(0).to(device) 

    # Run inference
    with torch.no_grad():
        output = model(input_batch)
        _, predicted_idx = torch.max(output, 1)
        
    predicted_class = CLASS_NAMES[predicted_idx.item()]
    return predicted_class

# 7. Run the prediction
if __name__ == "__main__":
    result = predict_image(IMAGE_PATH)
    print(f"The predicted class is: {result}")
