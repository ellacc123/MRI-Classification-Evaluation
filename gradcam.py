import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('baseline_resnet18.pth'))
model.eval()

# Hook to capture gradients
gradients = []
activations = []

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

def forward_hook(module, input, output):
    activations.append(output)

target_layer = model.layer4[1].conv2
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# Load and preprocess one image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

img_path = 'sample.jpg'   # replace with a real image path
img = Image.open(img_path).convert('RGB')
input_tensor = transform(img).unsqueeze(0)

# Forward + backward pass
output = model(input_tensor)
pred_class = output.argmax(1).item()
model.zero_grad()
output[0, pred_class].backward()

# Compute Grad-CAM heatmap
grads = gradients[0].squeeze().mean(dim=[1, 2])
acts  = activations[0].squeeze()
cam   = torch.zeros(acts.shape[1:])
for i, w in enumerate(grads):
    cam += w * acts[i]
cam = torch.relu(cam)
cam = cam.detach().numpy()
cam = cv2.resize(cam, (224, 224))
cam = (cam - cam.min()) / (cam.max() - cam.min())

# Overlay on image
img_resized = np.array(img.resize((224, 224)))
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
overlay = (0.5 * img_resized + 0.5 * heatmap).astype(np.uint8)

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1); plt.imshow(img_resized); plt.title('Original'); plt.axis('off')
plt.subplot(1, 3, 2); plt.imshow(cam, cmap='jet'); plt.title('Grad-CAM'); plt.axis('off')
plt.subplot(1, 3, 3); plt.imshow(overlay); plt.title('Overlay'); plt.axis('off')
plt.suptitle(f'Predicted: {"Tumor" if pred_class == 1 else "Healthy"}')
plt.savefig('gradcam_output.png')
plt.show()