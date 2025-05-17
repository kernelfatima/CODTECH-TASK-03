import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image loading and preprocessing
def load_image(image_path, max_size=400):
    image = Image.open(image_path)
    size = max(image.size)
    if size > max_size:
        size = max_size
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

# Loss calculation functions
def gram_matrix(tensor):
    b, d, h, w = tensor.size()
    tensor = tensor.view(b * d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram.div(b * d * h * w)

class StyleTransferModel(nn.Module):
    def __init__(self, content_img, style_img):
        super(StyleTransferModel, self).__init__()
        self.content_img = content_img
        self.style_img = style_img
        self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
        self.content_layers = ["conv_4"]
        self.style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

    def forward(self, x):
        content_loss = torch.tensor(0.0, device=device, requires_grad=True)
        style_loss = torch.tensor(0.0, device=device, requires_grad=True)
        style_features = []

        for name, layer in self.model._modules.items():
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                if name in self.style_layers:
                    style_features.append(gram_matrix(x))
                if name in self.content_layers:
                    content_loss += torch.mean((x - self.content_img) ** 2)

        for sf, style_target in zip(style_features, self.style_img):
            style_loss += torch.mean((sf - style_target) ** 2)

        total_loss = content_loss + 1000 * style_loss
        return total_loss

# Image optimization and training
def transfer_style(content_path, style_path):
    content_img = load_image(content_path)
    style_img = load_image(style_path)
    model = StyleTransferModel(content_img, style_img)
    target = content_img.clone().requires_grad_(True)
    optimizer = optim.Adam([target], lr=0.003)

    for step in range(300):
        optimizer.zero_grad()
        loss = model(target)
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"Step {step}, Loss: {loss.item()}")

    return target

# Displaying the result
def show_image(tensor):
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Usage example
result = transfer_style("content1.jpg", "style 1.jpg")
show_image(result)

