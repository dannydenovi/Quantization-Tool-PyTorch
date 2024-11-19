import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import os
from PIL import Image
import questionary
import threading
import time
import itertools
from rich.console import Console
import io
from models.lenet import LeNet_CIFAR10
from models.spatio_temporal import Transformer

console = Console()

# Dictionary of custom models
custom_models = {
    "LeNet_CIFAR10" : LeNet_CIFAR10,
    "Spatio-Temporal Transformer" : Transformer
}

def hourglass_animation(message, event):
    spinner = itertools.cycle(["⏳", "⌛"]) 
    while not event.is_set(): 
        console.print(f"{message} {next(spinner)}", end="\r", style="bold cyan")
        time.sleep(0.3)
    console.print(f"{message} [bold green]Done![/bold green]")

def run_with_hourglass(message, func, *args, **kwargs):
    event = threading.Event()  
    thread = threading.Thread(target=hourglass_animation, args=(message, event))
    thread.start()
    
    result = func(*args, **kwargs)
    
    event.set()
    thread.join()
    
    return result

def load_dataset():
    dataset_choice = questionary.select(
        "Select the dataset to use:",
        choices=[
            "CIFAR10",
            "MNIST",
            "Custom Dataset"
        ]
    ).ask()
    
    if dataset_choice == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset_choice == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_choice == "Custom Dataset":
        dataset_dir = questionary.path("Enter the path to the custom dataset directory:").ask()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = CustomDataset(dataset_dir, transform)
    else:
        raise ValueError("Invalid dataset choice.")
    
    return DataLoader(dataset, batch_size=64, shuffle=False)

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        
        for label_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.data.append(img_path)
                    self.labels.append(label_idx)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def initialize_model(device):
    all_models = {**custom_models, **{name: getattr(models, name) for name in dir(models) if not name.startswith("_")}}
    
    model_name = questionary.select(
        "Select the model to use:",
        choices=list(all_models.keys())
    ).ask()

    num_classes = int(questionary.text("Enter the number of classes in the dataset:").ask())

    model_files = [f for f in os.listdir() if f.endswith('.pth')]
    model_files.append("Enter custom path")
    
    model_file_choice = questionary.select(
        "Select a model file or enter a custom path:",
        choices=model_files
    ).ask()
    
    if model_file_choice == "Enter custom path":
        model_path = questionary.path("Enter the full path to the model file:").ask()
    else:
        model_path = model_file_choice

    model_class = all_models[model_name]
    model = model_class(num_classes=num_classes) if model_name in custom_models else model_class(pretrained=False)
    
    if hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, 'classifier'):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    console.print(f"\n[bold green]Model '{model_name}' initialized successfully![/bold green]\n")
    return model

def get_model_size_in_memory(model):
    buffer = io.BytesIO() 
    torch.save(model.state_dict(), buffer) 
    size = buffer.tell() / 1024 
    return size

def dynamic_quantization(model):
    return torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main():
    device = torch.device("cpu")  # Dynamic quantization supported only on CPU
    console.print(f"[bold yellow]Using device: {device}[/bold yellow]")
    torch.backends.quantized.engine = 'fbgemm'
    
    test_loader = load_dataset()
    console.print("[bold cyan]Dataset loaded successfully![/bold cyan]\n")
    
    model = initialize_model(device)
    
    original_accuracy = run_with_hourglass(
        "Calculating original model accuracy",
        evaluate_model,
        model,
        test_loader,
        device
    )
    original_size = get_model_size_in_memory(model)
    console.print(f"Original Model Accuracy: [bold green]{original_accuracy:.2f}%[/bold green]")
    console.print(f"Original Model Size: [bold green]{original_size:.2f} KB[/bold green]")
    
    quantized_model = run_with_hourglass(
        "Applying dynamic quantization",
        dynamic_quantization,
        model
    )
    quantized_accuracy = run_with_hourglass(
        "Calculating quantized model accuracy",
        evaluate_model,
        quantized_model,
        test_loader,
        device
    )
    quantized_size = get_model_size_in_memory(quantized_model)
    size_reduction = ((original_size - quantized_size) / original_size) * 100

    # Save the quantized model
    quantized_model_path = "quantized_model.pth"
    torch.save(quantized_model.state_dict(), quantized_model_path)
    console.print(f"[bold cyan]Quantized model saved as: '{quantized_model_path}'[/bold cyan]")
    
    console.print(f"Quantized Model Accuracy: [bold green]{quantized_accuracy:.2f}%[/bold green]")
    console.print(f"Quantized Model Size: [bold green]{quantized_size:.2f} KB[/bold green]")
    console.print(f"Model Size Reduction: [bold green]{size_reduction:.2f}%[/bold green]")


if __name__ == "__main__":
    main()

