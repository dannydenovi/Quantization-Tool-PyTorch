# **Dynamic Model Quantization Tool**

This repository provides an interactive tool for dynamic quantization of machine learning models. It enables users to evaluate and quantize models dynamically, offering insights into accuracy, model size, and performance trade-offs.

## **Features**
- **Interactive Interface**:
  - User-friendly terminal interface powered by `questionary`.
  - Allows the selection of datasets, models, and custom file paths.
- **Dynamic Quantization**:
  - Applies PyTorch's dynamic quantization to reduce model size while maintaining accuracy.
- **Model Evaluation**:
  - Calculates and compares the accuracy of the original and quantized models.
- **Model Size Insights**:
  - Displays original and quantized model sizes, along with the percentage reduction.
- **Hourglass Animations**:
  - Real-time feedback during evaluation and quantization with elegant loading animations.
- **Custom Dataset Support**:
  - Process and evaluate datasets from a directory structure with subfolders for each class.

## **Requirements**
### **Python Version**
- Python 3.8 or later

### **Dependencies**
Install the required Python packages using `pip`:
```bash
pip install torch torchvision questionary rich
```

### **Optional Custom Model**
If using a custom model (e.g., `LeNet_CIFAR10`), ensure it's included in your project and importable.

## **Usage**
### **Step 1: Clone the Repository**
```bash
git clone <repository-url>
cd <repository-folder>
```

### **Step 2: Run the Program**
```bash
python main.py
```

### **Step 3: Follow Interactive Prompts**
1. **Select Dataset**:
   - Choose between `CIFAR10`, `MNIST`, or a custom dataset.
2. **Select Model**:
   - Choose from predefined models or provide a custom model path.
3. **Quantize and Evaluate**:
   - The program evaluates the original model, applies quantization, and re-evaluates the quantized model.

### **Step 4: View Results**
- **Accuracy**:
  - See the original and quantized model accuracies.
- **Model Size**:
  - Observe size reduction in KB and percentage.

### **Example Output**
```plaintext
Using device: cpu
Dataset loaded successfully!

Model 'LeNet_CIFAR10' initialized successfully!

Calculating original model accuracy ⏳
Calculating original model accuracy [bold green]Done![/bold green]
Original Model Accuracy: [bold green]92.50%[/bold green]
Original Model Size: [bold green]356.78 KB[/bold green]

Applying dynamic quantization ⏳
Applying dynamic quantization [bold green]Done![/bold green]

Calculating quantized model accuracy ⏳
Calculating quantized model accuracy [bold green]Done![/bold green]
Quantized Model Accuracy: [bold green]91.00%[/bold green]
Quantized Model Size: [bold green]89.56 KB[/bold green]
Model Size Reduction: [bold green]74.90%[/bold green]
Quantized model saved as: 'quantized_model.pth'
```

## **Directory Structure**
```
.
├── data/                     # Directory for downloading standard datasets
├── lenet.py                  # Example custom model (LeNet_CIFAR10)
├── main.py                   # Main program script
├── README.md                 # Documentation
```

## **Customization**
### **Adding New Models**
1. Add your model to the `lenet.py` file or any Python file in the directory.
2. Include the model in the `custom_models` dictionary in `main.py`:
   ```python
   custom_models = {
       "LeNet_CIFAR10": LeNet_CIFAR10,
       "YourModelName": YourModelClass,
   }
   ```

### **Using Custom Datasets**
1. Prepare your dataset with a folder structure like:
   ```
   custom_dataset/
   ├── class1/
   │   ├── img1.jpg
   │   ├── img2.jpg
   ├── class2/
   │   ├── img1.jpg
   │   ├── img2.jpg
   ```
2. Select `Custom Dataset` during the dataset selection prompt.

## **Known Limitations**
- Dynamic quantization is supported only for CPU devices.
- Works primarily with PyTorch models.

## **Contributing**
Contributions are welcome! If you encounter any issues or have suggestions, feel free to open an issue or submit a pull request.

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
