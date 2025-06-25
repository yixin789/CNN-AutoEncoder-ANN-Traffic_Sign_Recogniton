import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import numpy as np
import os
import json # Import json library
from torchvision import transforms # Import transforms

# Import necessary components from your project
from models.traffic_sign_ann import TrafficSignCNN_AE_ANN
from datasets.traffic_sign_dataset import TrafficSignDataset, NUM_CLASSES # Changed: Import TrafficSignDataset and NUM_CLASSES
from utils.model_utils import load_model

class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Classifier")
        self.root.geometry("800x700")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} for GUI app")

        self.model = None
        self.class_names = self.load_class_names() # Load class names
        self.load_model()

        # Define standard normalization parameters (must match training/evaluation)
        self.NORM_MEAN = [0.485, 0.456, 0.406]
        self.NORM_STD = [0.229, 0.224, 0.225]

        # Define the transformation for inference (similar to val_test_transform in main.py)
        self.infer_transform = transforms.Compose([
            transforms.Resize((224, 224)), # Resize to model input size
            transforms.ToTensor(),         # Convert PIL Image to Tensor and scale to [0, 1]
            transforms.Normalize(mean=self.NORM_MEAN, std=self.NORM_STD) # Normalize
        ])

        # --- GUI Elements ---
        self.label_title = tk.Label(root, text="Select Image for Classification", font=("Helvetica", 16))
        self.label_title.pack(pady=20)

        self.image_frame = tk.Frame(root, bd=2, relief="groove", width=400, height=400)
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False) # Prevent frame from resizing to content

        self.image_label = tk.Label(self.image_frame, text="Drop Image Here or Click to Select", width=400, height=400)
        self.image_label.pack(expand=True)
        self.image_label.bind("<Button-1>", self.open_image_dialog)
        self.image_label.bind("<ButtonRelease-1>", self.on_drop) # For drag and drop

        self.label_prediction_title = tk.Label(root, text="Predicted Label:", font=("Helvetica", 12, "bold"))
        self.label_prediction_title.pack(pady=(10, 5))

        self.label_prediction = tk.Label(root, text="", font=("Helvetica", 12), wraplength=700)
        self.label_prediction.pack(pady=5)

        self.label_probabilities_title = tk.Label(root, text="Top Probabilities:", font=("Helvetica", 12, "bold"))
        self.label_probabilities_title.pack(pady=(10, 5))

        self.text_probabilities = tk.Text(root, height=8, width=50, state=tk.DISABLED, font=("Courier", 10))
        self.text_probabilities.pack(pady=5)

        self.current_image_path = None # To store the path of the currently displayed image

    def load_class_names(self):
        # Path to a dummy annotation file just to get class names
        # Use a valid JSON path that exists in your project, e.g., from train/valid/test split
        dummy_json_path = 'data_stratified/valid/_annotations.coco.json' 
        if not os.path.exists(dummy_json_path):
            print(f"Warning: Annotation file not found at {dummy_json_path}. Cannot load class names.")
            # Fallback if the JSON is not available, but this is less robust
            return [f"Class {i}" for i in range(NUM_CLASSES)]
        
        with open(dummy_json_path, 'r') as f:
            coco_data = json.load(f)
        
        categories = coco_data['categories']
        
        # Instantiate a dummy dataset to get the label mapping, ensuring consistency
        try:
            # We don't need image_dir for just loading class names
            dummy_dataset = TrafficSignDataset(dummy_json_path, "", self.device, transform=None) 
            
            class_names_dict = {cat['id']: cat['name'] for cat in categories}
            sorted_class_names = [None] * NUM_CLASSES
            for original_id, mapped_idx in dummy_dataset.label_mapping.items():
                if mapped_idx < NUM_CLASSES: # Ensure index is within bounds
                    sorted_class_names[mapped_idx] = class_names_dict.get(original_id, f"Unknown Class {original_id}")
            
            # Replace any None with a placeholder if a mapping was missing
            return [name if name is not None else f"Class {i}" for i, name in enumerate(sorted_class_names)]

        except Exception as e:
            print(f"Error loading class names from dataset: {e}. Falling back to default.")
            return [f"Class {i}" for i in range(NUM_CLASSES)]


    def load_model(self):
        # *** 确保这个路径和 main.py 中保存模型的路径一致 ***
        model_path = 'traffic_sign_model.pth' 
        try:
            # Pass the class itself, not an instance
            self.model = load_model(TrafficSignCNN_AE_ANN, model_path, self.device)
            print("Model loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}. Please train the model first.")
            self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def open_image_dialog(self, event=None):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            self.display_image(file_path)
            self.process_image(file_path)

    def display_image(self, image_path):
        try:
            image = Image.open(image_path)
            image.thumbnail((400, 400)) # Resize for display
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo # Keep a reference
            self.current_image_path = image_path
        except Exception as e:
            self.image_label.config(image="", text=f"Error loading image: {e}")
            self.label_prediction.config(text="", fg="black")
            self.text_probabilities.config(state=tk.NORMAL)
            self.text_probabilities.delete(1.0, tk.END)
            self.text_probabilities.config(state=tk.DISABLED)

    def on_drop(self, event):
        # This function is usually for handling drag-and-drop.
        # Tkinter's dnd is a bit more complex. For simplicity,
        # we'll use the click-to-select dialog.
        # If you need full DND, you'd integrate a dnd library or more complex bindings.
        pass

    def process_image(self, image_path):
        if self.model is None:
            self.label_prediction.config(text="Model not loaded. Cannot classify.", fg="red")
            return

        try:
            # Load image using PIL
            image = Image.open(image_path).convert('RGB')
            
            # Apply the defined inference transformations (on CPU)
            input_tensor = self.infer_transform(image)
            
            # Add a batch dimension and move to device
            input_batch = input_tensor.unsqueeze(0).to(self.device)

            self.model.eval() # Set model to evaluation mode
            with torch.no_grad():
                output = self.model(input_batch)
                
                # *** 关键更改：对于多类别分类，使用 softmax 获取概率 ***
                probabilities = torch.softmax(output, dim=1).cpu().numpy().flatten() 

            # *** 关键更改：获取单个预测类别索引 (最高概率的类别) ***
            predicted_index = np.argmax(probabilities)

            predicted_label_name = "Unknown"
            if predicted_index < len(self.class_names):
                predicted_label_name = self.class_names[predicted_index]
                self.label_prediction.config(text=f"Predicted: {predicted_label_name}", fg="blue")
            else:
                self.label_prediction.config(text=f"Predicted: {predicted_label_name} (Index: {predicted_index}) - Out of bounds", fg="orange")
            
            # Display top probabilities
            sorted_indices = np.argsort(probabilities)[::-1] # Sort descending
            top_k = 5 # Display top 5 probabilities
            top_probabilities_text = "Top Probabilities:\n" # Add title to text output
            for i in range(min(top_k, len(self.class_names))): # Use len(self.class_names) for robustness
                idx = sorted_indices[i]
                top_probabilities_text += f"{self.class_names[idx]}: {probabilities[idx]:.4f}\n"

            self.text_probabilities.config(state=tk.NORMAL)
            self.text_probabilities.delete(1.0, tk.END)
            self.text_probabilities.insert(tk.END, top_probabilities_text)
            self.text_probabilities.config(state=tk.DISABLED)

        except Exception as e:
            self.label_prediction.config(text=f"Error processing image: {e}", fg="red")
            self.image_label.config(image="", text="Error loading image. Drop Image Here or Click to Select")
            self.text_probabilities.config(state=tk.NORMAL)
            self.text_probabilities.delete(1.0, tk.END)
            self.text_probabilities.insert(tk.END, "")
            self.text_probabilities.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignApp(root)
    root.mainloop()