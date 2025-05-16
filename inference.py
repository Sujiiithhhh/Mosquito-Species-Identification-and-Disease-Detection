import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
import matplotlib.pyplot as plt

# Load the trained model
checkpoint_dir = 'checkpoints'
best_model_path = os.path.join(checkpoint_dir, 'BEST_MODEL.pth')

# Load the model checkpoint to get the number of classes
checkpoint = torch.load(best_model_path, map_location=torch.device('cpu'))
num_classes = checkpoint['fc.1.weight'].shape[0]

# Define the model with the correct number of classes
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_features, num_classes)
)

# Load the best model weights directly
model.load_state_dict(checkpoint)
model.eval()

# Define enhanced image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define mosquito species and their associated diseases
classes = [
    'aedes_cinereus', 'aedes_excrucians', 'aedes_melanimon', 'aedes_pullatus',
    'culex_pipiens', 'culex_restuans', 'culex_tarsalis', 'culiseta_inornata'
]

disease_mapping = {
    'aedes_cinereus': 'Dengue, Chikungunya, Zika Virus',
    'aedes_excrucians': 'Dengue, Yellow Fever',
    'aedes_melanimon': 'Zika Virus, West Nile Virus',
    'aedes_pullatus': 'Dengue',
    'culex_pipiens': 'West Nile Virus, Japanese Encephalitis',
    'culex_restuans': 'West Nile Virus',
    'culex_tarsalis': 'Western Equine Encephalitis',
    'culiseta_inornata': 'No major diseases associated'
}

# Function to predict mosquito species and related disease risk
def predict_species_and_disease(image_path: str):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = classes[predicted.item()]
        disease_info = disease_mapping.get(predicted_class, 'No known diseases')

    return predicted_class, disease_info, image

# GUI using Tkinter
def upload_and_predict():
    file_path = filedialog.askopenfilename()
    if file_path:
        predicted_class, disease_info, img = predict_species_and_disease(file_path)
        result_label.config(text=f'Predicted Species: {predicted_class}\nDiseases: {disease_info}')
        
        # Display the uploaded image
        img = img.resize((200, 200))
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img

# Create the main application window
app = tk.Tk()
app.title('Mosquito Species & Disease Detection Tool')
app.geometry('600x400')
app.configure(bg='#f4f4f4')

# Create a frame for layout
frame = Frame(app, bg='#f4f4f4')
frame.pack(pady=20)

# Upload Button
upload_button = Button(frame, text='Upload Image', command=upload_and_predict, font=('Arial', 12), bg='#4CAF50', fg='white', padx=10, pady=5)
upload_button.pack(pady=10)

# Image Display
img_label = Label(frame, bg='#f4f4f4')
img_label.pack()

# Result Label
result_label = Label(frame, text='Prediction will appear here', font=('Arial', 14), bg='#f4f4f4')
result_label.pack(pady=20)

# Run the application
app.mainloop()