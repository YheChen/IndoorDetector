import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
from train import IndoorOutdoorCNN  # reuse your model class

# Setup
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IndoorOutdoorCNN().to(device)
model.load_state_dict(torch.load('models/cnn_indoor_outdoor.pth', map_location=device))
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
classes = ['indoor', 'outdoor']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        f = request.files['file']
        if not f:
            return render_template('index.html', label="No file uploaded.")

        filename = secure_filename(f.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(filepath)

        image = Image.open(filepath).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred = torch.max(outputs, 1)
            label = classes[pred.item()]

        return render_template('index.html', label=label, image_url=filepath)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
