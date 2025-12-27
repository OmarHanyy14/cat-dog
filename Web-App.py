import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from arch import Cat_Dog_CNN

# --- Load Model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Cat_Dog_CNN().to(device)
model.load_state_dict(torch.load("models/depoly-85-91.pth", map_location=device))
model.eval()

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Prediction Function ---
def predict_image(image, threshold=0.75):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        prediction_prob, predicted = torch.max(probs, dim=0)
        if prediction_prob < threshold:
            label = "No one (Unknown)"
        else:
            label = "Dog" if predicted == 1 else "Cat"
    return label, prediction_prob.item()

# --- Streamlit App ---
st.title("Simple Cat vs Dog Classifier")
uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.9)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    if st.button("Predict"):
        label, prob = predict_image(image, threshold)
        st.write(f"Prediction: **{label}** ({prob*100:.2f}%)")

  

    if camera_image:
        image = Image.open(camera_image)
        st.subheader("Camera Image Prediction:")
        # Call the processing function we defined earlier
        process_and_predict(image,conf_threshold, "Captured Image")
