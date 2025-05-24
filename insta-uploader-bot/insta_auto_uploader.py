import os
import time
from instagrapi import Client
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Login credentials (loaded from environment variables)
USERNAME = os.environ.get("INSTA_USERNAME")
PASSWORD = os.environ.get("INSTA_PASSWORD")

# Folder containing media
UPLOAD_FOLDER = "InstaUpload"
PHOTO_EXT = ['.jpg', '.jpeg', '.png']
VIDEO_EXT = ['.mp4']

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# Authenticate to Instagram
cl = Client()
cl.login(USERNAME, PASSWORD)

# Process media files
folder_path = os.path.join(os.getcwd(), UPLOAD_FOLDER)
if not os.path.exists(folder_path):
    print(f"Folder {UPLOAD_FOLDER} not found.")
    exit()

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    ext = os.path.splitext(filename)[1].lower()
    try:
        if ext in PHOTO_EXT:
            caption = generate_caption(file_path)
            print(f"Uploading photo: {filename} with caption: {caption}")
            cl.photo_upload(file_path, caption)
            os.remove(file_path)
        elif ext in VIDEO_EXT:
            print(f"Uploading video: {filename}")
            cl.video_upload(file_path, f"Auto-uploaded 🎥: {filename}")
            os.remove(file_path)
    except Exception as e:
        print(f"Failed to upload {filename}: {e}")
