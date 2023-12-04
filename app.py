import gradio as gr
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import cv2
import numpy as np

# Initialize the model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

def process_frame(webcam_image):
    # Convert the webcam image from Gradio to the format expected by the model
    img = cv2.cvtColor(np.array(webcam_image), cv2.COLOR_RGB2BGR)
    pil_image = Image.fromarray(img)

    # Process the image
    inputs = processor(images=pil_image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([pil_image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # Draw bounding boxes and labels on the image
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [int(round(i, 0)) for i in box.tolist()]
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
        label_text = f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}"
        cv2.putText(img, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Convert back to RGB for Gradio display
    processed_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(processed_image)


demo = gr.Interface(
    process_frame,
    gr.Image(type="pil"),
    "image"
)

if __name__ == "__main__":
    demo.launch()


