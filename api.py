import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import json
from fastapi import FastAPI, File, UploadFile
import base64
from random import choice
import uvicorn
import os

app = FastAPI()

image_processor_tiny = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
model_tiny = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")

COLORS = ["#ff7f7f", "#ff7fbf", "#ff7fff", "#bf7fff",
          "#7f7fff", "#7fbfff", "#7fffff", "#7fffbf",
          "#7fff7f", "#bfff7f", "#ffff7f", "#ffbf7f"]

fdic = {
    "family": "DejaVu Serif",
    "style": "normal",
    "size": 18,
    "color": "yellow",
    "weight": "bold"
}

def get_figure(in_pil_img, in_results):
    plt.figure(figsize=(16, 10))
    plt.imshow(in_pil_img)
    ax = plt.gca()

    for score, label, box in zip(in_results["scores"], in_results["labels"], in_results["boxes"]):
        selected_color = choice(COLORS)

        box_int = [i.item() for i in torch.round(box).to(torch.int32)]
        x, y, w, h = box_int[0], box_int[1], box_int[2] - box_int[0], box_int[3] - box_int[1]

        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, color=selected_color, linewidth=3, alpha=0.8))
        ax.text(x, y, f"{model_tiny.config.id2label[label.item()]}: {round(score.item()*100, 2)}%", fontdict=fdic, alpha=0.8)

    plt.axis("off")
    return plt.gcf()

def infer(in_pil_img, threshold=0.9):
    target_sizes = torch.tensor([in_pil_img.size[::-1]])

    inputs = image_processor_tiny(images=in_pil_img, return_tensors="pt")
    outputs = model_tiny(**inputs)

    results = image_processor_tiny.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]

    figure = get_figure(in_pil_img, results)

    detected_objects = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box_coords = [int(i.item()) for i in box]
        detected_objects.append({
            "label": model_tiny.config.id2label[label.item()],
            "score": round(score.item() * 100, 2),
            "bounding_box": {
                "x1": box_coords[0],
                "y1": box_coords[1],
                "x2": box_coords[2],
                "y2": box_coords[3]
            }
        })

    json_output = {"Detected Objects": detected_objects}
    print(json.dumps(json_output, indent=4))
    
    with open('data.json', 'w') as f:
        json.dump(json_output, f, indent=4)

    return figure, json_output

def pil_image_to_base64(img: Image.Image):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode("utf-8")
    return img_str

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    img_bytes = await file.read()
    in_pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    figure, json_output = infer(in_pil_img)

    buf = io.BytesIO()
    figure.savefig(buf, bbox_inches='tight')
    buf.seek(0)
    output_pil_img = Image.open(buf)

    output_pil_img = output_pil_img.convert("RGB")
    out_img_path = "output.jpg"
    output_pil_img.save(out_img_path)

    img_base64 = pil_image_to_base64(output_pil_img)

    return {
        "filename": file.filename,
        "json_result": json_output,
        "image_path": os.path.abspath(out_img_path),
        "image_base64": img_base64
    }



# # Add the following block to run the FastAPI server using uvicorn
# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)