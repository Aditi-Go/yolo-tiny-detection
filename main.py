import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
from random import choice
import json

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
def infer(input_image_path, output_image_path, threshold=0.9):

    in_pil_img = Image.open(input_image_path).convert("RGB")
    target_sizes = torch.tensor([in_pil_img.size[::-1]])

    inputs = image_processor_tiny(images=in_pil_img, return_tensors="pt")
    outputs = model_tiny(**inputs)

    results = image_processor_tiny.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]

    figure = get_figure(in_pil_img, results)

    buf = io.BytesIO()
    figure.savefig(buf, bbox_inches='tight')
    buf.seek(0)
    output_pil_img = Image.open(buf)
    output_pil_img = output_pil_img.convert("RGB")
    output_pil_img.save(output_image_path, format="JPEG")
    print(f"Saved output image with detections to: {output_image_path}")

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

    return json_output


if __name__ == "__main__":
    input_image_path = "samples/cats.jpg"  
    output_image_path = "output_detected.jpg"  
    result=infer(input_image_path, output_image_path)


    # Example: Save JSON result to file
    with open("detection_results.json", "w") as f:
        json.dump(result, f, indent=4)