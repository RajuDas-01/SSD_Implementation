from flask import Flask, request, send_file
import cv2
import torch
from torchvision import transforms
from PIL import Image
import os
from ssd import SSD300
from box_utils import decode, nms

app = Flask(__name__, static_folder="static", template_folder="templates")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SSD300(num_classes=3).to(device)
model.load_state_dict(torch.load("ssd_catdog.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
])

CLASSES = ["background", "cat", "dog"]

def detect_and_save(input_path, output_path, conf_thresh=0.5):
    image = Image.open(input_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        loc_preds, cls_preds = model(img_tensor)

    dboxes = model.default_boxes.to(device)
    loc = loc_preds.squeeze(0)
    conf = torch.softmax(cls_preds.squeeze(0), dim=1)

    boxes = decode(loc, dboxes)
    boxes = boxes.clamp(min=0, max=1)

    img_cv = cv2.imread(input_path)
    h, w, _ = img_cv.shape

    for cls_id in range(1, len(CLASSES)):
        scores = conf[:, cls_id]
        mask = scores > conf_thresh
        if mask.sum() == 0:
            continue
        boxes_cls = boxes[mask]
        scores_cls = scores[mask]
        keep = nms(boxes_cls, scores_cls, threshold=0.45)

        for i in keep:
            box = boxes_cls[i] * torch.tensor([w, h, w, h])
            xmin, ymin, xmax, ymax = box.int().tolist()
            cv2.rectangle(img_cv, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(img_cv, f"{CLASSES[cls_id]}:{scores_cls[i]:.2f}",
                        (xmin, max(15, ymin-5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)

    cv2.imwrite(output_path, img_cv)
    return output_path

@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]
        input_path = os.path.join("static", file.filename)
        output_path = os.path.join("static", "result.jpg")
        file.save(input_path)

        detect_and_save(input_path, output_path)

        return send_file(output_path, mimetype="image/jpeg")

    return '''
    <h1>Upload Image</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Detect">
    </form>
    '''

if __name__ == "__main__":
    app.run(debug=True)
