import cv2
import torch
from torchvision import transforms
from PIL import Image
from ssd import SSD300
from box_utils import decode, nms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SSD300(num_classes=3).to(device)
model.load_state_dict(torch.load("ssd_catdog.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
])

CLASSES = ["background", "cat", "dog"]

def detect_image(img_path, conf_thresh=0.5):
    image = Image.open(img_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        loc_preds, cls_preds = model(img_tensor)

    dboxes = model.default_boxes.to(device)
    loc = loc_preds.squeeze(0)
    conf = torch.softmax(cls_preds.squeeze(0), dim=1)

    boxes = decode(loc, dboxes)
    boxes = boxes.clamp(min=0, max=1)

    img_cv = cv2.imread(img_path)
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

    cv2.imshow("Detections", img_cv)
    cv2.waitKey(0)

if __name__ == "__main__":
    detect_image("test.jpg")
