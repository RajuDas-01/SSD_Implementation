import torch

def decode(loc, dboxes):
    cxcy = loc[:, :2] * 0.1 * dboxes[:, 2:] + dboxes[:, :2]
    wh = torch.exp(loc[:, 2:] * 0.2) * dboxes[:, 2:]
    boxes = torch.cat([cxcy - wh/2, cxcy + wh/2], 1)
    return boxes

def nms(boxes, scores, threshold=0.45, top_k=200):
    keep = []
    _, idxs = scores.sort(descending=True)
    while idxs.numel() > 0:
        i = idxs[0]
        keep.append(i.item())
        if idxs.numel() == 1:
            break
        ious = jaccard(boxes[i].unsqueeze(0), boxes[idxs[1:]])[0]
        idxs = idxs[1:][ious <= threshold]
    return torch.tensor(keep, dtype=torch.long)[:top_k]

def jaccard(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(1)
    area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0)
    union = area_a + area_b - inter
    return inter / union

def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]
