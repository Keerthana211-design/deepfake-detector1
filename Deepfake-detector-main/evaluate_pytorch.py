#!/usr/bin/env python3
"""
Evaluate a binary deepfake detector (PyTorch).
Usage:
  python Deepfake-detector-main/evaluate_pytorch.py --weights path/to/model.pt \
       --test_csv data/crops.csv --outdir results --img_size 224

Notes:
- CSV format: image,label (label 0=real, 1=fake)
- Edit load_model() to match how your model is saved (entire model vs state_dict).
"""
import argparse
import csv
from pathlib import Path
from PIL import Image
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--test_csv", required=True)
    p.add_argument("--outdir", default="results")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    return p.parse_args()


def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def load_model(weights_path, device):
    # Modify this to match your model saving format.
    try:
        model = torch.load(weights_path, map_location=device)
        model.to(device)
        model.eval()
        return model
    except Exception:
        pass
    raise RuntimeError("Could not auto-load model. Edit load_model() to instantiate and load your model/state_dict.")


def predict_image(model, img_path, transform, device):
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    if isinstance(out, tuple):
        out = out[0]
    out = out.detach().cpu().squeeze()
    if out.numel() == 1:
        prob = torch.sigmoid(out).item()
    elif out.dim() == 1 and out.numel() == 2:
        prob = F.softmax(out.unsqueeze(0), dim=1)[0,1].item()
    elif out.dim() == 1:
        probs = F.softmax(out.unsqueeze(0), dim=1)[0]
        prob = probs[1].item() if probs.size(0) > 1 else probs[0].item()
    else:
        prob = float(out.numpy())
    return prob


def find_best_threshold(y_true, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
    best_idx = f1_scores.argmax()
    best_t = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return float(best_t), float(f1_scores[best_idx])


def plot_roc_pr(y_true, y_probs, outdir):
    try:
        from sklearn.metrics import roc_curve, auc, precision_recall_curve
n    except Exception:
        return
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.4f}')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, 'roc.png'))
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, 'pr.png'))
    plt.close()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    transform = get_transform(args.img_size)
    model = load_model(args.weights, args.device)

    rows = []
    with open(args.test_csv, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            if len(r) < 2: continue
            rows.append((r[0].strip(), int(r[1])))

    y_true = []
    y_probs = []
    records = []
    for img_path, label in rows:
        if not os.path.exists(img_path):
            print('Missing:', img_path)
            continue
        prob = predict_image(model, img_path, transform, args.device)
        y_true.append(label)
        y_probs.append(prob)
        records.append({'image': img_path, 'label': label, 'prob': prob})

    # find best threshold on this set
    best_t, best_f1 = find_best_threshold(y_true, y_probs)
    y_pred = [1 if p >= best_t else 0 for p in y_probs]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc_score = roc_auc_score(y_true, y_probs)
    except Exception:
        auc_score = None

    print('Best threshold:', best_t, 'Best F1@threshold:', best_f1)
    print('Accuracy:', acc, 'Precision:', prec, 'Recall:', rec, 'F1:', f1, 'ROC AUC:', auc_score)

    # save predictions
    import json
    with open(outdir/'predictions.json','w') as f:
        json.dump({'best_threshold': best_t, 'metrics': {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': auc_score}, 'records': records}, f, indent=2)

    # save misclassified images
    fp_dir = outdir/'false_positives_real_as_fake'
    fn_dir = outdir/'false_negatives_fake_as_real'
    fp_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)
    for rec, pred in zip(records, y_pred):
        if rec['label'] == 0 and pred == 1:
            # real labeled fake
            dst = fp_dir / Path(rec['image']).name
            try:
                from shutil import copyfile
                copyfile(rec['image'], dst)
            except Exception:
                pass
        if rec['label'] == 1 and pred == 0:
            dst = fn_dir / Path(rec['image']).name
            try:
                from shutil import copyfile
                copyfile(rec['image'], dst)
            except Exception:
                pass

    # plots
    try:
        plot_roc_pr(y_true, y_probs, str(outdir))
    except Exception as e:
        print('Could not plot ROC/PR:', e)

    print('Wrote results to', outdir)

if __name__ == '__main__':
    main()
