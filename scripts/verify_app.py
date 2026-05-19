"""Verifies app.py's model-loading + classification pipeline on real, labeled
images. Run: python verify_app.py"""
import os, glob, random
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import timm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
ARCH_CANDIDATES = ['convnext_tiny', 'resnet18', 'resnet50', 'convnextv2_base',
                   'tf_efficientnetv2_m', 'swin_base_patch4_window7_224',
                   'efficientnet_b0']


def load_model(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    classes, arch, img_size = None, None, 224
    if isinstance(ckpt, dict) and isinstance(ckpt.get('model'), dict):
        state = ckpt['model']
        classes = ckpt.get('classes')
        cfg = ckpt.get('cfg') or {}
        arch = cfg.get('model_name')
        img_size = int(cfg.get('img_size', 224) or 224)
    elif isinstance(ckpt, dict) and isinstance(ckpt.get('state_dict'), dict):
        state = ckpt['state_dict']
    else:
        state = ckpt
    n = len(classes) if classes else int(list(state.values())[-1].shape[0])
    for name in ([arch.split('.')[0]] if arch else []) + ARCH_CANDIDATES:
        if not name:
            continue
        try:
            m = timm.create_model(name, pretrained=False, num_classes=n)
            m.load_state_dict(state, strict=True)
            m.eval()
            return m, list(classes) if classes else None, name, img_size
        except Exception:
            continue
    raise RuntimeError('no architecture matched the checkpoint')


def build_transform(s):
    return T.Compose([
        T.Resize((s, s), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


@torch.no_grad()
def classify(img, model, classes, tfm):
    x = tfm(img.convert('RGB')).unsqueeze(0)
    p = F.softmax(model(x), dim=1)[0].numpy()
    i = int(p.argmax())
    return classes[i], float(p[i])


if __name__ == '__main__':
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_path = os.path.join(ROOT, 'app', 'best_model.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = sorted(glob.glob(os.path.join(ROOT, 'models', '*.pt')))[0]
    model, classes, arch, img_size = load_model(ckpt_path)
    if classes is None:
        classes = sorted(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') +
                          ['del', 'nothing', 'space'])
    print(f'checkpoint : {ckpt_path}')
    print(f'architecture: {arch}   classes: {len(classes)}   input: {img_size}px')

    tfm = build_transform(img_size)
    base = next((c for c in [
        os.path.join(ROOT, 'data', 'competition_data', 'competition_data'),
        os.path.join(ROOT, 'data', 'competition_data'),
        os.path.join(ROOT, 'data')]
        if os.path.isdir(os.path.join(c, 'train'))), None)
    if base is None:
        print('no local data to test against — skipping accuracy check')
        raise SystemExit(0)

    train_dir = os.path.join(base, 'train')
    random.seed(0)
    correct = total = 0
    for cls in random.sample(sorted(os.listdir(train_dir)), 12):
        d = os.path.join(train_dir, cls)
        for f in random.sample(os.listdir(d), 2):
            pred, conf = classify(Image.open(os.path.join(d, f)),
                                  model, classes, tfm)
            total += 1
            correct += (pred == cls)
            flag = '' if pred == cls else '   <-- MISMATCH'
            print(f'  true={cls:>7}  pred={pred:>7}  conf={conf:.3f}{flag}')
    acc = correct / total
    print(f'\naccuracy on {total} labeled images: {acc:.1%}')
    print('PIPELINE OK' if acc >= 0.9 else 'PIPELINE SUSPECT — check preprocessing')
