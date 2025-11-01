import json, cv2
from pathlib import Path
from tqdm import tqdm

def det2coco(split, det_root, out_json):
    """把 split 文件夹下的 txt 转成 COCO 格式"""
    det_root = Path(det_root)
    img_dir  = det_root / split / 'images'
    ann_dir  = det_root / split / 'annotations'
    images, annotations = [], []
    ann_id = 1
    for txt in tqdm(list(ann_dir.glob('*.txt')), desc=split):
        name = txt.stem
        img_file = img_dir / (name + '.jpg')
        H, W = cv2.imread(str(img_file)).shape[:2]
        img_id = len(images) + 1
        images.append(dict(id=img_id, file_name=img_file.name,
                           height=H, width=W))
        for line in txt.read_text().splitlines():
            parts = line.split(',')
            if len(parts) < 4 or '' in parts[:4]:  continue # 跳过空值
            x1, y1, w, h = map(int, parts[:4])
            if w <= 0 or h <= 0:
                continue
            annotations.append(
                dict(id=ann_id, image_id=img_id, category_id=1,
                     bbox=[x1, y1, w, h], area=w * h, iscrowd=0))
            ann_id += 1
    coco = dict(
        images=images,
        annotations=annotations,
        categories=[dict(id=1, name='object')])
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(coco, open(out_json, 'w'), indent=2)

if __name__ == '__main__':
    root = 'data/VisDrone2019-DET-'
    det2coco('train', root, f'{root}/annotations/instances_train.json')
    det2coco('val',   root, f'{root}/annotations/instances_val.json')

