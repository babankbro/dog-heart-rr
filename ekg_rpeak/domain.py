# -*- coding: utf-8 -*-
"""วัดว่าภาพที่จะป้อนโมเดล หน้าตาใกล้กับชุดที่โมเดลเคยเห็นตอนเทรนแค่ไหน

โมเดลครอปเทรนกับข้อมูลชุดหนึ่ง แต่ถูกใช้กับข้อมูลอีกชุด ความต่างของ "หน้าตา"
หลัง binarization อธิบายได้ว่าทำไมกล่องถึงหลุดหรือคร่อมไม่มิด ตัวนี้วัดสามอย่าง
ที่บอกหน้าตาได้จริงหลังภาพถูกย่อลง imgsz ตามที่โมเดลทำ

  ink    สัดส่วนพิกเซลที่เป็นเส้น — บอกว่าภาพ "เข้ม" แค่ไหน
  w_p90  ความหนาเส้นแนวนอน
  h_med  ความต่อเนื่องแนวตั้ง — ค่า 1 แปลว่าเป็นจุดประ ไม่ใช่เส้น

อ้างอิงฝั่งเทรนดึงจาก mosaic ที่ ultralytics เซฟไว้ตอนเทรน (`val_batch*_labels.jpg`)
ซึ่งคือภาพที่โมเดลเห็นจริง ไม่ใช่ภาพต้นฉบับ จึงเทียบกันได้ตรง ๆ
"""
import glob
import os
from typing import Dict, List, Optional

import cv2
import numpy as np

from .config import Config
from .imageio import imread_u, list_images
from .preprocess import crop_preprocess

PAD_GRAY = 113          # สีเทาที่ ultralytics ใช้เติมขอบใน mosaic
ANNO_SAT = 60           # กล่อง/ตัวอักษรที่วาดทับเป็นสี ภาพจริงเป็นขาวดำ
KEYS = ('ink', 'w_p90', 'h_med', 'h_p90')


def _runs(line: np.ndarray) -> np.ndarray:
    d = np.diff(np.concatenate(([0], line.astype(np.uint8), [0])))
    return np.where(d == -1)[0] - np.where(d == 1)[0]


def features(mask: np.ndarray) -> Dict[str, float]:
    """สถิติหน้าตาของภาพ binary หนึ่งภาพ (True = เส้น)"""
    hr = [_runs(r) for r in mask[::2] if r.any()]
    vr = [_runs(c) for c in mask.T[::2] if c.any()]
    hr = np.concatenate(hr) if hr else np.array([0])
    vr = np.concatenate(vr) if vr else np.array([0])
    return {'ink': float(mask.mean()), 'w_p90': float(np.percentile(hr, 90)),
            'h_med': float(np.median(vr)), 'h_p90': float(np.percentile(vr, 90))}


def longest_run(flags: np.ndarray):
    """ช่วงที่ True ติดกันยาวที่สุด คืน (ความยาว, เริ่ม, จบ)"""
    best, i = (0, 0, 0), 0
    while i < len(flags):
        if flags[i]:
            j = i
            while j < len(flags) and flags[j]:
                j += 1
            if j - i > best[0]:
                best = (j - i, i, j)
            i = j
        else:
            i += 1
    return best


def train_features(mosaic_dir: str, grid: int = 4) -> List[Dict[str, float]]:
    """สถิติของภาพที่โมเดลเห็นตอนเทรน อ่านจาก mosaic ที่ ultralytics เซฟไว้"""
    out = []
    for path in sorted(glob.glob(os.path.join(mosaic_dir, 'val_batch*_labels.jpg'))):
        img = imread_u(path)
        if img is None:
            continue
        H, W = img.shape[:2]
        th, tw = H // grid, W // grid
        for r in range(grid):
            for c in range(grid):
                tile = img[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
                g = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
                sat = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)[:, :, 1]
                content = ~((np.abs(g.astype(int) - PAD_GRAY) < 4) | (sat > ANNO_SAT))
                n, y1, y2 = longest_run(content.mean(axis=1) > 0.6)
                if n < 30:                     # ไทล์ที่เป็น padding ล้วน
                    continue
                f = features((g[y1:y2] < 128) & content[y1:y2])
                f['strip_h'] = float(n)
                out.append(f)
    return out


def model_view(raw: np.ndarray, cfg: Config) -> np.ndarray:
    """ภาพ binary ที่โมเดลเห็นจริง — ผ่าน binarization แล้วย่อลง imgsz"""
    pre = cv2.cvtColor(crop_preprocess(raw, cfg), cv2.COLOR_BGR2GRAY) < 128
    h, w = pre.shape
    s = cfg.crop_imgsz / max(h, w)
    return cv2.resize(pre.astype(np.uint8), (max(1, int(w * s)), max(1, int(h * s))),
                      interpolation=cv2.INTER_AREA) > 0


def image_features(images_dir: str, cfg: Config) -> List[Dict[str, float]]:
    out = []
    for p in list_images(images_dir):
        raw = imread_u(p)
        if raw is None:
            continue
        view = model_view(raw, cfg)
        f = features(view)
        f['strip_h'] = float(view.shape[0])
        out.append(f)
    return out


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return {k: float(np.median([r[k] for r in rows]))
            for k in list(KEYS) + ['strip_h']} if rows else {}


def distance(a: Dict[str, float], b: Dict[str, float]) -> Optional[float]:
    """ระยะห่างของโดเมน — ผลรวมของ log ratio จึงไม่ขึ้นกับหน่วยของแต่ละตัว"""
    if not a or not b:
        return None
    return float(sum(abs(np.log(max(a[k], 1e-6) / max(b[k], 1e-6))) for k in KEYS))
