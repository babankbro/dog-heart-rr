"""ไปป์ไลน์หลัก: กล่องจังหวะ -> ครอปต่อจังหวะ -> จุด R -> รวมผล

โมเดลถูกโหลดแบบ lazy เพื่อให้ชุดทดสอบรันได้โดยไม่ต้องมี torch/ultralytics
"""
import os
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from .config import Config
from .geometry import (dedup_landmarks, dedup_limit, dedup_peaks, expected_center, group_rows,
                       median_rr, pick_point, row_pitch, square_crop, unmap_point)
from .grid import find_grid, grid_origin, refine_grid, summarize_rr, to_mm
from .imageio import imread_u
from .preprocess import (crop_preprocess, drop_edge_non_beats,
                         find_r_anchor, point_preprocess)
from .scale import check_scale, px_per_mm_from_major, resolve_px_per_mm


@dataclass
class Models:
    crop: Any
    point: Optional[Any] = None

    @property
    def has_point(self) -> bool:
        return self.point is not None


def load_models(cfg: Config) -> Models:
    """โหลด YOLO สองตัว โมเดลจุดเป็น optional (ไปป์ไลน์ยังทำงานได้ด้วย anchor อย่างเดียว)"""
    from ultralytics import YOLO  # import ตรงนี้เพื่อไม่ให้ชุดทดสอบต้องมี torch

    if not os.path.exists(cfg.crop_weights):
        raise FileNotFoundError(
            f'ไม่พบ weights ของโมเดลครอป: {cfg.crop_weights}')
    crop = YOLO(cfg.crop_weights)
    point = YOLO(cfg.point_weights) if os.path.exists(cfg.point_weights) else None
    return Models(crop=crop, point=point)


def batch_predict(model, crops: List[np.ndarray], conf: float, iou: float,
                  cfg: Config) -> List:
    """รันโมเดลจุดเป็นชุด แทนการเรียกทีละครอป"""
    out: List = []
    for i in range(0, len(crops), cfg.batch):
        out += model.predict(crops[i:i + cfg.batch], conf=conf, iou=iou,
                             imgsz=cfg.point_imgsz, save=False, verbose=False)
    return out


def detect_boxes(raw: np.ndarray, models: Models, cfg: Config) -> np.ndarray:
    """ชั้นที่ 1 — กล่องของแต่ละจังหวะ (โมเดลนี้เทรนบนภาพ Blackhat)"""
    res = models.crop.predict(crop_preprocess(raw, cfg), conf=cfg.crop_conf,
                              iou=cfg.crop_iou, imgsz=cfg.crop_imgsz,
                              save=False, verbose=False)[0]
    if len(res.boxes) == 0:
        return np.zeros((0, 4), dtype=int)
    return np.round(res.boxes.xyxy.cpu().numpy()).astype(int)


def _landmarks_of(pred, meta: dict, row: int) -> List[dict]:
    """detection ทุกจุดของครอปหนึ่ง map กลับเป็นพิกัดบนภาพเต็ม

    โมเดลจุดตรวจ landmark หลายชนิด (P/Q/R/S/T ...) ไปป์ไลน์เลือกไปใช้แค่จุด R
    ที่เหลือเก็บไว้แสดงผลเพื่อให้ตรวจสอบด้วยตาได้ว่าโมเดลเห็นอะไรบ้าง
    """
    if len(pred.boxes) == 0:
        return []
    xy = pred.boxes.xywh.cpu().numpy()[:, :2]
    conf = pred.boxes.conf.cpu().numpy()
    cls = pred.boxes.cls.cpu().numpy().astype(int)
    out = []
    for (px, py), c, k in zip(xy, conf, cls):
        x, y = unmap_point(float(px), float(py), meta)
        out.append({'row': row, 'x': x, 'y': y, 'conf': float(c), 'cls': int(k)})
    return out


def detect_r_peaks(image_path: str, models: Models, cfg: Config) -> dict:
    """คืน dict: raw, boxes, rows, peaks, stats

    peaks = [{'row', 'index', 'x', 'y', 'conf', 'src', 'cls'}, ...]
    src = 'model' (โมเดลจุดยืนยัน) หรือ 'anchor' (ได้จาก image processing ล้วน)
    """
    raw = imread_u(image_path)
    if raw is None:
        raise FileNotFoundError(f'อ่านภาพไม่ได้: {image_path}')
    px_mm = resolve_px_per_mm(raw, cfg)

    boxes = detect_boxes(raw, models, cfg)
    rows = group_rows(boxes, cfg)

    n_edge = 0
    trimmed = []
    for row in rows:
        kept, dropped = drop_edge_non_beats(raw, row, cfg)
        n_edge += dropped
        if kept:
            trimmed.append(kept)
    rows = trimmed
    boxes = np.array([b for row in rows for b in row], dtype=int) if rows else np.zeros((0, 4), int)

    # ตรวจว่าสเกลที่ได้จากกริดเล็กให้อัตราการเต้นที่เป็นไปได้ไหม ถ้าไม่ ให้วัดใหม่จาก
    # ช่องกริดหลัก (หนึ่งช่อง = cfg.grid_mm มิลลิเมตร) ซึ่งเป็นเส้นที่เข้มและนับได้ชัดกว่า
    scale_source = 'manual' if cfg.px_per_mm else ('minor' if px_mm else 'none')
    ref_pitch_px = next((row_pitch(r) for r in rows if row_pitch(r)), None)
    # เคารพการปิดการประมาณสเกล — ผู้ใช้ที่สั่งไม่ให้เดา ต้องไม่ได้ค่าที่เดามาให้
    if not cfg.px_per_mm and cfg.auto_px_per_mm and cfg.scale_source in ('auto', 'major'):
        _, ok = check_scale(px_mm, ref_pitch_px, cfg.paper_speed_mm_s,
                            cfg.scale_hr_lo, cfg.scale_hr_hi)
        if cfg.scale_source == 'major' or not ok:
            alt = px_per_mm_from_major(raw, cfg, ref_pitch_px)
            if alt:
                px_mm, scale_source = alt, 'major'

    # ระบบพิกัดอ้างอิงจากเส้นกริดหลัก 5 mm — แม่นกว่ากริดเล็กเพราะเฉลี่ยจากหลายสิบเส้น
    grid = find_grid(raw, cfg, px_mm)
    if grid and cfg.grid_refine:
        grid = refine_grid(raw, grid, cfg)
    if grid and cfg.grid_from_lines:
        px_mm = grid['px_per_mm']
    origin = grid_origin(grid, float(rows[0][0][0]), cfg) if (grid and rows) else None

    crops, metas, owner, anchors, pitches = [], [], [], [], []
    for ri, row in enumerate(rows):
        pitch = row_pitch(row)
        for box in row:
            a = find_r_anchor(raw, box, cfg)
            if a is None:
                a = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
            sq, m = square_crop(raw, box, cfg, pitch=pitch, px_per_mm=px_mm)
            if sq is None:
                continue
            crops.append(point_preprocess(sq, cfg))
            metas.append(m)
            owner.append(ri)
            anchors.append(a)
            pitches.append(pitch)

    use_point = cfg.point_mode != 'anchor_only' and models.has_point and crops
    preds = batch_predict(models.point, crops, cfg.point_conf, cfg.point_iou, cfg) if use_point else []
    expect = expected_center(cfg)

    per_row: List[List] = [[] for _ in rows]
    landmarks: List[dict] = []
    n_model = n_anchor = n_reject = n_far = 0
    for i, (m, ri, a) in enumerate(zip(metas, owner, anchors)):
        X, Y, conf, src, cls = a[0], a[1], cfg.anchor_conf, 'anchor', -1
        if i < len(preds):
            landmarks += _landmarks_of(preds[i], m, ri)
        got = pick_point(preds[i], cfg, expect=expect) if i < len(preds) else None
        if got is not None:
            mx, my = unmap_point(got[0], got[1], m)
            lim = cfg.max_refine_ratio * pitches[i] if pitches[i] else float('inf')
            near = abs(mx - a[0]) <= lim
            if cfg.point_mode == 'model_only' or near or got[2] >= cfg.trust_model_conf:
                X, Y, conf, src, cls = mx, my, got[2], 'model', got[3]
                if not near:
                    n_far += 1     # โมเดลชี้ไกลแต่มั่นใจสูง เชื่อโมเดล ไม่ให้ anchor ผิดมาทับ
            else:
                n_reject += 1
        if src == 'model':
            n_model += 1
        else:
            n_anchor += 1
            if cfg.point_mode == 'model_only':
                continue
        per_row[ri].append((X, Y, conf, src, cls))

    ref_pitch = next((p for p in pitches if p), None)
    landmarks = dedup_landmarks(landmarks, cfg.landmark_dedup_ratio * ref_pitch
                                if ref_pitch else 3.0)

    peaks, n_dup = [], 0
    for ri, pts in enumerate(per_row):
        med = median_rr(pts)
        kept = (dedup_peaks(pts, dedup_limit(med, px_mm, cfg)) if med > 0
                else sorted(pts, key=lambda p: p[0]))
        n_dup += len(pts) - len(kept)
        for i, p in enumerate(kept):
            peaks.append({'row': ri, 'index': i, 'x': p[0], 'y': p[1],
                          'conf': p[2], 'src': p[3], 'cls': p[4],
                          'x_mm': (to_mm(p[0], origin, px_mm)
                                   if origin is not None and px_mm else None)})

    stats = {'n_boxes': len(boxes), 'n_rows': len(rows), 'n_peaks': len(peaks),
             'n_dup': n_dup, 'n_model': n_model, 'n_anchor': n_anchor,
             'n_reject': n_reject, 'n_far': n_far, 'n_landmarks': len(landmarks),
             'n_edge_dropped': n_edge, 'px_per_mm': px_mm,
             'scale_source': scale_source,
             'grid_spacing_px': grid['spacing'] if grid else None,
             'grid_lines': len(grid['lines']) if grid else 0,
             'grid_resid_px': grid.get('resid_rms_px') if grid else None,
             'grid_drift_px': grid.get('drift_px') if grid else None,
             'grid_origin_px': origin}
    rr = {ri: summarize_rr([p['x'] for p in peaks if p['row'] == ri], px_mm, cfg)
          for ri in range(len(rows))}
    # แถวหลัก = แถวที่มีจังหวะมากที่สุด ไม่ใช่แถวแรกเสมอไป — เศษกล่องที่ขอบภาพ
    # ถูกจัดเป็นแถวของตัวเองได้ ถ้ารายงานจากแถวแรกดื้อ ๆ ตัวเลขทั้งภาพจะหายไป
    main_row = max(range(len(rows)), key=lambda i: len(rows[i])) if rows else 0
    return {'raw': raw, 'boxes': boxes, 'rows': rows, 'peaks': peaks,
            'landmarks': landmarks, 'stats': stats, 'grid': grid,
            'origin': origin, 'rr': rr, 'main_row': main_row}
