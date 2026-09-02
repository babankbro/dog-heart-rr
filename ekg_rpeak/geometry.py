"""เรขาคณิตของครอป การจัดแถว และการเลือกจุด R หนึ่งจุดต่อจังหวะ"""
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import Config
from .preprocess import find_r_anchor


# ---------------------------------------------------------------- rows / pitch

def group_rows(boxes: np.ndarray, cfg: Config) -> List[List[np.ndarray]]:
    """จัดกลุ่มกล่องเป็นแถว (lead) ตามแกน y แล้วเรียงซ้ายไปขวาในแต่ละแถว

    ภาพหลายบรรทัดถ้าเรียงด้วยแกน x อย่างเดียวจะคำนวณ RR ข้ามบรรทัด
    """
    if len(boxes) == 0:
        return []
    cy = np.array([(b[1] + b[3]) / 2.0 for b in boxes])
    h = float(np.median([b[3] - b[1] for b in boxes])) or 1.0
    order = list(np.argsort(cy))
    rows, cur = [], [order[0]]
    for prev, j in zip(order, order[1:]):
        if cy[j] - cy[prev] > cfg.row_tol_ratio * h:
            rows.append(cur)
            cur = []
        cur.append(j)
    rows.append(cur)
    return [sorted([boxes[i] for i in r], key=lambda b: b[0]) for r in rows]


def row_pitch(row_boxes: Sequence) -> Optional[float]:
    """ระยะระหว่างจังหวะ (~RR เป็นพิกเซล) จากกล่องในแถวเดียวกัน"""
    if len(row_boxes) < 3:
        return None
    cx = np.sort(np.array([(b[0] + b[2]) / 2.0 for b in row_boxes], dtype=float))
    d = np.diff(cx)
    return float(np.median(d)) if d.size else None


def median_rr(pts: Sequence) -> float:
    if len(pts) < 3:
        return 0.0
    xs = np.sort(np.array([p[0] for p in pts], dtype=float))
    return float(np.median(np.diff(xs)))


def dedup_limit(median_rr_px: float, px_per_mm: Optional[float], cfg: Config) -> float:
    """ระยะที่ใกล้กว่านี้ถือว่าเป็นจุดซ้ำจากกล่องที่ซ้อนกัน

    เกณฑ์จากมัธยฐาน RR สมมติว่าจังหวะสม่ำเสมอ ถ้าหัวใจเต้นไม่สม่ำเสมอ เช่นเต้นเป็นคู่
    แล้วเว้นช่วงยาว มัธยฐานจะสะท้อนช่วงยาว แล้วไปตัดจังหวะคู่ที่ชิดกันทิ้งทั้งที่เป็นของจริง
    จึงคุมด้วยระยะ RR ที่สั้นที่สุดเท่าที่เป็นไปได้ทางสรีรวิทยา — สองจุดที่ห่างกว่านั้น
    เป็นคนละจังหวะแน่นอน
    """
    lim = cfg.dedup_ratio * median_rr_px
    if px_per_mm and cfg.scale_hr_hi > 0:
        lim = min(lim, 60.0 / cfg.scale_hr_hi * cfg.paper_speed_mm_s * px_per_mm)
    return lim


def dedup_peaks(pts: Sequence, min_dist_px: float) -> List:
    """refractory period — จุดที่ใกล้กันเกินไปคือจุดซ้ำจากกล่องที่ซ้อนกัน"""
    out: List = []
    for p in sorted(pts, key=lambda q: q[0]):
        if out and (p[0] - out[-1][0]) < min_dist_px:
            if p[2] > out[-1][2]:          # เก็บตัวที่ conf สูงกว่า
                out[-1] = p
        else:
            out.append(p)
    return out


# ---------------------------------------------------------------- crop geometry

def dedup_landmarks(landmarks: List[dict], min_dist: float) -> List[dict]:
    """รวม landmark ชนิดเดียวกันที่อยู่ใกล้กันให้เหลือจุดเดียว

    ครอปของจังหวะข้างเคียงซ้อนทับกัน จุดเดียวกันจึงถูกรายงานซ้ำหลายครั้ง
    เก็บตัวที่ conf สูงสุดไว้
    """
    if min_dist <= 0:
        return landmarks
    out: List[dict] = []
    for p in sorted(landmarks, key=lambda q: -q['conf']):
        if any(q['cls'] == p['cls'] and q['row'] == p['row']
               and abs(q['x'] - p['x']) < min_dist and abs(q['y'] - p['y']) < min_dist
               for q in out):
            continue
        out.append(p)
    return sorted(out, key=lambda q: (q['row'], q['x']))


def crop_region(raw: np.ndarray, box, cfg: Config, pitch: Optional[float] = None,
                px_per_mm: Optional[float] = None, mode: Optional[str] = None,
                ratio: Optional[float] = None, anchor: Optional[bool] = None):
    """คำนวณกรอบที่จะตัด คืน (x, y, w, h)

    โหมด 'mm' คิดขนาดจากเส้นกริดให้ครอบคลุมระยะเป็นมิลลิเมตรเท่าชุดเทรน
    และวางยอด R ตามสัดส่วนตำแหน่งของชุดเทรน ไม่ใช่กึ่งกลางภาพ
    """
    mode = cfg.crop_mode if mode is None else mode
    ratio = cfg.crop_side_ratio if ratio is None else ratio
    anchor = cfg.anchor_on_rpeak if anchor is None else anchor

    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if anchor:
        a = find_r_anchor(raw, box, cfg)
        if a is not None:
            cx, cy = a[0], a[1]

    if mode == 'mm' and px_per_mm:
        # กว้างกับสูงแยกกันได้ เพราะ dataset ยืดครอปที่ไม่จัตุรัสให้เป็นจัตุรัสตอน resize
        w = cfg.out_size * px_per_mm / cfg.train_px_per_mm
        h = (cfg.out_size * px_per_mm / cfg.train_px_per_mm_y
             if cfg.train_px_per_mm_y else w)
        return (cx - cfg.train_anchor_xfrac * w,
                cy - cfg.train_anchor_yfrac * h, w, h)

    if mode == 'train_match':
        # ทำตามสัดส่วนที่วัดได้จาก label ของชุดเทรน: กว้างตามระยะ RR สูงตามแอมพลิจูด
        # แล้วยืดเป็นจัตุรัสตอน resize เหมือนที่ Roboflow ทำ ("Stretch to 512x512")
        a, dbg = find_r_anchor(raw, box, cfg, return_debug=True)
        if a is not None and dbg is not None and 'base' in dbg and pitch:
            ax, ay = a
            baseline = dbg['Y1'] + float(dbg['base'])
            amp = max(8.0, abs(baseline - ay))
            w = cfg.train_frame_w_ratio * pitch
            h = cfg.train_frame_h_ratio * amp
            return (ax - cfg.train_anchor_xfrac * w,
                    ay - cfg.train_anchor_yfrac * h, w, h)
        mode = 'height'          # ขาดข้อมูล ถอยไปใช้โหมดที่ไม่ต้องรู้สัดส่วน

    if mode == 'anchored':
        # ทำแบบเดียวกับที่ dataset สร้างครอป: เอากล่องของโมเดลครอปมาขยายเล็กน้อย
        # วางให้ยอด R อยู่ตรงสัดส่วนที่วัดได้จาก label แล้วยืดเป็นจัตุรัสตอน resize
        w = bw * (1 + 2 * cfg.anchored_pad)
        h = bh * (1 + 2 * cfg.anchored_pad)
        return cx - cfg.train_anchor_xfrac * w, cy - cfg.train_anchor_yfrac * h, w, h

    if mode == 'stretch':
        w, h = bw * (1 + 2 * cfg.pad_ratio), bh * (1 + 2 * cfg.pad_ratio)
    else:
        if mode == 'pitch' and pitch:
            side = ratio * pitch
        elif mode == 'box':
            side = ratio * bw
        else:
            side = max(bw, bh) * (1 + cfg.pad_ratio)
        w = h = max(16.0, side)
    return cx - w / 2.0, cy - h / 2.0, w, h


def crop_to_square(raw: np.ndarray, region, cfg: Config):
    """ตัดตามกรอบแล้ว resize เป็นจัตุรัส เก็บสเกล x/y แยกกันเพื่อ map พิกัดกลับ

    ขนาดกรอบถูกตรึงตามที่คำนวณไว้เสมอ กล่องที่อยู่ติดขอบภาพจึงไม่ถูกซูมต่างจากกล่องอื่น
    และ crop ถูกวางตามตำแหน่งที่ถูกตัดออกไปจริง ไม่ recenter
    """
    x, y, w, h = region
    H, W = raw.shape[:2]
    wx1, wy1 = int(round(x)), int(round(y))
    ww, hh = max(2, int(round(w))), max(2, int(round(h)))
    if cfg.shift_inside:
        # เลื่อนกรอบเข้ามาแทนการเติมขอบ ลดแถบเทาที่ไม่มีในภาพชุดเทรน
        if ww <= W:
            wx1 = min(max(wx1, 0), W - ww)
        if hh <= H:
            wy1 = min(max(wy1, 0), H - hh)
    X1, Y1 = max(0, wx1), max(0, wy1)
    X2, Y2 = min(W, wx1 + ww), min(H, wy1 + hh)
    if X2 <= X1 or Y2 <= Y1:
        return None, None

    crop = raw[Y1:Y2, X1:X2]
    ch, cw = crop.shape[:2]
    ox, oy = X1 - wx1, Y1 - wy1
    top, bottom, left, right = oy, hh - oy - ch, ox, ww - ox - cw
    if cfg.pad_mode == 'replicate':
        canvas = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_REPLICATE)
    else:
        canvas = np.full((hh, ww, 3), 255, np.uint8)
        canvas[oy:oy + ch, ox:ox + cw] = crop

    sq = cv2.resize(canvas, (cfg.out_size, cfg.out_size), interpolation=cv2.INTER_CUBIC)
    meta = {'X1': X1, 'Y1': Y1, 'ox': ox, 'oy': oy,
            'sx': ww / cfg.out_size, 'sy': hh / cfg.out_size}
    return sq, meta


def square_crop(raw: np.ndarray, box, cfg: Config, pitch: Optional[float] = None,
                px_per_mm: Optional[float] = None, **kw):
    return crop_to_square(raw, crop_region(raw, box, cfg, pitch, px_per_mm, **kw), cfg)


def unmap_point(px: float, py: float, m: dict) -> Tuple[float, float]:
    """แปลงพิกัดในครอป กลับเป็นพิกัดบนภาพเต็ม"""
    return px * m['sx'] - m['ox'] + m['X1'], py * m['sy'] - m['oy'] + m['Y1']


def expected_center(cfg: Config) -> Tuple[float, float]:
    """ตำแหน่งที่ R ควรอยู่ในครอป (โหมด mm ไม่ใช่กึ่งกลางภาพ)"""
    if cfg.crop_mode in ('mm', 'anchored', 'train_match'):
        return cfg.train_anchor_xfrac * cfg.out_size, cfg.train_anchor_yfrac * cfg.out_size
    return cfg.out_size / 2.0, cfg.out_size / 2.0


def pick_point(pred, cfg: Config, expect: Optional[Tuple[float, float]] = None):
    """เลือกจุด R หนึ่งจุดจากผลของโมเดลจุด คืน (x, y, conf, cls) หรือ None

    โมเดลจุดตรวจ landmark หลายชนิด (P/Q/R/S/T ...) จึงต้องกรองเฉพาะคลาส R ก่อน
    ที่เหลือให้คะแนนด้วย conf คูณความใกล้ตำแหน่งที่คาดว่า R จะอยู่
    """
    if len(pred.boxes) == 0:
        return None
    xy = pred.boxes.xywh.cpu().numpy()[:, :2]
    conf = pred.boxes.conf.cpu().numpy()
    cls = pred.boxes.cls.cpu().numpy().astype(int)

    keep = np.ones(len(conf), dtype=bool) if cfg.r_class_id is None else (cls == cfg.r_class_id)
    if not keep.any():
        return None
    xy, conf, cls = xy[keep], conf[keep], cls[keep]

    ex, ey = expect if expect is not None else expected_center(cfg)
    d = np.hypot(xy[:, 0] - ex, xy[:, 1] - ey) / (cfg.out_size / 2.0)
    k = int((conf * np.exp(-(d ** 2) / (2 * cfg.center_sigma ** 2))).argmax())
    return float(xy[k, 0]), float(xy[k, 1]), float(conf[k]), int(cls[k])
