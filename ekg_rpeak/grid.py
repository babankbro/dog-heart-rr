"""ระบบพิกัดอ้างอิงจากเส้นกริดหลักของกระดาษ EKG

เส้นกริดหลักห่างกัน 5 mm ตายตัว จึงใช้เป็นไม้บรรทัดที่แม่นกว่าการวัดกริดเล็ก
เพราะเฉลี่ยจากเส้นหลายสิบเส้นตลอดความกว้างภาพ
"""
from typing import List, Optional

import cv2
import numpy as np

from .config import Config
from .scale import estimate_px_per_mm


def chroma_profile(img_bgr: np.ndarray, hp_sigma: float = 60.0) -> np.ndarray:
    """โปรไฟล์ความมีสีตามแนวนอน — เส้นกริดมีสี เส้นคลื่นสีดำไม่มี"""
    a = img_bgr.astype(np.int16)
    chroma = (a.max(axis=2) - a.min(axis=2)).astype(np.float32)
    prof = chroma.mean(axis=0)
    return prof - cv2.GaussianBlur(prof.reshape(1, -1), (0, 0), hp_sigma).ravel()


def find_grid(raw: np.ndarray, cfg: Config, px_per_mm: Optional[float] = None) -> Optional[dict]:
    """หาเส้นกริดหลักแนวตั้ง คืน dict(spacing, phase, lines, px_per_mm) หรือ None

    ใช้ comb matching: ลองระยะห่างรอบ ๆ ค่าที่คาด แล้วเลื่อนเฟสไปทุกตำแหน่ง
    เลือกชุดที่ทับกับสันของโปรไฟล์ได้แรงที่สุด ทนต่อเส้นที่ขาดหายบางเส้น
    """
    ppm = px_per_mm or estimate_px_per_mm(raw)
    if not ppm:
        return None
    prof = chroma_profile(raw)
    n = prof.size
    expect = cfg.grid_mm * ppm
    if expect < 6 or expect > n / 4:
        return None

    def best_phase(S):
        k = int(np.floor((n - 1) / S))
        if k < 3:
            return -np.inf, 0.0
        phases = np.arange(0.0, S, 0.5)
        idx = np.round(phases[:, None] + S * np.arange(k)[None, :]).astype(int)
        idx = np.clip(idx, 0, n - 1)
        scores = prof[idx].mean(axis=1)
        j = int(scores.argmax())
        return float(scores[j]), float(phases[j])

    best = (-np.inf, expect, 0.0)
    for S in np.arange(expect * 0.85, expect * 1.15, 0.1):       # หยาบ
        sc, ph = best_phase(S)
        if sc > best[0]:
            best = (sc, S, ph)
    for S in np.arange(best[1] - 0.15, best[1] + 0.15, 0.01):    # ละเอียด
        sc, ph = best_phase(S)
        if sc > best[0]:
            best = (sc, S, ph)

    score, S, ph = best
    lines = [float(x) for x in np.arange(ph, n, S)]
    if len(lines) < 3:
        return None
    return {'spacing': float(S), 'phase': float(ph), 'lines': lines,
            'px_per_mm': float(S / cfg.grid_mm), 'score': float(score),
            'n_lines': len(lines)}


def refine_grid(raw: np.ndarray, grid: dict, cfg: Config) -> dict:
    """ปรับระยะกริดด้วย least squares จากตำแหน่งเส้นจริง แล้วรายงานความคลาดเคลื่อน

    comb matching ให้ค่าหยาบระดับ 0.01 px จากการค้นหาแบบกริด ขั้นนี้ไปวัดยอด
    ของแต่ละเส้นจริง ๆ แล้ว fit x_k = a + b*k
    ถ้าสูตรระยะต่อพิกเซลถูก residual จะกระจายรอบศูนย์โดยไม่ไหลไปทางใดทางหนึ่ง
    """
    prof = chroma_profile(raw)
    n = prof.size
    S = grid['spacing']
    half = max(2, int(S / 4))

    ks, xs = [], []
    for k, x in enumerate(grid['lines']):
        c = int(round(x))
        lo, hi = max(0, c - half), min(n, c + half + 1)
        if hi - lo < 3:
            continue
        seg = prof[lo:hi]
        j = int(seg.argmax())
        if j == 0 or j == seg.size - 1:
            continue                       # ยอดติดขอบหน้าต่าง เชื่อไม่ได้
        y0, y1, y2 = seg[j - 1], seg[j], seg[j + 1]
        d = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / d if abs(d) > 1e-9 else 0.0
        ks.append(k)
        xs.append(lo + j + float(np.clip(off, -0.5, 0.5)))

    if len(ks) < 4:
        return {**grid, 'refined': False}

    ks_a, xs_a = np.array(ks, float), np.array(xs, float)
    b, a = np.polyfit(ks_a, xs_a, 1)          # x_k = a + b*k
    resid = xs_a - (a + b * ks_a)
    # ความไหลสะสม: เทียบ residual ครึ่งแรกกับครึ่งหลังของภาพ
    mid = len(ks_a) // 2
    drift = float(resid[mid:].mean() - resid[:mid].mean()) if mid else 0.0

    return {**grid,
            'refined': True,
            'spacing': float(b),
            'phase': float(a),
            # ต่อเส้นให้คลุมเต็มความกว้างภาพ รวมช่วงก่อนเส้นแรกที่วัดได้
            'lines': [float(a + b * k)
                      for k in range(int(np.ceil(-a / b)), int(np.floor((n - 1 - a) / b)) + 1)],
            'px_per_mm': float(b / cfg.grid_mm),
            'n_measured': len(ks),
            'resid_rms_px': float(np.sqrt((resid ** 2).mean())),
            'resid_max_px': float(np.abs(resid).max()),
            'drift_px': drift,
            'spacing_comb': float(S)}


def grid_origin(grid: dict, x_ref: float, cfg: Config) -> Optional[float]:
    """เส้นกริดที่ใช้เป็น x = 0

    ค่าเริ่มต้นคือเส้นแรกที่เลยขอบซ้ายของจังหวะซ้ายสุดออกไป (อยู่นอกจังหวะพอดี)
    ถ้าไม่มีเส้นอยู่ทางซ้ายเลย ใช้เส้นแรกที่ถัดจากขอบไปทางขวาแทน
    """
    if not grid or not grid['lines']:
        return None
    lines = np.array(grid['lines'])
    if cfg.grid_origin_mode == 'nearest':
        return float(lines[int(np.argmin(np.abs(lines - x_ref)))])
    before = lines[lines <= x_ref]
    if before.size:
        return float(before[-1])
    after = lines[lines > x_ref]
    return float(after[0]) if after.size else None


def to_mm(x: float, origin: float, px_per_mm: float) -> float:
    """พิกัดพิกเซลบนภาพ -> ระยะเป็นมิลลิเมตรจากจุดอ้างอิง"""
    return (x - origin) / px_per_mm


def summarize_rr(xs: List[float], px_per_mm: float, cfg: Config) -> dict:
    """สรุป RR interval ของจุด R ชุดหนึ่ง (พิกเซล -> mm -> วินาที -> bpm)"""
    xs = sorted(float(x) for x in xs)
    if len(xs) < 2 or not px_per_mm:
        return {'n': len(xs), 'mean_mm': None, 'median_mm': None, 'sd_mm': None,
                'mean_sec': None, 'mean_bpm': None, 'median_bpm': None}
    rr_px = np.diff(xs)
    rr_mm = rr_px / px_per_mm
    mean_mm = float(rr_mm.mean())
    median_mm = float(np.median(rr_mm))
    mean_sec = mean_mm / cfg.paper_speed_mm_s
    return {
        'n': len(xs),
        'mean_mm': mean_mm,
        'median_mm': median_mm,
        'sd_mm': float(rr_mm.std(ddof=1)) if rr_mm.size > 1 else 0.0,
        'min_mm': float(rr_mm.min()),
        'max_mm': float(rr_mm.max()),
        'mean_sec': float(mean_sec),
        'mean_bpm': float(60.0 / mean_sec) if mean_sec > 0 else None,
        'median_bpm': float(60.0 / (median_mm / cfg.paper_speed_mm_s)) if median_mm > 0 else None,
    }
