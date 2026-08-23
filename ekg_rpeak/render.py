"""วาดผลตรวจจับลงบนภาพเต็ม

แยกออกจาก webapp เพื่อให้เรียกใช้จากสคริปต์และทดสอบได้โดยไม่ต้องยกเซิร์ฟเวอร์
"""
from typing import Optional

import cv2
import numpy as np

# BGR
COL_BOX = (60, 220, 60)
COL_R_MODEL = (0, 0, 235)
COL_R_ANCHOR = (0, 150, 255)
COL_LANDMARK = (80, 80, 235)

# ขนาดวงกลม landmark เทียบกับครึ่งความยาวกากบาท และความทึบตอนผสมกับพื้นหลัง
LANDMARK_SCALE = 0.5
LANDMARK_ALPHA = 0.7


def mark_size(width: int, base: int = 9) -> int:
    """ครึ่งความยาวของกากบาท ปรับตามความกว้างภาพเพื่อให้เห็นเท่ากันทุกสเกล"""
    return max(4, int(base * max(1.0, width / 1600.0)))


COL_ANCHOR = (60, 200, 60)
COL_ORIGIN = (200, 120, 0)
COL_GRID = (200, 170, 120)

# ไม้บรรทัดกริดวาดจางกว่าเส้นอื่นครึ่งหนึ่ง เพื่อไม่ให้บังคลื่น
GRID_ALPHA = 0.25


def draw_mask_panel(dbg: dict, peaks=(), landmarks=(), anchor=None, size: int = 260,
                    landmark_alpha: float = LANDMARK_ALPHA,
                    landmark_scale: float = LANDMARK_SCALE) -> np.ndarray:
    """แผงตรวจสอบของจังหวะเดียว: ซ้าย = ภาพจริง ขวา = ink mask

    วาดจุดชุดเดียวกันทั้งสองฝั่ง เพื่อเทียบว่าตำแหน่งที่ได้มาตรงกับหมึกที่ mask เห็นไหม
    กากบาทเขียว = anchor จาก image processing, กากบาทแดง = R จากโมเดล
    (ส้มคือ R ที่สุดท้ายแล้วใช้ค่าจาก anchor) วงกลมจาง = landmark อื่นของโมเดล
    """
    roi = cv2.resize(dbg['roi'], (size, size))
    m = cv2.resize(dbg['mask'].astype(np.uint8) * 255, (size, size),
                   interpolation=cv2.INTER_NEAREST)
    mask = cv2.cvtColor(255 - m, cv2.COLOR_GRAY2BGR)

    h, w = dbg['roi'].shape[:2]
    sx, sy = size / max(1, w), size / max(1, h)

    def to_panel(x, y):
        return int(round((x - dbg['X1']) * sx)), int(round((y - dbg['Y1']) * sy))

    def inside(pt):
        return 0 <= pt[0] < size and 0 <= pt[1] < size

    r = max(6, size // 26)
    rad = max(2, int(round(r * landmark_scale)))
    for panel in (roi, mask):
        if landmarks:
            layer = panel.copy()
            drew = False
            for p in landmarks:
                pt = to_panel(p['x'], p['y'])
                if inside(pt):
                    cv2.circle(layer, pt, rad, COL_LANDMARK, 1, cv2.LINE_AA)
                    drew = True
            if drew:
                cv2.addWeighted(layer, landmark_alpha, panel, 1 - landmark_alpha, 0, panel)
        for p in peaks:
            pt = to_panel(p['x'], p['y'])
            if not inside(pt):
                continue
            col = COL_R_MODEL if p.get('src') == 'model' else COL_R_ANCHOR
            x, y = pt
            cv2.line(panel, (x - r, y - r), (x + r, y + r), col, 2, cv2.LINE_AA)
            cv2.line(panel, (x - r, y + r), (x + r, y - r), col, 2, cv2.LINE_AA)
        if anchor is not None:
            pt = to_panel(anchor[0], anchor[1])
            if inside(pt):
                cv2.drawMarker(panel, pt, COL_ANCHOR, cv2.MARKER_CROSS, int(r * 2.2), 2)
    return np.hstack([roi, mask])


def draw_overlay(result: dict, boxes: bool = True, marks: bool = True,
                 landmarks: bool = True, landmark_alpha: float = LANDMARK_ALPHA,
                 base: int = 9, landmark_scale: float = LANDMARK_SCALE,
                 origin: bool = True, grid: bool = False) -> np.ndarray:
    """คืนภาพเต็มที่วาดกรอบจังหวะ จุด R และ landmark อื่น ๆ ของโมเดลจุด

    จุด R เป็นกากบาท (แดง = โมเดลยืนยัน, ส้ม = มาจาก anchor)
    landmark อื่น ๆ เป็นวงกลมแดงจาง รัศมีราวครึ่งหนึ่งของครึ่งความยาวกากบาท
    """
    out = result['raw'].copy()
    r = mark_size(out.shape[1], base)
    thick = max(2, r // 4)

    if boxes:
        for (x1, y1, x2, y2) in result['boxes']:
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                          COL_BOX, max(1, r // 8))

    if grid and result.get('grid'):
        layer = out.copy()
        for x in result['grid']['lines']:
            xi = int(round(x))
            if 0 <= xi < out.shape[1]:
                cv2.line(layer, (xi, 0), (xi, out.shape[0]), COL_GRID, 1)
        cv2.addWeighted(layer, GRID_ALPHA, out, 1 - GRID_ALPHA, 0, out)

    if origin and result.get('origin') is not None:
        x0 = int(round(result['origin']))
        cv2.line(out, (x0, 0), (x0, out.shape[0]), COL_ORIGIN, max(2, r // 5))

    peaks = result.get('peaks', [])
    if landmarks:
        others = [p for p in result.get('landmarks', [])
                  if not any(abs(p['x'] - q['x']) <= r and abs(p['y'] - q['y']) <= r
                             for q in peaks)]           # ไม่วาดซ้ำจุดที่เป็น R อยู่แล้ว
        if others:
            layer = out.copy()
            rad = max(2, int(round(r * landmark_scale)))
            for p in others:
                cv2.circle(layer, (int(round(p['x'])), int(round(p['y']))),
                           rad, COL_LANDMARK, max(1, thick // 2), cv2.LINE_AA)
            cv2.addWeighted(layer, landmark_alpha, out, 1 - landmark_alpha, 0, out)

    if marks:
        for p in peaks:
            x, y = int(round(p['x'])), int(round(p['y']))
            col = COL_R_MODEL if p.get('src') == 'model' else COL_R_ANCHOR
            cv2.line(out, (x - r, y - r), (x + r, y + r), col, thick, cv2.LINE_AA)
            cv2.line(out, (x - r, y + r), (x + r, y - r), col, thick, cv2.LINE_AA)
    return out
