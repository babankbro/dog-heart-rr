"""แปลงผลตรวจจับเป็นแถว CSV พร้อม RR-interval, heart rate และ flag คุณภาพ"""
import csv
import os
from typing import Dict, List, Optional

import numpy as np

from .config import Config

FIELDS = ['image', 'row', 'r_index', 'x_px', 'y_px', 'x_mm', 'conf', 'src', 'cls',
          'rr_px', 'rr_mm', 'rr_sec', 'bpm', 'flag', 'px_per_mm']


def local_median(rr: np.ndarray, i: int, half: int) -> float:
    """มัธยฐานของช่วงข้างเคียง ใช้เป็นฐานเทียบแทนมัธยฐานทั้งภาพ

    ถ้าอัตราการเต้นค่อย ๆ เปลี่ยนระหว่างบันทึก การเทียบกับมัธยฐานทั้งภาพ
    จะแจ้ง missed_beat ผิดทั้งที่ไม่มีจังหวะไหนหาย จังหวะที่หายจริงจะกระโดด
    เป็นสองเท่าเมื่อเทียบกับเพื่อนบ้านทันที ไม่ใช่ค่อย ๆ ไต่
    """
    lo, hi = max(0, i - half), min(rr.size, i + half + 1)
    return float(np.median(rr[lo:hi]))


def result_to_rows(image_path: str, result: dict, cfg: Config) -> List[Dict]:
    """หนึ่งแถวต่อจุด R คำนวณ RR แยกทีละแถว (lead) เสมอ"""
    s = result['stats']
    px_mm = s['px_per_mm']
    out: List[Dict] = []
    for ri in range(s['n_rows']):
        pk = [p for p in result['peaks'] if p['row'] == ri]
        xs = [p['x'] for p in pk]
        rr = np.diff(xs) if len(xs) > 1 else np.array([])
        med = float(np.median(rr)) if rr.size else 0.0
        for i, p in enumerate(pk):
            rr_px = float(rr[i - 1]) if i > 0 else None

            flag = ''
            if rr_px is not None and rr.size:
                ref = (local_median(rr, i - 1, cfg.flag_window)
                       if rr.size >= cfg.flag_window * 2 + 1 else med)
                if ref > 0:
                    if rr_px > cfg.flag_high_ratio * ref:
                        flag = 'missed_beat?'  # จังหวะที่ชั้นครอปพลาดจะทำให้ bpm เหลือครึ่ง
                    elif rr_px < cfg.flag_low_ratio * ref:
                        flag = 'duplicate?'

            rr_mm = rr_sec = bpm = None
            if rr_px is not None and px_mm:
                rr_mm = rr_px / px_mm
                rr_sec = rr_mm / cfg.paper_speed_mm_s
                bpm = 60.0 / rr_sec if rr_sec > 0 else None

            out.append({
                'image': os.path.basename(image_path),
                'row': ri,
                'r_index': i,
                'x_px': round(p['x'], 1),
                'y_px': round(p['y'], 1),
                'x_mm': round(p['x_mm'], 2) if p.get('x_mm') is not None else '',
                'conf': round(p['conf'], 3),
                'src': p['src'],
                'cls': p['cls'],
                'rr_px': round(rr_px, 1) if rr_px is not None else '',
                'rr_mm': round(rr_mm, 2) if rr_mm is not None else '',
                'rr_sec': round(rr_sec, 3) if rr_sec is not None else '',
                'bpm': round(bpm, 1) if bpm is not None else '',
                'flag': flag,
                'px_per_mm': round(px_mm, 2) if px_mm else '',
            })
    return out


def median_hr(result: dict, cfg: Config, row: Optional[int] = None):
    """HR สรุปคิดจาก RR มัธยฐาน — ค่าเฉลี่ยของ bpm รายคู่ถูก outlier ลากง่าย

    ไม่ระบุแถว = ใช้แถวหลัก (แถวที่มีจังหวะมากที่สุด)
    """
    if row is None:
        row = result.get('main_row', 0)
    px_mm = result['stats']['px_per_mm']
    xs = [p['x'] for p in result['peaks'] if p['row'] == row]
    if not px_mm or len(xs) < 2:
        return None
    med = float(np.median(np.diff(sorted(xs))))
    rr_sec = med / px_mm / cfg.paper_speed_mm_s
    return 60.0 / rr_sec if rr_sec > 0 else None


def write_csv(rows: List[Dict], out_csv: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    # utf-8-sig เพื่อให้ Excel อ่านชื่อไฟล์ภาษาไทยได้ถูกต้อง
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return out_csv
