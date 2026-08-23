"""หาสเกลของกระดาษ (พิกเซลต่อมิลลิเมตร) จากเส้นกริด และตรวจความสมเหตุสมผล"""
from typing import Optional, Tuple

import cv2
import numpy as np


def grid_profile(img_bgr: np.ndarray) -> np.ndarray:
    """โปรไฟล์ตามแนวนอนที่มีแต่เส้นกริด โดยตัดเส้นคลื่นออก

    เส้นคลื่นมีโครงสร้างละเอียด (ขาขึ้น-ขาลงห่างกันไม่กี่พิกเซล) ซึ่งสร้างยอด
    autocorrelation ที่ lag สั้น ๆ แรงกว่ากริด ทำให้วัดระยะกริดพลาด
    กระดาษกริดสี: ใช้ค่าความมีสี (chroma) ซึ่งเส้นคลื่นสีดำไม่มี
    สแกนขาวดำ: ใช้ความเข้ม แต่ตัดพิกเซลที่เข้มที่สุด (เส้นคลื่น) ทิ้ง
    """
    a = img_bgr.astype(np.int16)
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    chroma = (mx - mn).astype(np.float32)
    if float(np.percentile(chroma, 99.5)) >= 25.0:
        return chroma.mean(axis=0)

    ink = 255.0 - cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    thr = float(np.percentile(ink, 98.0))
    ink = np.where(ink > thr, 0.0, ink)
    return ink.mean(axis=0)


def estimate_px_per_mm(img_bgr: np.ndarray, min_px: int = 3, max_px: int = 40,
                       rel: float = 0.35) -> Optional[float]:
    """ประมาณ px ต่อ 1 mm จากคาบเส้นกริด (ช่องเล็ก = 1 mm)

    ใช้ autocorrelation ไม่ใช่ FFT เพราะสเปกตรัมของเส้นหวีมี harmonic เต็มไปหมด
    จนแยกไม่ออกว่าคาบไหนของจริง และมักไปล็อกกับเส้นกริดหลัก (ทุก 5 mm)
    ทำให้ได้ค่าผิดไป 5 เท่า autocorrelation มียอดเฉพาะที่คาบจริงกับพหุคูณของมัน
    จึงเลือก "คาบสั้นที่สุดที่ยอดแรงพอ" = ช่องเล็ก 1 mm ได้ตรง
    """
    prof = grid_profile(img_bgr)
    n = prof.size
    if n < 4 * max_px:
        return None
    # ตัดเฉพาะความสว่างพื้นหลังที่ค่อย ๆ เปลี่ยน — sigma ต้องใหญ่กว่าคาบกริดที่สนใจมาก
    # ไม่งั้นกริดที่ห่าง (เช่น 25 px) จะถูกกรองทิ้งไปด้วย แล้วไปจับสัญญาณรบกวนแทน
    hp_sigma = max(15.0, 2.0 * max_px)
    prof = prof - cv2.GaussianBlur(prof.reshape(1, -1), (0, 0), hp_sigma).ravel()
    prof = prof - prof.mean()

    f = np.fft.rfft(prof, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]

    lo, hi = int(min_px), int(min(max_px, n // 4))
    seg = ac[lo:hi + 1]
    if seg.size < 3:
        return None
    pk = [i for i in range(1, seg.size - 1) if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1]]
    if not pk:
        return None
    top = max(seg[i] for i in pk)
    i = next((i for i in pk if seg[i] >= rel * top), max(pk, key=lambda i: seg[i]))

    y0, y1, y2 = seg[i - 1], seg[i], seg[i + 1]        # ปรับละเอียดระดับ sub-pixel
    d = y0 - 2 * y1 + y2
    off = 0.5 * (y0 - y2) / d if abs(d) > 1e-9 else 0.0
    return float(lo + i + float(np.clip(off, -0.5, 0.5)))


def check_scale(px_per_mm: Optional[float], pitch_px: Optional[float],
                paper_speed: float = 25.0, lo_bpm: float = 20.0,
                hi_bpm: float = 400.0) -> Tuple[Optional[float], bool]:
    """ตรวจความสมเหตุสมผลของ px/mm ด้วยอัตราการเต้นที่คำนวณได้ คืน (hr, ok)"""
    if not px_per_mm or not pitch_px:
        return None, False
    rr_sec = pitch_px / px_per_mm / paper_speed
    if rr_sec <= 0:
        return None, False
    hr = 60.0 / rr_sec
    return hr, (lo_bpm <= hr <= hi_bpm)


def resolve_px_per_mm(raw: np.ndarray, cfg) -> Optional[float]:
    """ค่าที่ตั้งเองมาก่อน ถ้าไม่ได้ตั้งจึงประมาณจากเส้นกริด"""
    if cfg.px_per_mm:
        return float(cfg.px_per_mm)
    return estimate_px_per_mm(raw) if cfg.auto_px_per_mm else None
