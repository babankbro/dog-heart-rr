"""Preprocessing ของทั้งสองโมเดล และการหายอด R ด้วย image processing

โมเดลครอปเทรนบนภาพ Blackhat ส่วนโมเดลจุดเทรนบนครอปภาพเทาที่ยังมีเส้นกริด
อินพุตของแต่ละตัวจึงต้องอยู่คนละโดเมนกัน
"""
from typing import Tuple

import cv2
import numpy as np

from .config import Config


CROP_PRE_MODES = ('blackhat', 'blackhat_otsu', 'tophat_gray', 'tophat_red',
                  'adaptive', 'ink', 'red')


def blackhat_preprocess(img_bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """ใช้กับโมเดลครอปเท่านั้น — ค่า ksize/thr ต้องตรงกับสคริปต์ที่ใช้เทรน"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.blackhat_ksize,) * 2)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, t = cv2.threshold(blackhat, cfg.blackhat_thr, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(cv2.bitwise_not(t), cv2.COLOR_GRAY2BGR)


def _black_on_white(mask: np.ndarray) -> np.ndarray:
    """mask ของเส้น (True = เส้น) -> ภาพ BGR เส้นดำพื้นขาว แบบเดียวกับที่โมเดลครอปเห็น"""
    out = np.full(mask.shape, 255, np.uint8)
    out[mask] = 0
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def hysteresis(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """เก็บกลุ่มพิกเซลของ weak เฉพาะกลุ่มที่แตะ strong

    ใช้กับเส้นที่ปลายจาง: แกนเส้นผ่านเกณฑ์เข้มอยู่แล้ว ส่วนปลายผ่านแค่เกณฑ์อ่อน
    การต่อยอดจากแกนจึงได้ปลายกลับมาโดยไม่ลากกระดาษหรือกริดที่จางพอกันเข้ามาด้วย
    """
    if not strong.any():
        return strong
    n, lab = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[np.unique(lab[strong])] = True
    keep[0] = False                       # พื้นหลัง
    return keep[lab]


def _join(mask: np.ndarray, cfg: Config) -> np.ndarray:
    """เชื่อมเส้นที่ขาดเป็นช่วง ๆ แล้วทำให้หนาขึ้น

    เส้นคลื่นที่สแกนมาจางบางช่วง binarization จึงทำให้เส้นขาดเป็นท่อน ๆ
    โมเดลที่เทรนบนเส้นต่อเนื่องจะอ่านเส้นขาดไม่ออก
    """
    out = mask.astype(np.uint8)
    if cfg.crop_pre_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.crop_pre_close,) * 2)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    if cfg.crop_pre_dilate > 0:
        out = cv2.dilate(out, np.ones((cfg.crop_pre_dilate,) * 2, np.uint8))
    return out.astype(bool)


def crop_preprocess(img_bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """ภาพที่ป้อนโมเดลครอป เลือกวิธี binarization ได้ด้วย cfg.crop_pre

    'blackhat' คือวิธีที่ weights ชุดนี้เห็นตอนเทรน ตัวเลือกอื่นมีไว้เปรียบเทียบ
    ว่าวิธีไหนเก็บปลายยอด R ที่หมึกจางไว้ได้ — วิธีเดิมตัดทิ้ง กล่องจึงคร่อมยอดไม่มิด
    การเปลี่ยนค่านี้คือการเปลี่ยนโดเมนที่โมเดลเห็น ต้องวัดผลก่อนใช้จริงเสมอ

    blackhat      blackhat + threshold คงที่ (ตามชุดเทรน)
    blackhat_otsu blackhat + Otsu — เกณฑ์ปรับตามภาพเอง
    tophat_gray   ลบพื้นหลังด้วย opening แล้ว threshold ตามคอนทราสต์ของภาพนั้น
    adaptive      adaptive threshold บนภาพเทา — ทนต่อความสว่างไม่สม่ำเสมอ
    ink           mask เดียวกับที่โมเดลจุดใช้ (ตัดกริดด้วยสี ไม่ใช่ความเข้ม)
    red           ใช้แชนเนลแดงอย่างเดียว กริดหมึกแดงจึงเกือบหายไปเอง

    crop_pre_close / crop_pre_dilate ใช้ต่อท้ายทุกโหมดยกเว้น 'blackhat'
    สำหรับเชื่อมเส้นที่ขาดเป็นช่วง ๆ และทำให้เส้นหนาขึ้นก่อนส่งเข้าโมเดล
    """
    mode = cfg.crop_pre
    if mode == 'blackhat':
        return blackhat_preprocess(img_bgr, cfg)   # ตรงกับชุดเทรน ห้ามแต่งเพิ่ม

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if mode == 'blackhat_otsu':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.blackhat_ksize,) * 2)
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, t = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return _black_on_white(_join(t > 0, cfg))
    if mode in ('tophat_gray', 'tophat_red'):
        # closing ด้วยแกนใหญ่ = ภาพพื้นหลัง (กระดาษ + กริด) ส่วนที่เข้มกว่าพื้นหลังคือเส้น
        #
        # ทำบนแชนเนลแดงได้ด้วย: กริดพิมพ์หมึกแดงจึงสว่างเกือบเท่ากระดาษในแชนเนลนั้น
        # พื้นหลังที่ต้องลบจึงเรียบกว่า เส้นคลื่นที่พาดทับเส้นกริดพอดีไม่ถูกกริดกลบ
        base = img_bgr[:, :, 2] if mode == 'tophat_red' else gray
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.crop_pre_ksize,) * 2)
        bg = cv2.morphologyEx(base, cv2.MORPH_CLOSE, k)
        diff = cv2.subtract(bg, base)
        return _black_on_white(_join(diff > cfg.crop_pre_thr, cfg))
    if mode == 'adaptive':
        t = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, cfg.crop_pre_block | 1, cfg.crop_pre_c)
        return _black_on_white(_join(t == 0, cfg))
    if mode == 'ink':
        return _black_on_white(_join(keep_trace(ink_mask(img_bgr, cfg), cfg), cfg))
    if mode == 'red':
        # แชนเนลแดง: กริดหมึกแดงสว่างเกือบเท่ากระดาษ เส้นคลื่นสีดำยังเข้ม
        red = img_bgr[:, :, 2]
        # cv2 นับ foreground เป็น > t เส้นจึงเป็น <= t — ใช้ < จะพลาดทั้งเส้น
        # เมื่อภาพมีระดับสีน้อยจน Otsu ตกลงบนค่าของเส้นพอดี
        t, _ = cv2.threshold(red, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = red <= t
        if cfg.crop_pre_hyst > 0:
            paper = float(np.percentile(red, 90))
            m = hysteresis(m, red <= t + cfg.crop_pre_hyst * (paper - t))
        return _black_on_white(_join(keep_trace(m, cfg), cfg))
    raise ValueError(f'crop_pre ไม่รู้จัก: {mode!r}')


POINT_PRE_MODES = ('ink', 'red_ink', 'red_contrast', 'gray', 'gray_contrast', 'none')


def stretch(g: np.ndarray) -> np.ndarray:
    """ดึงคอนทราสต์ให้เต็มช่วง โดยตัดหางบน-ล่างอย่างละ 1% กันพิกเซลหลุดลากสเกล"""
    lo, hi = np.percentile(g, (1, 99))
    if hi - lo <= 5:
        return g
    return np.clip((g.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def red_ink_mask(bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """ink mask ที่หากริดด้วยแชนเนลสีแทน saturation

    ink_mask ตัดกริดด้วย saturation ซึ่งพลาดเมื่อกริดจางจนสีอ่อน แชนเนลแดงตัดกริด
    ได้ตรงกว่าเพราะกริดพิมพ์ด้วยหมึกแดง แล้วค่อยดึงคอนทราสต์ก่อน threshold
    เพื่อให้ปลายยอดที่หมึกจางไม่หลุด
    """
    red = stretch(bgr[:, :, 2])
    t, _ = cv2.threshold(red, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return keep_trace(red <= t, cfg)


def point_preprocess(bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """ทำให้ครอปหน้าตาเหมือนชุดเทรนโมเดลจุด: กริดหาย เส้นคลื่นดำ พื้นขาว"""
    mode = cfg.point_pre
    if mode == 'none':
        return bgr
    if mode in ('ink', 'red_ink'):
        # ลบพื้นหลังกริดออกทั้งหมด เหลือเส้นคลื่นดำบนพื้นขาว
        # ใช้ได้ก็ต่อเมื่อชุดเทรนถูกทำแบบเดียวกัน ไม่งั้นจะหลุดโดเมน
        m = (red_ink_mask(bgr, cfg) if mode == 'red_ink'
             else keep_trace(ink_mask(bgr, cfg), cfg))
        out = np.full_like(bgr, 255)
        out[m] = 0
        return out
    g = bgr[:, :, 2] if mode == 'red_contrast' else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if mode in ('gray_contrast', 'red_contrast'):
        g = stretch(g)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def mask_quality(m: np.ndarray) -> Tuple[float, float]:
    """คืน (สัดส่วนคอลัมน์ที่มีหมึก, สัดส่วนพิกเซลที่เป็นหมึก)"""
    if m.size == 0:
        return 0.0, 0.0
    return float(m.any(axis=0).mean()), float(m.mean())


def ink_mask(bgr: np.ndarray, cfg: Config, return_thr: bool = False):
    """เส้นคลื่น = เข้ม + ไม่มีสี

    การตัดกริดด้วยความเข้มอย่างเดียวใช้ไม่ได้ เพราะเส้นกริดหลัก (ทุก 5 mm)
    เข้มพอ ๆ กับเส้นคลื่น แต่มันมีสีเสมอ จึงแยกด้วย saturation แทน

    ถ้าครอปสว่างหรือเส้นจางจน mask แทบว่าง จะปรับ threshold เองด้วย Otsu
    แล้วไล่เพิ่มทีละขั้นจนกว่าจะได้เส้นต่อเนื่อง และหยุดถ้ากระดาษเริ่มกลายเป็นหมึก
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v, sat = hsv[:, :, 2], hsv[:, :, 1]
    neutral = sat < cfg.ink_sat_max

    def build(th):
        m = neutral & (v < th)
        if cfg.ink_drop_full_cols and m.shape[0] > 4:
            m = m.copy()
            m[:, m.sum(axis=0) > 0.97 * m.shape[0]] = False   # เส้นกริดตั้งที่พาดเต็มความสูง
        return m

    def good(m):
        cols, frac = mask_quality(m)
        return cols >= cfg.ink_min_col_frac and frac <= cfg.ink_max_frac

    m, thr = build(cfg.ink_dark_v), float(cfg.ink_dark_v)
    if not cfg.ink_adaptive or good(m):
        return (m, thr) if return_thr else m

    # 1) Otsu เฉพาะพิกเซลสีกลาง — ปรับตามความสว่างของครอปนั้นเอง
    vals = v[neutral]
    if vals.size > 50:
        t = float(cv2.threshold(vals.reshape(-1, 1), 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
        if cfg.ink_dark_v < t <= cfg.ink_dark_v_max:
            cand = build(t)
            if good(cand):
                return (cand, t) if return_thr else cand

    # 2) ไล่เพิ่มเพดานทีละขั้นจนกว่าจะได้เส้นต่อเนื่อง
    best, best_thr, best_cols = m, thr, mask_quality(m)[0]
    for t in range(int(cfg.ink_dark_v) + 10, int(cfg.ink_dark_v_max) + 1, 10):
        cand = build(t)
        cols, frac = mask_quality(cand)
        if frac > cfg.ink_max_frac:
            break
        if cols > best_cols:
            best, best_thr, best_cols = cand, float(t), cols
        if good(cand):
            return (cand, float(t)) if return_thr else cand
    return (best, best_thr) if return_thr else best


def keep_trace(mask: np.ndarray, cfg: Config) -> np.ndarray:
    """เก็บเฉพาะกลุ่มพิกเซลที่ใหญ่พอ

    เส้นคลื่นเป็นเส้นต่อเนื่องผืนเดียว ส่วนจุดของเส้นกริดแบบประที่รอดตัวกรองสีมาได้
    เป็นจุดเล็ก ๆ แยกกัน ถ้าไม่ตัดทิ้ง จุดที่บังเอิญอยู่แถวบนสุดจะถูกเข้าใจผิดว่าเป็นยอด R
    """
    if not cfg.ink_keep_trace_only:
        return mask
    m8 = mask.astype(np.uint8)
    if m8.sum() == 0:
        return mask
    closed = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    if n <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(n, dtype=bool)
    keep[1:] = areas >= max(1.0, cfg.ink_min_area_frac * float(areas.max()))
    return keep[lab] & mask


def solid_ink(mask: np.ndarray, cfg: Config) -> np.ndarray:
    """เหลือเฉพาะหมึกที่ต่อเนื่องในแนวตั้งอย่างน้อย ink_min_run พิกเซล

    ขาขึ้นของ QRS เป็นเส้นยาว ส่วนจุดรบกวนสั้น ๆ จะหายไป
    """
    k = int(cfg.ink_min_run)
    if k <= 1:
        return mask
    er = cv2.erode(mask.astype(np.uint8), np.ones((k, 1), np.uint8))
    return er.astype(bool) if er.any() else mask


def beat_shape(raw: np.ndarray, box, cfg: Config):
    """วัดรูปร่างของเส้นในกล่องหนึ่ง คืน dict หรือ None ถ้าไม่มีเส้นให้วัด

    amp   = ระยะเบี่ยงจากเส้นฐานมากที่สุด (ความสูงของยอด)
    width = จำนวนคอลัมน์ที่เบี่ยงเกินครึ่งหนึ่งของ amp (ความกว้างที่ครึ่งความสูง)

    ยอด R เป็นเส้นแหลม width จึงแคบมาก ส่วน calibration pulse เป็นสี่เหลี่ยม
    มียอดแบนกว้าง width จะกว้างกว่าหลายเท่า
    """
    H, W = raw.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    X1, X2 = max(0, x1), min(W, x2)
    Y1, Y2 = max(0, y1), min(H, y2)
    if X2 - X1 < 3 or Y2 - Y1 < 3:
        return None

    mask = keep_trace(ink_mask(raw[Y1:Y2, X1:X2], cfg), cfg)
    has = mask.any(axis=0)
    cols_frac, _ = mask_quality(mask)
    if not has.any():
        return None

    h = mask.shape[0]
    top = np.full(mask.shape[1], np.nan)
    bot = np.full(mask.shape[1], np.nan)
    top[has] = np.argmax(mask[:, has], axis=0)
    bot[has] = h - 1 - np.argmax(mask[::-1][:, has], axis=0)
    base = np.nanmedian((top + bot) / 2.0)
    dev = np.fmax(base - top, bot - base)
    amp = float(np.nanmax(dev))
    if not np.isfinite(amp) or amp <= 0:
        return None
    width = int(np.nansum(dev >= 0.5 * amp))
    peak_at = float(np.nanargmax(dev)) / max(1, mask.shape[1] - 1)
    return {'amp': amp, 'width': width, 'cols_frac': cols_frac,
            'peak_at': peak_at, 'box_w': X2 - X1}


def is_beat_like(shape, ref, cfg: Config) -> bool:
    """กล่องนี้หน้าตาเหมือนจังหวะจริงไหม เทียบกับจังหวะอื่นในแถวเดียวกัน"""
    if shape is None:
        return False
    if shape['cols_frac'] < cfg.anchor_min_col_frac:
        return False                                        # เส้นขาด ๆ ไม่ใช่คลื่นเต็มจังหวะ
    if shape['amp'] < cfg.edge_amp_ratio * ref['amp']:
        return False                                        # ยอดเตี้ยผิดปกติ
    if ref['width'] > 0 and shape['width'] > cfg.edge_width_ratio * ref['width']:
        return False                                        # ยอดแบนกว้าง เช่น calibration pulse
    if shape['box_w'] < cfg.edge_min_box_ratio * ref['box_w']:
        return False                                        # กล่องแคบผิดปกติ = เศษที่ขอบภาพ
    m = cfg.edge_peak_margin
    if not (m <= shape['peak_at'] <= 1 - m):
        return False                                        # ยอดไปเกาะขอบกล่อง ไม่ได้อยู่กลาง
    return True


def drop_edge_non_beats(raw: np.ndarray, row, cfg: Config):
    """ตัดกล่องหัวและท้ายแถวที่รูปร่างไม่ใช่คลื่น คืน (กล่องที่เหลือ, จำนวนที่ตัด)

    ตัดเฉพาะสองปลายเท่านั้น กล่องตรงกลางไม่แตะ เพราะจังหวะผิดปกติกลางแถว
    เป็นข้อมูลที่ต้องเก็บไว้ ไม่ใช่สิ่งแปลกปลอม
    """
    if not cfg.drop_edge_non_beats or len(row) < 4:
        return list(row), 0

    shapes = [beat_shape(raw, b, cfg) for b in row]
    inner = [s for s in shapes[1:-1] if s]
    if len(inner) < 2:
        return list(row), 0
    ref = {k: float(np.median([s[k] for s in inner])) for k in ('amp', 'width', 'box_w')}

    keep = list(range(len(row)))
    while keep and not is_beat_like(shapes[keep[0]], ref, cfg):
        keep.pop(0)
    while keep and not is_beat_like(shapes[keep[-1]], ref, cfg):
        keep.pop()
    return [row[i] for i in keep], len(row) - len(keep)


def find_r_anchor(raw: np.ndarray, box, cfg: Config, return_debug: bool = False):
    """หายอด R ในกล่องจากภาพดิบ รองรับทั้ง R ขึ้นและ R ลง

    คืน (x, y) หรือ None เมื่อ mask ได้เส้นไม่ต่อเนื่องพอจนเชื่อไม่ได้
    """
    H, W = raw.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    ex = int((x2 - x1) * cfg.anchor_expand)
    ey = int((y2 - y1) * cfg.anchor_expand_y)
    X1, X2 = max(0, x1 - ex), min(W, x2 + ex)
    Y1, Y2 = max(0, y1 - ey), min(H, y2 + ey)
    if X2 - X1 < 3 or Y2 - Y1 < 3:
        return (None, None) if return_debug else None

    roi = raw[Y1:Y2, X1:X2]
    mask, thr = ink_mask(roi, cfg, return_thr=True)
    mask = keep_trace(mask, cfg)
    cols_frac, ink_frac = mask_quality(mask)
    dbg = {'roi': roi, 'mask': mask, 'X1': X1, 'Y1': Y1,
           'thr': thr, 'cols_frac': cols_frac, 'ink_frac': ink_frac}

    has = mask.any(axis=0)
    if not has.any() or cols_frac < cfg.anchor_min_col_frac:
        return (None, dbg) if return_debug else None

    # ตำแหน่งยอดคิดจากหมึกที่ต่อเนื่องเท่านั้น แต่เส้นฐานคิดจาก mask เต็ม
    solid = solid_ink(mask, cfg)
    pad = int(cfg.ink_min_run) // 2
    h = mask.shape[0]
    top = np.full(mask.shape[1], np.nan)
    bot = np.full(mask.shape[1], np.nan)
    has_s = solid.any(axis=0)
    if not has_s.any():
        solid, has_s, pad = mask, has, 0
    top[has_s] = np.clip(np.argmax(solid[:, has_s], axis=0) - pad, 0, h - 1)
    bot[has_s] = np.clip(h - 1 - np.argmax(solid[::-1][:, has_s], axis=0) + pad, 0, h - 1)
    base = np.nanmedian((top + bot) / 2.0)
    up, dn = base - top, bot - base
    if np.nanmax(up) >= np.nanmax(dn):
        j = int(np.nanargmax(up)); ry = top[j]
    else:
        j = int(np.nanargmax(dn)); ry = bot[j]

    a = (float(X1 + j), float(Y1 + ry))
    dbg.update({'top': top, 'base': base})
    return (a, dbg) if return_debug else a
