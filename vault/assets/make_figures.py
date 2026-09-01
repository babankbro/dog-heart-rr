# -*- coding: utf-8 -*-
"""สร้างภาพประกอบสำหรับ vault — รันซ้ำได้เมื่อค่าตั้งเปลี่ยน

    docker compose run --rm --no-deps -T web python - < vault/assets/make_figures.py

ป้ายกำกับในภาพเป็นภาษาอังกฤษ เพราะ OpenCV วาดอักษรไทยไม่ได้ คำอธิบายภาษาไทยอยู่ในโน้ต
ภาพถูกเขียนลง out/figures แล้วคัดลอกเข้า vault/assets ภายหลัง
"""
import os
import numpy as np
import cv2

from ekg_rpeak.config import Config
from ekg_rpeak.imageio import imread_u
from ekg_rpeak.preprocess import (CROP_PRE_MODES, POINT_PRE_MODES, crop_preprocess,
                                  find_r_anchor, point_preprocess)
from ekg_rpeak.geometry import expected_center, row_pitch, square_crop
from ekg_rpeak.pipeline import detect_r_peaks, load_models
from ekg_rpeak.render import draw_mask_panel, draw_overlay
from ekg_rpeak.scale import resolve_px_per_mm

OUT = 'out/figures'
FONT = cv2.FONT_HERSHEY_SIMPLEX
IMAGE = 'data/661627/661627 methung 1.jpg'
REGION = (1380, 1820)          # ช่วงที่ยอด R พาดทับเส้นกริดหลัก


def label(img, text, h=32, scale=0.62):
    bar = np.full((h, img.shape[1], 3), 248, np.uint8)
    cv2.putText(bar, text, (8, int(h * 0.7)), FONT, scale, (0, 0, 0), 2)
    return np.vstack([bar, img])


def frame(img, pad=2):
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(180, 180, 180))


def title_bar(width, text, h=44, scale=0.85):
    bar = np.full((h, width, 3), 255, np.uint8)
    cv2.putText(bar, text, (10, int(h * 0.68)), FONT, scale, (0, 0, 0), 2)
    return bar


def stack(panels, title):
    w = max(p.shape[1] for p in panels)
    out = [title_bar(w, title)]
    for p in panels:
        if p.shape[1] < w:
            p = cv2.copyMakeBorder(p, 0, 0, 0, w - p.shape[1], cv2.BORDER_CONSTANT,
                                   value=(255, 255, 255))
        out += [p, np.full((10, w, 3), 255, np.uint8)]
    return np.vstack(out)


def save(name, img):
    os.makedirs(OUT, exist_ok=True)
    cv2.imwrite(os.path.join(OUT, name), img)
    print(f'  {name}  {img.shape[1]}x{img.shape[0]}')


def fig_crop_pre(raw):
    """เทียบวิธี binarization ทุกแบบบนช่วงเดียวกัน"""
    x1, x2 = REGION
    sub = raw[:, x1:x2]
    panels = [label(frame(sub), 'original  (red grid on paper)')]
    for mode in CROP_PRE_MODES:
        cfg = Config().with_(crop_pre=mode)
        out = crop_preprocess(sub, cfg)
        ink = (cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) < 128).mean()
        tag = f'crop_pre = {mode}   ink={ink:.3f}'
        if mode == Config().crop_pre:
            tag += '   <-- default'
        panels.append(label(frame(out), tag))
    return stack(panels, 'Binarization before the crop model  -  same region, one method per row')


def fig_point_pre(raw, cfg, det):
    """เทียบวิธีเตรียมครอปก่อนหาจุด R บนจังหวะชุดเดียวกัน"""
    main = det.get('main_row', 0)
    row = det['rows'][main]
    pitch = row_pitch(row)
    px_mm = resolve_px_per_mm(raw, cfg)
    S, N = 220, 4
    panels = []
    for mode in POINT_PRE_MODES:
        c = cfg.with_(point_pre=mode)
        tiles = []
        for box in row[:N]:
            sq, _ = square_crop(raw, box, c, pitch=pitch, px_per_mm=px_mm)
            t = (cv2.resize(point_preprocess(sq, c), (S, S)) if sq is not None
                 else np.full((S, S, 3), 255, np.uint8))
            ex, ey = expected_center(c)
            cv2.drawMarker(t, (int(ex * S / c.out_size), int(ey * S / c.out_size)),
                           (60, 200, 60), cv2.MARKER_CROSS, 18, 2)
            tiles.append(frame(t))
        tag = f'point_pre = {mode}'
        if mode == Config().point_pre:
            tag += '   <-- default'
        panels.append(label(np.hstack(tiles), tag))
    return stack(panels, 'Crop preprocessing before the landmark model  (green + = trained R position)')


def fig_pipeline(path, raw, cfg, det):
    """ไล่ทีละขั้นของไปป์ไลน์บนภาพเดียวกัน"""
    x1, x2 = REGION
    main = det.get('main_row', 0)
    row = det['rows'][main]
    pitch = row_pitch(row)
    px_mm = resolve_px_per_mm(raw, cfg)
    S = 240
    panels = []

    panels.append(label(frame(raw[:, x1:x2]), '1. input  -  scanned EKG paper'))
    panels.append(label(frame(crop_preprocess(raw[:, x1:x2], cfg)),
                        f'2. binarization ({cfg.crop_pre})  -  input to the crop model'))

    boxed = draw_overlay(det, boxes=True, marks=False, landmarks=False, origin=False)
    panels.append(label(frame(boxed[:, x1:x2]), '3. crop model  -  one box per beat'))

    box = next((b for b in row if x1 <= b[0] <= x2), row[0])
    sq, _ = square_crop(raw, box, cfg, pitch=pitch, px_per_mm=px_mm)
    tiles = []
    if sq is not None:
        tiles.append(label(frame(cv2.resize(sq, (S, S))), '4. square crop'))
        t = cv2.resize(point_preprocess(sq, cfg), (S, S))
        ex, ey = expected_center(cfg)
        cv2.drawMarker(t, (int(ex * S / cfg.out_size), int(ey * S / cfg.out_size)),
                       (60, 200, 60), cv2.MARKER_CROSS, 18, 2)
        tiles.append(label(frame(t), f'5. point_pre ({cfg.point_pre})'))
    a, dbg = find_r_anchor(raw, box, cfg, return_debug=True)
    if dbg is not None:
        panel = draw_mask_panel(dbg, peaks=det['peaks'], landmarks=(), anchor=a, size=S)
        tiles.append(label(frame(panel), '6. ink mask  -  anchor from image processing'))
    if tiles:
        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                                    value=(255, 255, 255)) for t in tiles]
        panels.append(np.hstack(tiles))

    final = draw_overlay(det, boxes=True, marks=True, landmarks=False, origin=True, grid=True)
    panels.append(label(frame(final[:, x1:x2]),
                        '7. merged result  -  red = model confirmed, orange = anchor'))
    return stack(panels, 'Processing steps, one image')


def fig_r_position(cfg, models):
    """ตำแหน่งของยอด R ภายในครอป วาดเป็นฮิสโทแกรมแนวตั้ง"""
    from ekg_rpeak.imageio import list_images
    ys = []
    for p in list_images('data')[::3]:
        raw = imread_u(p)
        det = detect_r_peaks(p, models, cfg)
        main = det.get('main_row', 0)
        rows = det['rows']
        if not rows or main >= len(rows):
            continue
        pitch = row_pitch(rows[main])
        px_mm = resolve_px_per_mm(raw, cfg)
        peaks = [q for q in det['peaks'] if q['row'] == main]
        for box in rows[main]:
            x1, y1, x2, y2 = [float(v) for v in box]
            inside = [q for q in peaks if x1 <= q['x'] <= x2]
            if not inside:
                continue
            sq, m = square_crop(raw, box, cfg, pitch=pitch, px_per_mm=px_mm)
            if sq is None or m is None:
                continue
            cy = (inside[0]['y'] - m['Y1'] + m['oy']) / m['sy']
            if 0 <= cy <= cfg.out_size:
                ys.append(cy / cfg.out_size)
    if not ys:
        return None

    W, H, pad = 900, 380, 60
    img = np.full((H, W, 3), 255, np.uint8)
    hist, edges = np.histogram(ys, bins=20, range=(0, 1))
    bw = (W - 2 * pad) / len(hist)
    top = max(hist.max(), 1)
    for i, c in enumerate(hist):
        x = int(pad + i * bw)
        h = int((H - 2 * pad) * c / top)
        cv2.rectangle(img, (x + 1, H - pad - h), (int(x + bw) - 1, H - pad), (90, 140, 220), -1)
    cv2.line(img, (pad, H - pad), (W - pad, H - pad), (60, 60, 60), 2)
    for frac, name in ((cfg.train_anchor_yfrac, 'configured'), (float(np.median(ys)), 'measured')):
        x = int(pad + frac * (W - 2 * pad))
        cv2.line(img, (x, pad), (x, H - pad), (0, 0, 200), 2)
        cv2.putText(img, f'{name} {frac:.3f}', (x + 6, pad + 20), FONT, 0.55, (0, 0, 200), 2)
    cv2.putText(img, '0 = top of crop', (pad, H - pad + 26), FONT, 0.55, (60, 60, 60), 1)
    cv2.putText(img, '1 = bottom', (W - pad - 130, H - pad + 26), FONT, 0.55, (60, 60, 60), 1)
    return stack([img], f'Where the R peak sits inside the crop  (n={len(ys)} beats)')


def main():
    cfg = Config()
    models = load_models(cfg)
    raw = imread_u(IMAGE)
    det = detect_r_peaks(IMAGE, models, cfg)
    print('creating figures...')
    save('fig-crop-pre.png', fig_crop_pre(raw))
    save('fig-point-pre.png', fig_point_pre(raw, cfg, det))
    save('fig-pipeline.png', fig_pipeline(IMAGE, raw, cfg, det))
    f = fig_r_position(cfg, models)
    if f is not None:
        save('fig-r-position.png', f)
    print('done ->', OUT)


main()
