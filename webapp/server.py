"""FastAPI backend สำหรับดูผลตรวจจับ R peak

เก็บผลล่าสุดของแต่ละภาพไว้ในหน่วยความจำ เพื่อให้ overlay / ครอป / CSV
ใช้ผลเดียวกับที่หน้าเว็บเพิ่งรัน โดยไม่ต้องรันโมเดลซ้ำ
(ออกแบบสำหรับใช้งานเครื่องเดียว ไม่ได้ทำ session แยกผู้ใช้)

ผลทุกครั้งที่คำนวณจริงถูกเขียนลงดิสก์ด้วย (`ekg_rpeak/results.py`) หน่วยความจำ
จึงเป็นแค่ชั้นหน้า รีสตาร์ตแล้วผลยังอยู่ครบ หน้าเว็บโหลดกลับมาเองโดยไม่ต้องกดวิเคราะห์
ภาพต้นฉบับไม่ถูกเก็บลงดิสก์ซ้ำ และถูกถือไว้ในหน่วยความจำแค่ไม่กี่ภาพล่าสุด
"""
import json
import os
import shutil
import threading
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from ekg_rpeak.config import Config
from ekg_rpeak.export import FIELDS, median_hr, result_to_rows, write_csv
from ekg_rpeak.geometry import expected_center, row_pitch, square_crop
from ekg_rpeak.imageio import imread_u, list_images
from ekg_rpeak import patients as pt
from ekg_rpeak.pipeline import detect_r_peaks, load_models
from ekg_rpeak import results as rs
from ekg_rpeak import rrstats
from ekg_rpeak.preprocess import CROP_PRE_MODES, crop_preprocess, point_preprocess
from ekg_rpeak.render import draw_mask_panel, draw_overlay
from ekg_rpeak.scale import check_scale, resolve_px_per_mm

DATA_DIR = os.getenv('EKG_DATA_DIR', 'data')
OUT_DIR = os.getenv('EKG_OUT_DIR', 'out')
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

# ค่าที่หน้าเว็บปรับได้ (จำกัดไว้เพื่อไม่ให้ตั้งค่าที่ทำให้ระบบพัง)
ALLOWED = {
    'point_mode': str, 'crop_mode': str, 'point_pre': str,
    'r_class_id': int, 'crop_conf': float, 'crop_iou': float,
    'point_conf': float, 'point_iou': float,
    'px_per_mm': float, 'train_px_per_mm': float,
    'train_frame_w_ratio': float, 'train_frame_h_ratio': float,
    'max_refine_ratio': float, 'trust_model_conf': float,
    # binarization ของภาพที่ป้อนโมเดลครอป — ปรับได้เพื่อดูว่ากินปลายยอดไปแค่ไหน
    'crop_pre': str, 'blackhat_thr': int,
    'crop_pre_ksize': int, 'crop_pre_thr': int,          # knob ของ tophat_gray
    'crop_pre_hyst': float, 'crop_pre_close': int, 'crop_pre_dilate': int,
}

app = FastAPI(title='EKG R-Peak Viewer')
_models = None
_lock = threading.Lock()
_cache: Dict[str, Dict[str, Any]] = {}       # ผลตรวจจับต่อภาพ
_png_cache: Dict[str, bytes] = {}            # ภาพที่วาดแล้ว กันเรนเดอร์ซ้ำตอนสลับแท็บ
_PNG_CACHE_MAX = 96
_RAW_CACHE_MAX = 12                          # จำนวนภาพต้นฉบับที่ถือไว้พร้อมกันได้


def cached_png(key: str, build):
    """คืนภาพจากแคชถ้ามี ไม่งั้นเรนเดอร์แล้วเก็บไว้"""
    hit = _png_cache.get(key)
    if hit is not None:
        return Response(hit, media_type='image/png',
                        headers={'Cache-Control': 'public, max-age=31536000, immutable'})
    img = build()
    ok, buf = cv2.imencode('.png', img)
    if not ok:
        raise HTTPException(500, 'สร้างภาพไม่สำเร็จ')
    data = buf.tobytes()
    if len(_png_cache) >= _PNG_CACHE_MAX:
        for k in list(_png_cache)[:_PNG_CACHE_MAX // 3]:
            _png_cache.pop(k, None)
    _png_cache[key] = data
    return Response(data, media_type='image/png',
                    headers={'Cache-Control': 'public, max-age=31536000, immutable'})


def drop_png_cache(image: str) -> None:
    for k in [k for k in _png_cache if k.startswith(image + '|')]:
        _png_cache.pop(k, None)


def build_config(overrides: Optional[dict]) -> Config:
    cfg = Config()
    mode = (overrides or {}).get('crop_pre')
    if mode not in (None, '') and mode not in CROP_PRE_MODES:
        raise HTTPException(400, f'crop_pre ไม่รู้จัก: {mode!r}')
    for k, v in (overrides or {}).items():
        if k not in ALLOWED or v is None or v == '':
            continue
        try:
            cfg = cfg.with_(**{k: ALLOWED[k](v)})
        except (TypeError, ValueError):
            raise HTTPException(400, f'ค่าของ {k} ไม่ถูกต้อง: {v!r}')
    return cfg


def get_models(cfg: Config):
    global _models
    with _lock:
        if _models is None:
            try:
                _models = load_models(cfg)
            except FileNotFoundError as e:
                raise HTTPException(400, str(e))
        return _models


def resolve_path(name: str) -> str:
    """กันไม่ให้หลุดออกนอกโฟลเดอร์ data"""
    p = os.path.abspath(os.path.join(DATA_DIR, name))
    if not p.startswith(os.path.abspath(DATA_DIR)) or not os.path.isfile(p):
        raise HTTPException(404, f'ไม่พบภาพ: {name}')
    return p


def cached(name: str) -> Dict[str, Any]:
    """ผลของภาพนี้พร้อมภาพต้นฉบับ สำหรับ endpoint ที่ต้องวาดภาพหรือส่ง CSV"""
    hit = cache_hit(name, need_raw=True)
    if hit is None:
        raise HTTPException(409, 'ยังไม่ได้รันภาพนี้ กดปุ่มตรวจจับก่อน')
    return hit


def trim_raw() -> None:
    """ปล่อยภาพต้นฉบับของรายการเก่า เหลือไว้เท่าที่กำหนด

    ผลตัวเลขยังอยู่ครบ ภาพอ่านคืนจากไฟล์เดิมได้เสมอ ถ้าไม่ทำแบบนี้ ชุดข้อมูล
    หลายสิบภาพจะกินหน่วยความจำหลายร้อยเมกะไบต์ทั้งที่ผู้ใช้ดูอยู่ทีละภาพ
    """
    holding = [k for k, v in _cache.items() if v['result'].get('raw') is not None]
    for k in holding[:-_RAW_CACHE_MAX] if len(holding) > _RAW_CACHE_MAX else []:
        _cache[k]['result']['raw'] = None
        drop_png_cache(k)


def record(name: str, path: str, cfg: Config, result: dict, rows: list,
           mtime: float, width: int, height: int) -> Dict[str, Any]:
    """เก็บผลหนึ่งภาพเข้าแคชหน่วยความจำ พร้อมเลขรุ่นสำหรับทำ URL ที่แคชได้"""
    _cache.pop(name, None)                       # ใส่ท้ายเสมอ ลำดับในดิกต์ = ลำดับการใช้
    _cache[name] = {'result': result, 'cfg': cfg, 'rows': rows, 'path': path,
                    'mtime': mtime, 'rev': rs.revision(name, mtime, cfg),
                    'width': width, 'height': height}
    trim_raw()
    return _cache[name]


def from_disk(name: str) -> Optional[Dict[str, Any]]:
    """ผลที่เขียนไว้บนดิสก์ — ใช้ตอนเพิ่งรีสตาร์ต หรือแคชถูกปล่อยไปแล้ว"""
    try:
        path = resolve_path(name)
    except HTTPException:
        return None
    saved = rs.load(OUT_DIR, name, image_path=path)
    if saved is None:
        return None
    return record(name, path, saved['cfg_obj'], saved['result'], saved['rows'],
                  saved['mtime'], saved['width'], saved['height'])


def cache_hit(name: str, cfg: Optional[Config] = None,
              need_raw: bool = False) -> Optional[Dict[str, Any]]:
    """ผลที่จำไว้ของภาพนี้ ถ้ายังใช้ได้ หาในหน่วยความจำก่อน ไม่เจอจึงอ่านจากดิสก์

    ใช้ไม่ได้เมื่อไฟล์ถูกแก้หลังรัน หรือ (ถ้าระบุ cfg) ค่าตั้งไม่ตรงกับตอนที่รัน
    need_raw=True เมื่อผู้เรียกต้องวาดภาพ จะอ่านภาพต้นฉบับคืนให้ถ้ายังไม่ได้ถือไว้
    """
    hit = _cache.get(name)
    if hit is None:
        hit = from_disk(name)
        if hit is None:
            return None
    try:
        if os.path.getmtime(hit['path']) != hit['mtime']:
            return None
    except OSError:                              # ไฟล์ถูกลบไปแล้ว
        return None
    if cfg is not None and hit['cfg'] != cfg:
        return None
    if need_raw and hit['result'].get('raw') is None:
        img = imread_u(hit['path'])
        if img is None:
            return None
        hit['result']['raw'] = img
        _cache[name] = _cache.pop(name)           # เพิ่งใช้ ย้ายไปท้ายแถว
        trim_raw()
    return hit


@app.get('/api/health')
def health():
    cfg = Config()
    return {'ok': True,
            'crop_weights': os.path.exists(cfg.crop_weights),
            'point_weights': os.path.exists(cfg.point_weights),
            'data_dir': DATA_DIR,
            'saved_results': rs.count(OUT_DIR)}


@app.get('/api/config')
def get_config():
    cfg = Config()
    return {k: getattr(cfg, k) for k in ALLOWED}


@app.get('/api/images')
def api_images():
    out = []
    for p in list_images(DATA_DIR):
        img = imread_u(p)
        h, w = img.shape[:2]
        out.append({'name': os.path.relpath(p, DATA_DIR).replace(os.sep, '/'),
                    'w': int(w), 'h': int(h),
                    'ran': os.path.relpath(p, DATA_DIR).replace(os.sep, '/') in _cache})
    return out


@app.post('/api/upload')
async def api_upload(files: List[UploadFile] = File(...)):
    os.makedirs(DATA_DIR, exist_ok=True)
    saved = []
    for f in files:
        name = os.path.basename(f.filename or 'upload.png')
        if not name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        dest = os.path.join(DATA_DIR, name)
        with open(dest, 'wb') as fh:
            fh.write(await f.read())
        if imread_u(dest) is None:
            os.remove(dest)
            continue
        saved.append(name)
    if not saved:
        raise HTTPException(400, 'ไม่มีไฟล์ภาพที่ใช้ได้ (รองรับ .jpg .jpeg .png)')
    return {'saved': saved}


def run_detect(name: str, cfg: Config, force: bool = False):
    """รันตรวจจับหนึ่งภาพ แล้วเก็บผลไว้ใช้ซ้ำ

    ถ้าค่าตั้งและไฟล์ไม่เปลี่ยน จะคืนผลเดิมทันทีโดยไม่รันโมเดลใหม่
    ทำให้การสลับแท็บหรือเปิดภาพเดิมซ้ำไม่ต้องรอประมวลผลอีก
    """
    path = resolve_path(name)
    mtime = os.path.getmtime(path)
    hit = None if force else cache_hit(name, cfg)
    if hit is not None:
        return hit['result'], hit['rows']

    models = get_models(cfg)
    result = detect_r_peaks(path, models, cfg)
    rows = result_to_rows(path, result, cfg)
    h, w = result['raw'].shape[:2]
    rs.save(OUT_DIR, name, path, cfg, result, rows)   # กดวิเคราะห์ครั้งเดียวพอ
    record(name, path, cfg, result, rows, mtime, int(w), int(h))
    drop_png_cache(name)                     # ผลเปลี่ยน ภาพที่วาดไว้ใช้ไม่ได้แล้ว
    return result, rows


def image_summary(name: str, rec: dict) -> dict:
    """สรุปหนึ่งภาพสำหรับหน้ารวมของสัตว์แต่ละตัว

    อ่านจากระเบียนล้วน ไม่แตะภาพต้นฉบับ หน้ารวมของสัตว์หลายภาพจึงไม่ต้อง decode
    ภาพสักใบ
    """
    result, rows, cfg = rec['result'], rec['rows'], rec['cfg']
    s = result['stats']
    rr = result.get('rr', {}).get(result.get('main_row', 0)) or {}
    return {
        'image': name,
        'rev': rec['rev'],
        'beats': s['n_boxes'], 'r_peaks': s['n_peaks'],
        'model': s['n_model'], 'anchor': s['n_anchor'],
        'reject': s['n_reject'], 'dup': s['n_dup'],
        'edge_dropped': s.get('n_edge_dropped', 0),
        'px_per_mm': s['px_per_mm'], 'scale_source': s.get('scale_source'),
        'rr_mean_mm': rr.get('mean_mm'), 'rr_sd_mm': rr.get('sd_mm'),
        'hr': median_hr(result, cfg),
        'flags': sum(1 for r in rows if r['flag']),
        'width': rec['width'], 'height': rec['height'],
    }


@app.post('/api/detect')
def api_detect(payload: dict = Body(...)):
    name = payload.get('image')
    if not name:
        raise HTTPException(400, 'ต้องระบุชื่อภาพ')
    cfg = build_config(payload.get('overrides'))
    if payload.get('cached_only'):
        # ใช้ตอนเรียกหน้าจอของสัตว์ตัวเดิมกลับมา — ห้ามเผลอรันโมเดลใหม่ให้ผู้ใช้รอ
        if cache_hit(name, cfg) is None:
            raise HTTPException(409, 'ยังไม่มีผลที่จำไว้สำหรับภาพนี้')
    else:
        run_detect(name, cfg)
    # ขนาดภาพมาจากระเบียน ไม่ใช่จาก result['raw'] เพราะผลที่โหลดจากดิสก์ยังไม่มีภาพต้นฉบับ
    rec = _cache[name]
    result, rows = rec['result'], rec['rows']

    s = result['stats']
    pitch = (row_pitch(result['rows'][result.get('main_row', 0)])
             if result['rows'] else None)
    hr, scale_ok = check_scale(s['px_per_mm'], pitch, cfg.paper_speed_mm_s)
    return {
        'image': name,
        'rev': rec['rev'],
        'width': rec['width'], 'height': rec['height'],
        'stats': {k: (float(v) if isinstance(v, float) else v) for k, v in s.items()},
        'has_point_model': get_models(cfg).has_point,
        'pitch_px': pitch,
        'hr_from_pitch': hr,
        'scale_ok': bool(scale_ok),
        'median_hr': median_hr(result, cfg),
        'crop_side_px': (cfg.out_size * s['px_per_mm'] / cfg.train_px_per_mm
                         if s['px_per_mm'] else None),
        'peaks': [{'row': p['row'], 'index': p['index'],
                   'x': round(p['x'], 1), 'y': round(p['y'], 1),
                   'conf': round(p['conf'], 3), 'src': p['src'], 'cls': p['cls']}
                  for p in result['peaks']],
        'boxes': [[int(v) for v in b] for b in result['boxes']],
        'landmarks': [{'row': p['row'], 'x': round(p['x'], 1), 'y': round(p['y'], 1),
                       'conf': round(p['conf'], 3), 'cls': p['cls']}
                      for p in result.get('landmarks', [])],
        'rows': rows,
        'fields': FIELDS,
        'grid': ({'spacing_px': round(result['grid']['spacing'], 3),
                  'n_lines': len(result['grid']['lines']),
                  'n_measured': result['grid'].get('n_measured'),
                  'px_per_mm': round(result['grid']['px_per_mm'], 3),
                  'resid_rms_px': (round(result['grid']['resid_rms_px'], 3)
                                   if result['grid'].get('resid_rms_px') is not None else None),
                  'drift_px': (round(result['grid']['drift_px'], 3)
                               if result['grid'].get('drift_px') is not None else None),
                  'mm': cfg.grid_mm} if result.get('grid') else None),
        'origin_px': result.get('origin'),
        'rr': result.get('rr', {}).get(result.get('main_row', 0)),
    }


def _png(img: np.ndarray) -> Response:
    ok, buf = cv2.imencode('.png', img)
    if not ok:
        raise HTTPException(500, 'สร้างภาพไม่สำเร็จ')
    return Response(buf.tobytes(), media_type='image/png')


@app.get('/api/overlay')
def api_overlay(image: str, boxes: int = 1, marks: int = 1, landmarks: int = 1,
                origin: int = 1, grid: int = 0, width: int = 0):
    """ภาพเต็มพร้อมกรอบจังหวะ จุด R และ landmark อื่น ๆ ของโมเดลจุด"""
    c = cached(image)

    def build():
        img = draw_overlay(c['result'], boxes=bool(boxes), marks=bool(marks),
                           landmarks=bool(landmarks), origin=bool(origin), grid=bool(grid))
        if width and 0 < width < img.shape[1]:
            h = max(1, int(round(img.shape[0] * width / img.shape[1])))
            img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
        return img

    key = f"{image}|overlay|{c['rev']}|{boxes}{marks}{landmarks}{origin}{grid}|{width}"
    return cached_png(key, build)


def render_prebin(label: str, path: str, width: int, opts: dict) -> Response:
    """ภาพที่ป้อนโมเดลครอปหลัง binarization — ใช้ร่วมกันทั้งหน้าหลักและหน้าเทียบสองชุด"""
    cfg = build_config(opts)
    key = (f'{label}|prebin|{os.path.getmtime(path)}|{cfg.crop_pre}|{cfg.blackhat_thr}'
           f'|{cfg.crop_pre_ksize}|{cfg.crop_pre_thr}'
           f'|{cfg.crop_pre_hyst}|{cfg.crop_pre_close}|{cfg.crop_pre_dilate}|{width}')

    def build():
        raw = imread_u(path)
        if raw is None:
            raise HTTPException(400, f'เปิดภาพไม่ได้: {label}')
        img = crop_preprocess(raw, cfg)
        if width and 0 < width < img.shape[1]:
            h = max(1, int(round(img.shape[0] * width / img.shape[1])))
            img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
        return img

    return cached_png(key, build)


@app.get('/api/prebin')
def api_prebin(image: str, width: int = 1600, crop_pre: Optional[str] = None,
               blackhat_thr: Optional[int] = None, crop_pre_ksize: Optional[int] = None,
               crop_pre_thr: Optional[int] = None, crop_pre_hyst: Optional[float] = None,
               crop_pre_close: Optional[int] = None, crop_pre_dilate: Optional[int] = None):
    """ภาพที่ป้อนโมเดลครอปจริง ๆ ก่อนตีกรอบ

    ดูได้โดยไม่ต้องรันตรวจจับก่อน เพราะเป็นแค่ preprocessing — ใช้ปรับค่าแล้วเห็นผลทันที
    ว่า binarization เก็บปลายยอด R ไว้ได้แค่ไหน และกริดเหลือมากเกินไปหรือเปล่า
    """
    return render_prebin(image, resolve_path(image), width,
                         {'crop_pre': crop_pre, 'blackhat_thr': blackhat_thr,
                          'crop_pre_ksize': crop_pre_ksize, 'crop_pre_thr': crop_pre_thr,
                          'crop_pre_hyst': crop_pre_hyst, 'crop_pre_close': crop_pre_close,
                          'crop_pre_dilate': crop_pre_dilate})


@app.get('/api/crops')
def api_crops(image: str, n: int = 8, size: int = 220):
    """ครอปที่ป้อนโมเดลจุดจริง ๆ — ใช้เทียบด้วยตากับภาพชุดเทรน"""
    c = cached(image)
    cfg, result = c['cfg'], c['result']
    key = f"{image}|crops|{c['rev']}|{n}|{size}"
    if key in _png_cache:
        return cached_png(key, lambda: None)

    raw = result['raw']
    px_mm = resolve_px_per_mm(raw, cfg)
    rows = result['rows']
    if not rows:
        raise HTTPException(409, 'ไม่พบกล่องจังหวะในภาพนี้')
    main = result.get('main_row', 0)
    pitch = row_pitch(rows[main])
    tiles = []
    for box in rows[main][:max(1, n)]:
        sq, _ = square_crop(raw, box, cfg, pitch=pitch, px_per_mm=px_mm)
        if sq is None:
            continue
        tile = cv2.resize(point_preprocess(sq, cfg), (size, size))
        # ต้องเป็นจุดเดียวกับที่ pick_point ใช้เป็นตำแหน่งคาดหวังของ R ไม่งั้นกากบาท
        # ที่ใช้เทียบกับภาพชุดเทรนชี้ผิดที่ — โหมด train_match เคยถูกวางไว้กลางภาพ
        cx, cy = expected_center(cfg)
        ex = int(cx * size / cfg.out_size)
        ey = int(cy * size / cfg.out_size)
        cv2.drawMarker(tile, (ex, ey), (60, 200, 60), cv2.MARKER_CROSS, 18, 2)
        cv2.rectangle(tile, (0, 0), (size - 1, size - 1), (200, 200, 200), 1)
        tiles.append(tile)
    if not tiles:
        raise HTTPException(409, 'สร้างครอปไม่ได้')
    return cached_png(key, lambda: np.hstack(tiles))


@app.get('/api/mask')
def api_mask(image: str, index: int = 0, size: int = 260, landmarks: int = 1):
    """ROI กับ ink mask ของจังหวะหนึ่ง พร้อมจุดที่ไปป์ไลน์ได้มา

    ใช้ตรวจว่าตำแหน่ง R ตรงกับหมึกที่ mask เห็นจริงไหม และกริดหลุดเข้ามาหรือเปล่า
    """
    from ekg_rpeak.preprocess import find_r_anchor
    c = cached(image)
    cfg, result = c['cfg'], c['result']
    rows = result['rows']
    main = result.get('main_row', 0)
    if not rows or index >= len(rows[main]):
        raise HTTPException(409, 'ไม่มีจังหวะลำดับนี้')

    key = f"{image}|mask|{c['rev']}|{index}|{size}|{landmarks}"
    if key in _png_cache:
        return cached_png(key, lambda: None)

    box = rows[main][index]
    a, dbg = find_r_anchor(result['raw'], box, cfg, return_debug=True)
    if dbg is None:
        raise HTTPException(409, 'ครอปเล็กเกินไป')

    return cached_png(key, lambda: draw_mask_panel(
        dbg,
        peaks=result['peaks'],
        landmarks=result.get('landmarks', []) if landmarks else (),
        anchor=a,
        size=size))


@app.get('/api/csv')
def api_csv(image: str):
    c = cached(image)
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image))[0]
    out = os.path.join(OUT_DIR, f'{stem}_r_peaks.csv')
    write_csv(c['rows'], out)
    return FileResponse(out, media_type='text/csv', filename=os.path.basename(out))


@app.get('/api/patients/{pid}/csv')
def api_patient_csv(pid: str):
    """CSV ของทุกภาพของสัตว์ตัวนี้รวมเป็นไฟล์เดียว

    อ่านจากผลที่คำนวณไว้แล้วเท่านั้น ไม่รันโมเดลใหม่ ภาพที่ยังไม่ได้วิเคราะห์จะถูกข้าม
    คอลัมน์ image บอกอยู่แล้วว่าแต่ละแถวมาจากภาพไหน จึงรวมไฟล์ได้โดยไม่เสียข้อมูล
    """
    p = pt.get_patient(DATA_DIR, pid)
    if not p:
        raise HTTPException(404, f'ไม่พบรหัส {pid}')
    rows, missing = [], []
    for name in p['images']:
        rec = cache_hit(name)
        if rec is None:
            missing.append(name)
            continue
        rows += rec['rows']
    if not rows:
        raise HTTPException(409, 'ยังไม่มีผลของสัตว์ตัวนี้ กด "วิเคราะห์ทั้งตัว" ก่อน')

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f'{pid}_r_peaks.csv')
    write_csv(rows, out)
    return FileResponse(out, media_type='text/csv', filename=os.path.basename(out),
                        headers={'X-Images-Used': str(len(p['images']) - len(missing)),
                                 'X-Images-Missing': str(len(missing))})


# ---------------------------------------------------------------- ทะเบียนสัตว์

@app.get('/api/patients')
def api_patients():
    return pt.list_patients(DATA_DIR)


@app.get('/api/groups')
def api_groups():
    return pt.list_groups(DATA_DIR)


@app.post('/api/patients')
def api_patient_create(payload: dict = Body(...)):
    try:
        return pt.create_patient(DATA_DIR, str(payload.get('id', '')).strip(),
                                 str(payload.get('name', '')), str(payload.get('note', '')),
                                 str(payload.get('group', '')))
    except FileExistsError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.patch('/api/patients/{pid}')
def api_patient_update(pid: str, payload: dict = Body(...)):
    if not pt.get_patient(DATA_DIR, pid):
        raise HTTPException(404, f'ไม่พบรหัส {pid}')
    return pt.update_patient(DATA_DIR, pid, payload.get('name'), payload.get('note'),
                             payload.get('group'))


@app.delete('/api/patients/{pid}')
def api_patient_delete(pid: str, with_images: int = 0):
    if not pt.get_patient(DATA_DIR, pid):
        raise HTTPException(404, f'ไม่พบรหัส {pid}')
    for name in pt.list_patient_images(DATA_DIR, pid):
        _cache.pop(name, None)
        drop_png_cache(name)
        rs.drop(OUT_DIR, name)
    pt.delete_patient(DATA_DIR, pid, with_images=bool(with_images))
    return {'deleted': pid, 'images_removed': bool(with_images)}


@app.post('/api/patients/{pid}/images')
async def api_patient_add_images(pid: str, files: List[UploadFile] = File(...)):
    if not pt.get_patient(DATA_DIR, pid):
        raise HTTPException(404, f'ไม่พบรหัส {pid}')
    saved, failed = [], []
    for f in files:
        try:
            saved.append(pt.add_image(DATA_DIR, pid, f.filename or '', await f.read()))
        except ValueError as e:
            failed.append({'file': f.filename, 'reason': str(e)})
    if not saved:
        raise HTTPException(400, failed[0]['reason'] if failed else 'ไม่มีไฟล์ที่ใช้ได้')
    return {'saved': saved, 'failed': failed}


@app.delete('/api/patients/{pid}/images')
def api_patient_delete_image(pid: str, name: str):
    try:
        pt.delete_image(DATA_DIR, pid, name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    key = f'{pid}/{os.path.basename(name)}'
    _cache.pop(key, None)
    drop_png_cache(key)
    rs.drop(OUT_DIR, key)
    return {'deleted': name}


def aggregate_images(images: List[dict]) -> dict:
    """สรุปรวมทุกภาพของสัตว์หนึ่งตัว"""
    hrs = [i['hr'] for i in images if i['hr']]
    rrs = [i['rr_mean_mm'] for i in images if i['rr_mean_mm']]
    return {
        'n_images': len(images),
        'beats': sum(i['beats'] for i in images),
        'r_peaks': sum(i['r_peaks'] for i in images),
        'model': sum(i['model'] for i in images),
        'anchor': sum(i['anchor'] for i in images),
        'flags': sum(i['flags'] for i in images),
        'hr_mean': float(np.mean(hrs)) if hrs else None,
        'hr_min': float(np.min(hrs)) if hrs else None,
        'hr_max': float(np.max(hrs)) if hrs else None,
        'rr_mean_mm': float(np.mean(rrs)) if rrs else None,
        'rr_spread_mm': float(np.max(rrs) - np.min(rrs)) if len(rrs) > 1 else 0.0,
    }


@app.get('/api/patients/{pid}/summary')
def api_patient_summary(pid: str):
    """ผลที่จำไว้ของสัตว์ตัวนี้ — ไม่รันโมเดลเลยแม้แต่ภาพเดียว

    ใช้ตอนสลับกลับมาดูตัวเดิม ผลที่วิเคราะห์ไปแล้วจึงไม่หายไปกับการเปลี่ยนตัว
    ภาพที่ยังไม่เคยรัน (หรือไฟล์ถูกแก้หลังรัน) ถูกคืนไว้ใน pending ให้หน้าเว็บบอกผู้ใช้
    """
    p = pt.get_patient(DATA_DIR, pid)
    if not p:
        raise HTTPException(404, f'ไม่พบรหัส {pid}')
    images, pending = [], []
    for name in p['images']:
        hit = cache_hit(name)
        if hit is None:
            pending.append(name)
            continue
        images.append(image_summary(name, hit))
    return {'patient': {k: p[k] for k in ('id', 'name', 'note', 'group', 'created')},
            'images': images, 'aggregate': aggregate_images(images),
            'errors': [], 'pending': pending}


@app.post('/api/patients/{pid}/analyze')
def api_patient_analyze(pid: str, payload: dict = Body(default={})):
    """รันตรวจจับทุกภาพของสัตว์ตัวนี้ แล้วสรุปรวมให้เห็นในหน้าเดียว"""
    p = pt.get_patient(DATA_DIR, pid)
    if not p:
        raise HTTPException(404, f'ไม่พบรหัส {pid}')
    cfg = build_config((payload or {}).get('overrides'))
    images, errors = [], []
    for name in p['images']:
        try:
            run_detect(name, cfg)
            images.append(image_summary(name, _cache[name]))
        except Exception as e:                       # ภาพเสียหนึ่งใบต้องไม่ล้มทั้งหน้า
            errors.append({'image': name, 'error': str(e)})

    return {'patient': {k: p[k] for k in ('id', 'name', 'note', 'group', 'created')},
            'images': images, 'aggregate': aggregate_images(images),
            'errors': errors, 'pending': []}


@app.post('/api/migrate')
def api_migrate():
    """จัดกลุ่มภาพที่วางแบน ๆ ใน data/ เข้าโฟลเดอร์ตามรหัสที่อ่านจากชื่อไฟล์"""
    res = pt.migrate_flat_images(DATA_DIR)
    _cache.clear()
    _png_cache.clear()
    return res


# ---------------------------------------------------------------- สรุป RR รายตัว

def image_rr(rec: Dict[str, Any]) -> Dict[str, Any]:
    """ค่า RR ทุกช่วงของภาพหนึ่ง เฉพาะแถวหลัก พร้อมสถานะความน่าเชื่อถือของสเกล

    ภาพที่วัด px/mm ผิดจะให้ RR ผิดตามไปทั้งภาพ จึงต้องแยกให้เห็น ไม่ใช่เอาไปรวมเงียบ ๆ
    """
    result, rows, cfg = rec['result'], rec['rows'], rec['cfg']
    main = result.get('main_row', 0)
    vals = [float(r['rr_mm']) for r in rows
            if r['row'] == main and r['rr_mm'] not in ('', None)]
    hr = median_hr(result, cfg)
    ppm = result['stats']['px_per_mm']
    ok = bool(ppm) and hr is not None and 40 <= hr <= 300
    reason = ''
    if not ppm:
        reason = 'วัดสเกลจากกริดไม่ได้'
    elif hr is None:
        reason = 'จุด R น้อยเกินคำนวณ'
    elif not ok:
        reason = f'HR {hr:.0f} bpm อยู่นอกช่วงที่เป็นไปได้ — px/mm น่าจะผิด'
    return {'rr': vals, 'n': len(vals), 'px_per_mm': ppm, 'hr': hr,
            'scale_ok': ok, 'reason': reason}


def rr_summary(exclude_bad_scale: bool = True) -> Dict[str, Any]:
    """สรุป RR ของสัตว์ทุกตัวจากผลที่คำนวณไว้แล้ว — ไม่รันโมเดลใหม่"""
    out, skipped = [], []
    for p in pt.list_patients(DATA_DIR):
        vals: List[float] = []
        used, dropped, pending = [], [], []
        for name in p['images']:
            rec = cache_hit(name)
            if rec is None:
                pending.append(name)
                continue
            info = image_rr(rec)
            entry = {'image': name.split('/')[-1], 'n': info['n'],
                     'px_per_mm': info['px_per_mm'], 'hr': info['hr'],
                     'scale_ok': info['scale_ok'], 'reason': info['reason']}
            if info['scale_ok'] or not exclude_bad_scale:
                vals += info['rr']
                used.append(entry)
            else:
                dropped.append(entry)
        row = {'id': p['id'], 'name': p['name'], 'group': p['group'],
               'n_images': p['n_images'], 'images_used': used,
               'images_dropped': dropped, 'images_pending': pending,
               'summary': rrstats.summarize(vals)}
        out.append(row)
        if row['summary'] is None:
            skipped.append(p['id'])
    return {'patients': out, 'exclude_bad_scale': exclude_bad_scale,
            'no_data': skipped, 'methods': list(rrstats.METHODS), 'mid_n': rrstats.MID_N}


@app.get('/api/rr-summary')
def api_rr_summary(exclude_bad_scale: int = 1):
    return rr_summary(bool(exclude_bad_scale))


@app.get('/api/rr-summary.csv')
def api_rr_summary_csv(exclude_bad_scale: int = 1):
    """ตารางเดียวกับหน้าเว็บ ในรูปแบบที่เปิดด้วย Excel ได้"""
    import csv
    import io as _io
    data = rr_summary(bool(exclude_bad_scale))
    cols = ['id', 'name', 'group', 'n_images', 'images_used', 'images_dropped',
            'n_rr', 'rr_min_mm', 'rr_max_mm', 'rr_sd_mm', 'rr_range_mm']
    for m in data['methods']:
        cols += [f'{m}_mm', f'{m}_n', f'{m}_sd', f'{m}_sem']
    cols += ['spread_mm', 'spread_pct', 'หมายเหตุ']

    buf = _io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for p in data['patients']:
        s = p['summary']
        row = {'id': p['id'], 'name': p['name'], 'group': p['group'],
               'n_images': p['n_images'], 'images_used': len(p['images_used']),
               'images_dropped': len(p['images_dropped']),
               'หมายเหตุ': '; '.join(d['reason'] for d in p['images_dropped'])}
        if s:
            row.update({'n_rr': s['n'], 'rr_min_mm': round(s['min'], 3),
                        'rr_max_mm': round(s['max'], 3), 'rr_sd_mm': round(s['sd'], 3),
                        'rr_range_mm': round(s['range'], 3),
                        'spread_mm': round(s['spread'], 3),
                        'spread_pct': round(s['spread_pct'], 2)})
            for m in data['methods']:
                row.update({f'{m}_mm': round(s[m]['value'], 3), f'{m}_n': s[m]['n_used'],
                            f'{m}_sd': round(s[m]['sd'], 3), f'{m}_sem': round(s[m]['sem'], 4)})
        w.writerow(row)
    # utf-8-sig เพื่อให้ Excel อ่านภาษาไทยได้ถูกต้อง
    return Response(buf.getvalue().encode('utf-8-sig'), media_type='text/csv',
                    headers={'Content-Disposition': 'attachment; filename="rr_summary.csv"'})


# ---------------------------------------------------------------- หน้าเทียบสองชุด

DEBUG_SIDES = ('train', 'test')
_debug: Dict[str, Dict[str, Any]] = {}       # ผลของภาพในหน้าเทียบ คีย์ = side/ชื่อไฟล์
PICK_FILE = 'picked.json'                    # รายการที่เลือก เก็บลงดิสก์เหมือนไฟล์ที่อัปโหลด


def debug_dir(side: str) -> str:
    """โฟลเดอร์เก็บภาพที่อัปโหลดเข้ามาเทียบ แยกจาก data/ เพราะไม่ใช่ข้อมูลผู้ป่วยของระบบ"""
    if side not in DEBUG_SIDES:
        raise HTTPException(404, f'ไม่รู้จักชุด {side!r} (ใช้ได้: {", ".join(DEBUG_SIDES)})')
    return os.path.join(OUT_DIR, 'debug', side)


def debug_path(side: str, name: str) -> str:
    d = os.path.abspath(debug_dir(side))
    p = os.path.abspath(os.path.join(d, os.path.basename(name)))
    if not p.startswith(d + os.sep) or not os.path.isfile(p):
        raise HTTPException(404, f'ไม่พบภาพ: {name}')
    return p


def debug_images(side: str) -> List[str]:
    d = debug_dir(side)
    return [os.path.relpath(p, d).replace(os.sep, '/') for p in list_images(d)]         if os.path.isdir(d) else []


def load_picks(side: str) -> List[str]:
    """ภาพใน data/ ที่ชุดนี้เลือกไว้ — เก็บลงดิสก์เพื่อให้ไม่หายตอนรีสตาร์ต
    เหมือนไฟล์ที่อัปโหลด ถ้าเก็บไว้ในหน่วยความจำอย่างเดียวจะหายไม่พร้อมกัน
    """
    p = os.path.join(debug_dir(side), PICK_FILE)
    try:
        with open(p, encoding='utf-8') as f:
            got = json.load(f)
    except (OSError, ValueError):
        return []
    return [n for n in got if isinstance(n, str)] if isinstance(got, list) else []


def save_picks(side: str, names: List[str]) -> None:
    d = debug_dir(side)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, PICK_FILE), 'w', encoding='utf-8') as f:
        json.dump(names, f, ensure_ascii=False)


def debug_items(side: str) -> List[Dict[str, str]]:
    """ภาพของชุดนี้ = ที่อัปโหลดเข้ามา + ที่เลือกจากข้อมูลที่มีอยู่แล้ว

    ภาพจาก data/ ถูกอ้างถึงเฉย ๆ ไม่คัดลอก เพราะเป็นข้อมูลผู้ป่วย ไม่ควรมีสำเนาเพิ่ม
    """
    items = [{'name': n, 'src': 'upload'} for n in debug_images(side)]
    items += [{'name': n, 'src': 'data'} for n in load_picks(side)]
    return items


def debug_item_path(side: str, item: Dict[str, str]) -> str:
    return debug_path(side, item['name']) if item['src'] == 'upload' else resolve_path(item['name'])


@app.get('/api/debug/{side}/images')
def api_debug_list(side: str):
    return {'side': side, 'images': debug_images(side),
            'picked': load_picks(side), 'items': debug_items(side)}


@app.post('/api/debug/{side}/pick')
def api_debug_pick(side: str, payload: dict = Body(default={})):
    """เลือกภาพจาก data/ มาเข้าชุดนี้ ส่งรายการว่างเพื่อยกเลิกทั้งหมด"""
    debug_dir(side)                                   # ตรวจชื่อชุดก่อน
    names = [str(n) for n in (payload or {}).get('images', [])]
    for n in names:
        resolve_path(n)                               # กัน path หลุดออกนอก data/
    save_picks(side, names)
    for k in [k for k in _debug if k.startswith(side + '/')]:
        _debug.pop(k, None)                           # รายการเปลี่ยน ผลเดิมใช้ไม่ได้
    return {'side': side, 'picked': names, 'items': debug_items(side)}


@app.post('/api/debug/{side}/images')
async def api_debug_upload(side: str, files: List[UploadFile] = File(...)):
    d = debug_dir(side)
    os.makedirs(d, exist_ok=True)
    saved, failed = [], []
    for f in files:
        base = os.path.basename(f.filename or 'image.png')
        if not base.lower().endswith(('.jpg', '.jpeg', '.png')):
            failed.append({'file': base, 'reason': 'รองรับเฉพาะ .jpg .jpeg .png'})
            continue
        dest = os.path.join(d, base)
        with open(dest, 'wb') as fh:
            fh.write(await f.read())
        if imread_u(dest) is None:
            os.remove(dest)
            failed.append({'file': base, 'reason': 'เปิดเป็นภาพไม่ได้'})
            continue
        saved.append(base)
    if not saved and failed:
        raise HTTPException(400, failed[0]['reason'])
    return {'saved': saved, 'failed': failed, 'images': debug_images(side)}


@app.delete('/api/debug/{side}/images')
def api_debug_clear(side: str):
    d = debug_dir(side)
    if os.path.isdir(d):
        shutil.rmtree(d)
    for k in [k for k in _debug if k.startswith(side + '/')]:
        _debug.pop(k, None)
    return {'cleared': side}


def debug_rec(side: str, name: str) -> Dict[str, Any]:
    rec = _debug.get(f'{side}/{name}')
    if rec is None:
        raise HTTPException(409, 'ยังไม่ได้รันชุดนี้ กดปุ่มรันก่อน')
    return rec


@app.post('/api/debug/{side}/run')
def api_debug_run(side: str, payload: dict = Body(default={})):
    """รันไปป์ไลน์เต็มกับทุกภาพของชุดนี้ ด้วยค่าตั้งของชุดนี้เอง

    สองชุดตั้งค่าไม่เหมือนกันได้ เพราะเป็นข้อมูลคนละที่มา จุดประสงค์คือเทียบ
    ผลลัพธ์ปลายทาง ไม่ใช่บังคับให้ประมวลผลเหมือนกัน
    """
    cfg = build_config((payload or {}).get('overrides'))
    items = debug_items(side)
    if not items:
        raise HTTPException(400, f'ยังไม่มีภาพในชุด {side}')
    models = get_models(cfg)
    images, errors = [], []
    for item in items:
        name = item['name']
        try:
            path = debug_item_path(side, item)
            result = detect_r_peaks(path, models, cfg)
            rows = result_to_rows(path, result, cfg)
            h, w = result['raw'].shape[:2]
            rec = {'result': result, 'rows': rows, 'cfg': cfg, 'path': path,
                   'mtime': os.path.getmtime(path),
                   'rev': rs.revision(f'{side}/{name}', os.path.getmtime(path), cfg),
                   'width': int(w), 'height': int(h)}
            _debug[f'{side}/{name}'] = rec
            images.append({**image_summary(name, rec), 'src': item['src']})
        except Exception as e:                       # ภาพเสียหนึ่งใบต้องไม่ล้มทั้งชุด
            errors.append({'image': name, 'error': str(e)})
    return {'side': side, 'images': images, 'aggregate': aggregate_images(images),
            'errors': errors, 'config': {k: getattr(cfg, k) for k in ALLOWED}}


@app.get('/api/debug/{side}/prebin')
def api_debug_prebin(side: str, image: str, width: int = 1200,
                     crop_pre: Optional[str] = None, blackhat_thr: Optional[int] = None,
                     crop_pre_ksize: Optional[int] = None, crop_pre_thr: Optional[int] = None,
                     crop_pre_hyst: Optional[float] = None, crop_pre_close: Optional[int] = None,
                     crop_pre_dilate: Optional[int] = None):
    """ภาพหลัง binarization ของภาพในชุดนี้ ก่อนตีกรอบ

    ดูได้ทันทีโดยไม่ต้องรัน จึงใช้ไล่ค่าจนกล่องที่หายกลับมาได้ก่อนค่อยรันจริง
    """
    item = next((i for i in debug_items(side) if i['name'] == image), None)
    if item is None:
        raise HTTPException(404, f'ไม่พบภาพ {image} ในชุด {side}')
    return render_prebin(f'{side}/{image}', debug_item_path(side, item), width,
                         {'crop_pre': crop_pre, 'blackhat_thr': blackhat_thr,
                          'crop_pre_ksize': crop_pre_ksize, 'crop_pre_thr': crop_pre_thr,
                          'crop_pre_hyst': crop_pre_hyst, 'crop_pre_close': crop_pre_close,
                          'crop_pre_dilate': crop_pre_dilate})


@app.get('/api/debug/{side}/overlay')
def api_debug_overlay(side: str, image: str, width: int = 900, rev: str = ''):
    rec = debug_rec(side, image)

    def build():
        img = draw_overlay(rec['result'], boxes=True, marks=True, landmarks=False, origin=False)
        if width and 0 < width < img.shape[1]:
            h = max(1, int(round(img.shape[0] * width / img.shape[1])))
            img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
        return img

    return cached_png(f'{side}/{image}|dbg-overlay|{rec["rev"]}|{width}', build)


@app.get('/api/debug/{side}/crops')
def api_debug_crops(side: str, image: str, n: int = 5, size: int = 220):
    """ครอปที่ป้อนโมเดลจุด — แถวที่ compare_domain ใช้เทียบว่าสองชุดหน้าตาตรงกันไหม"""
    rec = debug_rec(side, image)
    cfg, result = rec['cfg'], rec['result']
    rows = result['rows']
    if not rows:
        raise HTTPException(409, 'ไม่พบกล่องจังหวะในภาพนี้')
    key = f'{side}/{image}|dbg-crops|{rec["rev"]}|{n}|{size}'

    def build():
        raw = result['raw']
        px_mm = resolve_px_per_mm(raw, cfg)
        pitch = row_pitch(rows[result.get('main_row', 0)])
        cx, cy = expected_center(cfg)
        tiles = []
        for box in rows[result.get('main_row', 0)][:max(1, n)]:
            sq, _ = square_crop(raw, box, cfg, pitch=pitch, px_per_mm=px_mm)
            if sq is None:
                continue
            tile = cv2.resize(point_preprocess(sq, cfg), (size, size))
            cv2.drawMarker(tile, (int(cx * size / cfg.out_size), int(cy * size / cfg.out_size)),
                           (60, 200, 60), cv2.MARKER_CROSS, 18, 2)
            cv2.rectangle(tile, (0, 0), (size - 1, size - 1), (200, 200, 200), 1)
            tiles.append(tile)
        if not tiles:
            raise HTTPException(409, 'สร้างครอปไม่ได้')
        return np.hstack(tiles)

    return cached_png(key, build)


class NoCacheStatic(StaticFiles):
    """บังคับให้เบราว์เซอร์ถามเซิร์ฟเวอร์ทุกครั้งว่าหน้าเว็บมีของใหม่ไหม

    ไม่งั้นแท็บที่เปิดค้างไว้จะรัน app.js ตัวเก่าต่อไปหลัง deploy โดยไม่มีอะไรบอก
    ผู้ใช้เห็นพฤติกรรมเก่าทั้งที่เซิร์ฟเวอร์แก้แล้ว — หาสาเหตุยากมาก
    ETag ยังทำงานตามปกติ การถามส่วนใหญ่จึงจบที่ 304 ไม่ได้โหลดไฟล์ซ้ำจริง
    """

    async def get_response(self, path: str, scope):
        res = await super().get_response(path, scope)
        res.headers['Cache-Control'] = 'no-cache'
        return res


app.mount('/', NoCacheStatic(directory=STATIC_DIR, html=True), name='static')
