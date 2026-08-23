"""FastAPI backend สำหรับดูผลตรวจจับ R peak

เก็บผลล่าสุดของแต่ละภาพไว้ในหน่วยความจำ เพื่อให้ overlay / ครอป / CSV
ใช้ผลเดียวกับที่หน้าเว็บเพิ่งรัน โดยไม่ต้องรันโมเดลซ้ำ
(ออกแบบสำหรับใช้งานเครื่องเดียว ไม่ได้ทำ session แยกผู้ใช้)
"""
import os
import threading
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from ekg_rpeak.config import Config
from ekg_rpeak.export import FIELDS, median_hr, result_to_rows, write_csv
from ekg_rpeak.geometry import row_pitch, square_crop
from ekg_rpeak.imageio import imread_u, list_images
from ekg_rpeak import patients as pt
from ekg_rpeak.pipeline import detect_r_peaks, load_models
from ekg_rpeak.preprocess import point_preprocess
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
    'max_refine_ratio': float, 'trust_model_conf': float,
}

app = FastAPI(title='EKG R-Peak Viewer')
_models = None
_lock = threading.Lock()
_cache: Dict[str, Dict[str, Any]] = {}       # ผลตรวจจับต่อภาพ
_png_cache: Dict[str, bytes] = {}            # ภาพที่วาดแล้ว กันเรนเดอร์ซ้ำตอนสลับแท็บ
_PNG_CACHE_MAX = 96
_rev = 0                                     # เพิ่มขึ้นทุกครั้งที่มีการคำนวณจริง


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
    if name not in _cache:
        raise HTTPException(409, 'ยังไม่ได้รันภาพนี้ กดปุ่มตรวจจับก่อน')
    return _cache[name]


@app.get('/api/health')
def health():
    cfg = Config()
    return {'ok': True,
            'crop_weights': os.path.exists(cfg.crop_weights),
            'point_weights': os.path.exists(cfg.point_weights),
            'data_dir': DATA_DIR}


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
    global _rev
    path = resolve_path(name)
    mtime = os.path.getmtime(path)
    hit = _cache.get(name)
    if not force and hit and hit['cfg'] == cfg and hit.get('mtime') == mtime:
        return hit['result'], hit['rows']

    models = get_models(cfg)
    result = detect_r_peaks(path, models, cfg)
    rows = result_to_rows(path, result, cfg)
    _rev += 1
    _cache[name] = {'result': result, 'cfg': cfg, 'rows': rows, 'path': path,
                    'mtime': mtime, 'rev': _rev}
    drop_png_cache(name)                     # ผลเปลี่ยน ภาพที่วาดไว้ใช้ไม่ได้แล้ว
    return result, rows


def image_summary(name: str, result: dict, rows: list, cfg: Config) -> dict:
    """สรุปหนึ่งภาพสำหรับหน้ารวมของสัตว์แต่ละตัว"""
    s = result['stats']
    rr = result.get('rr', {}).get(0) or {}
    return {
        'image': name,
        'rev': _cache[name]['rev'],
        'beats': s['n_boxes'], 'r_peaks': s['n_peaks'],
        'model': s['n_model'], 'anchor': s['n_anchor'],
        'reject': s['n_reject'], 'dup': s['n_dup'],
        'edge_dropped': s.get('n_edge_dropped', 0),
        'px_per_mm': s['px_per_mm'],
        'rr_mean_mm': rr.get('mean_mm'), 'rr_sd_mm': rr.get('sd_mm'),
        'hr': median_hr(result, cfg),
        'flags': sum(1 for r in rows if r['flag']),
        'width': int(result['raw'].shape[1]), 'height': int(result['raw'].shape[0]),
    }


@app.post('/api/detect')
def api_detect(payload: dict = Body(...)):
    name = payload.get('image')
    if not name:
        raise HTTPException(400, 'ต้องระบุชื่อภาพ')
    cfg = build_config(payload.get('overrides'))
    result, rows = run_detect(name, cfg)
    path = _cache[name]['path']

    s = result['stats']
    pitch = row_pitch(result['rows'][0]) if result['rows'] else None
    hr, scale_ok = check_scale(s['px_per_mm'], pitch, cfg.paper_speed_mm_s)
    raw = result['raw']
    return {
        'image': name,
        'rev': _cache[name]['rev'],
        'width': int(raw.shape[1]), 'height': int(raw.shape[0]),
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
        'rr': result.get('rr', {}).get(0),
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
    pitch = row_pitch(rows[0])
    tiles = []
    for box in rows[0][:max(1, n)]:
        sq, _ = square_crop(raw, box, cfg, pitch=pitch, px_per_mm=px_mm)
        if sq is None:
            continue
        tile = cv2.resize(point_preprocess(sq, cfg), (size, size))
        ex = int(cfg.train_anchor_xfrac * size) if cfg.crop_mode == 'mm' else size // 2
        ey = int(cfg.train_anchor_yfrac * size) if cfg.crop_mode == 'mm' else size // 2
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
    if not rows or index >= len(rows[0]):
        raise HTTPException(409, 'ไม่มีจังหวะลำดับนี้')

    key = f"{image}|mask|{c['rev']}|{index}|{size}|{landmarks}"
    if key in _png_cache:
        return cached_png(key, lambda: None)

    box = rows[0][index]
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


# ---------------------------------------------------------------- ทะเบียนสัตว์

@app.get('/api/patients')
def api_patients():
    return pt.list_patients(DATA_DIR)


@app.post('/api/patients')
def api_patient_create(payload: dict = Body(...)):
    try:
        return pt.create_patient(DATA_DIR, str(payload.get('id', '')).strip(),
                                 str(payload.get('name', '')), str(payload.get('note', '')))
    except FileExistsError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.patch('/api/patients/{pid}')
def api_patient_update(pid: str, payload: dict = Body(...)):
    if not pt.get_patient(DATA_DIR, pid):
        raise HTTPException(404, f'ไม่พบรหัส {pid}')
    return pt.update_patient(DATA_DIR, pid, payload.get('name'), payload.get('note'))


@app.delete('/api/patients/{pid}')
def api_patient_delete(pid: str, with_images: int = 0):
    if not pt.get_patient(DATA_DIR, pid):
        raise HTTPException(404, f'ไม่พบรหัส {pid}')
    for name in pt.list_patient_images(DATA_DIR, pid):
        _cache.pop(name, None)
        drop_png_cache(name)
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
    return {'deleted': name}


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
            result, rows = run_detect(name, cfg)
            images.append(image_summary(name, result, rows, cfg))
        except Exception as e:                       # ภาพเสียหนึ่งใบต้องไม่ล้มทั้งหน้า
            errors.append({'image': name, 'error': str(e)})

    hrs = [i['hr'] for i in images if i['hr']]
    rrs = [i['rr_mean_mm'] for i in images if i['rr_mean_mm']]
    agg = {
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
    return {'patient': {k: p[k] for k in ('id', 'name', 'note', 'created')},
            'images': images, 'aggregate': agg, 'errors': errors}


@app.post('/api/migrate')
def api_migrate():
    """จัดกลุ่มภาพที่วางแบน ๆ ใน data/ เข้าโฟลเดอร์ตามรหัสที่อ่านจากชื่อไฟล์"""
    res = pt.migrate_flat_images(DATA_DIR)
    _cache.clear()
    _png_cache.clear()
    return res


app.mount('/', StaticFiles(directory=STATIC_DIR, html=True), name='static')
