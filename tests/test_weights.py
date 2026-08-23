"""Integration — ต้องมี weights จริงใน models/ และติดตั้ง ultralytics แล้ว

รันด้วย:  docker compose run --rm test-full
ถ้าไม่มีไฟล์หรือไม่มี ultralytics จะข้ามอัตโนมัติ
"""
import os

import pytest

from ekg_rpeak.config import Config

pytestmark = pytest.mark.integration

ultralytics = pytest.importorskip('ultralytics', reason='ไม่ได้ติดตั้ง ultralytics')
CFG = Config()


def _need(path):
    if not os.path.exists(path):
        pytest.skip(f'ไม่พบ weights: {path}')


def _train_args(model):
    return (getattr(model, 'ckpt', None) or {}).get('train_args', {}) or {}


def test_crop_weights_load_and_imgsz_matches_config():
    """args.yaml ของ run ระบุ imgsz=512 ถ้า Config ไม่ตรง ผลจะเพี้ยนทั้งไปป์ไลน์"""
    _need(CFG.crop_weights)
    m = ultralytics.YOLO(CFG.crop_weights)
    imgsz = _train_args(m).get('imgsz')
    assert imgsz == CFG.crop_imgsz, f'weights เทรนที่ imgsz={imgsz} แต่ Config={CFG.crop_imgsz}'


def test_point_weights_class_id_is_configured():
    """โมเดลจุดตรวจ landmark หลายชนิด ถ้าไม่ตั้ง r_class_id จะหยิบ P/T มาเป็น R ได้"""
    _need(CFG.point_weights)
    m = ultralytics.YOLO(CFG.point_weights)
    names = m.names
    assert len(names) > 1, 'โมเดลจุดควรมีหลายคลาส'
    assert CFG.r_class_id is not None, (
        f'ต้องตั้ง r_class_id ให้ตรงกับคลาส R คลาสที่มี: {names}')
    assert CFG.r_class_id in names, f'r_class_id={CFG.r_class_id} ไม่มีในคลาส {names}'


def test_end_to_end_on_real_images():
    """รันจริงกับภาพใน data/ ถ้ามี"""
    from ekg_rpeak.imageio import list_images
    from ekg_rpeak.pipeline import detect_r_peaks, load_models
    _need(CFG.crop_weights)
    paths = list_images('data')
    if not paths:
        pytest.skip('ไม่มีภาพใน data/')
    models = load_models(CFG)
    r = detect_r_peaks(paths[0], models, CFG)
    s = r['stats']
    assert s['n_boxes'] > 0, 'โมเดลครอปไม่พบกล่องเลย'
    assert s['n_peaks'] > 0, 'ไม่ได้จุด R เลยแม้แต่จาก anchor'
    assert s['n_peaks'] <= s['n_boxes']
