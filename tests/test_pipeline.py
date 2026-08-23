"""ไปป์ไลน์ทั้งเส้นด้วยโมเดลจำลอง — ไม่ต้องมี torch หรือ weights"""
import numpy as np
import pytest

from ekg_rpeak.pipeline import Models, detect_r_peaks
from conftest import FakeCropModel, FakePointModel, make_ekg


def _models(cfg, boxes, mode='good'):
    return Models(crop=FakeCropModel(boxes), point=FakePointModel(cfg, mode))


def test_model_confirms_every_beat(cfg, ekg_path):
    path, boxes, truth = ekg_path
    r = detect_r_peaks(path, _models(cfg, boxes, 'good'), cfg)
    s = r['stats']
    assert s['n_peaks'] == len(truth)          # กล่องซ้ำถูก dedup ออกแล้ว
    assert s['n_model'] == s['n_boxes'] and s['n_anchor'] == 0
    assert s['n_dup'] == 1
    assert all(p['cls'] == cfg.r_class_id for p in r['peaks'])
    err = max(abs(p['x'] - truth[p['index']]) for p in r['peaks'])
    assert err < 3, f'ตำแหน่ง R คลาดเคลื่อน {err:.1f}px'


def test_falls_back_to_anchor_when_model_silent(cfg, ekg_path):
    """โมเดลจุดไม่คืนอะไรเลย ต้องยังได้ R ครบจาก image processing"""
    path, boxes, truth = ekg_path
    r = detect_r_peaks(path, _models(cfg, boxes, 'silent'), cfg)
    s = r['stats']
    assert s['n_peaks'] == len(truth)
    assert s['n_anchor'] == s['n_boxes'] and s['n_model'] == 0
    assert max(abs(p['x'] - truth[p['index']]) for p in r['peaks']) < 3


def test_rejects_far_low_confidence_point(cfg, ekg_path):
    """โมเดลชี้ไกลจาก anchor ด้วย conf ต่ำ ต้องถูกปฏิเสธ แล้วใช้ anchor แทน"""
    path, boxes, truth = ekg_path
    r = detect_r_peaks(path, _models(cfg, boxes, 'far'), cfg)
    s = r['stats']
    assert s['n_reject'] == s['n_boxes'] and s['n_far'] == 0
    assert s['n_peaks'] == len(truth)
    assert max(abs(p['x'] - truth[p['index']]) for p in r['peaks']) < 3


def test_trusts_far_point_when_confident(cfg, ekg_path):
    """ถ้า conf ถึงเกณฑ์ ให้เชื่อโมเดล เพื่อไม่ให้ anchor ที่ผิดมาทับของที่ถูก"""
    path, boxes, _ = ekg_path
    c = cfg.with_(trust_model_conf=0.10)
    r = detect_r_peaks(path, _models(c, boxes, 'far'), c)
    assert r['stats']['n_reject'] == 0 and r['stats']['n_far'] > 0


def test_model_only_mode_drops_undetected_beats(cfg, ekg_path):
    path, boxes, _ = ekg_path
    c = cfg.with_(point_mode='model_only')
    assert detect_r_peaks(path, _models(c, boxes, 'silent'), c)['stats']['n_peaks'] == 0


def test_anchor_only_mode_skips_point_model(cfg, ekg_path):
    path, boxes, truth = ekg_path
    c = cfg.with_(point_mode='anchor_only')
    r = detect_r_peaks(path, _models(c, boxes, 'good'), c)
    assert r['stats']['n_model'] == 0 and r['stats']['n_peaks'] == len(truth)


def test_runs_without_point_weights(cfg, ekg_path):
    """weights ของโมเดลจุดหายไป ไปป์ไลน์ต้องยังทำงานได้"""
    path, boxes, truth = ekg_path
    r = detect_r_peaks(path, Models(crop=FakeCropModel(boxes), point=None), cfg)
    assert r['stats']['n_peaks'] == len(truth)
    assert all(p['src'] == 'anchor' for p in r['peaks'])


def test_no_boxes_gives_empty_result(cfg, ekg_path):
    path, _, _ = ekg_path
    r = detect_r_peaks(path, Models(crop=FakeCropModel([], duplicate=False)), cfg)
    assert r['stats']['n_boxes'] == 0 and r['peaks'] == []


def test_missing_file_raises(cfg, tmp_path):
    with pytest.raises(FileNotFoundError):
        detect_r_peaks(str(tmp_path / 'nope.png'), Models(crop=FakeCropModel([])), cfg)


def test_multi_lead_rows_are_independent(cfg, tmp_path):
    """สองบรรทัด (lead) ต้องแยกแถวกัน ไม่คำนวณ RR ข้ามบรรทัด"""
    import cv2
    top, boxes_t, truth = make_ekg(h=420)
    bottom, boxes_b, _ = make_ekg(h=420)
    img = np.vstack([top, bottom])
    boxes = np.vstack([boxes_t, boxes_b + np.array([0, 420, 0, 420])])
    p = tmp_path / 'two_leads.png'
    cv2.imwrite(str(p), img)
    r = detect_r_peaks(str(p), Models(crop=FakeCropModel(boxes, duplicate=False)), cfg)
    assert r['stats']['n_rows'] == 2
    assert len({pk['row'] for pk in r['peaks']}) == 2
    assert r['stats']['n_peaks'] == 2 * len(truth)
