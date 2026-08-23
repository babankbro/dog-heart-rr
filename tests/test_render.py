"""การวาดผลลงภาพเต็ม: กากบาท R และวงกลมจางของ landmark อื่น ๆ"""
import numpy as np

from ekg_rpeak.pipeline import Models, detect_r_peaks
from ekg_rpeak.render import COL_LANDMARK, draw_overlay, mark_size
from conftest import FakeCropModel, FakePointModel, make_ekg


def _result(with_landmarks=True):
    img, boxes, _ = make_ekg(w=800)
    peaks = [{'row': 0, 'index': 0, 'x': 200.0, 'y': 100.0,
              'conf': .9, 'src': 'model', 'cls': 5}]
    lms = [{'row': 0, 'x': 260.0, 'y': 150.0, 'conf': .5, 'cls': 3},
           {'row': 0, 'x': 320.0, 'y': 160.0, 'conf': .4, 'cls': 1}]
    return {'raw': img, 'boxes': boxes[:1], 'rows': [list(boxes[:1])],
            'peaks': peaks, 'landmarks': lms if with_landmarks else [],
            'stats': {}}


def _diff(a, b):
    return int((np.abs(a.astype(int) - b.astype(int)).sum(axis=2) > 0).sum())


def test_layers_can_be_toggled_independently():
    r = _result()
    plain = draw_overlay(r, boxes=False, marks=False, landmarks=False)
    assert _diff(plain, r['raw']) == 0

    only_marks = draw_overlay(r, boxes=False, marks=True, landmarks=False)
    only_lms = draw_overlay(r, boxes=False, marks=False, landmarks=True)
    only_boxes = draw_overlay(r, boxes=True, marks=False, landmarks=False)
    for layer in (only_marks, only_lms, only_boxes):
        assert _diff(layer, r['raw']) > 0


def test_landmark_circle_is_smaller_than_cross():
    r = _result()
    cross = _diff(draw_overlay(r, boxes=False, marks=True, landmarks=False), r['raw'])
    circles = _diff(draw_overlay(r, boxes=False, marks=False, landmarks=True), r['raw'])
    assert circles < cross, 'วงกลม landmark ต้องเล็กกว่ากากบาท R'
    assert circles > 0


def test_landmark_drawn_faintly():
    """วงกลมต้องจาง คือไม่ทับสีเต็มความเข้มของสีที่กำหนด"""
    r = _result()
    out = draw_overlay(r, boxes=False, marks=False, landmarks=True)
    changed = np.abs(out.astype(int) - r['raw'].astype(int)).sum(axis=2) > 0
    assert changed.any()
    exact = np.all(out[changed] == np.array(COL_LANDMARK), axis=-1)
    assert not exact.any(), 'สีที่วาดต้องผสมกับพื้นหลัง ไม่ใช่ทับเต็ม'


def test_landmark_at_r_position_is_not_drawn_twice():
    """landmark ที่เป็นจุด R อยู่แล้ว ไม่ต้องวาดวงกลมซ้ำ"""
    r = _result(with_landmarks=False)
    p = r['peaks'][0]
    r['landmarks'] = [{'row': 0, 'x': p['x'] + 1, 'y': p['y'] - 1, 'conf': .9, 'cls': 5}]
    assert _diff(draw_overlay(r, boxes=False, marks=False, landmarks=True), r['raw']) == 0


def test_mark_size_scales_with_width():
    assert mark_size(800) == mark_size(1600)          # ภาพเล็กใช้ขนาดฐาน
    assert mark_size(3200) > mark_size(1600)


def test_pipeline_collects_landmarks(cfg, ekg_path):
    """โมเดลจุดจำลองคืน 2 detection ต่อครอป (คลาส R และคลาสอื่น) ต้องเก็บครบ"""
    path, boxes, truth = ekg_path
    c = cfg.with_(landmark_dedup_ratio=0.0)           # ปิด dedup เพื่อนับดิบ ๆ
    models = Models(crop=FakeCropModel(boxes), point=FakePointModel(c, 'good'))
    r = detect_r_peaks(path, models, c)
    s = r['stats']
    assert s['n_landmarks'] == 2 * s['n_boxes']
    assert {p['cls'] for p in r['landmarks']} == {c.r_class_id, 3}
    assert len(r['peaks']) == len(truth)              # แต่เลือกเป็น R แค่จุดเดียวต่อจังหวะ


def test_pipeline_dedups_landmarks(cfg, ekg_path):
    """ครอปที่ซ้อนกันรายงาน landmark เดียวกันซ้ำ ต้องรวมให้เหลือจุดเดียว"""
    path, boxes, _ = ekg_path
    models = Models(crop=FakeCropModel(boxes), point=FakePointModel(cfg, 'good'))
    raw_count = detect_r_peaks(path, models, cfg.with_(landmark_dedup_ratio=0.0))
    deduped = detect_r_peaks(path, models, cfg)
    assert deduped['stats']['n_landmarks'] <= raw_count['stats']['n_landmarks']
    for i, p in enumerate(deduped['landmarks']):      # เรียงตาม x และไม่มีคู่ที่ทับกัน
        for q in deduped['landmarks'][i + 1:]:
            if q['cls'] == p['cls'] and q['row'] == p['row']:
                assert abs(q['x'] - p['x']) >= 1.0 or abs(q['y'] - p['y']) >= 1.0


def test_no_landmarks_without_point_model(cfg, ekg_path):
    path, boxes, _ = ekg_path
    r = detect_r_peaks(path, Models(crop=FakeCropModel(boxes), point=None), cfg)
    assert r['landmarks'] == [] and r['stats']['n_landmarks'] == 0


def _dbg():
    img, boxes, _ = make_ekg(w=900)
    x1, y1, x2, y2 = [int(v) for v in boxes[1]]
    roi = img[y1:y2, x1:x2]
    mask = np.zeros(roi.shape[:2], bool)
    mask[:, roi.shape[1] // 2] = True
    return {'roi': roi, 'mask': mask, 'X1': x1, 'Y1': y1}


def test_mask_panel_shows_both_views():
    from ekg_rpeak.render import draw_mask_panel
    d = _dbg()
    out = draw_mask_panel(d, size=200)
    assert out.shape == (200, 400, 3)          # ซ้าย = ภาพจริง ขวา = mask


def test_mask_panel_marks_model_point_in_red():
    from ekg_rpeak.render import COL_R_ANCHOR, COL_R_MODEL, draw_mask_panel
    d = _dbg()
    cx = d['X1'] + d['roi'].shape[1] // 2
    cy = d['Y1'] + d['roi'].shape[0] // 3
    model = draw_mask_panel(d, peaks=[{'x': cx, 'y': cy, 'src': 'model'}], size=200)
    anchor = draw_mask_panel(d, peaks=[{'x': cx, 'y': cy, 'src': 'anchor'}], size=200)
    assert np.all(model == np.array(COL_R_MODEL), axis=-1).any(), 'ต้องมีกากบาทแดง'
    assert np.all(anchor == np.array(COL_R_ANCHOR), axis=-1).any(), 'anchor ต้องเป็นสีส้ม'
    assert not np.all(anchor == np.array(COL_R_MODEL), axis=-1).any()


def test_mask_panel_draws_on_both_halves():
    from ekg_rpeak.render import COL_R_MODEL, draw_mask_panel
    d = _dbg()
    p = [{'x': d['X1'] + d['roi'].shape[1] // 2,
          'y': d['Y1'] + d['roi'].shape[0] // 3, 'src': 'model'}]
    out = draw_mask_panel(d, peaks=p, size=200)
    red = np.all(out == np.array(COL_R_MODEL), axis=-1)
    assert red[:, :200].any() and red[:, 200:].any()


def test_mask_panel_skips_points_outside_roi():
    from ekg_rpeak.render import COL_R_MODEL, draw_mask_panel
    d = _dbg()
    far = [{'x': d['X1'] - 500, 'y': d['Y1'], 'src': 'model'}]
    out = draw_mask_panel(d, peaks=far, size=200)
    assert not np.all(out == np.array(COL_R_MODEL), axis=-1).any()
