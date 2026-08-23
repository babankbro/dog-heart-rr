"""ครอปต้องสเกลคงที่ map พิกัดกลับได้ และเลือกจุด R ให้ถูกคลาส"""
import numpy as np
import pytest

from ekg_rpeak.geometry import (crop_region, crop_to_square, dedup_peaks,
                                expected_center, group_rows, pick_point,
                                row_pitch, square_crop, unmap_point)
from ekg_rpeak.preprocess import find_r_anchor
from ekg_rpeak.scale import estimate_px_per_mm
from conftest import _Boxes, _Res, make_ekg


def test_crop_scale_is_constant_at_image_edges(cfg, ekg):
    """กล่องติดขอบภาพต้องไม่ถูกซูมต่างจากกล่องกลางภาพ"""
    img, boxes, _ = ekg
    ppm = estimate_px_per_mm(img)
    scales = []
    for b in [boxes[0], boxes[len(boxes) // 2], boxes[-1]]:
        sq, m = square_crop(img, b, cfg, px_per_mm=ppm)
        assert sq.shape == (cfg.out_size, cfg.out_size, 3)
        scales.append(round(m['sx'], 6))
    assert len(set(scales)) == 1, f'สเกลไม่คงที่: {scales}'


def test_expected_center_round_trip(cfg, ekg):
    """map ตำแหน่งที่คาดว่า R อยู่ กลับไปภาพเต็ม ต้องตรงกับ anchor"""
    img, boxes, _ = ekg
    cfg = cfg.with_(crop_mode='mm')
    ppm = estimate_px_per_mm(img)
    worst = 0.0
    for b in boxes:
        sq, m = square_crop(img, b, cfg, px_per_mm=ppm)
        a = find_r_anchor(img, b, cfg)
        bx, by = unmap_point(*expected_center(cfg), m)
        worst = max(worst, abs(bx - a[0]), abs(by - a[1]))
    assert worst < 1.5, f'คลาดเคลื่อนสูงสุด {worst:.2f}px'


def test_mm_mode_side_follows_grid(cfg, ekg):
    """โหมด mm: ด้านครอปต้องแปรตาม px/mm เพื่อให้ครอบคลุมมิลลิเมตรเท่าชุดเทรน"""
    img, boxes, _ = ekg
    cfg = cfg.with_(crop_mode='mm')
    ppm = estimate_px_per_mm(img)
    _, _, w, h = crop_region(img, boxes[1], cfg, px_per_mm=ppm)
    assert w == h
    assert abs(w - cfg.out_size * ppm / cfg.train_px_per_mm) < 1e-6
    assert abs(w / ppm - cfg.out_size / cfg.train_px_per_mm) < 1e-6   # กี่ mm ต่อครอป


def test_crop_outside_image_returns_none(cfg, ekg):
    img, _, _ = ekg
    assert crop_to_square(img, (-5000.0, -5000.0, 100.0, 100.0), cfg) == (None, None)


def test_group_rows_splits_leads(cfg):
    boxes = np.array([[10, 100, 30, 200], [60, 100, 80, 200], [110, 102, 130, 202],
                      [10, 400, 30, 500], [60, 398, 80, 498]])
    rows = group_rows(boxes, cfg)
    assert [len(r) for r in rows] == [3, 2]
    assert [r[0] for r in rows[0]] == [10, 60, 110]      # เรียงซ้ายไปขวาในแถว


def test_group_rows_empty(cfg):
    assert group_rows(np.zeros((0, 4), int), cfg) == []


def test_row_pitch(cfg, ekg):
    _, boxes, truth = ekg
    assert abs(row_pitch(boxes) - (truth[1] - truth[0])) < 1.0
    assert row_pitch(boxes[:2]) is None                  # กล่องน้อยเกินไป


def test_dedup_keeps_higher_conf():
    pts = [(100., 0., .9, 'a'), (105., 0., .95, 'b'), (300., 0., .8, 'c')]
    kept = dedup_peaks(pts, 100.)
    assert [p[0] for p in kept] == [105., 300.]
    assert kept[0][2] == .95


def test_pick_point_filters_class(cfg):
    """โมเดลจุดตรวจ landmark หลายชนิด — ต้องเลือกเฉพาะคลาส R แม้ conf ต่ำกว่า"""
    ex, ey = expected_center(cfg)
    pred = _Res(_Boxes([[ex - 2, ey - 2, ex + 2, ey + 2], [ex - 2, ey - 2, ex + 2, ey + 2]],
                       [0.40, 0.99], [cfg.r_class_id, 3]))
    got = pick_point(pred, cfg)
    assert got is not None and got[3] == cfg.r_class_id and got[2] == pytest.approx(0.40)


def test_pick_point_no_matching_class(cfg):
    ex, ey = expected_center(cfg)
    pred = _Res(_Boxes([[ex, ey, ex + 2, ey + 2]], [0.9], [3]))
    assert pick_point(pred, cfg) is None


def test_pick_point_prefers_expected_position(cfg):
    """คลาสเดียวกันหลายจุด ให้เลือกจุดที่อยู่ตรงตำแหน่งที่ชุดเทรนวาง R ไว้"""
    ex, ey = expected_center(cfg)
    pred = _Res(_Boxes([[ex, ey, ex + 1, ey + 1], [500, 500, 502, 502]],
                       [0.35, 0.80], [cfg.r_class_id, cfg.r_class_id]))
    got = pick_point(pred, cfg)
    assert abs(got[0] - ex) < 2 and abs(got[1] - ey) < 2


def test_pick_point_empty(cfg):
    assert pick_point(_Res(_Boxes(np.zeros((0, 4)), [], [])), cfg) is None


def test_dedup_landmarks_keeps_highest_conf():
    """จุดชนิดเดียวกันที่อยู่ใกล้กันเหลือตัวเดียว ตัวที่ conf สูงสุด"""
    from ekg_rpeak.geometry import dedup_landmarks
    lms = [{'row': 0, 'x': 100.0, 'y': 50.0, 'conf': 0.3, 'cls': 5},
           {'row': 0, 'x': 102.0, 'y': 51.0, 'conf': 0.8, 'cls': 5},
           {'row': 0, 'x': 101.0, 'y': 50.0, 'conf': 0.4, 'cls': 3},   # คนละคลาส ต้องอยู่
           {'row': 1, 'x': 100.0, 'y': 50.0, 'conf': 0.2, 'cls': 5},   # คนละแถว ต้องอยู่
           {'row': 0, 'x': 300.0, 'y': 50.0, 'conf': 0.1, 'cls': 5}]
    out = dedup_landmarks(lms, 20.0)
    assert len(out) == 4
    kept = [p for p in out if p['cls'] == 5 and p['row'] == 0 and p['x'] < 200]
    assert len(kept) == 1 and kept[0]['conf'] == 0.8


def test_dedup_landmarks_disabled():
    from ekg_rpeak.geometry import dedup_landmarks
    lms = [{'row': 0, 'x': 100.0, 'y': 50.0, 'conf': 0.3, 'cls': 5},
           {'row': 0, 'x': 100.5, 'y': 50.0, 'conf': 0.8, 'cls': 5}]
    assert len(dedup_landmarks(lms, 0.0)) == 2


def test_train_match_frame_follows_beat_proportions(cfg, ekg):
    """โหมด train_match: กว้างตามระยะ RR สูงตามแอมพลิจูด ตามสัดส่วนที่วัดจาก label"""
    img, boxes, _ = ekg
    c = cfg.with_(crop_mode='train_match')
    pitch = row_pitch(boxes)
    x, y, w, h = crop_region(img, boxes[2], c, pitch=pitch)
    assert abs(w - c.train_frame_w_ratio * pitch) < 1e-6
    a = find_r_anchor(img, boxes[2], c)
    assert abs((a[0] - x) / w - c.train_anchor_xfrac) < 0.01     # R อยู่ตรงสัดส่วนที่กำหนด
    assert abs((a[1] - y) / h - c.train_anchor_yfrac) < 0.01
    assert w != h, 'เฟรมต้องไม่จัตุรัส เพราะยืดตอน resize เหมือนชุดเทรน'


def test_train_match_falls_back_without_pitch(cfg, ekg):
    """ไม่มีระยะ RR (กล่องน้อยเกินไป) ต้องถอยไปโหมดที่ไม่ต้องใช้ ไม่ใช่พัง"""
    img, boxes, _ = ekg
    c = cfg.with_(crop_mode='train_match')
    x, y, w, h = crop_region(img, boxes[0], c, pitch=None)
    assert w == h and w > 0


def test_shift_inside_avoids_padding(cfg, ekg):
    """กรอบที่ล้นขอบภาพ ถ้าเลื่อนเข้ามาได้ต้องไม่เหลือขอบเติม"""
    img, boxes, _ = ekg
    c = cfg.with_(shift_inside=True)
    sq, m = crop_to_square(img, (-30.0, 10.0, 120.0, 120.0), c)
    assert m['X1'] == 0 and m['ox'] == 0
    plain = crop_to_square(img, (-30.0, 10.0, 120.0, 120.0), cfg)[1]
    assert plain['ox'] > 0                       # โหมดปกติยังเติมขอบเหมือนเดิม
