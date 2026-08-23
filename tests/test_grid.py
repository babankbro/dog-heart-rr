"""ระบบพิกัดอ้างอิงจากเส้นกริดหลัก 5 mm"""
import numpy as np
import pytest

from ekg_rpeak.grid import (chroma_profile, find_grid, grid_origin,
                            summarize_rr, to_mm)
from ekg_rpeak.pipeline import Models, detect_r_peaks
from conftest import PPM, FakeCropModel, FakePointModel, make_ekg


@pytest.mark.parametrize('ppm', [7.9, 12.0])
def test_find_grid_measures_major_line_spacing(cfg, ppm):
    """เส้นหลักห่างกัน 5 mm จึงใช้คำนวณ px ต่อ mm ได้โดยตรง"""
    img, _, _ = make_ekg(w=2000, ppm=ppm)
    g = find_grid(img, cfg)
    assert g is not None
    assert abs(g['spacing'] - 5 * ppm) < 0.6, f"ระยะเส้นหลัก {g['spacing']:.2f} ควรใกล้ {5*ppm:.2f}"
    assert abs(g['px_per_mm'] - ppm) < 0.15
    assert g['n_lines'] > 5


def test_find_grid_without_major_lines(cfg):
    """ไม่มีเส้นหลักก็ยังต้องไม่พัง (อาจได้ค่าจากกริดเล็กหรือ None)"""
    img, _, _ = make_ekg(w=1600, major_grid=False)
    g = find_grid(img, cfg)
    assert g is None or g['spacing'] > 0


def test_grid_origin_is_line_before_first_beat(cfg):
    """x = 0 คือเส้นกริดที่อยู่นอกจังหวะซ้ายสุดพอดี"""
    img, boxes, _ = make_ekg(w=1600)
    g = find_grid(img, cfg)
    x_ref = float(boxes[0][0])
    x0 = grid_origin(g, x_ref, cfg)
    assert x0 is not None and x0 <= x_ref
    assert x_ref - x0 < g['spacing'], 'ต้องเป็นเส้นที่ใกล้ขอบที่สุด ไม่ใช่เส้นไกล ๆ'
    assert any(abs(x0 - L) < 1e-6 for L in g['lines'])


def test_grid_origin_nearest_mode(cfg):
    img, boxes, _ = make_ekg(w=1600)
    g = find_grid(img, cfg)
    x_ref = float(boxes[0][0])
    near = grid_origin(g, x_ref, cfg.with_(grid_origin_mode='nearest'))
    before = grid_origin(g, x_ref, cfg)
    assert abs(near - x_ref) <= abs(before - x_ref)


def test_grid_origin_handles_missing_grid(cfg):
    assert grid_origin(None, 100.0, cfg) is None
    assert grid_origin({'lines': []}, 100.0, cfg) is None


def test_to_mm_is_relative_to_origin():
    assert to_mm(100.0, 20.0, 8.0) == pytest.approx(10.0)
    assert to_mm(20.0, 20.0, 8.0) == 0.0
    assert to_mm(4.0, 20.0, 8.0) == pytest.approx(-2.0)     # ก่อนจุดอ้างอิงเป็นลบ


def test_summarize_rr_math(cfg):
    """RR 80 px ที่ 8 px/mm = 10 mm = 0.4 s ที่ 25 mm/s = 150 bpm"""
    xs = [100.0, 180.0, 260.0, 340.0]
    s = summarize_rr(xs, 8.0, cfg)
    assert s['n'] == 4
    assert s['mean_mm'] == pytest.approx(10.0)
    assert s['sd_mm'] == pytest.approx(0.0, abs=1e-9)
    assert s['mean_sec'] == pytest.approx(0.4)
    assert s['mean_bpm'] == pytest.approx(150.0)
    assert s['min_mm'] == pytest.approx(10.0) and s['max_mm'] == pytest.approx(10.0)


def test_summarize_rr_needs_two_points(cfg):
    assert summarize_rr([100.0], 8.0, cfg)['mean_mm'] is None
    assert summarize_rr([100.0, 180.0], None, cfg)['mean_mm'] is None


def test_summarize_rr_reports_spread(cfg):
    s = summarize_rr([0.0, 80.0, 200.0], 8.0, cfg)      # 10 mm แล้ว 15 mm
    assert s['mean_mm'] == pytest.approx(12.5)
    assert s['min_mm'] == pytest.approx(10.0) and s['max_mm'] == pytest.approx(15.0)
    assert s['sd_mm'] > 0


def test_pipeline_reports_mm_coordinates(cfg, ekg_path):
    """ทุกจุด R ต้องมีระยะเป็นมิลลิเมตรจากจุดอ้างอิง และ RR สรุปได้"""
    path, boxes, truth = ekg_path
    models = Models(crop=FakeCropModel(boxes), point=FakePointModel(cfg, 'good'))
    r = detect_r_peaks(path, models, cfg)
    s = r['stats']
    assert s['grid_spacing_px'] is not None and s['grid_lines'] > 5
    assert s['grid_origin_px'] is not None
    assert abs(s['px_per_mm'] - PPM) < 0.15

    xs_mm = [p['x_mm'] for p in r['peaks']]
    assert all(v is not None for v in xs_mm)
    assert xs_mm == sorted(xs_mm)                        # เรียงตามแนวนอน
    rr = r['rr'][0]
    expect_mm = (truth[1] - truth[0]) / PPM
    assert rr['mean_mm'] == pytest.approx(expect_mm, abs=0.3)


def test_refine_grid_matches_true_spacing(cfg):
    """least squares ต้องได้ระยะตรงกับที่วาดไว้ และไม่มีความคลาดเคลื่อนสะสม"""
    from ekg_rpeak.grid import refine_grid
    ppm = 7.9
    img, _, _ = make_ekg(w=2400, ppm=ppm)
    g = refine_grid(img, find_grid(img, cfg), cfg)
    true_spacing = round(ppm * 5)          # fixture วาดเส้นหลักที่ระยะจำนวนเต็ม
    assert g['refined'] is True
    assert abs(g['spacing'] - true_spacing) < 0.2
    assert g['resid_rms_px'] < 1.0, 'เส้นที่คำนวณต้องตกตรงยอดจริงในระดับต่ำกว่าพิกเซล'
    assert abs(g['drift_px']) < 0.5, 'residual ต้องไม่ไหลไปทางเดียวตลอดภาพ'
    assert g['n_measured'] >= 0.8 * len(g['lines'])


def test_refine_grid_lines_span_whole_image(cfg):
    """ไม้บรรทัดต้องต่อเนื่องตลอดความกว้างภาพ ไม่ใช่แค่ช่วงที่วัดได้"""
    from ekg_rpeak.grid import refine_grid
    img, _, _ = make_ekg(w=2400)
    g = refine_grid(img, find_grid(img, cfg), cfg)
    assert min(g['lines']) < g['spacing']
    assert max(g['lines']) > img.shape[1] - 2 * g['spacing']
    d = np.diff(sorted(g['lines']))
    assert np.allclose(d, g['spacing'], atol=1e-6)      # ระยะเท่ากันทุกช่วง


def test_refine_grid_needs_enough_lines(cfg):
    from ekg_rpeak.grid import refine_grid
    img, _, _ = make_ekg(w=2400)
    g = find_grid(img, cfg)
    assert refine_grid(img, {**g, 'lines': g['lines'][:2]}, cfg)['refined'] is False


def test_pipeline_reports_grid_accuracy(cfg, ekg_path):
    """สถิติต้องมีตัวเลขคุณภาพของไม้บรรทัดให้ตรวจสอบได้"""
    from ekg_rpeak.pipeline import Models, detect_r_peaks
    from conftest import FakeCropModel
    path, boxes, _ = ekg_path
    s = detect_r_peaks(path, Models(crop=FakeCropModel(boxes)), cfg)['stats']
    assert s['grid_resid_px'] is not None and s['grid_resid_px'] < 1.5
    assert abs(s['grid_drift_px']) < 0.5
