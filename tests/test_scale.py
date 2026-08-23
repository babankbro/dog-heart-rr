"""px/mm ต้องล็อกกับช่องเล็ก 1 mm ไม่ใช่เส้นกริดหลักทุก 5 mm"""
import cv2
import numpy as np
import pytest

from ekg_rpeak.scale import check_scale, estimate_px_per_mm, resolve_px_per_mm
from conftest import make_ekg


@pytest.mark.parametrize('ppm', [5.0, 7.9, 12.0, 25.6])
def test_estimate_px_per_mm_with_major_gridlines(ppm):
    """เคสที่ FFT เคยพลาด: ไปล็อกกับเส้นหลัก ได้ค่าผิด 5 เท่า"""
    img, _, _ = make_ekg(w=2400, ppm=ppm)
    est = estimate_px_per_mm(img)
    assert est is not None
    assert abs(est - ppm) < 0.6, f'จริง {ppm} แต่ได้ {est:.2f}'


def test_estimate_px_per_mm_faint_small_grid():
    """ช่องเล็กจางกว่าเส้นหลักมาก ยิ่งล่อให้ไปล็อกกับเส้นหลัก 5 mm"""
    img, _, _ = make_ekg(w=2400, ppm=7.9)
    faint = np.all(img == np.array([200, 195, 245], np.uint8), axis=2)
    img[faint] = (238, 236, 250)
    est = estimate_px_per_mm(img)
    assert abs(est - 7.9) < 0.6, f'ได้ {est:.2f} (ถ้าได้ ~39.5 คือล็อกกับเส้นหลัก)'


def test_estimate_px_per_mm_grayscale_scan():
    """สแกนขาวดำไม่มีสีให้แยกกริด ต้องยังวัดได้ผ่านทางเลือกสำรอง"""
    img, _, _ = make_ekg(w=2400, ppm=8.0)
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    est = estimate_px_per_mm(gray)
    assert est is not None and abs(est - 8.0) < 0.6, f'ได้ {est}'


def test_estimate_px_per_mm_no_major_grid():
    img, _, _ = make_ekg(w=2400, ppm=9.0, major_grid=False)
    assert abs(estimate_px_per_mm(img) - 9.0) < 0.6


def test_estimate_px_per_mm_too_small_image():
    img, _, _ = make_ekg(w=60, h=60, ppm=8.0)
    assert estimate_px_per_mm(img) is None


def test_check_scale_flags_impossible_hr():
    """px/mm ที่เพี้ยน 5 เท่า จะให้ HR หลายร้อย bpm ต้องจับได้"""
    hr, ok = check_scale(px_per_mm=7.9, pitch_px=90)
    assert ok and 100 < hr < 200
    hr_bad, ok_bad = check_scale(px_per_mm=39.5, pitch_px=90)
    assert not ok_bad and hr_bad > 400


def test_check_scale_missing_inputs():
    assert check_scale(None, 90) == (None, False)
    assert check_scale(7.9, None) == (None, False)


def test_resolve_px_per_mm_prefers_explicit(cfg, ekg):
    img, _, _ = ekg
    assert resolve_px_per_mm(img, cfg.with_(px_per_mm=13.0)) == 13.0
    assert resolve_px_per_mm(img, cfg.with_(auto_px_per_mm=False)) is None
    assert abs(resolve_px_per_mm(img, cfg) - 8.0) < 0.6
