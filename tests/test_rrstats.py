# -*- coding: utf-8 -*-
"""สรุปช่วง RR ของสัตว์แต่ละตัวด้วยหลายวิธี"""
import pytest

from ekg_rpeak import rrstats as rs


def test_middle_values_trims_both_tails():
    v = list(range(1, 31))                       # 30 ค่า
    mid = rs.middle_values(v, n=20)
    assert len(mid) == 20
    assert mid[0] > min(v) and mid[-1] < max(v)


def test_middle_values_keeps_everything_when_short():
    v = [3.0, 1.0, 2.0]
    assert rs.middle_values(v, n=20) == [1.0, 2.0, 3.0]


def test_middle_values_trims_the_long_side_harder_on_odd_counts():
    """ช่วงที่ยาวผิดปกติมาจากจังหวะที่หาย อันตรายกว่าช่วงที่สั้น จึงตัดข้างล่างมากกว่า"""
    v = list(range(1, 24))                       # 23 ค่า ต้องตัดออก 3
    mid = rs.middle_values(v, n=20)
    assert len(mid) == 20
    assert mid[0] == 3 and mid[-1] == 22


def test_describe_handles_a_single_value():
    d = rs.describe([5.0])
    assert d['n'] == 1 and d['sd'] == 0.0 and d['sem'] == 0.0
    assert d['mean'] == d['median'] == d['min'] == d['max'] == 5.0


def test_describe_on_empty_input():
    assert rs.describe([]) is None
    assert rs.summarize([]) is None


def test_three_methods_agree_on_clean_data():
    v = [10.0] * 25
    s = rs.summarize(v)
    for m in rs.METHODS:
        assert s[m]['value'] == pytest.approx(10.0)
    assert s['spread'] == pytest.approx(0.0)
    assert s['sd'] == pytest.approx(0.0)


def test_outliers_separate_the_methods():
    """จังหวะที่หายทำให้เกิดช่วงยาวผิดปกติ ค่าเฉลี่ยรวมต้องโดนลากมากที่สุด"""
    v = [10.0] * 24 + [200.0]
    s = rs.summarize(v)
    assert s['mean_all']['value'] > s['mid20']['value']
    assert s['median']['value'] == pytest.approx(10.0)
    assert s['spread'] > 5


def test_spread_is_reported_relative_to_the_median():
    s = rs.summarize([10.0] * 24 + [200.0])
    assert s['spread_pct'] == pytest.approx(100.0 * s['spread'] / 10.0)


def test_median_error_is_wider_than_the_mean_error():
    """มัธยฐานทิ้งข้อมูลไปมากกว่า ความไม่แน่นอนของมันจึงกว้างกว่าเมื่อข้อมูลกระจายปกติ"""
    v = [9.0, 10.0, 11.0, 10.5, 9.5, 10.2, 9.8, 10.1]
    s = rs.summarize(v)
    assert s['median']['sem'] > s['mean_all']['sem']


def test_counts_report_how_much_data_each_method_used():
    v = [float(i) for i in range(30)]
    s = rs.summarize(v)
    assert s['mean_all']['n_used'] == 30
    assert s['mid20']['n_used'] == 20
    assert s['median']['n_used'] == 30
    assert s['n'] == 30
