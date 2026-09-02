# -*- coding: utf-8 -*-
"""หน้าสรุป RR รายตัว: รวมค่าจากทุกภาพ แยกภาพที่สเกลผิด และส่งออก CSV"""
import os

import cv2
import numpy as np
import pytest

pytest.importorskip('fastapi', reason='ไม่ได้ติดตั้ง fastapi')
from fastapi.testclient import TestClient          # noqa: E402

from conftest import make_ekg                      # noqa: E402
from webapp import server                          # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    data.mkdir()
    monkeypatch.setattr(server, 'DATA_DIR', str(data))
    monkeypatch.setattr(server, 'OUT_DIR', str(tmp_path / 'out'))
    monkeypatch.setattr(server, '_cache', {})
    return TestClient(server.app)


def make_rec(path, rr_mm, px_per_mm=8.0, hr_ok=True):
    """ระเบียนผลจำลอง — สนใจแค่ rr_mm กับสเกล ไม่ต้องรันโมเดล"""
    xs = [0.0]
    step = (60.0 / 120.0) * 25.0 * px_per_mm if hr_ok else 1.0
    for _ in rr_mm:
        xs.append(xs[-1] + step)
    result = {
        'raw': np.zeros((10, 10, 3), np.uint8), 'main_row': 0,
        'stats': {'px_per_mm': px_per_mm, 'n_rows': 1, 'n_peaks': len(xs)},
        'rows': [[np.array([0, 0, 5, 5])] * len(xs)],
        'peaks': [{'row': 0, 'x': x, 'y': 0.0} for x in xs],
        'rr': {}, 'boxes': np.zeros((0, 4), int), 'landmarks': [], 'grid': None, 'origin': None,
    }
    rows = [{'row': 0, 'rr_mm': '' if i == 0 else rr_mm[i - 1], 'flag': ''}
            for i in range(len(xs))]
    return {'result': result, 'rows': rows, 'cfg': server.Config(), 'path': path,
            'mtime': os.path.getmtime(path), 'rev': 'x', 'width': 10, 'height': 10}


def add_image(client, pid, name, rr_mm, px_per_mm=8.0, hr_ok=True):
    client.post('/api/patients', json={'id': pid})
    img, _, _ = make_ekg(w=300)
    blob = cv2.imencode('.png', img)[1].tobytes()
    client.post(f'/api/patients/{pid}/images', files={'files': (name, blob, 'image/png')})
    key = f'{pid}/{name}'
    path = os.path.join(server.DATA_DIR, pid, name)
    server._cache[key] = make_rec(path, rr_mm, px_per_mm, hr_ok)
    return key


def test_rr_of_one_animal_pools_every_image(client):
    add_image(client, 'D001', 'a.png', [10.0] * 5)
    add_image(client, 'D001', 'b.png', [12.0] * 5)
    p = client.get('/api/rr-summary').json()['patients'][0]
    assert p['summary']['n'] == 10
    assert p['summary']['min'] == 10.0 and p['summary']['max'] == 12.0
    assert len(p['images_used']) == 2


def test_all_three_methods_are_reported(client):
    add_image(client, 'D001', 'a.png', [float(i) for i in range(1, 26)])
    s = client.get('/api/rr-summary').json()['patients'][0]['summary']
    for m in ('mean_all', 'mid20', 'median'):
        assert {'value', 'n_used', 'sd', 'sem'} <= set(s[m])
    assert s['mid20']['n_used'] == 20 and s['mean_all']['n_used'] == 25


def test_images_with_a_broken_scale_are_excluded_by_default(client):
    add_image(client, 'D001', 'good.png', [10.0] * 5)
    add_image(client, 'D001', 'bad.png', [99.0] * 5, px_per_mm=40.0, hr_ok=False)
    p = client.get('/api/rr-summary').json()['patients'][0]
    assert [i['image'] for i in p['images_dropped']] == ['bad.png']
    assert p['summary']['max'] == 10.0
    assert 'px/mm' in p['images_dropped'][0]['reason']


def test_broken_scale_can_be_included_on_request(client):
    add_image(client, 'D001', 'good.png', [10.0] * 5)
    add_image(client, 'D001', 'bad.png', [99.0] * 5, px_per_mm=40.0, hr_ok=False)
    p = client.get('/api/rr-summary', params={'exclude_bad_scale': 0}).json()['patients'][0]
    assert p['images_dropped'] == [] and p['summary']['max'] == 99.0


def test_an_animal_with_no_usable_image_reports_no_summary(client):
    add_image(client, 'D001', 'bad.png', [99.0] * 5, px_per_mm=40.0, hr_ok=False)
    d = client.get('/api/rr-summary').json()
    assert d['patients'][0]['summary'] is None
    assert d['no_data'] == ['D001']


def test_images_without_results_are_listed_as_pending(client):
    client.post('/api/patients', json={'id': 'D001'})
    img, _, _ = make_ekg(w=300)
    client.post('/api/patients/D001/images',
                files={'files': ('never-run.png', cv2.imencode('.png', img)[1].tobytes(),
                                 'image/png')})
    p = client.get('/api/rr-summary').json()['patients'][0]
    assert p['images_pending'] == ['D001/never-run.png'] and p['summary'] is None


def test_csv_has_a_row_per_animal_and_all_methods(client):
    add_image(client, 'D001', 'a.png', [10.0] * 5)
    add_image(client, 'D002', 'b.png', [12.0] * 5)
    r = client.get('/api/rr-summary.csv')
    assert r.status_code == 200
    assert 'attachment' in r.headers['content-disposition']
    text = r.content.decode('utf-8-sig')
    head, *body = [ln for ln in text.splitlines() if ln.strip()]
    for col in ('mean_all_mm', 'mid20_mm', 'median_mm', 'rr_sd_mm', 'spread_mm'):
        assert col in head, col
    assert len(body) == 2


def test_csv_explains_why_an_image_was_dropped(client):
    add_image(client, 'D001', 'good.png', [10.0] * 5)
    add_image(client, 'D001', 'bad.png', [99.0] * 5, px_per_mm=40.0, hr_ok=False)
    text = client.get('/api/rr-summary.csv').content.decode('utf-8-sig')
    assert 'px/mm' in text


def test_the_dashboard_page_is_served_and_wired():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html = open(os.path.join(root, 'webapp', 'static', 'rr.html'), encoding='utf-8').read()
    js = open(os.path.join(root, 'webapp', 'static', 'rr.js'), encoding='utf-8').read()
    assert 'rr.js' in html and 'style.css' in html
    assert 'id="csv"' in html and 'id="excl"' in html
    assert 'rr-summary.csv' in js and 'exclude_bad_scale' in js
