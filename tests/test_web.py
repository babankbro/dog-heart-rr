"""ทดสอบ API ของหน้าเว็บ ในส่วนที่ไม่ต้องใช้โมเดลจริง"""
import os

import cv2
import pytest

pytest.importorskip('fastapi', reason='ไม่ได้ติดตั้ง fastapi')
from fastapi.testclient import TestClient          # noqa: E402

from webapp import server                          # noqa: E402
from conftest import make_ekg                      # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    data.mkdir()
    img, _, _ = make_ekg(w=800)
    cv2.imwrite(str(data / 'demo.png'), img)
    monkeypatch.setattr(server, 'DATA_DIR', str(data))
    monkeypatch.setattr(server, 'OUT_DIR', str(tmp_path / 'out'))
    monkeypatch.setattr(server, '_cache', {})
    return TestClient(server.app)


def test_health(client):
    r = client.get('/api/health')
    assert r.status_code == 200 and r.json()['ok'] is True


def test_config_exposes_only_allowed_keys(client):
    keys = set(client.get('/api/config').json())
    assert keys == set(server.ALLOWED)
    assert 'crop_weights' not in keys       # path ของ weights ต้องไม่ให้แก้จากหน้าเว็บ


def test_images_lists_data_dir(client):
    items = client.get('/api/images').json()
    assert [i['name'] for i in items] == ['demo.png']
    assert items[0]['w'] == 800 and items[0]['ran'] is False


def test_index_page_served(client):
    r = client.get('/')
    assert r.status_code == 200 and 'EKG R-Peak Viewer' in r.text


def test_detect_rejects_unknown_image(client):
    r = client.post('/api/detect', json={'image': 'nope.png'})
    assert r.status_code == 404


def test_detect_rejects_path_traversal(client):
    r = client.post('/api/detect', json={'image': '../../etc/passwd'})
    assert r.status_code == 404


def test_detect_rejects_bad_override(client):
    r = client.post('/api/detect', json={'image': 'demo.png',
                                         'overrides': {'r_class_id': 'ห้า'}})
    assert r.status_code == 400


def test_overlay_before_detect_is_conflict(client):
    r = client.get('/api/overlay', params={'image': 'demo.png'})
    assert r.status_code == 409


def test_upload_rejects_non_image(client):
    r = client.post('/api/upload', files={'files': ('a.txt', b'hello', 'text/plain')})
    assert r.status_code == 400


def test_upload_saves_image(client, tmp_path):
    img, _, _ = make_ekg(w=400)
    ok, buf = cv2.imencode('.png', img)
    r = client.post('/api/upload', files={'files': ('new.png', buf.tobytes(), 'image/png')})
    assert r.status_code == 200 and r.json()['saved'] == ['new.png']
    assert os.path.exists(os.path.join(server.DATA_DIR, 'new.png'))
    assert len(client.get('/api/images').json()) == 2


def test_build_config_applies_overrides():
    cfg = server.build_config({'point_mode': 'anchor_only', 'r_class_id': '5',
                               'point_conf': '0.3', 'px_per_mm': ''})
    assert cfg.point_mode == 'anchor_only'
    assert cfg.r_class_id == 5 and cfg.point_conf == 0.3
    assert cfg.px_per_mm is None            # ค่าว่างต้องไม่ทับค่า default


def test_build_config_ignores_unknown_keys():
    cfg = server.build_config({'crop_weights': '/etc/passwd', 'batch': 999})
    assert cfg.crop_weights == 'models/crop_best.pt' and cfg.batch == 16


# ---------------------------------------------------------------- ทะเบียนสัตว์

def test_patient_crud_via_api(client):
    assert client.get('/api/patients').json() == []

    r = client.post('/api/patients', json={'id': 'D001', 'name': 'Buddy'})
    assert r.status_code == 200 and r.json()['name'] == 'Buddy'

    assert client.post('/api/patients', json={'id': 'D001'}).status_code == 409
    assert client.post('/api/patients', json={'id': '../evil'}).status_code == 400

    r = client.patch('/api/patients/D001', json={'name': 'Buddy ใหม่', 'note': 'x'})
    assert r.json()['name'] == 'Buddy ใหม่'
    assert client.patch('/api/patients/ไม่มี', json={'name': 'a'}).status_code == 404


def test_patient_image_upload_and_delete(client):
    client.post('/api/patients', json={'id': 'A1'})
    img, _, _ = make_ekg(w=400)
    blob = cv2.imencode('.png', img)[1].tobytes()

    r = client.post('/api/patients/A1/images', files={'files': ('a.png', blob, 'image/png')})
    assert r.status_code == 200 and r.json()['saved'] == ['A1/a.png']
    assert client.get('/api/patients').json()[0]['n_images'] == 1

    bad = client.post('/api/patients/A1/images', files={'files': ('a.txt', b'x', 'text/plain')})
    assert bad.status_code == 400

    assert client.delete('/api/patients/A1/images', params={'name': 'a.png'}).status_code == 200
    assert client.get('/api/patients').json()[0]['n_images'] == 0
    assert client.delete('/api/patients/A1/images', params={'name': 'a.png'}).status_code == 404


def test_patient_upload_requires_existing_patient(client):
    img, _, _ = make_ekg(w=400)
    blob = cv2.imencode('.png', img)[1].tobytes()
    r = client.post('/api/patients/ไม่มีตัวนี้/images',
                    files={'files': ('a.png', blob, 'image/png')})
    assert r.status_code == 404


def test_patient_delete_with_and_without_images(client):
    client.post('/api/patients', json={'id': 'A1'})
    img, _, _ = make_ekg(w=400)
    client.post('/api/patients/A1/images',
                files={'files': ('a.png', cv2.imencode('.png', img)[1].tobytes(), 'image/png')})
    client.delete('/api/patients/A1')                       # ไม่ลบภาพ
    assert client.get('/api/patients').json()[0]['n_images'] == 1
    client.delete('/api/patients/A1', params={'with_images': 1})
    assert client.get('/api/patients').json() == []


def test_analyze_unknown_patient(client):
    assert client.post('/api/patients/ไม่มี/analyze', json={}).status_code == 404


def test_migrate_endpoint_groups_images(client, tmp_path):
    from ekg_rpeak import patients as pt
    img, _, _ = make_ekg(w=400)
    cv2.imwrite(os.path.join(server.DATA_DIR, 'D001 Buddy 1.jpg'), img)
    r = client.post('/api/migrate').json()
    assert len(r['moved']) == 1
    ids = [p['id'] for p in client.get('/api/patients').json()]
    assert 'D001' in ids


# ---------------------------------------------------------------- การใช้ผลซ้ำ

def _fake_result():
    import numpy as np
    return {'raw': np.zeros((10, 10, 3), np.uint8), 'boxes': np.zeros((0, 4), int),
            'rows': [], 'peaks': [], 'landmarks': [], 'grid': None, 'origin': None, 'rr': {},
            'stats': {'n_boxes': 0, 'n_rows': 0, 'n_peaks': 0, 'n_dup': 0, 'n_model': 0,
                      'n_anchor': 0, 'n_reject': 0, 'n_far': 0, 'n_landmarks': 0,
                      'n_edge_dropped': 0, 'px_per_mm': 8.0, 'grid_spacing_px': 40.0,
                      'grid_lines': 5, 'grid_resid_px': 0.1, 'grid_drift_px': 0.0,
                      'grid_origin_px': 0.0}}


@pytest.fixture
def counting_detect(client, monkeypatch):
    """นับจำนวนครั้งที่โมเดลถูกเรียกจริง"""
    calls = []
    monkeypatch.setattr(server, 'get_models', lambda cfg: object())
    monkeypatch.setattr(server, 'detect_r_peaks',
                        lambda path, models, cfg: (calls.append(path), _fake_result())[1])
    return calls


def test_result_is_reused_when_config_unchanged(counting_detect):
    """สลับแท็บหรือเปิดภาพเดิมซ้ำ ต้องไม่รันโมเดลใหม่"""
    cfg = server.Config()
    server.run_detect('demo.png', cfg)
    server.run_detect('demo.png', cfg)
    server.run_detect('demo.png', cfg)
    assert len(counting_detect) == 1


def test_result_recomputed_when_config_changes(counting_detect):
    server.run_detect('demo.png', server.Config())
    server.run_detect('demo.png', server.Config().with_(point_pre='gray'))
    assert len(counting_detect) == 2


def test_force_recomputes(counting_detect):
    cfg = server.Config()
    server.run_detect('demo.png', cfg)
    server.run_detect('demo.png', cfg, force=True)
    assert len(counting_detect) == 2


def test_result_recomputed_when_file_replaced(counting_detect, tmp_path):
    """ภาพถูกอัปโหลดทับด้วยชื่อเดิม ต้องคำนวณใหม่ ไม่ใช่คืนผลเก่า"""
    cfg = server.Config()
    server.run_detect('demo.png', cfg)
    p = os.path.join(server.DATA_DIR, 'demo.png')
    os.utime(p, (0, 0))                      # จำลองว่าไฟล์เปลี่ยน
    server.run_detect('demo.png', cfg)
    assert len(counting_detect) == 2


def test_rev_increases_only_on_real_compute(counting_detect):
    cfg = server.Config()
    server.run_detect('demo.png', cfg)
    rev1 = server._cache['demo.png']['rev']
    server.run_detect('demo.png', cfg)
    assert server._cache['demo.png']['rev'] == rev1
    server.run_detect('demo.png', cfg, force=True)
    assert server._cache['demo.png']['rev'] > rev1


def test_png_cache_dropped_on_recompute(counting_detect):
    cfg = server.Config()
    server.run_detect('demo.png', cfg)
    server._png_cache['demo.png|overlay|1|11110|0'] = b'stale'
    server.run_detect('demo.png', cfg, force=True)
    assert not any(k.startswith('demo.png|') for k in server._png_cache)
