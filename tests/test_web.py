"""ทดสอบ API ของหน้าเว็บ ในส่วนที่ไม่ต้องใช้โมเดลจริง"""
import os
from types import SimpleNamespace

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
    models = SimpleNamespace(has_point=True)      # โครงเท่าที่ api_detect ใช้จริง
    monkeypatch.setattr(server, 'get_models', lambda cfg: models)
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


def test_rev_tracks_content_not_load_order(counting_detect):
    """rev ไปอยู่ใน URL ของภาพที่แคชแบบ immutable จึงต้องเปลี่ยนตามเนื้อหาเท่านั้น

    ถ้าใช้ตัวนับ เลขจะเริ่มใหม่ทุกครั้งที่รีสตาร์ต ภาพคนละใบได้เลขซ้ำกัน
    แล้วเบราว์เซอร์จะหยิบภาพเก่ามาแสดงคู่กับตัวเลขชุดใหม่
    """
    cfg = server.Config()
    server.run_detect('demo.png', cfg)
    rev1 = server._cache['demo.png']['rev']

    server.run_detect('demo.png', cfg, force=True)          # คำนวณใหม่ ผลเท่าเดิม
    assert server._cache['demo.png']['rev'] == rev1

    server.run_detect('demo.png', cfg.with_(point_pre='gray'))
    assert server._cache['demo.png']['rev'] != rev1          # ค่าตั้งเปลี่ยน = คนละรุ่น


def test_rev_differs_between_images(client, counting_detect):
    """เลขรุ่นซ้ำกันข้ามภาพคือที่มาของการแสดงภาพผิดใบ"""
    import shutil
    shutil.copy(os.path.join(server.DATA_DIR, 'demo.png'),
                os.path.join(server.DATA_DIR, 'demo2.png'))
    cfg = server.Config()
    server.run_detect('demo.png', cfg)
    server.run_detect('demo2.png', cfg)
    assert server._cache['demo.png']['rev'] != server._cache['demo2.png']['rev']


def test_rev_survives_a_restart(client, counting_detect, monkeypatch):
    cfg = server.Config()
    server.run_detect('demo.png', cfg)
    before = server._cache['demo.png']['rev']
    monkeypatch.setattr(server, '_cache', {})
    assert server.cache_hit('demo.png')['rev'] == before


def test_png_cache_dropped_on_recompute(counting_detect):
    cfg = server.Config()
    server.run_detect('demo.png', cfg)
    server._png_cache['demo.png|overlay|1|11110|0'] = b'stale'
    server.run_detect('demo.png', cfg, force=True)
    assert not any(k.startswith('demo.png|') for k in server._png_cache)


# ---------------------------------------------------------------- ความจำข้ามการสลับสัตว์

@pytest.fixture
def patient_with_image(client):
    """สัตว์หนึ่งตัวที่มีภาพหนึ่งใบ พร้อมให้วิเคราะห์"""
    client.post('/api/patients', json={'id': 'D001', 'name': 'Buddy'})
    img, _, _ = make_ekg(w=400)
    client.post('/api/patients/D001/images',
                files={'files': ('a.png', cv2.imencode('.png', img)[1].tobytes(), 'image/png')})
    return client


def test_summary_unknown_patient(client):
    assert client.get('/api/patients/ไม่มี/summary').status_code == 404


def test_summary_before_analyze_lists_everything_as_pending(patient_with_image):
    r = patient_with_image.get('/api/patients/D001/summary').json()
    assert r['images'] == [] and r['pending'] == ['D001/a.png']
    assert r['aggregate']['n_images'] == 0


def test_summary_returns_remembered_result_without_rerunning(patient_with_image, counting_detect):
    """สลับไปสัตว์ตัวอื่นแล้วกลับมา ต้องได้ผลเดิมคืนโดยไม่รันโมเดลซ้ำ"""
    patient_with_image.post('/api/patients/D001/analyze', json={})
    assert len(counting_detect) == 1

    r = patient_with_image.get('/api/patients/D001/summary').json()
    assert [i['image'] for i in r['images']] == ['D001/a.png']
    assert r['pending'] == [] and r['aggregate']['n_images'] == 1
    assert len(counting_detect) == 1          # summary ต้องไม่แตะโมเดลเลย


def test_summary_drops_result_when_file_replaced(patient_with_image, counting_detect):
    """ภาพถูกอัปโหลดทับ ผลที่จำไว้ใช้ไม่ได้แล้ว ต้องไม่เอามาแสดง"""
    patient_with_image.post('/api/patients/D001/analyze', json={})
    os.utime(os.path.join(server.DATA_DIR, 'D001', 'a.png'), (0, 0))
    r = patient_with_image.get('/api/patients/D001/summary').json()
    assert r['images'] == [] and r['pending'] == ['D001/a.png']


def test_detect_cached_only_never_runs_the_model(client, counting_detect):
    r = client.post('/api/detect', json={'image': 'demo.png', 'cached_only': True})
    assert r.status_code == 409 and not counting_detect

    client.post('/api/detect', json={'image': 'demo.png'})
    assert len(counting_detect) == 1
    assert client.post('/api/detect', json={'image': 'demo.png', 'cached_only': True}).status_code == 200
    assert len(counting_detect) == 1


def test_detect_cached_only_rejects_result_from_other_config(client, counting_detect):
    """เปลี่ยนค่าตั้งแล้ว ผลเก่าไม่ใช่คำตอบของค่าตั้งใหม่ ต้องไม่เอามาแสดงเงียบ ๆ"""
    client.post('/api/detect', json={'image': 'demo.png'})
    r = client.post('/api/detect', json={'image': 'demo.png',
                                         'overrides': {'point_pre': 'gray'}, 'cached_only': True})
    assert r.status_code == 409 and len(counting_detect) == 1


def test_static_files_must_be_revalidated(client):
    """แท็บที่เปิดค้างไว้ต้องไม่รันสคริปต์เก่าต่อหลัง deploy"""
    for path in ('/', '/app.js'):
        r = client.get(path)
        assert r.status_code == 200
        assert r.headers['cache-control'] == 'no-cache', path


def test_patient_group_round_trips_through_api(client):
    r = client.post('/api/patients', json={'id': 'D001', 'name': 'Buddy', 'group': 'Normal'})
    assert r.status_code == 200 and r.json()['group'] == 'Normal'
    assert client.get('/api/groups').json() == ['Normal']

    assert client.patch('/api/patients/D001', json={'group': 'B1'}).json()['group'] == 'B1'
    assert client.get('/api/patients').json()[0]['group'] == 'B1'


def test_patient_summary_carries_group(client):
    client.post('/api/patients', json={'id': 'D001', 'group': 'Normal'})
    assert client.get('/api/patients/D001/summary').json()['patient']['group'] == 'Normal'


# ---------------------------------------------------------------- ผลรอดการรีสตาร์ต

def restart(monkeypatch):
    """จำลองรีสตาร์ต: หน่วยความจำหายหมด ดิสก์ยังอยู่"""
    monkeypatch.setattr(server, '_cache', {})
    monkeypatch.setattr(server, '_png_cache', {})


def test_result_survives_restart_without_rerunning(client, counting_detect, monkeypatch):
    client.post('/api/detect', json={'image': 'demo.png'})
    assert len(counting_detect) == 1
    assert client.get('/api/health').json()['saved_results'] == 1

    restart(monkeypatch)
    r = client.post('/api/detect', json={'image': 'demo.png', 'cached_only': True})
    assert r.status_code == 200                      # ยังได้ผลเดิมทั้งที่แคชว่าง
    assert len(counting_detect) == 1                 # และไม่ได้แตะโมเดลเลย


def test_patient_summary_reads_from_disk_after_restart(client, counting_detect, monkeypatch):
    client.post('/api/patients', json={'id': 'D001'})
    img, _, _ = make_ekg(w=400)
    client.post('/api/patients/D001/images',
                files={'files': ('a.png', cv2.imencode('.png', img)[1].tobytes(), 'image/png')})
    client.post('/api/patients/D001/analyze', json={})
    assert len(counting_detect) == 1

    restart(monkeypatch)
    r = client.get('/api/patients/D001/summary').json()
    assert [i['image'] for i in r['images']] == ['D001/a.png'] and r['pending'] == []
    assert len(counting_detect) == 1


def test_overlay_works_after_restart(client, counting_detect, monkeypatch):
    """endpoint ที่ต้องวาดภาพต้องอ่านภาพต้นฉบับคืนเองได้ ไม่ใช่ตอบว่ายังไม่ได้รัน"""
    client.post('/api/detect', json={'image': 'demo.png'})
    restart(monkeypatch)
    assert client.get('/api/overlay', params={'image': 'demo.png'}).status_code == 200


def test_stored_result_ignored_when_image_replaced(client, counting_detect, monkeypatch):
    client.post('/api/detect', json={'image': 'demo.png'})
    os.utime(os.path.join(server.DATA_DIR, 'demo.png'), (0, 0))
    restart(monkeypatch)
    r = client.post('/api/detect', json={'image': 'demo.png', 'cached_only': True})
    assert r.status_code == 409 and len(counting_detect) == 1


def test_deleting_image_removes_its_stored_result(client, counting_detect, monkeypatch):
    client.post('/api/patients', json={'id': 'D001'})
    img, _, _ = make_ekg(w=400)
    client.post('/api/patients/D001/images',
                files={'files': ('a.png', cv2.imencode('.png', img)[1].tobytes(), 'image/png')})
    client.post('/api/detect', json={'image': 'D001/a.png'})
    assert client.get('/api/health').json()['saved_results'] == 1

    client.delete('/api/patients/D001/images', params={'name': 'a.png'})
    assert client.get('/api/health').json()['saved_results'] == 0


def test_deleting_patient_removes_stored_results(client, counting_detect):
    client.post('/api/patients', json={'id': 'D001'})
    img, _, _ = make_ekg(w=400)
    client.post('/api/patients/D001/images',
                files={'files': ('a.png', cv2.imencode('.png', img)[1].tobytes(), 'image/png')})
    client.post('/api/detect', json={'image': 'D001/a.png'})
    client.delete('/api/patients/D001', params={'with_images': 1})
    assert client.get('/api/health').json()['saved_results'] == 0


def test_only_a_few_source_images_are_held_in_memory(client, counting_detect, monkeypatch):
    """ชุดข้อมูลหลายสิบภาพต้องไม่ทำให้ภาพต้นฉบับค้างในหน่วยความจำทั้งหมด"""
    monkeypatch.setattr(server, '_RAW_CACHE_MAX', 2)
    img, _, _ = make_ekg(w=200)
    blob = cv2.imencode('.png', img)[1].tobytes()
    for i in range(4):
        with open(os.path.join(server.DATA_DIR, f'x{i}.png'), 'wb') as f:
            f.write(blob)
        client.get('/api/overlay', params={'image': f'x{i}.png'})   # 409 ก็ไม่เป็นไร
        client.post('/api/detect', json={'image': f'x{i}.png'})

    holding = [k for k, v in server._cache.items() if v['result'].get('raw') is not None]
    assert len(holding) <= 2 and len(server._cache) == 4             # ผลตัวเลขยังอยู่ครบ


# ---------------------------------------------------------------- ดู binarization ก่อนตีกรอบ

def test_prebin_works_without_running_detection(client):
    """เป็นแค่ preprocessing จึงต้องดูได้ทันที ไม่ต้องรอผลตรวจจับ"""
    r = client.get('/api/prebin', params={'image': 'demo.png', 'width': 400})
    assert r.status_code == 200 and r.headers['content-type'] == 'image/png'


def test_prebin_reflects_the_chosen_method(client):
    """คนละวิธี = คนละภาพ ไม่งั้นปรับค่าแล้วดูไม่ออกว่าเปลี่ยนอะไร"""
    a = client.get('/api/prebin', params={'image': 'demo.png', 'crop_pre': 'blackhat'}).content
    b = client.get('/api/prebin', params={'image': 'demo.png', 'crop_pre': 'red'}).content
    assert a != b


def test_prebin_reflects_tuning_knobs(client):
    base = client.get('/api/prebin', params={'image': 'demo.png', 'crop_pre': 'red'}).content
    thick = client.get('/api/prebin', params={'image': 'demo.png', 'crop_pre': 'red',
                                              'crop_pre_dilate': 3}).content
    assert base != thick


def test_prebin_rejects_an_unknown_method(client):
    r = client.get('/api/prebin', params={'image': 'demo.png', 'crop_pre': 'ไม่มีวิธีนี้'})
    assert r.status_code == 400


def test_prebin_guards_path_traversal(client):
    assert client.get('/api/prebin', params={'image': '../secret.png'}).status_code == 404


def test_binarization_knobs_are_tunable_from_the_web(client):
    cfg = client.get('/api/config').json()
    for k in ('crop_pre', 'blackhat_thr', 'crop_pre_hyst', 'crop_pre_close', 'crop_pre_dilate'):
        assert k in cfg, k
    r = client.post('/api/detect', json={'image': 'demo.png', 'overrides': {'crop_pre': 'มั่ว'}})
    assert r.status_code == 400


def test_crop_marker_uses_the_training_anchor(client, counting_detect, monkeypatch):
    """กากบาทในแท็บครอปคือจุดที่ชุดเทรนวาง R ไว้ ใช้เทียบว่าครอปของเราหน้าตาตรงกันไหม

    เคยวางไว้กลางภาพเมื่อ crop_mode=train_match ซึ่งเป็นค่าเริ่มต้น ทำให้เทียบผิด
    """
    from ekg_rpeak.geometry import expected_center
    cfg = server.Config()
    assert cfg.crop_mode == 'train_match'
    ex, ey = expected_center(cfg)
    assert (ex, ey) != (cfg.out_size / 2, cfg.out_size / 2)

    drawn = []
    monkeypatch.setattr(server.cv2, 'drawMarker',
                        lambda img, pt, *a, **k: drawn.append(pt))
    img, _, _ = make_ekg(w=900)
    cv2.imwrite(os.path.join(server.DATA_DIR, 'wide.png'), img)
    monkeypatch.setattr(server, 'detect_r_peaks',
                        lambda path, models, c: _real_result(path, c))
    client.post('/api/detect', json={'image': 'wide.png'})
    r = client.get('/api/crops', params={'image': 'wide.png', 'n': 1, 'size': 200})
    if r.status_code == 200 and drawn:
        assert drawn[0] == (int(ex * 200 / cfg.out_size), int(ey * 200 / cfg.out_size))


def _real_result(path, cfg):
    """ผลจำลองที่มีกล่องจริงพอให้ /api/crops ทำงานได้ โดยไม่ต้องใช้โมเดล"""
    import numpy as np
    from ekg_rpeak.imageio import imread_u
    raw = imread_u(path)
    h, w = raw.shape[:2]
    boxes = np.array([[80, 40, 140, h - 40], [240, 40, 300, h - 40],
                      [400, 40, 460, h - 40]], dtype=int)
    res = _fake_result()
    res['raw'] = raw
    res['boxes'] = boxes
    res['rows'] = [list(boxes)]
    return res


def test_tophat_knobs_are_tunable_from_the_web(client):
    """tophat_gray เป็นค่าเริ่มต้นแล้ว knob ของมันต้องปรับได้จากเว็บเหมือน blackhat thr"""
    cfg = client.get('/api/config').json()
    assert cfg['crop_pre_ksize'] and cfg['crop_pre_thr']
    a = client.get('/api/prebin', params={'image': 'demo.png', 'crop_pre': 'tophat_gray'}).content
    b = client.get('/api/prebin', params={'image': 'demo.png', 'crop_pre': 'tophat_gray',
                                          'crop_pre_thr': 45}).content
    assert a != b
