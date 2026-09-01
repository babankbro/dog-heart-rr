# -*- coding: utf-8 -*-
"""หน้าเทียบสองชุด: อัปโหลดภาพเทรน/เทส แล้วรันไปป์ไลน์เต็มด้วยค่าตั้งของแต่ละชุดเอง"""
import os

import cv2
import pytest

pytest.importorskip('fastapi', reason='ไม่ได้ติดตั้ง fastapi')
from fastapi.testclient import TestClient          # noqa: E402

from conftest import make_ekg                      # noqa: E402
from webapp import server                          # noqa: E402
from test_web import _fake_result                  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    data.mkdir()
    monkeypatch.setattr(server, 'DATA_DIR', str(data))
    monkeypatch.setattr(server, 'OUT_DIR', str(tmp_path / 'out'))
    monkeypatch.setattr(server, '_debug', {})
    monkeypatch.setattr(server, '_png_cache', {})
    return TestClient(server.app)


@pytest.fixture
def stub_pipeline(client, monkeypatch):
    """ไม่ต้องมี weights จริง — สนใจว่าเส้นทางของหน้าเทียบทำงานครบ"""
    calls = []
    monkeypatch.setattr(server, 'get_models', lambda cfg: object())

    def fake(path, models, cfg):
        calls.append((path, cfg))
        from ekg_rpeak.imageio import imread_u
        res = _fake_result()
        res['raw'] = imread_u(path)
        return res

    monkeypatch.setattr(server, 'detect_r_peaks', fake)
    return calls


def upload(client, side, name='a.png', w=500):
    img, _, _ = make_ekg(w=w)
    blob = cv2.imencode('.png', img)[1].tobytes()
    return client.post(f'/api/debug/{side}/images',
                       files={'files': (name, blob, 'image/png')})


def test_unknown_side_is_rejected(client):
    assert client.get('/api/debug/staging/images').status_code == 404
    assert client.post('/api/debug/staging/run', json={}).status_code == 404


def test_upload_list_and_clear(client):
    assert client.get('/api/debug/train/images').json()['images'] == []
    assert upload(client, 'train').status_code == 200
    assert client.get('/api/debug/train/images').json()['images'] == ['a.png']

    client.delete('/api/debug/train/images')
    assert client.get('/api/debug/train/images').json()['images'] == []


def test_upload_rejects_non_image(client):
    r = client.post('/api/debug/test/images',
                    files={'files': ('note.txt', b'hello', 'text/plain')})
    assert r.status_code == 400


def test_debug_images_never_land_in_the_patient_folder(client):
    """ภาพ debug ไม่ใช่ข้อมูลผู้ป่วย ต้องไม่ไปโผล่ในทะเบียนสัตว์"""
    upload(client, 'test')
    assert client.get('/api/images').json() == []
    assert client.get('/api/patients').json() == []


def test_upload_guards_path_traversal(client):
    img, _, _ = make_ekg(w=300)
    blob = cv2.imencode('.png', img)[1].tobytes()
    client.post('/api/debug/train/images',
                files={'files': ('../../escaped.png', blob, 'image/png')})
    assert client.get('/api/debug/train/images').json()['images'] == ['escaped.png']
    assert not os.path.exists(os.path.join(server.OUT_DIR, 'escaped.png'))


def test_run_returns_per_image_and_aggregate(client, stub_pipeline):
    upload(client, 'test', 'one.png')
    upload(client, 'test', 'two.png')
    r = client.post('/api/debug/test/run', json={'overrides': {}}).json()
    assert sorted(i['image'] for i in r['images']) == ['one.png', 'two.png']
    assert r['aggregate']['n_images'] == 2 and r['errors'] == []
    assert len(stub_pipeline) == 2


def test_each_side_keeps_its_own_settings(client, stub_pipeline):
    """สองชุดมาจากคนละที่ ต้องตั้งค่าคนละแบบได้"""
    upload(client, 'train', 't.png')
    upload(client, 'test', 'x.png')
    client.post('/api/debug/train/run', json={'overrides': {'crop_pre': 'red'}})
    client.post('/api/debug/test/run', json={'overrides': {'crop_pre': 'blackhat'}})
    used = [cfg.crop_pre for _, cfg in stub_pipeline]
    assert used == ['red', 'blackhat']


def test_run_reports_the_config_it_used(client, stub_pipeline):
    upload(client, 'train')
    r = client.post('/api/debug/train/run', json={'overrides': {'crop_conf': 0.3}}).json()
    assert r['config']['crop_conf'] == 0.3


def test_run_without_images_is_a_clear_error(client, stub_pipeline):
    r = client.post('/api/debug/train/run', json={})
    assert r.status_code == 400


def test_overlay_and_crops_need_a_run_first(client, stub_pipeline):
    upload(client, 'test')
    assert client.get('/api/debug/test/overlay', params={'image': 'a.png'}).status_code == 409
    client.post('/api/debug/test/run', json={})
    assert client.get('/api/debug/test/overlay', params={'image': 'a.png'}).status_code == 200


def test_a_broken_image_does_not_sink_the_whole_run(client, stub_pipeline, monkeypatch):
    upload(client, 'test', 'ok.png')
    upload(client, 'test', 'bad.png')

    def flaky(path, models, cfg):
        if path.endswith('bad.png'):
            raise RuntimeError('ภาพนี้พัง')
        from ekg_rpeak.imageio import imread_u
        res = _fake_result()
        res['raw'] = imread_u(path)
        return res

    monkeypatch.setattr(server, 'detect_r_peaks', flaky)
    r = client.post('/api/debug/test/run', json={}).json()
    assert [i['image'] for i in r['images']] == ['ok.png']
    assert [e['image'] for e in r['errors']] == ['bad.png']


def test_debug_page_is_served_and_wired():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html = open(os.path.join(root, 'webapp', 'static', 'debug.html'), encoding='utf-8').read()
    js = open(os.path.join(root, 'webapp', 'static', 'debug.js'), encoding='utf-8').read()
    assert 'style.css' in html and 'debug.js' in html
    assert 'data-side="train"' in html and 'data-side="test"' in html
    for k in ('crop_pre', 'point_pre', 'crop_conf', 'train_frame_w_ratio', 'train_frame_h_ratio'):
        assert f'data-k="{k}"' in html, k
        assert k in server.ALLOWED, k


# ---------------------------------------------------------------- เลือกจากภาพที่มีอยู่แล้ว

def add_patient_image(client, pid='D001', name='p.png', w=500):
    client.post('/api/patients', json={'id': pid})
    img, _, _ = make_ekg(w=w)
    blob = cv2.imencode('.png', img)[1].tobytes()
    client.post(f'/api/patients/{pid}/images', files={'files': (name, blob, 'image/png')})
    return f'{pid}/{name}'


def test_pick_selects_images_that_already_exist(client, stub_pipeline):
    name = add_patient_image(client)
    r = client.post('/api/debug/test/pick', json={'images': [name]}).json()
    assert r['picked'] == [name]
    assert r['items'] == [{'name': name, 'src': 'data'}]

    run = client.post('/api/debug/test/run', json={}).json()
    assert [i['image'] for i in run['images']] == [name]
    assert run['images'][0]['src'] == 'data'


def test_picked_images_are_not_copied(client, stub_pipeline):
    """เป็นข้อมูลผู้ป่วย อ้างถึงพอ ไม่ควรมีสำเนาเพิ่มใน out/"""
    name = add_patient_image(client)
    client.post('/api/debug/test/pick', json={'images': [name]})
    assert client.get('/api/debug/test/images').json()['images'] == []
    side_dir = os.path.join(server.OUT_DIR, 'debug', 'test')
    copies = [f for f in os.listdir(side_dir)] if os.path.isdir(side_dir) else []
    assert copies == [server.PICK_FILE]      # มีแค่รายการที่เลือก ไม่มีสำเนาภาพ


def test_pick_rejects_paths_outside_the_data_folder(client):
    assert client.post('/api/debug/test/pick',
                       json={'images': ['../escape.png']}).status_code == 404
    assert client.post('/api/debug/test/pick',
                       json={'images': ['ไม่มีจริง.png']}).status_code == 404


def test_uploads_and_picks_run_together(client, stub_pipeline):
    name = add_patient_image(client)
    upload(client, 'test', 'up.png')
    client.post('/api/debug/test/pick', json={'images': [name]})
    run = client.post('/api/debug/test/run', json={}).json()
    got = {i['image']: i['src'] for i in run['images']}
    assert got == {'up.png': 'upload', name: 'data'}


def test_changing_the_selection_drops_stale_results(client, stub_pipeline):
    name = add_patient_image(client)
    client.post('/api/debug/test/pick', json={'images': [name]})
    client.post('/api/debug/test/run', json={})
    assert client.get('/api/debug/test/overlay', params={'image': name}).status_code == 200

    client.post('/api/debug/test/pick', json={'images': []})
    assert client.get('/api/debug/test/overlay', params={'image': name}).status_code == 409


def test_clearing_a_side_also_clears_the_selection(client, stub_pipeline):
    name = add_patient_image(client)
    client.post('/api/debug/test/pick', json={'images': [name]})
    client.delete('/api/debug/test/images')
    assert client.get('/api/debug/test/images').json()['picked'] == []


def test_debug_page_has_the_dataset_picker():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html = open(os.path.join(root, 'webapp', 'static', 'debug.html'), encoding='utf-8').read()
    js = open(os.path.join(root, 'webapp', 'static', 'debug.js'), encoding='utf-8').read()
    assert html.count('class="dataList"') == 2      # มีทั้งสองฝั่ง
    assert 'usePicked' in js and '/pick' in js and 'loadCatalog' in js


# ---------------------------------------------------------------- ดู binarization ในหน้า debug

def test_prebin_works_before_running(client, stub_pipeline):
    """ต้องดูภาพก่อนตีกรอบได้ทันที ไม่งั้นไล่ค่าไม่ได้ — ต้องรันก่อนถึงจะเห็นก็สายไป"""
    upload(client, 'test', 'a.png')
    r = client.get('/api/debug/test/prebin', params={'image': 'a.png', 'width': 400})
    assert r.status_code == 200 and r.headers['content-type'] == 'image/png'


def test_prebin_works_for_picked_dataset_images(client, stub_pipeline):
    name = add_patient_image(client)
    client.post('/api/debug/test/pick', json={'images': [name]})
    assert client.get('/api/debug/test/prebin', params={'image': name}).status_code == 200


def test_prebin_reflects_the_method_and_knobs(client, stub_pipeline):
    upload(client, 'test', 'a.png')
    q = {'image': 'a.png', 'width': 300}
    a = client.get('/api/debug/test/prebin', params={**q, 'crop_pre': 'blackhat'}).content
    b = client.get('/api/debug/test/prebin', params={**q, 'crop_pre': 'red'}).content
    c = client.get('/api/debug/test/prebin', params={**q, 'crop_pre': 'red',
                                                     'crop_pre_dilate': 3}).content
    assert a != b and b != c


def test_prebin_rejects_images_outside_the_side(client, stub_pipeline):
    upload(client, 'train', 'only-train.png')
    assert client.get('/api/debug/test/prebin',
                      params={'image': 'only-train.png'}).status_code == 404


def test_prebin_rejects_an_unknown_method(client, stub_pipeline):
    upload(client, 'test', 'a.png')
    r = client.get('/api/debug/test/prebin', params={'image': 'a.png', 'crop_pre': 'มั่ว'})
    assert r.status_code == 400


def test_debug_page_wires_the_preview():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html = open(os.path.join(root, 'webapp', 'static', 'debug.html'), encoding='utf-8').read()
    js = open(os.path.join(root, 'webapp', 'static', 'debug.js'), encoding='utf-8').read()
    assert html.count('class="prebinImg"') == 2 and html.count('class="prebinPick"') == 2
    for k in ('crop_pre_hyst', 'crop_pre_close', 'crop_pre_dilate'):
        assert f'data-k="{k}"' in html, k
    assert 'refreshPrebin' in js and 'PRE_KEYS' in js


def test_selection_survives_a_restart(client, stub_pipeline, monkeypatch):
    """ไฟล์ที่อัปโหลดอยู่บนดิสก์ รายการที่เลือกก็ควรอยู่ด้วย ไม่งั้นหายไม่พร้อมกัน"""
    name = add_patient_image(client)
    client.post('/api/debug/test/pick', json={'images': [name]})
    monkeypatch.setattr(server, '_debug', {})            # จำลองรีสตาร์ต
    assert client.get('/api/debug/test/images').json()['picked'] == [name]


def test_the_selection_file_is_not_mistaken_for_an_image(client, stub_pipeline):
    name = add_patient_image(client)
    client.post('/api/debug/test/pick', json={'images': [name]})
    assert client.get('/api/debug/test/images').json()['images'] == []
