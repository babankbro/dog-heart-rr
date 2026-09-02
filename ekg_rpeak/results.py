"""เก็บผลตรวจจับลงดิสก์ เพื่อให้ผลที่วิเคราะห์แล้วไม่หายไปกับการรีสตาร์ต

เก็บเป็น JSON หนึ่งไฟล์ต่อหนึ่งภาพ ใน `out/results/` โดย **ไม่เก็บภาพต้นฉบับ**
(`result['raw']`) เพราะอ่านคืนจากไฟล์เดิมได้ ไฟล์ผลจึงเล็กระดับสิบกิโลไบต์
แทนที่จะเป็นหลายเมกะไบต์ต่อภาพ

ผลที่เก็บไว้ใช้ไม่ได้เมื่อภาพต้นทางถูกแก้ (เทียบ mtime) หรือค่าตั้งไม่ตรงกับตอนที่รัน
กรณีแบบนั้นถือว่าไม่มีผล ให้รันใหม่ ดีกว่าแสดงตัวเลขที่ไม่ตรงกับภาพตรงหน้า
"""
import hashlib
import json
import os
from dataclasses import asdict, fields
from typing import Any, Dict, Optional

import numpy as np

from .config import Config

VERSION = 2          # ขึ้นเลขเมื่อรูปแบบที่เก็บเปลี่ยน ไฟล์รุ่นเก่าจะถูกมองว่าใช้ไม่ได้
STORE_NAME = 'results'


def store_dir(out_dir: str) -> str:
    return os.path.join(out_dir, STORE_NAME)


def jsonable(v: Any) -> Any:
    """แปลงค่าจาก numpy ให้ json เขียนได้ โดยคงโครงเดิมไว้ทั้งหมด"""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    return v


def revision(image: str, mtime: float, cfg: Config) -> str:
    """รหัสรุ่นของผลหนึ่งชิ้น ใช้ทำ URL ที่เบราว์เซอร์แคชได้อย่างปลอดภัย

    ต้องคิดจากเนื้อหา (ภาพ + เวลาแก้ไข + ค่าตั้ง) ไม่ใช่ตัวนับ เพราะตัวนับเริ่มใหม่
    ทุกครั้งที่รีสตาร์ต ภาพคนละใบจึงได้เลขเดียวกัน แล้วเบราว์เซอร์ที่แคชไว้แบบ
    immutable จะหยิบภาพเก่ามาแสดงคู่กับตัวเลขชุดใหม่ — ผิดแบบที่มองไม่ออก
    """
    key = f'{image}|{mtime!r}|{sorted(asdict(cfg).items())!r}|{VERSION}'
    return hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]


def path_for(out_dir: str, image: str) -> str:
    """ชื่อไฟล์ผลของภาพหนึ่ง — hash เพราะชื่อภาพมีทั้งภาษาไทย ช่องว่าง และ /"""
    key = hashlib.sha1(image.encode('utf-8')).hexdigest()[:20]
    return os.path.join(store_dir(out_dir), f'{key}.json')


def save(out_dir: str, image: str, image_path: str, cfg: Config,
         result: dict, rows: list) -> str:
    """เขียนผลของภาพหนึ่งลงดิสก์แบบ atomic คืน path ของไฟล์ที่เขียน"""
    raw = result['raw']
    payload = {
        'version': VERSION,
        'image': image,
        'mtime': os.path.getmtime(image_path),
        'width': int(raw.shape[1]),
        'height': int(raw.shape[0]),
        'cfg': asdict(cfg),
        'rows': jsonable(rows),
        'result': {k: jsonable(v) for k, v in result.items() if k != 'raw'},
    }
    d = store_dir(out_dir)
    os.makedirs(d, exist_ok=True)
    dest = path_for(out_dir, image)
    tmp = dest + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, dest)                      # กันไฟล์ผลพังถ้าโดนขัดจังหวะกลางคัน
    return dest


def _restore(payload: dict) -> Optional[dict]:
    """ปรับสิ่งที่ json ทำเพี้ยนให้กลับเป็นแบบเดิม คืน None ถ้าไฟล์ใช้ไม่ได้"""
    if payload.get('version') != VERSION:
        return None
    known = {f.name for f in fields(Config)}
    saved_cfg = payload.get('cfg') or {}
    if set(saved_cfg) != known:                # Config เปลี่ยนหน้าตาไปแล้ว ผลเก่าเทียบไม่ได้
        return None
    try:
        payload['cfg_obj'] = Config(**saved_cfg)
    except TypeError:
        return None
    # คีย์ของ rr เป็นดัชนีแถว json บังคับให้เป็น string ต้องแปลงกลับ
    rr = payload['result'].get('rr') or {}
    payload['result']['rr'] = {int(k): v for k, v in rr.items()}
    return payload


def load(out_dir: str, image: str, image_path: Optional[str] = None,
         cfg: Optional[Config] = None) -> Optional[Dict[str, Any]]:
    """ผลที่เก็บไว้ของภาพนี้ ถ้ายังใช้ได้ — ไม่อ่านภาพต้นฉบับ จึงเร็วพอจะเรียกทีละหลายภาพ

    คืน None เมื่อไม่มีไฟล์ผล ไฟล์เสีย รุ่นไม่ตรง ภาพถูกแก้หลังรัน
    หรือค่าตั้งที่ขอไม่ตรงกับตอนที่รัน
    """
    p = path_for(out_dir, image)
    try:
        with open(p, encoding='utf-8') as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return None
    payload = _restore(payload)
    if payload is None:
        return None
    if image_path is not None:
        try:
            if os.path.getmtime(image_path) != payload['mtime']:
                return None
        except OSError:
            return None
    if cfg is not None and payload['cfg_obj'] != cfg:
        return None
    return payload


def drop(out_dir: str, image: str) -> None:
    try:
        os.remove(path_for(out_dir, image))
    except OSError:
        pass


def count(out_dir: str) -> int:
    d = store_dir(out_dir)
    return len([f for f in os.listdir(d) if f.endswith('.json')]) if os.path.isdir(d) else 0
