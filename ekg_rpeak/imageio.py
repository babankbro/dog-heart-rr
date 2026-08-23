"""อ่านภาพให้ทนกับ path ที่มีอักขระไม่ใช่ ASCII"""
import glob
import os
from typing import List, Optional

import cv2
import numpy as np


def imread_u(path: str) -> Optional[np.ndarray]:
    """cv2.imread คืน None เงียบ ๆ เมื่อ path มีอักขระไม่ใช่ ASCII จึงอ่านผ่าน numpy แทน"""
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def list_images(root: str, exts=('.jpg', '.jpeg', '.png')) -> List[str]:
    """ไล่หาไฟล์ภาพในโฟลเดอร์ ข้ามขยะจาก zip ของ macOS และไฟล์ที่เปิดไม่ได้"""
    if os.path.isfile(root):
        return [root] if imread_u(root) is not None else []
    cands = [p for p in glob.glob(os.path.join(root, '**', '*.*'), recursive=True)
             if p.lower().endswith(exts)
             and '__MACOSX' not in p.replace(os.sep, '/')
             and not os.path.basename(p).startswith('._')]
    return [p for p in sorted(cands) if imread_u(p) is not None]
