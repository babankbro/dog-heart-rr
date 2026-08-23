"""ภาพ EKG สังเคราะห์ + โมเดลจำลอง สำหรับทดสอบโดยไม่ต้องมี torch หรือ weights"""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ekg_rpeak.config import Config          # noqa: E402
from ekg_rpeak.geometry import expected_center  # noqa: E402

PPM = 8.0                 # พิกเซลต่อมิลลิเมตรของภาพทดสอบ
BEAT_MM = 22              # ระยะระหว่างจังหวะ (มิลลิเมตร)


AMP_MM = 22       # ความสูงของ R เหนือเส้นฐาน (มิลลิเมตร)
BELOW_MM = 6      # ความลึกของ S ใต้เส้นฐาน
MARGIN_MM = 6


def make_ekg(w=1600, h=None, ppm=PPM, beat_mm=BEAT_MM, trace_gray=35,
             major_grid=True, with_trace=True):
    """กระดาษ EKG: กริดเล็กสีชมพู 1 mm + กริดหลักสีแดงเข้มทุก 5 mm + คลื่นสีดำ

    ขนาดทุกอย่างคิดเป็นมิลลิเมตรแล้วคูณ ppm เพื่อให้คลื่นอยู่ในกรอบภาพเสมอ
    ไม่ว่าจะทดสอบที่ความละเอียดเท่าไร

    คืน (ภาพ, กล่องของแต่ละจังหวะ, ตำแหน่ง x ของยอด R จริง)
    """
    if h is None:
        h = int(round(ppm * (AMP_MM + BELOW_MM + 2 * MARGIN_MM)))
    img = np.full((h, w, 3), 255, np.uint8)
    step = max(1, int(round(ppm)))
    for x in range(0, w, step):
        img[:, x] = (200, 195, 245)
    for y in range(0, h, step):
        img[y, :] = (200, 195, 245)
    if major_grid:
        big = max(1, int(round(ppm * 5)))
        for x in range(0, w, big):
            img[:, x] = (70, 70, 190)
        for y in range(0, h, big):
            img[y, :] = (70, 70, 190)
    if not with_trace:
        return img, np.zeros((0, 4), int), []

    g = (trace_gray,) * 3
    base = min(int(round(ppm * (AMP_MM + MARGIN_MM))), h - int(round(ppm * BELOW_MM)) - 2)
    rr = int(ppm * beat_mm)
    cv2.line(img, (0, base), (w, base), g, 2)
    boxes, truth = [], []
    for cx in range(rr, w - rr, rr):
        apex = int(base - AMP_MM * ppm)
        nadir = int(base + BELOW_MM * ppm)
        cv2.line(img, (cx - 3, base), (cx, apex), g, 2)
        cv2.line(img, (cx, apex), (cx + 4, nadir), g, 2)
        cv2.line(img, (cx + 4, nadir), (cx + 9, base), g, 2)
        boxes.append([cx - 12, apex - 3, cx + 14, nadir + 3])
        truth.append(cx)
    return img, np.array(boxes), truth


# ---------------------------------------------------------------- fake models

class _T:
    """เลียนแบบ tensor ของ ultralytics ที่ต้อง .cpu().numpy()"""

    def __init__(self, a):
        self.a = a

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Boxes:
    def __init__(self, xyxy, conf, cls=None):
        self._xyxy = np.array(xyxy, np.float32).reshape(-1, 4)
        self._conf = np.array(conf, np.float32)
        self._cls = np.array(cls if cls is not None else [0] * len(self._conf), np.float32)

    def __len__(self):
        return len(self._conf)

    @property
    def xyxy(self):
        return _T(self._xyxy)

    @property
    def conf(self):
        return _T(self._conf)

    @property
    def cls(self):
        return _T(self._cls)

    @property
    def xywh(self):
        a = self._xyxy
        return _T(np.stack([(a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2,
                            a[:, 2] - a[:, 0], a[:, 3] - a[:, 1]], axis=1))


class _Res:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeCropModel:
    """คืนกล่องจริงของทุกจังหวะ บวกกล่องซ้ำหนึ่งอันไว้ทดสอบ dedup"""

    def __init__(self, boxes, duplicate=True):
        b = list(boxes)
        if duplicate and len(b) > 1:
            b.append(np.array(b[1]) + np.array([2, 0, 2, 0]))
        self._boxes = b

    def predict(self, img, **kw):
        return [_Res(_Boxes(self._boxes, [0.9] * len(self._boxes)))]


class FakePointModel:
    """โมเดลจุดจำลอง

    mode 'good'  : คืนคลาส R ตรงตำแหน่งที่คาด + landmark คลาสอื่นที่ conf สูงกว่า
    mode 'far'   : ชี้ผิดที่ไกลมาก conf ต่ำ (ต้องถูกปฏิเสธ)
    mode 'silent': ไม่คืนอะไรเลย (ต้อง fallback ไป anchor)
    """

    def __init__(self, cfg: Config, mode='good', r_cls=5):
        self.cfg = cfg
        self.mode = mode
        self.r_cls = r_cls

    def predict(self, imgs, **kw):
        ex, ey = expected_center(self.cfg)
        out = []
        for _ in imgs:
            if self.mode == 'good':
                out.append(_Res(_Boxes(
                    [[ex - 3, ey - 3, ex + 3, ey + 3], [40, 400, 60, 420]],
                    [0.62, 0.93],
                    [self.r_cls, 3])))          # คลาส 3 (เช่น T) conf สูงกว่า R
            elif self.mode == 'far':
                out.append(_Res(_Boxes([[500, 480, 510, 500]], [0.12], [self.r_cls])))
            else:
                out.append(_Res(_Boxes(np.zeros((0, 4)), [], [])))
        return out


@pytest.fixture
def cfg():
    return Config(r_class_id=5, train_px_per_mm=25.6)


@pytest.fixture
def ekg():
    return make_ekg()


@pytest.fixture
def ekg_path(tmp_path, ekg):
    img, boxes, truth = ekg
    p = tmp_path / 'ekg.png'
    cv2.imwrite(str(p), img)
    return str(p), boxes, truth
