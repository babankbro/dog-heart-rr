"""บรรทัดคำสั่งของไปป์ไลน์

    python -m ekg_rpeak.cli info
    python -m ekg_rpeak.cli scale --images data
    python -m ekg_rpeak.cli calibrate --image train_sample.png
    python -m ekg_rpeak.cli detect --images data --out out/r_peaks.csv
"""
import argparse
import os
import sys

from .config import Config
from .export import median_hr, result_to_rows, write_csv
from .imageio import imread_u, list_images
from .preprocess import find_r_anchor
from .scale import check_scale, estimate_px_per_mm


def build_config(args) -> Config:
    cfg = Config()
    for k in ('crop_weights', 'point_weights', 'crop_conf', 'point_conf',
              'r_class_id', 'crop_mode', 'point_mode', 'px_per_mm',
              'train_px_per_mm', 'point_pre'):
        v = getattr(args, k, None)
        if v is not None:
            cfg = cfg.with_(**{k: v})
    return cfg


def cmd_info(args) -> int:
    """พิมพ์ข้อมูลของ weights ที่มี — ที่สำคัญคือรายชื่อคลาสของโมเดลจุด"""
    from ultralytics import YOLO
    cfg = build_config(args)
    for label, path in (('crop', cfg.crop_weights), ('point', cfg.point_weights)):
        if not os.path.exists(path):
            print(f'{label:6s} ไม่พบไฟล์: {path}')
            continue
        m = YOLO(path)
        args_ = (getattr(m, 'ckpt', None) or {}).get('train_args', {}) or {}
        print(f'{label:6s} {path}')
        print(f'       imgsz ตอนเทรน = {args_.get("imgsz")}')
        print(f'       คลาส = {m.names}')
        if label == 'point':
            print('       -> เลือก id ของคลาส R แล้วส่งเข้า --r-class-id')
    return 0


def cmd_scale(args) -> int:
    """ตรวจ px/mm ของแต่ละภาพก่อนรันจริง"""
    cfg = build_config(args)
    paths = list_images(args.images)
    if not paths:
        print(f'ไม่พบภาพใน {args.images}')
        return 1
    for p in paths:
        img = imread_u(p)
        ppm = estimate_px_per_mm(img)
        h, w = img.shape[:2]
        print(f'{os.path.basename(p):34s} {w}x{h}  px/mm={ppm:.2f}' if ppm
              else f'{os.path.basename(p):34s} {w}x{h}  วัดกริดไม่ได้')
    return 0


def cmd_calibrate(args) -> int:
    """วัดค่าจากภาพชุดเทรนโมเดลจุดหนึ่งภาพ เพื่อตั้ง train_* ใน Config"""
    cfg = build_config(args)
    img = imread_u(args.image)
    if img is None:
        print(f'อ่านไม่ได้: {args.image}')
        return 1
    h, w = img.shape[:2]
    ppm = estimate_px_per_mm(img)
    a = find_r_anchor(img, (0, 0, w, h), cfg)
    print(f'ขนาดภาพ {w}x{h}')
    if ppm:
        print(f'train_px_per_mm    = {ppm:.2f}   (ครอปครอบคลุม {w / ppm:.1f} mm)')
    else:
        print('วัดระยะกริดไม่ได้ ตั้ง train_px_per_mm เอง')
    if a:
        print(f'train_anchor_xfrac = {a[0] / w:.3f}')
        print(f'train_anchor_yfrac = {a[1] / h:.3f}')
    return 0


def cmd_detect(args) -> int:
    from .pipeline import detect_r_peaks, load_models
    cfg = build_config(args)
    paths = list_images(args.images)
    if not paths:
        print(f'ไม่พบภาพใน {args.images}')
        return 1
    models = load_models(cfg)
    if not models.has_point:
        print(f'ไม่พบ weights ของโมเดลจุด ({cfg.point_weights}) '
              f'-> ใช้ anchor จาก image processing อย่างเดียว')
    if models.has_point and cfg.r_class_id is None:
        print('เตือน: ไม่ได้ตั้ง --r-class-id โมเดลจุดตรวจ landmark หลายชนิด '
              '(P/Q/R/S/T) การไม่กรองคลาสอาจได้จุดที่ไม่ใช่ R')

    rows = []
    for p in paths:
        r = detect_r_peaks(p, models, cfg)
        s = r['stats']
        rows += result_to_rows(p, r, cfg)
        hr = median_hr(r, cfg)
        ppm = f"{s['px_per_mm']:.1f}" if s['px_per_mm'] else '?'
        _, ok = check_scale(s['px_per_mm'], None)
        print(f"{os.path.basename(p):30s} beats={s['n_boxes']:3d} R={s['n_peaks']:3d} "
              f"model={s['n_model']:3d} anchor={s['n_anchor']:3d} "
              f"reject={s['n_reject']:2d} dup={s['n_dup']:2d} px/mm={ppm}"
              + (f'  HR~{hr:.0f}bpm' if hr else ''))
    write_csv(rows, args.out)
    print(f'บันทึก {len(rows)} จุด -> {args.out}')
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='ekg_rpeak', description='EKG R-peak two-stage pipeline')
    ap.add_argument('--crop-weights', dest='crop_weights')
    ap.add_argument('--point-weights', dest='point_weights')
    ap.add_argument('--crop-conf', dest='crop_conf', type=float)
    ap.add_argument('--point-conf', dest='point_conf', type=float)
    ap.add_argument('--r-class-id', dest='r_class_id', type=int,
                    help='id ของคลาส R ในโมเดลจุด (ดูจากคำสั่ง info)')
    ap.add_argument('--crop-mode', dest='crop_mode',
                    choices=['mm', 'height', 'pitch', 'box', 'stretch'])
    ap.add_argument('--point-mode', dest='point_mode',
                    choices=['refine', 'model_only', 'anchor_only'])
    ap.add_argument('--point-pre', dest='point_pre',
                    choices=['gray', 'gray_contrast', 'none'])
    ap.add_argument('--px-per-mm', dest='px_per_mm', type=float)
    ap.add_argument('--train-px-per-mm', dest='train_px_per_mm', type=float)

    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('info', help='พิมพ์ imgsz และรายชื่อคลาสของ weights').set_defaults(fn=cmd_info)

    p = sub.add_parser('scale', help='ตรวจ px/mm ของภาพ')
    p.add_argument('--images', default='data')
    p.set_defaults(fn=cmd_scale)

    p = sub.add_parser('calibrate', help='วัดค่าจากภาพชุดเทรนโมเดลจุด')
    p.add_argument('--image', required=True)
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser('detect', help='ตรวจหาจุด R แล้ว export CSV')
    p.add_argument('--images', default='data')
    p.add_argument('--out', default='out/r_peaks.csv')
    p.set_defaults(fn=cmd_detect)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
