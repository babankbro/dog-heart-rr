"""บรรทัดคำสั่งของไปป์ไลน์

    python -m ekg_rpeak.cli info
    python -m ekg_rpeak.cli scale --images data
    python -m ekg_rpeak.cli calibrate --image train_sample.png
    python -m ekg_rpeak.cli detect --images data --out out/r_peaks.csv
    python -m ekg_rpeak.cli import-tree --src "/src" --data data --replace
    python -m ekg_rpeak.cli warm --images data --out out
    python -m ekg_rpeak.cli domain --images data --ref artifacts/yolo11m_crop_clean-2
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
              'train_px_per_mm', 'point_pre', 'crop_pre', 'crop_pre_hyst',
              'crop_pre_close', 'crop_pre_dilate'):
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


def cmd_import_tree(args) -> int:
    """ดึงภาพจากโครงโฟลเดอร์ <ประเภท>/<รหัส ชื่อ>/<crop> เข้าทะเบียนสัตว์"""
    from .importer import import_tree
    r = import_tree(args.src, args.data, subdir=args.subdir,
                    replace=args.replace, limit=args.limit)
    by_group = {}
    for a in r['added']:
        by_group.setdefault(a['group'], []).append(a)
    for group, items in sorted(by_group.items()):
        n_img = sum(len(a['images']) for a in items)
        print(f'{group or "(ไม่มีประเภท)"}: {len(items)} ตัว {n_img} ภาพ')
        for a in items:
            print(f'   {a["id"]:10s} {a["name"][:24]:24s} {len(a["images"])} ภาพ')
    for sk in r['skipped']:
        print(f'ข้าม {sk["folder"]}: {sk["reason"]}')
    print()
    print(f'รวม {r["n_patients"]} ตัว {r["n_images"]} ภาพ -> {args.data}')
    return 0


def cmd_warm(args) -> int:
    """รันทุกภาพไว้ล่วงหน้าแล้วเก็บผลลงดิสก์

    หน้าเว็บอ่านผลชุดเดียวกันนี้ ผู้ใช้จึงเปิดดูได้ทันทีโดยไม่ต้องกดวิเคราะห์
    ภาพที่มีผลตรงกับค่าตั้งปัจจุบันอยู่แล้วจะถูกข้าม สั่งซ้ำได้ไม่เสียเวลา
    """
    from .pipeline import detect_r_peaks, load_models
    from . import results as rs
    cfg = build_config(args)
    paths = list_images(args.images)
    if not paths:
        print(f'ไม่พบภาพใน {args.images}')
        return 1

    todo = []
    for p in paths:
        name = os.path.relpath(p, args.images).replace(os.sep, '/')
        if not args.force and rs.load(args.out, name, image_path=p, cfg=cfg):
            print(f'ข้าม (มีผลอยู่แล้ว) {name}')
            continue
        todo.append((name, p))
    if not todo:
        print(f'ครบแล้วทั้ง {len(paths)} ภาพ ไม่มีอะไรต้องรัน')
        return 0

    models = load_models(cfg)
    fails = []
    for i, (name, p) in enumerate(todo, 1):
        try:
            result = detect_r_peaks(p, models, cfg)
            rows = result_to_rows(p, result, cfg)
            rs.save(args.out, name, p, cfg, result, rows)
            s = result['stats']
            hr = median_hr(result, cfg)
            print(f'[{i}/{len(todo)}] {name}  beats={s["n_boxes"]:3d} R={s["n_peaks"]:3d} '
                  f'model={s["n_model"]:3d} anchor={s["n_anchor"]:3d}'
                  + (f'  HR~{hr:.0f}bpm' if hr else '  HR วัดไม่ได้'))
        except Exception as e:                   # ภาพเสียหนึ่งใบต้องไม่ล้มทั้งชุด
            fails.append((name, str(e)))
            print(f'[{i}/{len(todo)}] {name}  ล้มเหลว: {e}')
    print()
    print(f'สำเร็จ {len(todo) - len(fails)}/{len(todo)} ภาพ -> {rs.store_dir(args.out)}')
    for name, err in fails:
        print(f'  ล้มเหลว {name}: {err}')
    return 1 if fails else 0


def cmd_domain(args) -> int:
    """เทียบหน้าตาของภาพที่จะป้อนโมเดล กับภาพที่โมเดลเห็นตอนเทรน

    ใช้ตอนเอาโมเดลไปใช้กับชุดข้อมูลใหม่ ตัวเลขที่ห่างกันมากบอกได้ว่ากล่องจะเพี้ยน
    ก่อนจะเสียเวลาไล่ปรับค่าทีละตัว
    """
    from . import domain as dm
    cfg = build_config(args)
    test = dm.summarize(dm.image_features(args.images, cfg))
    if not test:
        print(f'ไม่พบภาพใน {args.images}')
        return 1
    ref = dm.summarize(dm.train_features(args.ref)) if args.ref else {}

    cols = ['ink', 'w_p90', 'h_med', 'h_p90', 'strip_h']
    print(f'{"ชุด":28s} ' + ' '.join(f'{c:>8s}' for c in cols))
    if ref:
        print(f'{"เทรน (val_batch)":28s} ' + ' '.join(f'{ref[c]:8.3f}' for c in cols))
    print(f'{args.images[:28]:28s} ' + ' '.join(f'{test[c]:8.3f}' for c in cols))
    if ref:
        print()
        print(f'ระยะห่างของโดเมน = {dm.distance(test, ref):.3f}  (0 = เหมือนกัน)')
        if test['h_med'] < ref['h_med']:
            print('เส้นในชุดนี้ต่อเนื่องน้อยกว่าที่โมเดลเคยเห็น — กริดมาเป็นจุดประ ไม่ใช่เส้นทึบ')
        if test['ink'] < ref['ink'] * 0.6:
            print('ภาพชุดนี้ "จาง" กว่าที่โมเดลเคยเห็นมาก')
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
    ap.add_argument('--crop-pre', dest='crop_pre',
                    choices=['blackhat', 'blackhat_otsu', 'tophat_gray', 'adaptive', 'ink', 'red'],
                    help='binarization ของภาพที่ป้อนโมเดลครอป (ค่าเริ่มต้นตรงกับชุดเทรน)')
    ap.add_argument('--crop-pre-hyst', dest='crop_pre_hyst', type=float,
                    help='ผ่อนเกณฑ์แบบ hysteresis เพื่อเก็บปลายยอดที่หมึกจาง')
    ap.add_argument('--crop-pre-close', dest='crop_pre_close', type=int,
                    help='เชื่อมเส้นที่ขาดเป็นช่วง ๆ')
    ap.add_argument('--crop-pre-dilate', dest='crop_pre_dilate', type=int,
                    help='ทำให้เส้นหนาขึ้นหลัง binarize')
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

    p = sub.add_parser('import-tree', help='ดึงภาพจากโครงโฟลเดอร์ภายนอกเข้าทะเบียนสัตว์')
    p.add_argument('--src', required=True, help='รากของโครง <ประเภท>/<รหัส ชื่อ>/<crop>')
    p.add_argument('--data', default='data')
    p.add_argument('--subdir', default='crop', help='ชื่อโฟลเดอร์ภาพที่จะเอา (ไม่สนตัวพิมพ์)')
    p.add_argument('--limit', type=int, default=0, help='จำกัดจำนวนภาพต่อตัว (0 = ทั้งหมด)')
    p.add_argument('--replace', action='store_true', help='ลบภาพเดิมของรหัสที่ชนกันก่อนใส่ชุดใหม่')
    p.set_defaults(fn=cmd_import_tree)

    p = sub.add_parser('warm', help='รันทุกภาพไว้ล่วงหน้าแล้วเก็บผลลงดิสก์')
    p.add_argument('--images', default='data')
    p.add_argument('--out', default='out')
    p.add_argument('--force', action='store_true', help='รันใหม่แม้มีผลเก็บไว้แล้ว')
    p.set_defaults(fn=cmd_warm)

    p = sub.add_parser('domain', help='เทียบหน้าตาภาพกับชุดที่โมเดลเห็นตอนเทรน')
    p.add_argument('--images', default='data')
    p.add_argument('--ref', help='โฟลเดอร์ผลเทรนที่มี val_batch*_labels.jpg')
    p.set_defaults(fn=cmd_domain)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
