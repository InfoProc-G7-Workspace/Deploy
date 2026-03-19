"""
人脸比对 — 使用 FaceMjpegSystem (FPGA Haar 并行检测) + ONNX 推理

use:
    python3 compare_face_fast.py verify face1.jpg face2.jpg \
      --bit bitstream/design_1.bit --onnx model/face_040.onnx

    python3 compare_face_fast.py detect photo.jpg \
      --bit bitstream/design_1.bit --onnx model/face_040.onnx
"""

import argparse
import cv2
import numpy as np
import time
import os

from driver.face_mjpeg_system import FaceMjpegSystem


def detect_and_align(img, system):
    """FPGA 并行多尺度人脸检测，取最大脸，裁剪 112x96。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_orig, w_orig = gray.shape

    faces = system.detect(gray, scale_factor=1.15, min_size=30, min_neighbors=3)
    if len(faces) == 0:
        return None

    areas = [w * h for (x, y, w, h) in faces]
    idx = np.argmax(areas)
    fx, fy, fw, fh = faces[idx]

    fx = max(0, min(fx, w_orig - 1))
    fy = max(0, min(fy, h_orig - 1))
    fw = min(fw, w_orig - fx)
    fh = min(fh, h_orig - fy)
    if fw <= 0 or fh <= 0:
        return None

    y1 = max(0, fy - int(fh * 0.15))
    y2 = min(img.shape[0], fy + fh + int(fh * 0.05))
    x1 = max(0, fx - int(fw * 0.05))
    x2 = min(img.shape[1], fx + fw + int(fw * 0.05))
    if y2 <= y1 or x2 <= x1:
        return None

    face = img[y1:y2, x1:x2]
    return cv2.resize(face, (96, 112))


def detect_and_align_all(img, system):
    """检测所有人脸，返回 [(bbox, face_img), ...]"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_orig, w_orig = gray.shape

    faces = system.detect(gray, scale_factor=1.15, min_size=30, min_neighbors=3)
    if len(faces) == 0:
        return []

    results = []
    for (fx, fy, fw, fh) in faces:
        fx = max(0, min(fx, w_orig - 1))
        fy = max(0, min(fy, h_orig - 1))
        fw = min(fw, w_orig - fx)
        fh = min(fh, h_orig - fy)
        if fw <= 0 or fh <= 0:
            continue

        y1 = max(0, fy - int(fh * 0.15))
        y2 = min(img.shape[0], fy + fh + int(fh * 0.05))
        x1 = max(0, fx - int(fw * 0.05))
        x2 = min(img.shape[1], fx + fw + int(fw * 0.05))
        if y2 <= y1 or x2 <= x1:
            continue

        face = img[y1:y2, x1:x2].copy()
        face = cv2.resize(face, (96, 112))
        results.append(((fx, fy, fw, fh), face))

    return results


def preprocess(face_img):
    """112x96 BGR -> (1, 3, 112, 96) float32"""
    img = face_img.astype(np.float32)
    img = (img - 127.5) / 128.0
    img = img.transpose(2, 0, 1)
    return img[np.newaxis, :, :, :]


def extract_feature(session, face_img):
    """原图 + 镜像，拼接得到 embedding。"""
    input_name = session.get_inputs()[0].name
    feat = session.run(None, {input_name: preprocess(face_img)})[0]
    feat_flip = session.run(None, {input_name: preprocess(face_img[:, ::-1, :].copy())})[0]
    return np.concatenate([feat, feat_flip], axis=1)[0]


def compare(img_path_a, img_path_b, bit_path, onnx_path, save_aligned=True):
    result_dir = 'result'
    try:
        os.makedirs(result_dir, exist_ok=True)
    except PermissionError:
        save_aligned = False

    print('--- 加载模型 ---')
    print(f"  FPGA bitstream: {bit_path}")
    system = FaceMjpegSystem(bit_path)
    print(f"  FPGA 人脸检测器: 已加载 ({system.num_haar_engines} 个并行 engine)")

    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path)
    print(f"  推理模型: {onnx_path}")

    print(f'\n--- 人脸检测 (FPGA {system.num_haar_engines}x 并行) ---')

    t0 = time.time()
    img_a = cv2.imread(img_path_a)
    if img_a is None:
        raise FileNotFoundError(f'无法读取: {img_path_a}')
    face_a = detect_and_align(img_a, system)
    t1 = time.time()

    img_b = cv2.imread(img_path_b)
    if img_b is None:
        raise FileNotFoundError(f'无法读取: {img_path_b}')
    face_b = detect_and_align(img_b, system)
    t2 = time.time()

    if face_a is None:
        print(f'  警告: {img_path_a} 未检测到人脸，直接缩放')
        face_a = cv2.resize(img_a, (96, 112))
    else:
        print(f'  A 检测完成: {t1-t0:.3f}s')

    if face_b is None:
        print(f'  警告: {img_path_b} 未检测到人脸，直接缩放')
        face_b = cv2.resize(img_b, (96, 112))
    else:
        print(f'  B 检测完成: {t2-t1:.3f}s')

    if save_aligned:
        try:
            cv2.imwrite(os.path.join(result_dir, 'aligned_a.jpg'), face_a)
            cv2.imwrite(os.path.join(result_dir, 'aligned_b.jpg'), face_b)
        except Exception:
            pass

    print('\n--- 特征提取 ---')
    t3 = time.time()
    feat_a = extract_feature(session, face_a)
    t4 = time.time()
    feat_b = extract_feature(session, face_b)
    t5 = time.time()
    print(f'  维度: {feat_a.shape[0]}')
    print(f'  耗时: A={t4-t3:.3f}s  B={t5-t4:.3f}s')

    feat_a = feat_a / np.linalg.norm(feat_a)
    feat_b = feat_b / np.linalg.norm(feat_b)
    similarity = float(np.dot(feat_a, feat_b))

    total_time = time.time() - t0
    print(f'\n--- 比对结果 ---')
    print(f'图片A: {img_path_a}')
    print(f'图片B: {img_path_b}')
    print(f'相似度: {similarity:.4f}')
    print(f'总耗时: {total_time:.3f}s  (检测: {t2-t0:.3f}s  推理: {t5-t3:.3f}s)')

    threshold = 0.5
    if similarity > threshold:
        print(f'判定: 同一人 (相似度 {similarity:.4f} > 阈值 {threshold})')
    else:
        print(f'判定: 不同人 (相似度 {similarity:.4f} <= 阈值 {threshold})')

    return similarity


def detect(img_path, bit_path, onnx_path, save_aligned=True):
    """检测所有人脸并提取 embedding。"""
    result_dir = 'result'
    try:
        os.makedirs(result_dir, exist_ok=True)
    except PermissionError:
        save_aligned = False

    print('--- 加载模型 ---')
    system = FaceMjpegSystem(bit_path)
    print(f"  FPGA 人脸检测器: {system.num_haar_engines} 个并行 engine")

    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path)
    print(f"  推理模型: {onnx_path}")

    print('\n--- 人脸检测 ---')
    t0 = time.time()
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'无法读取: {img_path}')

    face_results = detect_and_align_all(img, system)
    t1 = time.time()

    if not face_results:
        print(f'  未检测到人脸: {img_path}')
        return []

    print(f'  检测到 {len(face_results)} 张人脸 ({t1-t0:.3f}s)')

    print('\n--- 特征提取 ---')
    embeddings = []
    t2 = time.time()
    for i, (bbox, face_img) in enumerate(face_results):
        feat = extract_feature(session, face_img)
        feat = feat / np.linalg.norm(feat)
        embeddings.append({'index': i, 'bbox': bbox, 'embedding': feat})

        if save_aligned:
            cv2.imwrite(os.path.join(result_dir, f'face_{i}.jpg'), face_img)
    t3 = time.time()

    print(f'  维度: {embeddings[0]["embedding"].shape[0]}')
    print(f'  耗时: {t3-t2:.3f}s ({len(embeddings)} 张脸)')

    total_time = time.time() - t0
    print(f'\n--- 结果 ---')
    print(f'图片: {img_path}')
    print(f'人脸数: {len(embeddings)}')
    for e in embeddings:
        x, y, w, h = e['bbox']
        emb = e['embedding']
        emb_str = ', '.join(f'{v:.4f}' for v in emb[:4]) + f', ... , {emb[-1]:.4f}'
        print(f'  Face {e["index"]}: bbox=({x},{y},{w},{h})  emb=[{emb_str}]')
    print(f'总耗时: {total_time:.3f}s')

    return embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='人脸比对 - FPGA 并行加速版')
    subparsers = parser.add_subparsers(dest='mode', required=True)

    p_verify = subparsers.add_parser('verify', help='比对两张人脸')
    p_verify.add_argument('img_a', type=str)
    p_verify.add_argument('img_b', type=str)
    p_verify.add_argument('--bit', type=str, default='bitstream/design_1.bit')
    p_verify.add_argument('--onnx', type=str, default='model/face_040.onnx')
    p_verify.add_argument('--no-save', action='store_true')

    p_detect = subparsers.add_parser('detect', help='检测所有人脸并提取 embedding')
    p_detect.add_argument('image', type=str)
    p_detect.add_argument('--bit', type=str, default='bitstream/design_1.bit')
    p_detect.add_argument('--onnx', type=str, default='model/face_040.onnx')
    p_detect.add_argument('--no-save', action='store_true')

    args = parser.parse_args()

    if args.mode == 'verify':
        compare(args.img_a, args.img_b, args.bit, args.onnx,
                save_aligned=not args.no_save)
    elif args.mode == 'detect':
        detect(args.image, args.bit, args.onnx,
               save_aligned=not args.no_save)
