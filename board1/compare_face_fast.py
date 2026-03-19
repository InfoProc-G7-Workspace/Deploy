"""
Lightweight face comparison for PYNQ.
Haar Cascade face/eye detection + ONNX Runtime inference.

Usage:
    # Face verification (compare two faces)
    python3 compare_face_fast.py verify face1.jpg face2.jpg --onnx mobilefacenet_pynq_fp32.onnx

    # Face detection (detect all faces and extract embeddings)
    python3 compare_face_fast.py detect photo.jpg --onnx mobilefacenet_pynq_fp32.onnx
"""

import argparse
import cv2
import numpy as np
import time
import os

HAAR_SEARCH_PATHS = [
    '/usr/share/opencv4/haarcascades/',
    '/usr/share/opencv/haarcascades/',
    '/usr/local/share/opencv4/haarcascades/',
    '/usr/local/share/OpenCV/haarcascades/',
]

def find_haar_file(filename):
    """Search common paths for a Haar cascade XML file."""
    try:
        path = cv2.data.haarcascades + filename
        if os.path.exists(path):
            return path
    except AttributeError:
        pass

    for base in HAAR_SEARCH_PATHS:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path

    import subprocess
    try:
        result = subprocess.run(
            ['find', '/usr', '-name', filename, '-type', 'f'],
            capture_output=True, text=True, timeout=5
        )
        paths = result.stdout.strip().split('\n')
        if paths and paths[0]:
            return paths[0]
    except Exception:
        pass

    return None


def load_detectors():
    """Load face and eye Haar cascade detectors."""
    face_xml = find_haar_file('haarcascade_frontalface_default.xml')
    eye_xml = find_haar_file('haarcascade_eye.xml')

    if face_xml is None:
        raise RuntimeError(
            "Cannot find haarcascade_frontalface_default.xml\n"
            "Try: sudo apt install libopencv-data"
        )

    face_cascade = cv2.CascadeClassifier(face_xml)
    print(f"  Face model: {face_xml}")

    eye_cascade = None
    if eye_xml:
        eye_cascade = cv2.CascadeClassifier(eye_xml)
        print(f"  Eye model: {eye_xml}")
    else:
        print("  Eye model: not found (skipping eye alignment)")

    return face_cascade, eye_cascade


def _detect_faces_raw(img, face_cascade):
    """Detect all faces in an image. Returns list of (x, y, w, h) in original coords."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    scale = 1.0
    if max(img.shape[:2]) > 640:
        scale = 640.0 / max(img.shape[:2])
        gray_small = cv2.resize(gray, None, fx=scale, fy=scale)
    else:
        gray_small = gray

    faces = face_cascade.detectMultiScale(
        gray_small,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(30, 30),
    )

    if len(faces) == 0:
        return []

    # Map back to original image coordinates
    result = []
    for (x, y, w, h) in faces:
        result.append((int(x / scale), int(y / scale),
                        int(w / scale), int(h / scale)))
    return result


def _align_and_crop(img, fx, fy, fw, fh, eye_cascade=None):
    """Align by eye positions and crop a single face to 112x96."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if eye_cascade is not None:
        face_roi_gray = gray[fy:fy + fh // 2 + 10, fx:fx + fw]
        eyes = eye_cascade.detectMultiScale(
            face_roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(int(fw * 0.1), int(fw * 0.1)),
        )

        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])
            left_eye = eyes[0]
            right_eye = eyes[-1]

            lx = fx + left_eye[0] + left_eye[2] // 2
            ly = fy + left_eye[1] + left_eye[3] // 2
            rx = fx + right_eye[0] + right_eye[2] // 2
            ry = fy + right_eye[1] + right_eye[3] // 2

            angle = np.degrees(np.arctan2(ry - ly, rx - lx))

            if abs(angle) > 0.5:
                eye_center = (int((lx + rx) // 2), int((ly + ry) // 2))
                M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
                img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    y1 = max(0, fy - int(fh * 0.15))
    y2 = min(img.shape[0], fy + fh + int(fh * 0.05))
    x1 = max(0, fx - int(fw * 0.05))
    x2 = min(img.shape[1], fx + fw + int(fw * 0.05))

    face = img[y1:y2, x1:x2]
    face = cv2.resize(face, (96, 112))

    return face


def detect_and_align(img, face_cascade, eye_cascade=None):
    """Detect the largest face, align and crop to 112x96. Returns None on failure."""
    faces = _detect_faces_raw(img, face_cascade)
    if not faces:
        return None

    # Pick the largest face
    areas = [w * h for (x, y, w, h) in faces]
    idx = np.argmax(areas)
    fx, fy, fw, fh = faces[idx]

    return _align_and_crop(img, fx, fy, fw, fh, eye_cascade)


def detect_and_align_all(img, face_cascade, eye_cascade=None):
    """Detect all faces, align and crop each to 112x96.
    Returns list of (bbox, face_img) where bbox is (x, y, w, h)."""
    faces = _detect_faces_raw(img, face_cascade)
    if not faces:
        return []

    results = []
    for (fx, fy, fw, fh) in faces:
        face_img = _align_and_crop(img.copy(), fx, fy, fw, fh, eye_cascade)
        results.append(((fx, fy, fw, fh), face_img))
    return results


def preprocess(face_img):
    """112x96 BGR -> (1, 3, 112, 96) float32"""
    img = face_img.astype(np.float32)
    img = (img - 127.5) / 128.0
    img = img.transpose(2, 0, 1)
    return img[np.newaxis, :, :, :]


def extract_feature(session, face_img):
    """Extract feature: original + flipped, concat to 128-dim."""
    input_name = session.get_inputs()[0].name
    feat = session.run(None, {input_name: preprocess(face_img)})[0]
    feat_flip = session.run(None, {input_name: preprocess(face_img[:, ::-1, :].copy())})[0]
    return np.concatenate([feat, feat_flip], axis=1)[0]


def detect(img_path, onnx_path, save_aligned=True):
    """Detect all faces in an image and extract embeddings."""
    result_dir = 'result'
    try:
        os.makedirs(result_dir, exist_ok=True)
    except PermissionError:
        save_aligned = False

    print('--- Loading models ---')
    face_cascade, eye_cascade = load_detectors()

    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path)
    print(f"  ONNX model: {onnx_path}")

    print('\n--- Face detection ---')
    t0 = time.time()
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Cannot read: {img_path}')

    face_results = detect_and_align_all(img, face_cascade, eye_cascade)
    t1 = time.time()

    if not face_results:
        print(f'  No faces found in {img_path}')
        return []

    print(f'  Found {len(face_results)} face(s) in {t1-t0:.3f}s')

    print('\n--- Feature extraction ---')
    embeddings = []
    t2 = time.time()
    for i, (bbox, face_img) in enumerate(face_results):
        feat = extract_feature(session, face_img)
        feat = feat / np.linalg.norm(feat)
        embeddings.append({'index': i, 'bbox': bbox, 'embedding': feat})

        if save_aligned:
            cv2.imwrite(os.path.join(result_dir, f'face_{i}.jpg'), face_img)
    t3 = time.time()

    print(f'  Dim: {embeddings[0]["embedding"].shape[0]}')
    print(f'  Time: {t3-t2:.3f}s ({len(embeddings)} faces)')

    total_time = time.time() - t0
    print(f'\n--- Result ---')
    print(f'Image: {img_path}')
    print(f'Faces: {len(embeddings)}')
    for e in embeddings:
        x, y, w, h = e['bbox']
        emb = e['embedding']
        emb_str = ', '.join(f'{v:.4f}' for v in emb[:4]) + f', ... , {emb[-1]:.4f}'
        print(f'  Face {e["index"]}: bbox=({x},{y},{w},{h})  emb=[{emb_str}]')
    print(f'Total: {total_time:.3f}s')

    return embeddings


def compare(img_path_a, img_path_b, onnx_path, save_aligned=True):
    """Compare two images by their largest face."""
    result_dir = 'result'
    try:
        os.makedirs(result_dir, exist_ok=True)
    except PermissionError:
        save_aligned = False

    print('--- Loading models ---')
    face_cascade, eye_cascade = load_detectors()

    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path)
    print(f"  ONNX model: {onnx_path}")

    print('\n--- Face detection ---')

    t0 = time.time()
    img_a = cv2.imread(img_path_a)
    if img_a is None:
        raise FileNotFoundError(f'Cannot read: {img_path_a}')
    face_a = detect_and_align(img_a, face_cascade, eye_cascade)
    t1 = time.time()

    img_b = cv2.imread(img_path_b)
    if img_b is None:
        raise FileNotFoundError(f'Cannot read: {img_path_b}')
    face_b = detect_and_align(img_b, face_cascade, eye_cascade)
    t2 = time.time()

    if face_a is None:
        print(f'  Warning: no face in {img_path_a}, resizing whole image')
        face_a = cv2.resize(img_a, (96, 112))
    else:
        print(f'  A done: {t1-t0:.3f}s')

    if face_b is None:
        print(f'  Warning: no face in {img_path_b}, resizing whole image')
        face_b = cv2.resize(img_b, (96, 112))
    else:
        print(f'  B done: {t2-t1:.3f}s')

    if save_aligned:
        cv2.imwrite(os.path.join(result_dir, 'aligned_a.jpg'), face_a)
        cv2.imwrite(os.path.join(result_dir, 'aligned_b.jpg'), face_b)

    print('\n--- Feature extraction ---')
    t3 = time.time()
    feat_a = extract_feature(session, face_a)
    t4 = time.time()
    feat_b = extract_feature(session, face_b)
    t5 = time.time()
    print(f'  Dim: {feat_a.shape[0]}')
    print(f'  Time: A={t4-t3:.3f}s  B={t5-t4:.3f}s')

    feat_a = feat_a / np.linalg.norm(feat_a)
    feat_b = feat_b / np.linalg.norm(feat_b)
    similarity = float(np.dot(feat_a, feat_b))

    total_time = time.time() - t0
    print(f'\n--- Result ---')
    print(f'Image A: {img_path_a}')
    print(f'Image B: {img_path_b}')
    print(f'Similarity: {similarity:.4f}')
    print(f'Total: {total_time:.3f}s  (detect: {t2-t0:.3f}s  infer: {t5-t3:.3f}s)')

    threshold = 0.5
    if similarity > threshold:
        print(f'Same person ({similarity:.4f} > {threshold})')
    else:
        print(f'Different person ({similarity:.4f} <= {threshold})')

    return similarity


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face tools - lightweight PYNQ version')
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # verify subcommand
    p_verify = subparsers.add_parser('verify', help='Compare two faces (largest face per image)')
    p_verify.add_argument('img_a', type=str, help='First image path')
    p_verify.add_argument('img_b', type=str, help='Second image path')
    p_verify.add_argument('--onnx', type=str, default='mobilefacenet_pynq_fp32.onnx',
                          help='ONNX model path')
    p_verify.add_argument('--no-save', action='store_true', help='Do not save aligned images')

    # detect subcommand
    p_detect = subparsers.add_parser('detect', help='Detect all faces and extract embeddings')
    p_detect.add_argument('image', type=str, help='Image path')
    p_detect.add_argument('--onnx', type=str, default='mobilefacenet_pynq_fp32.onnx',
                          help='ONNX model path')
    p_detect.add_argument('--no-save', action='store_true', help='Do not save aligned face images')

    args = parser.parse_args()

    if args.mode == 'verify':
        compare(args.img_a, args.img_b, args.onnx, save_aligned=not args.no_save)
    elif args.mode == 'detect':
        detect(args.image, args.onnx, save_aligned=not args.no_save)