"""
FLIR Spinnaker Camera Scanner — Scene capture via AWS IoT Core MQTT
===================================================================
Subscribes to robot/scan for scan commands, captures high-quality stills
using a FLIR Spinnaker camera, uploads them to S3, and reports progress
on robot/scan/status.

If PySpin is not installed, runs in simulation mode — generates test
images with OpenCV so the full pipeline can be tested without hardware.

Install: pip install boto3 awsiotsdk opencv-python
         (+ PySpin from Spinnaker SDK when FLIR camera is available)
Usage:
    python flir_scanner.py              # auto-detects PySpin
    python flir_scanner.py --simulate   # force simulation mode
"""

import os
import sys
import json
import time
import uuid
import threading
import numpy as np
import cv2
import boto3
from awscrt import mqtt
from awsiot import mqtt_connection_builder
from driver.face_mjpeg_system import FaceMjpegSystem
from compare_face_fast import detect_and_align_all, extract_feature

# Try to import PySpin; fall back to simulation mode if unavailable
try:
    import PySpin
    PYSPIN_AVAILABLE = True
except ImportError:
    PYSPIN_AVAILABLE = False

# # --- ONNX inference (uncomment when model is ready) ---
# import onnxruntime as ort
# ONNX_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                                 'model', 'face_040.onnx')
# _ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
# print(f'[FACE] ONNX model loaded: {ONNX_MODEL_PATH}')
# # --- End ONNX init ---

# --- FPGA face detection (Haar hardware accelerated) ---
BITSTREAM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'bitstream', 'design_1.bit')
_fpga_system = FaceMjpegSystem(BITSTREAM_PATH)
print(f'[FACE] FPGA Haar loaded: {_fpga_system.num_haar_engines} engines')

SIMULATE = '--simulate' in sys.argv or not PYSPIN_AVAILABLE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CERTS_DIR = os.path.join(SCRIPT_DIR, '..', 'certs')
KEYS_PATH = os.path.join(os.path.expanduser('~'), 'keys.txt')

SCAN_TOPIC = 'robot/scan'
SCAN_STATUS_TOPIC = 'robot/scan/status'
JPEG_QUALITY = 95


def load_keys():
    """Load KEY=VALUE pairs from ~/keys.txt."""
    keys = {}
    if os.path.isfile(KEYS_PATH):
        with open(KEYS_PATH) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    k, v = line.split('=', 1)
                    keys[k.strip()] = v.strip()
    return keys


_keys = load_keys()
IOT_ENDPOINT = os.getenv('IOT_ENDPOINT', _keys.get('IOT_ENDPOINT', ''))
AWS_REGION = os.getenv('AWS_REGION', _keys.get('AWS_REGION', 'eu-west-2'))
IMAGE_BUCKET = os.getenv('IMAGE_BUCKET', _keys.get('IMAGE_BUCKET', 'robot-raw-images-eu-west-2'))

# boto3 clients (credentials from keys.txt or environment)
_aws_kwargs = {'region_name': AWS_REGION}
_ak = os.getenv('AWS_ACCESS_KEY_ID', _keys.get('AWS_ACCESS_KEY_ID', ''))
_sk = os.getenv('AWS_SECRET_ACCESS_KEY', _keys.get('AWS_SECRET_ACCESS_KEY', ''))
if _ak and _sk:
    _aws_kwargs['aws_access_key_id'] = _ak
    _aws_kwargs['aws_secret_access_key'] = _sk

s3 = boto3.client('s3', **_aws_kwargs)
dynamodb = boto3.resource('dynamodb', **_aws_kwargs)
sessions_table = dynamodb.Table('sessions')

# Global MQTT connection (set in main)
_connection = None
_scanning = False


def publish_status(scene_id, status, current_image=0, total_images=0, message=''):
    """Publish scan progress to robot/scan/status."""
    payload = {
        'status': status,
        'scene_id': scene_id,
        'current_image': current_image,
        'total_images': total_images,
    }
    if message:
        payload['message'] = message
    _connection.publish(
        topic=SCAN_STATUS_TOPIC,
        payload=json.dumps(payload),
        qos=mqtt.QoS.AT_LEAST_ONCE,
    )


def extract_session_id(image_s3_prefix):
    """Extract session_id from image_s3_prefix (user_id/session_id/)."""
    parts = image_s3_prefix.strip('/').split('/')
    if len(parts) >= 2:
        return parts[1]
    return None


def increment_image_count(session_id):
    """Atomically increment image_count in DynamoDB sessions table."""
    sessions_table.update_item(
        Key={'session_id': session_id},
        UpdateExpression='ADD image_count :one',
        ExpressionAttributeValues={':one': 1},
    )


def generate_test_image(frame_number, total, scene_name):
    """Generate a labelled test image for simulation mode."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Gradient background
    img[:, :, 0] = np.linspace(30, 80, 480, dtype=np.uint8).reshape(-1, 1)
    img[:, :, 1] = np.linspace(20, 60, 640, dtype=np.uint8).reshape(1, -1)
    img[:, :, 2] = 40
    # Text overlay
    cv2.putText(img, f'SIMULATED FLIR CAPTURE', (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 212, 255), 2)
    cv2.putText(img, f'Scene: {scene_name}', (40, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(img, f'Frame {frame_number} / {total}', (40, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(img, time.strftime('%Y-%m-%d %H:%M:%S'), (40, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    return img


def process_faces(image_data, image_s3_prefix, frame_index):
    """Detect faces via FPGA Haar, crop, UUID-name, upload to S3."""
    face_results = detect_and_align_all(image_data, _fpga_system)
    if not face_results:
        print(f'[FACE] Frame {frame_index}: no faces detected')
        return []

    faces_info = []
    for bbox, face_img in face_results:
        face_id = str(uuid.uuid4())

        # # --- Embedding extraction (uncomment when ONNX model is ready) ---
        # feat = extract_feature(_ort_session, face_img)
        # feat = feat / np.linalg.norm(feat)
        # embedding_list = feat.tolist()
        # # --- End embedding ---
        embedding_list = []

        # Encode aligned face as JPEG
        _, face_jpeg = cv2.imencode('.jpg', face_img,
                                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        # Upload: <prefix>faces/<uuid>.jpg
        s3_key = f'{image_s3_prefix}faces/{face_id}.jpg'
        s3.put_object(
            Bucket=IMAGE_BUCKET,
            Key=s3_key,
            Body=face_jpeg.tobytes(),
            ContentType='image/jpeg',
        )

        x, y, w, h = bbox
        faces_info.append({
            'face_id': face_id,
            'bbox': [x, y, w, h],
            'embedding': embedding_list,
            's3_key': s3_key,
        })
        print(f'[FACE] Frame {frame_index}: {face_id} bbox=({x},{y},{w},{h}) → s3://{IMAGE_BUCKET}/{s3_key}')

    # Upload metadata JSON
    meta_key = f'{image_s3_prefix}faces/frame{str(frame_index).zfill(6)}_meta.json'
    s3.put_object(
        Bucket=IMAGE_BUCKET,
        Key=meta_key,
        Body=json.dumps(faces_info).encode(),
        ContentType='application/json',
    )

    # Publish face detection result via MQTT
    _connection.publish(
        topic='robot/scan/faces',
        payload=json.dumps({
            'scene_id': '',
            'frame': frame_index,
            'faces': [{'face_id': f['face_id'], 'bbox': f['bbox'],
                        's3_key': f['s3_key']} for f in faces_info],
        }),
        qos=mqtt.QoS.AT_LEAST_ONCE,
    )
    return faces_info


def capture_flir_frame(cam, processor):
    """Capture a single frame from a FLIR camera via PySpin."""
    cam.BeginAcquisition()
    image_result = cam.GetNextImage()

    if image_result.IsIncomplete():
        image_result.Release()
        cam.EndAcquisition()
        return None

    # Convert Bayer to BGR8 using ImageProcessor
    image_converted = processor.Convert(image_result, PySpin.PixelFormat_BGR8)
    image_data = image_converted.GetNDArray()
    image_result.Release()
    cam.EndAcquisition()
    return image_data


def run_scan(command):
    """Execute a scan: capture images and upload to S3."""
    global _scanning

    scene_id = command.get('scene_id', '')
    scene_name = command.get('scene_name', '')
    image_s3_prefix = command.get('image_s3_prefix', '')
    total_images = int(command.get('total_images', 1))
    image_interval = float(command.get('image_interval', 5))
    session_id = extract_session_id(image_s3_prefix)

    mode_label = 'SIMULATION' if SIMULATE else 'FLIR'
    print(f'[SCANNER] Starting scan ({mode_label}): scene="{scene_name}" ({scene_id})')
    print(f'[SCANNER]   S3 prefix: {image_s3_prefix}')
    print(f'[SCANNER]   Images: {total_images}, Interval: {image_interval}s')

    cam = None
    system = None
    cam_list = None

    if not SIMULATE:
        # Initialise PySpin
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()

        if cam_list.GetSize() == 0:
            print('[SCANNER] ERROR: No FLIR cameras detected')
            publish_status(scene_id, 'error', message='No FLIR cameras detected')
            cam_list.Clear()
            system.ReleaseInstance()
            _scanning = False
            return

        cam = cam_list[0]
        cam.Init()

        nodemap = cam.GetNodeMap()

        # Set pixel format to BayerRG8 (camera native) — conversion to BGR8
        # happens in capture_flir_frame via PySpin's image processor
        try:
            pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
            if PySpin.IsAvailable(pixel_format) and PySpin.IsWritable(pixel_format):
                entry = pixel_format.GetEntryByName('BayerRG8')
                if PySpin.IsAvailable(entry):
                    pixel_format.SetIntValue(entry.GetValue())
                    print('[SCANNER] Pixel format set to BayerRG8')
        except PySpin.SpinnakerException as e:
            print(f'[SCANNER] WARNING: Could not set pixel format: {e}')

        # Enable auto white balance for correct color
        try:
            balance_white_auto = PySpin.CEnumerationPtr(nodemap.GetNode('BalanceWhiteAuto'))
            if PySpin.IsAvailable(balance_white_auto) and PySpin.IsWritable(balance_white_auto):
                entry = balance_white_auto.GetEntryByName('Continuous')
                if PySpin.IsAvailable(entry):
                    balance_white_auto.SetIntValue(entry.GetValue())
                    print('[SCANNER] Balance White Auto set to Continuous')
        except PySpin.SpinnakerException as e:
            print(f'[SCANNER] WARNING: Could not set white balance: {e}')

        cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_SingleFrame)

        # Create image processor for Bayer -> BGR8 conversion
        processor = PySpin.ImageProcessor()
        processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)

    try:
        for i in range(1, total_images + 1):
            t0 = time.monotonic()

            if SIMULATE:
                image_data = generate_test_image(i, total_images, scene_name)
            else:
                image_data = capture_flir_frame(cam, processor)
                if image_data is None:
                    print(f'[SCANNER] WARNING: Frame {i} incomplete, skipping')
                    continue

            # Encode as high-quality JPEG
            _, jpeg_buf = cv2.imencode('.jpg', image_data,
                                       [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            jpeg_bytes = jpeg_buf.tobytes()

            # Upload to S3
            padded = str(i).zfill(6)
            s3_key = f'{image_s3_prefix}frame_{padded}.jpg'
            s3.put_object(
                Bucket=IMAGE_BUCKET,
                Key=s3_key,
                Body=jpeg_bytes,
                ContentType='image/jpeg',
            )

            # Update DynamoDB image_count
            if session_id:
                increment_image_count(session_id)

            print(f'[SCANNER] Captured & uploaded frame {i}/{total_images} → s3://{IMAGE_BUCKET}/{s3_key}')

            # Face detection (FPGA Haar) + upload per frame
            process_faces(image_data, image_s3_prefix, i)

            # Publish progress
            publish_status(scene_id, 'capturing', current_image=i, total_images=total_images)

            # Wait for the interval (minus elapsed time)
            if i < total_images:
                elapsed = time.monotonic() - t0
                sleep_for = image_interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

        # Scan complete
        print(f'[SCANNER] Scan complete: {total_images} images captured for "{scene_name}"')
        publish_status(scene_id, 'complete', current_image=total_images, total_images=total_images)

    except Exception as e:
        print(f'[SCANNER] ERROR during scan: {e}')
        publish_status(scene_id, 'error', message=str(e))
    finally:
        if cam is not None:
            try:
                cam.EndAcquisition()
            except PySpin.SpinnakerException:
                pass
            cam.DeInit()
            del cam
        if cam_list is not None:
            cam_list.Clear()
        if system is not None:
            system.ReleaseInstance()
        _scanning = False


def on_scan_message(topic, payload, **kwargs):
    """Handle incoming scan commands on robot/scan topic."""
    global _scanning

    try:
        command = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        print(f'[SCANNER] Bad message on {topic}: {payload}')
        return

    action = command.get('action', '')
    if action != 'start_scan':
        print(f'[SCANNER] Unknown action: {action}')
        return

    if _scanning:
        print('[SCANNER] Scan already in progress, ignoring new command')
        return

    _scanning = True
    # Run scan in a separate thread so MQTT stays responsive
    thread = threading.Thread(target=run_scan, args=(command,), daemon=True)
    thread.start()


def main():
    global _connection

    if not IOT_ENDPOINT:
        raise RuntimeError('Set IOT_ENDPOINT env var or in ~/keys.txt')

    cert_path = os.path.join(CERTS_DIR, 'device-cert.pem')
    key_path = os.path.join(CERTS_DIR, 'private-key.pem')
    ca_path = os.path.join(CERTS_DIR, 'root-ca.pem')

    for p in (cert_path, key_path, ca_path):
        if not os.path.isfile(p):
            raise RuntimeError(f'Missing certificate: {p}')

    _connection = mqtt_connection_builder.mtls_from_path(
        endpoint=IOT_ENDPOINT,
        cert_filepath=cert_path,
        pri_key_filepath=key_path,
        ca_filepath=ca_path,
        client_id='flir-scanner-' + str(int(time.time())),
        clean_session=True,
        keep_alive_secs=30,
    )

    _connection.connect().result()
    print(f'[SCANNER] Connected to IoT Core — listening on {SCAN_TOPIC}')

    subscribe_future, _ = _connection.subscribe(
        topic=SCAN_TOPIC,
        qos=mqtt.QoS.AT_LEAST_ONCE,
        callback=on_scan_message,
    )
    subscribe_future.result()
    print(f'[SCANNER] Subscribed to {SCAN_TOPIC}')
    print(f'[SCANNER] Mode: {"SIMULATION (no PySpin)" if SIMULATE else "FLIR Spinnaker"}')
    print(f'[SCANNER] S3 bucket: {IMAGE_BUCKET}')
    print(f'[SCANNER] Waiting for scan commands...')

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('[SCANNER] Shutting down...')
    finally:
        _connection.disconnect().result()


if __name__ == '__main__':
    main()
