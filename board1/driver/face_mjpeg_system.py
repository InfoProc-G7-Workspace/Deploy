"""
Unified PYNQ driver for Face Detection + MJPEG Encoding system.

Hardware:
    2x Haar Face Detection IP  (haar_face_detect_0/1 + axi_dma_0/1)
    1x MJPEG Encoder pipeline  (mjpeg_dma -> yuyv_unpack -> image_preproc -> mjpeg_enc)
    1x AXI GPIO for preproc mode (axi_gpio_mode)

Usage:
    from face_mjpeg_system import FaceMjpegSystem

    system = FaceMjpegSystem("/home/xilinx/haar_mjpeg_system.bit")

    # Face detection only
    bboxes = system.detect(gray_image)

    # JPEG encoding only (raw YUYV422 — hardware does unpacking)
    jpeg_bytes = system.encode_jpeg(yuyv_frame)

    # Combined: detect faces AND encode frame in parallel
    bboxes, jpeg_bytes = system.detect_and_encode(gray_image, yuyv_frame)

    # Change preprocessing mode (0=Normal, 1=Sharpen, 2=Blue, 3=Binarize)
    system.set_preproc_mode(1)
"""

import time
import threading
import numpy as np
from pynq import Overlay, allocate


# =========================================================================
# MJPEG Encoder register map (matches mjpeg_encoder_top.vhd / hostif.vhd)
# =========================================================================
REG_ENC_START   = 0x000   # bit[0] = SOF (write 1 to start)
REG_IMAGE_SIZE  = 0x004   # [31:16] = width, [15:0] = height
REG_ENC_STS     = 0x00C   # bit[1] = done, bit[0] = busy (read-only)
REG_COD_ADDR    = 0x010   # output base address (leave 0)
REG_ENC_LENGTH  = 0x014   # JPEG byte count (read-only, valid after done)
REG_QUANT_LUM   = 0x100   # luminance quant table  (64 x 32-bit)
REG_QUANT_CHR   = 0x200   # chrominance quant table (64 x 32-bit)

# Standard JPEG quantisation tables (ITU-T T.81, Annex K)
STD_QUANT_LUM = np.array([
    16,  11,  10,  16,  24,  40,  51,  61,
    12,  12,  14,  19,  26,  58,  60,  55,
    14,  13,  16,  24,  40,  57,  69,  56,
    14,  17,  22,  29,  51,  87,  80,  62,
    18,  22,  37,  56,  68, 109, 103,  77,
    24,  35,  55,  64,  81, 104, 113,  92,
    49,  64,  78,  87, 103, 121, 120, 101,
    72,  92,  95,  98, 112, 100, 103,  99,
], dtype=np.uint8)

STD_QUANT_CHR = np.array([
    17,  18,  24,  47,  99,  99,  99,  99,
    18,  21,  26,  66,  99,  99,  99,  99,
    24,  26,  56,  99,  99,  99,  99,  99,
    47,  66,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
], dtype=np.uint8)


def _scale_quant_table(base_table, quality):
    """Scale a quantisation table by JPEG quality factor (1-100)."""
    quality = max(1, min(100, quality))
    if quality < 50:
        scale = 5000 // quality
    else:
        scale = 200 - quality * 2
    table = ((base_table.astype(np.int32) * scale + 50) // 100).clip(1, 255)
    return table.astype(np.uint8)


# =========================================================================
# Haar Engine (reused from haar_detect_parallel_driver.py)
# =========================================================================
class _HaarEngine:
    """One Haar IP + DMA pair."""

    _REGS = {
        "AP_CTRL": 0x00,
        "NUM_SCALES": 0x10,
        "NUM_DETS": 0x18,
        "NUM_DETS_CTRL": 0x1C,
        "SCALE_WIDTHS_BASE": 0x100,
        "SCALE_HEIGHTS_BASE": 0x140,
        "SCALE_Q8_BASE": 0x180,
        "DETECTIONS_BASE": 0x400,
    }

    AP_START = 0x01
    AP_DONE = 0x02
    NUM_DETS_VLD = 0x01
    MAX_DETECTIONS = 64
    MAX_SCALES = 16

    def __init__(self, ip, dma, engine_id):
        self.ip = ip
        self.dma = dma
        self.engine_id = engine_id

    def _write_reg(self, offset, value):
        self.ip.write(offset, int(value))

    def _read_reg(self, offset):
        return int(self.ip.read(offset))

    def run(self, levels, timeout_s=5.0):
        if not levels:
            return []

        num_scales = len(levels)
        total_pixels = sum(img.shape[0] * img.shape[1] for img, _ in levels)

        in_buffer = allocate(shape=(total_pixels,), dtype=np.uint8)
        try:
            offset = 0
            for i, (level_img, _) in enumerate(levels):
                n = level_img.shape[0] * level_img.shape[1]
                np.copyto(in_buffer[offset:offset + n], level_img.reshape(-1))
                offset += n

            self._write_reg(self._REGS["NUM_SCALES"], num_scales)
            for i, (level_img, sq8) in enumerate(levels):
                h, w = level_img.shape
                self._write_reg(self._REGS["SCALE_WIDTHS_BASE"] + i * 4, w)
                self._write_reg(self._REGS["SCALE_HEIGHTS_BASE"] + i * 4, h)
                self._write_reg(self._REGS["SCALE_Q8_BASE"] + i * 4, sq8)

            self.dma.sendchannel.transfer(in_buffer)
            self._write_reg(self._REGS["AP_CTRL"], self.AP_START)
            self.dma.sendchannel.wait()

            t0 = time.monotonic()
            while (self._read_reg(self._REGS["AP_CTRL"]) & self.AP_DONE) == 0:
                if (time.monotonic() - t0) > timeout_s:
                    raise TimeoutError(
                        f"Engine {self.engine_id}: timeout waiting for ap_done")
                time.sleep(0.0001)

            return self._read_detections(timeout_s)
        finally:
            in_buffer.freebuffer()

    def _read_detections(self, timeout_s):
        t0 = time.monotonic()
        while (self._read_reg(self._REGS["NUM_DETS_CTRL"])
               & self.NUM_DETS_VLD) == 0:
            if (time.monotonic() - t0) > timeout_s:
                raise TimeoutError(
                    f"Engine {self.engine_id}: timeout waiting for num_dets")
            time.sleep(0.0001)

        num_dets = min(self._read_reg(self._REGS["NUM_DETS"]),
                       self.MAX_DETECTIONS)
        det_base = self._REGS["DETECTIONS_BASE"]

        bboxes = []
        for i in range(num_dets):
            word0 = self._read_reg(det_base + i * 8)
            word1 = self._read_reg(det_base + i * 8 + 4)
            x = word0 & 0xFFFF
            y = (word0 >> 16) & 0xFFFF
            w = word1 & 0xFFFF
            h = (word1 >> 16) & 0xFFFF
            bboxes.append((x, y, w, h))

        return bboxes


# =========================================================================
# Unified System Driver
# =========================================================================
class FaceMjpegSystem:
    """
    Combined Face Detection + MJPEG Encoding system.

    Loads a single overlay containing:
      - haar_face_detect_0/1 + axi_dma_0/1  (face detection, HP0/HP1)
      - mjpeg_dma -> yuyv_unpack -> image_preproc -> mjpeg_enc  (JPEG, HP2)
      - axi_gpio_mode (preprocessing mode select)

    Face detection and JPEG encoding use separate HP ports, enabling
    true hardware parallelism via detect_and_encode().

    Parameters
    ----------
    bitstream : str
        Path to the .bit file (matching .hwh must be alongside).
    jpeg_width, jpeg_height : int
        Frame dimensions for JPEG encoding (default 640x480).
    jpeg_quality : int
        JPEG quality 1-100 (default 75).
    """

    MAX_IMG_W = 200
    MAX_IMG_H = 150
    WIN_SIZE = 24
    MAX_SCALES = 16

    MODE_NAMES = ["Normal", "Sharpen", "Blue Filter", "Binarize"]

    def __init__(self, bitstream, jpeg_width=640, jpeg_height=480,
                 jpeg_quality=75):
        assert jpeg_width % 16 == 0, f"width must be multiple of 16"
        assert jpeg_height % 8 == 0, f"height must be multiple of 8"

        self.jpeg_width = jpeg_width
        self.jpeg_height = jpeg_height
        self.n_pixels = jpeg_width * jpeg_height

        # Load overlay
        self.overlay = (bitstream if isinstance(bitstream, Overlay)
                        else Overlay(bitstream))

        # --- Discover Haar engines ---
        self.haar_engines = []
        idx = 0
        while True:
            ip_name = f"haar_face_detect_{idx}"
            dma_name = f"axi_dma_{idx}"
            if hasattr(self.overlay, ip_name) and hasattr(self.overlay, dma_name):
                ip = getattr(self.overlay, ip_name)
                dma = getattr(self.overlay, dma_name)
                self.haar_engines.append(_HaarEngine(ip, dma, idx))
                idx += 1
            else:
                break

        if not self.haar_engines:
            raise RuntimeError("No haar_face_detect_N + axi_dma_N pairs found")

        # --- MJPEG encoder ---
        self.mjpeg_ctrl = self.overlay.mjpeg_enc
        self.mjpeg_dma = self.overlay.mjpeg_dma

        # --- Preprocessing mode GPIO ---
        self._preproc_mode = 0
        if hasattr(self.overlay, 'axi_gpio_mode'):
            self.gpio_mode = self.overlay.axi_gpio_mode
            self.gpio_mode.write(0x0, 0)
        else:
            self.gpio_mode = None

        # Allocate persistent DMA buffers for MJPEG
        # YUYV422: 2 bytes per pixel, packed as uint32 (2 pixels per word)
        self.px_buf = allocate(shape=(self.n_pixels // 2,), dtype=np.uint32)
        max_jpeg = max(200000, self.n_pixels)
        self.jpeg_out_buf = allocate(shape=(max_jpeg,), dtype=np.uint8)

        # Configure MJPEG encoder
        self.mjpeg_ctrl.write(REG_COD_ADDR, 0)
        self.mjpeg_ctrl.write(REG_IMAGE_SIZE,
                              (jpeg_width << 16) | jpeg_height)
        self.set_jpeg_quality(jpeg_quality)

    @property
    def num_haar_engines(self):
        return len(self.haar_engines)

    # -----------------------------------------------------------------
    # MJPEG Encoding
    # -----------------------------------------------------------------
    def set_jpeg_quality(self, quality):
        """Change JPEG quality (1-100)."""
        lum = _scale_quant_table(STD_QUANT_LUM, quality)
        chm = _scale_quant_table(STD_QUANT_CHR, quality)
        for i in range(64):
            self.mjpeg_ctrl.write(REG_QUANT_LUM + i * 4, int(lum[i]))
            self.mjpeg_ctrl.write(REG_QUANT_CHR + i * 4, int(chm[i]))

    def set_preproc_mode(self, mode):
        """Set image preprocessing mode.

        Parameters
        ----------
        mode : int
            0 = Normal (pass-through)
            1 = Sharpen (1D high-pass on Y)
            2 = Blue filter (shift Cb/Cr)
            3 = Binarize (threshold Y, neutral chroma)
        """
        self._preproc_mode = mode & 0x3
        if self.gpio_mode is not None:
            self.gpio_mode.write(0x0, self._preproc_mode)

    def next_preproc_mode(self):
        """Cycle to the next preprocessing mode. Returns new mode name."""
        self.set_preproc_mode((self._preproc_mode + 1) % 4)
        return self.MODE_NAMES[self._preproc_mode]

    def encode_jpeg(self, yuyv_frame):
        """Encode one YUYV422 frame through FPGA MJPEG encoder.

        The raw YUYV bytes are sent directly to DMA. The PL-side
        yuyv_unpack module converts them to per-pixel format, then
        image_preproc applies the selected filter before encoding.

        Parameters
        ----------
        yuyv_frame : bytes, bytearray, or np.ndarray (uint8)
            Raw YUYV422 frame (width * height * 2 bytes).

        Returns
        -------
        bytes : JPEG-compressed frame
        """
        if isinstance(yuyv_frame, (bytes, bytearray)):
            raw = np.frombuffer(yuyv_frame, dtype=np.uint32)
        elif yuyv_frame.dtype == np.uint8:
            raw = np.frombuffer(yuyv_frame.tobytes(), dtype=np.uint32)
        else:
            raw = yuyv_frame.view(np.uint32).ravel()

        expected = self.n_pixels // 2
        assert len(raw) == expected, \
            f"YUYV frame size mismatch: got {len(raw)*4} bytes, " \
            f"expected {expected*4} (for {self.jpeg_width}x{self.jpeg_height})"

        # Copy raw YUYV into DMA buffer (hardware does unpacking)
        np.copyto(self.px_buf, raw)
        self.jpeg_out_buf[:64] = 0

        # Arm S2MM (receive) FIRST, then start encoder, then send pixels
        self.mjpeg_dma.recvchannel.transfer(self.jpeg_out_buf)
        self.mjpeg_ctrl.write(REG_ENC_START, 1)
        self.mjpeg_dma.sendchannel.transfer(self.px_buf)

        self.mjpeg_dma.sendchannel.wait()
        self.mjpeg_dma.recvchannel.wait()

        jpeg_len = self.mjpeg_ctrl.read(REG_ENC_LENGTH)
        if jpeg_len == 0 or jpeg_len > len(self.jpeg_out_buf):
            raise RuntimeError(
                f"Unexpected ENC_LENGTH={jpeg_len}, "
                f"status=0x{self.mjpeg_ctrl.read(REG_ENC_STS):08x}")

        return bytes(self.jpeg_out_buf[:jpeg_len])

    # -----------------------------------------------------------------
    # Face Detection
    # -----------------------------------------------------------------
    def _build_pyramid(self, gray_img, scale_factor, min_size):
        import cv2

        h0, w0 = gray_img.shape
        levels = []
        scale = 1.0

        while len(levels) < self.MAX_SCALES:
            new_h = int(h0 / scale)
            new_w = int(w0 / scale)

            if new_h < self.WIN_SIZE or new_w < self.WIN_SIZE:
                break

            win_at_orig = int(round(self.WIN_SIZE * scale))
            if win_at_orig < int(min_size):
                scale *= scale_factor
                continue

            if new_h > self.MAX_IMG_H or new_w > self.MAX_IMG_W:
                scale *= scale_factor
                continue

            level_img = cv2.resize(gray_img, (new_w, new_h),
                                   interpolation=cv2.INTER_LINEAR)
            scale_q8 = int(round(scale * 256))
            levels.append((level_img, scale_q8))
            scale *= scale_factor

        return levels

    def _distribute_levels(self, levels):
        n = self.num_haar_engines
        if n == 1:
            return [levels]

        costs = [img.shape[0] * img.shape[1] for img, _ in levels]
        buckets = [[] for _ in range(n)]
        bucket_costs = [0] * n

        sorted_indices = sorted(range(len(levels)), key=lambda i: -costs[i])
        for idx in sorted_indices:
            min_engine = min(range(n), key=lambda e: bucket_costs[e])
            buckets[min_engine].append(levels[idx])
            bucket_costs[min_engine] += costs[idx]

        return buckets

    def detect(self, gray_img, scale_factor=1.15, min_size=30,
               min_neighbors=3, timeout_s=5.0):
        """Multi-scale face detection using parallel Haar engines.

        Parameters
        ----------
        gray_img : np.ndarray (uint8, 2D)
            Grayscale input image.

        Returns
        -------
        list of (x, y, w, h) tuples in original image coordinates.
        """
        import cv2

        if gray_img.ndim != 2 or gray_img.dtype != np.uint8:
            raise ValueError("gray_img must be a 2D uint8 array")

        levels = self._build_pyramid(gray_img, scale_factor, min_size)
        if not levels:
            return []

        buckets = self._distribute_levels(levels)

        all_bboxes = []
        errors = []

        if self.num_haar_engines == 1:
            all_bboxes = self.haar_engines[0].run(buckets[0], timeout_s)
        else:
            results = [None] * self.num_haar_engines
            threads = []

            def _run_engine(engine_idx):
                try:
                    engine_levels = buckets[engine_idx]
                    if engine_levels:
                        results[engine_idx] = self.haar_engines[engine_idx].run(
                            engine_levels, timeout_s)
                    else:
                        results[engine_idx] = []
                except Exception as e:
                    errors.append((engine_idx, e))
                    results[engine_idx] = []

            for i in range(self.num_haar_engines):
                t = threading.Thread(target=_run_engine, args=(i,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            if errors:
                msgs = [f"Engine {eid}: {err}" for eid, err in errors]
                raise RuntimeError(
                    f"Parallel detection failed:\n" + "\n".join(msgs))

            for r in results:
                if r:
                    all_bboxes.extend(r)

        if not all_bboxes:
            return []

        rects = np.array(all_bboxes, dtype=np.int32).tolist()
        grouped, _ = cv2.groupRectangles(rects, int(min_neighbors), 0.2)
        if len(grouped) == 0:
            return []
        return [tuple(int(v) for v in r) for r in grouped]

    # -----------------------------------------------------------------
    # Combined: Detect + Encode in parallel
    # -----------------------------------------------------------------
    def detect_and_encode(self, gray_img, yuyv_frame,
                          scale_factor=1.15, min_size=30, min_neighbors=3,
                          timeout_s=5.0):
        """Run face detection and JPEG encoding concurrently.

        Face detection uses Haar engines via DMA on HP0/HP1.
        JPEG encoding uses MJPEG pipeline via DMA on HP2.
        They use completely separate DMA channels and HP ports,
        enabling true hardware parallelism with no bus contention.

        Parameters
        ----------
        gray_img : np.ndarray (uint8, 2D)
            Grayscale image for face detection.
        yuyv_frame : bytes, bytearray, or np.ndarray (uint8)
            Raw YUYV422 frame for JPEG encoding.

        Returns
        -------
        (bboxes, jpeg_bytes) : tuple
            bboxes: list of (x, y, w, h)
            jpeg_bytes: bytes
        """
        detect_result = [None]
        encode_result = [None]
        detect_error = [None]
        encode_error = [None]

        def _do_detect():
            try:
                detect_result[0] = self.detect(
                    gray_img, scale_factor, min_size,
                    min_neighbors, timeout_s)
            except Exception as e:
                detect_error[0] = e
                detect_result[0] = []

        def _do_encode():
            try:
                encode_result[0] = self.encode_jpeg(yuyv_frame)
            except Exception as e:
                encode_error[0] = e
                encode_result[0] = b''

        t_detect = threading.Thread(target=_do_detect)
        t_encode = threading.Thread(target=_do_encode)

        t_detect.start()
        t_encode.start()

        t_detect.join()
        t_encode.join()

        if detect_error[0]:
            raise detect_error[0]
        if encode_error[0]:
            raise encode_error[0]

        return detect_result[0], encode_result[0]

    # -----------------------------------------------------------------
    # Debug: MJPEG Encoder internal state
    # -----------------------------------------------------------------
    # Debug register offsets (in mjpeg_encoder_top, read-only)
    _DBG_PIXEL_IN_CNT   = 0x300
    _DBG_CORE_OUT_CNT   = 0x304
    _DBG_MAXIS_OUT_CNT  = 0x308
    _DBG_FIFO_LEVEL     = 0x30C
    _DBG_FLAGS          = 0x310
    _DBG_LAST_WRADDR    = 0x314
    _DBG_SAXIS_BEAT_CNT = 0x318

    def debug_mjpeg_state(self):
        """Read and print MJPEG encoder debug registers."""
        enc = self.mjpeg_ctrl

        pixel_in   = enc.read(self._DBG_PIXEL_IN_CNT)
        core_out   = enc.read(self._DBG_CORE_OUT_CNT)
        maxis_out  = enc.read(self._DBG_MAXIS_OUT_CNT)
        fifo_level = enc.read(self._DBG_FIFO_LEVEL)
        flags      = enc.read(self._DBG_FLAGS)
        last_addr  = enc.read(self._DBG_LAST_WRADDR)
        saxis_beat = enc.read(self._DBG_SAXIS_BEAT_CNT)

        # Decode status register
        enc_sts = enc.read(REG_ENC_STS)
        enc_len = enc.read(REG_ENC_LENGTH)
        img_size = enc.read(REG_IMAGE_SIZE)
        img_w = (img_size >> 16) & 0xFFFF
        img_h = img_size & 0xFFFF

        print("=" * 60)
        print("MJPEG Encoder Debug State")
        print("=" * 60)
        print(f"  IMAGE_SIZE reg:    {img_w}x{img_h}")
        print(f"  ENC_STS:           0x{enc_sts:08x}  "
              f"(busy={enc_sts & 1}, done={(enc_sts >> 1) & 1})")
        print(f"  ENC_LENGTH:        {enc_len} bytes")
        print(f"  --- Pixel Input ---")
        print(f"  S_AXIS beats:      {saxis_beat}")
        print(f"  Pixels consumed:   {pixel_in}  (expected: {img_w * img_h})")
        print(f"  --- JPEG Output ---")
        print(f"  Core bytes out:    {core_out}")
        print(f"  M_AXIS bytes out:  {maxis_out}")
        print(f"  Output FIFO level: {fifo_level}")
        print(f"  Last write addr:   0x{last_addr:06x}")
        print(f"  --- Live Flags ---")
        flag_names = {
            0: "s_axis_tvalid",
            1: "s_axis_tready",
            2: "iram_fifo_afull",
            3: "m_axis_tvalid",
            4: "m_axis_tready",
            8: "outif_afull",
            16: "ram_wren",
            24: "tlast_seen(sticky)",
        }
        active = []
        for bit, name in flag_names.items():
            if flags & (1 << bit):
                active.append(name)
        print(f"  Flags: 0x{flags:08x}  [{', '.join(active) if active else 'none'}]")
        print("=" * 60)

        return {
            "pixel_in": pixel_in, "core_out": core_out,
            "maxis_out": maxis_out, "fifo_level": fifo_level,
            "flags": flags, "enc_sts": enc_sts, "enc_len": enc_len,
            "img_w": img_w, "img_h": img_h,
            "saxis_beat": saxis_beat, "last_addr": last_addr,
        }

    def debug_dma_state(self, dma, name="DMA"):
        """Read and print DMA channel status registers."""
        print(f"  --- {name} ---")
        # Read raw MMIO registers
        # MM2S: DMACR=0x00, DMASR=0x04, LENGTH=0x28
        # S2MM: DMACR=0x30, DMASR=0x34, LENGTH=0x58
        try:
            mm2s_cr = dma.mmio.read(0x00)
            mm2s_sr = dma.mmio.read(0x04)
            mm2s_len = dma.mmio.read(0x28)
            print(f"  MM2S DMACR: 0x{mm2s_cr:08x}  "
                  f"(RS={mm2s_cr & 1}, Reset={(mm2s_cr >> 2) & 1})")
            print(f"  MM2S DMASR: 0x{mm2s_sr:08x}  "
                  f"(Halted={mm2s_sr & 1}, Idle={(mm2s_sr >> 1) & 1}, "
                  f"IOC={(mm2s_sr >> 12) & 1}, Err={(mm2s_sr >> 4) & 0xF})")
            print(f"  MM2S LENGTH: {mm2s_len}")
        except Exception as e:
            print(f"  MM2S read error: {e}")

        try:
            s2mm_cr = dma.mmio.read(0x30)
            s2mm_sr = dma.mmio.read(0x34)
            s2mm_len = dma.mmio.read(0x58)
            print(f"  S2MM DMACR: 0x{s2mm_cr:08x}  "
                  f"(RS={s2mm_cr & 1}, Reset={(s2mm_cr >> 2) & 1})")
            print(f"  S2MM DMASR: 0x{s2mm_sr:08x}  "
                  f"(Halted={s2mm_sr & 1}, Idle={(s2mm_sr >> 1) & 1}, "
                  f"IOC={(s2mm_sr >> 12) & 1}, Err={(s2mm_sr >> 4) & 0xF})")
            print(f"  S2MM LENGTH: {s2mm_len}")
        except Exception as e:
            print(f"  S2MM read error (might be MM2S-only DMA): {e}")

    def debug_haar_state(self, engine_idx=None):
        """Read and print Haar engine + DMA state."""
        engines = (self.haar_engines if engine_idx is None
                   else [self.haar_engines[engine_idx]])

        print("=" * 60)
        print("Haar Engine Debug State")
        print("=" * 60)
        for eng in engines:
            ap_ctrl = eng._read_reg(eng._REGS["AP_CTRL"])
            num_sc = eng._read_reg(eng._REGS["NUM_SCALES"])
            num_det = eng._read_reg(eng._REGS["NUM_DETS"])
            num_det_ctrl = eng._read_reg(eng._REGS["NUM_DETS_CTRL"])

            print(f"  --- Engine {eng.engine_id} ---")
            print(f"  AP_CTRL:     0x{ap_ctrl:08x}  "
                  f"(start={ap_ctrl & 1}, done={(ap_ctrl >> 1) & 1}, "
                  f"idle={(ap_ctrl >> 2) & 1}, ready={(ap_ctrl >> 3) & 1})")
            print(f"  NUM_SCALES:  {num_sc}")
            print(f"  NUM_DETS:    {num_det}  "
                  f"(valid={(num_det_ctrl & 1)})")

            # Print scale metadata
            for i in range(min(num_sc, 4)):
                w = eng._read_reg(eng._REGS["SCALE_WIDTHS_BASE"] + i * 4)
                h = eng._read_reg(eng._REGS["SCALE_HEIGHTS_BASE"] + i * 4)
                q = eng._read_reg(eng._REGS["SCALE_Q8_BASE"] + i * 4)
                print(f"    Scale[{i}]: {w}x{h}, q8={q} "
                      f"(scale={q / 256:.2f})")
            if num_sc > 4:
                print(f"    ... ({num_sc - 4} more scales)")

            self.debug_dma_state(eng.dma, f"Haar DMA {eng.engine_id}")

        print("=" * 60)

    def debug_pipeline_gpio(self):
        """Read debug GPIO blocks for yuyv_unpack + image_preproc counters.
        Only works if the bitstream includes the debug GPIO blocks."""
        print("=" * 60)
        print("Pipeline Debug (GPIO)")
        print("=" * 60)
        try:
            gpio_unpack = self.overlay.axi_gpio_dbg_unpack
            in_count = gpio_unpack.read(0x00)   # channel 1
            out_count = gpio_unpack.read(0x08)  # channel 2
            print(f"  yuyv_unpack:")
            print(f"    Input words:   {in_count}")
            print(f"    Output pixels: {out_count}")
        except AttributeError:
            print("  yuyv_unpack debug GPIO not found (old bitstream?)")

        try:
            gpio_preproc = self.overlay.axi_gpio_dbg_preproc
            px_count = gpio_preproc.read(0x00)
            print(f"  image_preproc:")
            print(f"    Pixels processed: {px_count}")
        except AttributeError:
            print("  image_preproc debug GPIO not found (old bitstream?)")

        try:
            gpio_state = self.overlay.axi_gpio_dbg_state
            state_val = gpio_state.read(0x00)
            unpack_state = state_val & 0xFF
            preproc_state = (state_val >> 8) & 0xFF
            print(f"  yuyv_unpack state:  0x{unpack_state:02x}  "
                  f"(phase={(unpack_state >> 1) & 1}, "
                  f"aresetn={unpack_state & 1})")
            print(f"  image_preproc state: 0x{preproc_state:02x}  "
                  f"(mode={((preproc_state >> 2) & 3)}, "
                  f"tready={(preproc_state >> 1) & 1}, "
                  f"tvalid={preproc_state & 1})")
        except AttributeError:
            print("  State debug GPIO not found (old bitstream?)")
        print("=" * 60)

    def debug_all(self):
        """Print comprehensive debug state of the entire system."""
        self.debug_haar_state()
        self.debug_dma_state(self.mjpeg_dma, "MJPEG DMA")
        self.debug_mjpeg_state()
        self.debug_pipeline_gpio()

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------
    def close(self):
        """Free DMA buffers."""
        self.px_buf.freebuffer()
        self.jpeg_out_buf.freebuffer()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
