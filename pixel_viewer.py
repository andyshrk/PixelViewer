#!/usr/bin/env python3
"""
PixelViewer - 原始图像查看器
支持 RGB888, RGB565, XRGB8888, NV12, NV21, NV16, NV61, NV24, NV42 等格式
"""

import sys
from enum import Enum
from typing import Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QMenuBar, QMenu, QLabel, QLineEdit, QComboBox, QPushButton,
        QToolBar, QStatusBar, QFileDialog, QMessageBox, QScrollArea,
        QButtonGroup, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    )
    from PyQt6.QtCore import Qt, QPoint, pyqtSignal
    from PyQt6.QtGui import (
        QAction, QImage, QPixmap, QPainter, QColor,
        QKeySequence
    )
except ImportError:
    print("Error: PyQt6 is required")
    print("Install with: pip install PyQt6")
    sys.exit(1)


class PixelFormat(Enum):
    """支持的像素格式"""
    RGB888 = "RGB888"
    BGR888 = "BGR888"
    RGB565 = "RGB565"
    BGR565 = "BGR565"
    XRGB8888 = "XRGB8888"
    XBGR8888 = "XBGR8888"
    NV12 = "NV12"
    NV21 = "NV21"
    NV16 = "NV16"
    NV61 = "NV61"
    NV24 = "NV24"
    NV42 = "NV42"


class YuvRange(Enum):
    """YUV 色彩范围"""
    LIMITED = "Limited"      # TV range: Y=16-235, UV=16-240
    FULL = "Full"            # PC range: Y=0-255, UV=0-255


class YuvRangeDetector:
    """YUV 色彩范围自动检测器"""

    @staticmethod
    def detect(data: bytes, width: int, height: int, fmt: PixelFormat) -> YuvRange:
        """自动检测 YUV 数据使用的色彩范围"""
        if not fmt.name.startswith('NV'):
            return YuvRange.LIMITED

        y_size = width * height
        if len(data) < y_size:
            return YuvRange.LIMITED

        # 采样分析
        sample_step = max(1, y_size // 10000)
        y_min = 255
        y_max = 0
        y_samples = []

        for i in range(0, min(y_size, len(data)), sample_step):
            y_val = data[i]
            y_min = min(y_min, y_val)
            y_max = max(y_max, y_val)
            y_samples.append(y_val)

        # 检测逻辑
        # 1. 如果有 Y=0 且 UV 接近中性 -> Full Range
        if y_min <= 4:
            for i in range(0, min(y_size, 1000), sample_step):
                if data[i] <= 4:
                    uv_idx = y_size + (i // width) * ((width + 1) // 2) * 2 + (i % width) // 2 * 2
                    if uv_idx + 1 < len(data):
                        u, v = data[uv_idx], data[uv_idx + 1]
                        if abs(u - 128) <= 20 and abs(v - 128) <= 20:
                            return YuvRange.FULL

        # 2. 如果 Y 值都在 16-235 范围内 -> Limited Range
        if y_min >= 14 and y_max <= 237:
            return YuvRange.LIMITED

        # 3. 如果有接近 255 的 Y 值 -> Full Range
        if y_max >= 250:
            return YuvRange.FULL

        # 4. 默认 Full Range
        return YuvRange.FULL


class PixelDecoder:
    """像素格式解码器"""
    _yuv_range = YuvRange.LIMITED
    _auto_detect = True  # 默认自动检测

    @staticmethod
    def set_yuv_range(range_mode: YuvRange):
        """设置 YUV 色彩范围"""
        PixelDecoder._yuv_range = range_mode
        PixelDecoder._auto_detect = False

    @staticmethod
    def get_yuv_range() -> YuvRange:
        """获取当前 YUV 色彩范围"""
        return PixelDecoder._yuv_range

    @staticmethod
    def decode(data: bytes, width: int, height: int, fmt: PixelFormat) -> QImage:
        """解码原始数据为 QImage"""
        # 自动检测 YUV 范围
        if PixelDecoder._auto_detect and fmt.name.startswith('NV'):
            detected = YuvRangeDetector.detect(data, width, height, fmt)
            PixelDecoder._yuv_range = detected

        if HAS_NUMPY:
            return PixelDecoder._decode_numpy(data, width, height, fmt)
        else:
            return PixelDecoder._decode_pure(data, width, height, fmt)

    @staticmethod
    def _decode_numpy(data: bytes, width: int, height: int, fmt: PixelFormat) -> QImage:
        """使用 NumPy 高效解码 - 支持部分数据"""
        arr = np.frombuffer(data, dtype=np.uint8)

        if fmt in [PixelFormat.RGB888, PixelFormat.BGR888]:
            expected = width * height * 3
            actual_size = min(len(arr), expected)
            actual_h = actual_size // (width * 3)
            actual_w = width
            actual_bytes = actual_h * actual_w * 3

            if actual_h > 0:
                rgb = arr[:actual_bytes].reshape(actual_h, actual_w, 3).copy()
                if fmt == PixelFormat.BGR888:
                    rgb = rgb[:, :, ::-1]
            else:
                rgb = np.zeros((1, width, 3), dtype=np.uint8)
                actual_h = 1

            img = QImage(rgb, width, actual_h, width * 3, QImage.Format.Format_RGB888)
            return img.copy()

        elif fmt in [PixelFormat.RGB565, PixelFormat.BGR565]:
            actual_h = min(height, len(arr) // (width * 2))
            actual_bytes = actual_h * width * 2

            if actual_h > 0:
                pixels = arr[:actual_bytes].view(np.uint16).reshape(actual_h, width)
                r = ((pixels & 0xF800) >> 8).astype(np.uint8)
                g = ((pixels & 0x07E0) >> 3).astype(np.uint8)
                b = ((pixels & 0x001F) << 3).astype(np.uint8)
                if fmt == PixelFormat.BGR565:
                    rgb = np.stack([b, g, r], axis=2)
                else:
                    rgb = np.stack([r, g, b], axis=2)
            else:
                rgb = np.zeros((1, width, 3), dtype=np.uint8)
                actual_h = 1

            img = QImage(rgb.reshape(-1), width, actual_h, width * 3, QImage.Format.Format_RGB888)
            return img.copy()

        elif fmt in [PixelFormat.XRGB8888, PixelFormat.XBGR8888]:
            actual_h = min(height, len(arr) // (width * 4))
            actual_bytes = actual_h * width * 4

            if actual_h > 0:
                rgb = arr[:actual_bytes].reshape(actual_h, width, 4)[:, :, :3].copy()
                if fmt == PixelFormat.XBGR8888:
                    rgb = rgb[:, :, ::-1]
            else:
                rgb = np.zeros((1, width, 3), dtype=np.uint8)
                actual_h = 1

            img = QImage(rgb.reshape(-1), width, actual_h, width * 3, QImage.Format.Format_RGB888)
            return img.copy()

        elif fmt in [PixelFormat.NV12, PixelFormat.NV21, PixelFormat.NV16, PixelFormat.NV61,
                     PixelFormat.NV24, PixelFormat.NV42]:
            y_size = width * height
            actual_y_size = min(len(arr), y_size)
            actual_h = actual_y_size // width if width > 0 else 0

            if actual_h == 0:
                return QImage(width, height, QImage.Format.Format_RGB888)

            # Y 平面
            y = arr[:actual_y_size].astype(np.float32).reshape(actual_h, width)

            # UV 平面
            uv_actual = max(0, len(arr) - actual_y_size)

            if fmt in [PixelFormat.NV12, PixelFormat.NV21]:
                uv_width = (width + 1) // 2
                uv_height = (height + 1) // 2
                uv_size = uv_width * uv_height * 2
                uv_data = arr[actual_y_size:actual_y_size + min(uv_actual, uv_size)]
                actual_uv_h = len(uv_data) // (uv_width * 2)
                if actual_uv_h > 0:
                    uv = uv_data[:actual_uv_h * uv_width * 2].reshape(actual_uv_h, uv_width, 2)
                    u = np.repeat(np.repeat(uv[:, :, 0], 2, axis=0), 2, axis=1)[:actual_h, :width]
                    v = np.repeat(np.repeat(uv[:, :, 1], 2, axis=0), 2, axis=1)[:actual_h, :width]
                else:
                    u = np.full((actual_h, width), 128, dtype=np.float32)
                    v = np.full((actual_h, width), 128, dtype=np.float32)
            elif fmt in [PixelFormat.NV16, PixelFormat.NV61]:
                uv_width = (width + 1) // 2
                uv_size = width * uv_width * 2
                uv_data = arr[actual_y_size:actual_y_size + min(uv_actual, uv_size)]
                actual_uv_w = min(uv_width, len(uv_data) // (2 * actual_h)) if actual_h > 0 else 0
                if actual_uv_w > 0:
                    uv = uv_data[:actual_h * actual_uv_w * 2].reshape(actual_h, actual_uv_w, 2)
                    u = np.repeat(uv[:, :, 0], (width // actual_uv_w) + 1, axis=1)[:, :width]
                    v = np.repeat(uv[:, :, 1], (width // actual_uv_w) + 1, axis=1)[:, :width]
                else:
                    u = np.full((actual_h, width), 128, dtype=np.float32)
                    v = np.full((actual_h, width), 128, dtype=np.float32)
            else:  # NV24, NV42
                uv_width = width
                uv_data = arr[actual_y_size:]
                actual_uv_w = min(width, len(uv_data) // (2 * actual_h)) if actual_h > 0 else 0
                if actual_uv_w > 0:
                    uv = uv_data[:actual_h * actual_uv_w * 2].reshape(actual_h, actual_uv_w, 2)
                    u = np.repeat(uv[:, :, 0], (width // actual_uv_w) + 1, axis=1)[:, :width]
                    v = np.repeat(uv[:, :, 1], (width // actual_uv_w) + 1, axis=1)[:, :width]
                else:
                    u = np.full((actual_h, width), 128, dtype=np.float32)
                    v = np.full((actual_h, width), 128, dtype=np.float32)

            # UV 顺序
            if fmt in [PixelFormat.NV21, PixelFormat.NV61, PixelFormat.NV42]:
                u, v = v, u

            # YUV to RGB 转换
            if PixelDecoder._yuv_range == YuvRange.FULL:
                # Full Range: Y=0-255, UV=0-255
                # 公式: R = Y + 1.402*(V-128), G = Y - 0.344*(U-128) - 0.714*(V-128), B = Y + 1.772*(U-128)
                u_f = u.astype(np.float32) - 128
                v_f = v.astype(np.float32) - 128
                r = np.clip(y + 1.402 * v_f, 0, 255).astype(np.uint8)
                g = np.clip(y - 0.344 * u_f - 0.714 * v_f, 0, 255).astype(np.uint8)
                b = np.clip(y + 1.772 * u_f, 0, 255).astype(np.uint8)
            else:
                # Limited Range: Y=16-235, UV=16-240 (中点 128)
                # 公式: R = 1.164*(Y-16) + 1.596*(V-128) + 128
                c = y - 16
                d = u.astype(np.float32) - 128
                e = v.astype(np.float32) - 128
                r = np.clip(1.164 * c + 1.596 * e + 128, 0, 255).astype(np.uint8)
                g = np.clip(1.164 * c - 0.392 * d - 0.813 * e + 128, 0, 255).astype(np.uint8)
                b = np.clip(1.164 * c + 2.017 * d + 128, 0, 255).astype(np.uint8)

            rgb = np.stack([r, g, b], axis=2).reshape(-1)
            img = QImage(rgb, width, actual_h, width * 3, QImage.Format.Format_RGB888)
            return img.copy()

        raise ValueError(f"Unsupported format: {fmt}")

    @staticmethod
    def _decode_pure(data: bytes, width: int, height: int, fmt: PixelFormat) -> QImage:
        """纯 Python 解码 (备用)"""
        if fmt in [PixelFormat.RGB888, PixelFormat.BGR888]:
            return PixelDecoder._decode_rgb888(data, width, height, fmt == PixelFormat.BGR888)
        elif fmt in [PixelFormat.RGB565, PixelFormat.BGR565]:
            return PixelDecoder._decode_rgb565(data, width, height, fmt == PixelFormat.BGR565)
        elif fmt in [PixelFormat.XRGB8888, PixelFormat.XBGR8888]:
            return PixelDecoder._decode_xrgb8888(data, width, height, fmt == PixelFormat.XBGR8888)
        elif fmt in [PixelFormat.NV12, PixelFormat.NV21]:
            return PixelDecoder._decode_nv12(data, width, height, fmt == PixelFormat.NV21)
        elif fmt in [PixelFormat.NV16, PixelFormat.NV61]:
            return PixelDecoder._decode_nv16(data, width, height, fmt == PixelFormat.NV61)
        elif fmt in [PixelFormat.NV24, PixelFormat.NV42]:
            return PixelDecoder._decode_nv24(data, width, height, fmt == PixelFormat.NV42)
        raise ValueError(f"Unsupported format: {fmt}")

    @staticmethod
    def _decode_rgb888(data: bytes, width: int, height: int, bgr: bool) -> QImage:
        """解码 RGB888/BGR888"""
        img = QImage(width, height, QImage.Format.Format_RGB888)
        for y in range(height):
            for x in range(width):
                idx = (y * width + x) * 3
                if idx + 2 >= len(data):
                    return img
                if bgr:
                    r, g, b = data[idx + 2], data[idx + 1], data[idx]
                else:
                    r, g, b = data[idx], data[idx + 1], data[idx + 2]
                img.setPixel(x, y, (r << 16) | (g << 8) | b)
        return img

    @staticmethod
    def _decode_rgb565(data: bytes, width: int, height: int, bgr: bool) -> QImage:
        """解码 RGB565/BGR565"""
        img = QImage(width, height, QImage.Format.Format_RGB888)
        for y in range(height):
            for x in range(width):
                idx = (y * width + x) * 2
                if idx + 1 >= len(data):
                    return img
                pixel = data[idx] | (data[idx + 1] << 8)
                if bgr:
                    b = ((pixel & 0x001F) << 3)
                    g = ((pixel & 0x07E0) >> 3)
                    r = ((pixel & 0xF800) >> 8)
                else:
                    r = ((pixel & 0xF800) >> 8)
                    g = ((pixel & 0x07E0) >> 3)
                    b = ((pixel & 0x001F) << 3)
                img.setPixel(x, y, (r << 16) | (g << 8) | b)
        return img

    @staticmethod
    def _decode_xrgb8888(data: bytes, width: int, height: int, bgr: bool) -> QImage:
        """解码 XRGB8888/XBGR8888"""
        img = QImage(width, height, QImage.Format.Format_RGB888)
        for y in range(height):
            for x in range(width):
                idx = (y * width + x) * 4
                if idx + 3 >= len(data):
                    return img
                if bgr:
                    b, g, r = data[idx], data[idx + 1], data[idx + 2]
                else:
                    r, g, b = data[idx], data[idx + 1], data[idx + 2]
                img.setPixel(x, y, (r << 16) | (g << 8) | b)
        return img

    @staticmethod
    def _decode_nv12(data: bytes, width: int, height: int, vu_order: bool) -> QImage:
        """解码 NV12/NV21 (YUV 4:2:0)"""
        img = QImage(width, height, QImage.Format.Format_RGB888)
        y_size = width * height
        uv_width = (width + 1) // 2

        for y in range(height):
            for x in range(width):
                y_idx = y * width + x
                if y_idx >= len(data):
                    return img
                y_val = data[y_idx]

                uv_x = x // 2
                uv_y = y // 2
                uv_idx = y_size + uv_y * uv_width * 2 + uv_x * 2
                if uv_idx + 1 >= len(data):
                    uv_idx = min(uv_idx, len(data) - 2)

                if vu_order:
                    v_val = data[uv_idx]
                    u_val = data[uv_idx + 1]
                else:
                    u_val = data[uv_idx]
                    v_val = data[uv_idx + 1]

                if PixelDecoder._yuv_range == YuvRange.FULL:
                    r = int(max(0, min(255, y_val + 1.402 * (v_val - 128))))
                    g = int(max(0, min(255, y_val - 0.344 * (u_val - 128) - 0.714 * (v_val - 128))))
                    b = int(max(0, min(255, y_val + 1.772 * (u_val - 128))))
                else:
                    c = y_val - 16
                    d = u_val - 128
                    e = v_val - 128
                    r = int(max(0, min(255, 1.164 * c + 1.596 * e + 128)))
                    g = int(max(0, min(255, 1.164 * c - 0.392 * d - 0.813 * e + 128)))
                    b = int(max(0, min(255, 1.164 * c + 2.017 * d + 128)))

                img.setPixel(x, y, (r << 16) | (g << 8) | b)
        return img

    @staticmethod
    def _decode_nv16(data: bytes, width: int, height: int, vu_order: bool) -> QImage:
        """解码 NV16/NV61 (YUV 4:2:2)"""
        img = QImage(width, height, QImage.Format.Format_RGB888)
        y_size = width * height
        uv_width = (width + 1) // 2

        for y in range(height):
            for x in range(width):
                y_idx = y * width + x
                if y_idx >= len(data):
                    return img
                y_val = data[y_idx]

                uv_x = x // 2
                uv_idx = y_size + y * uv_width * 2 + uv_x * 2
                if uv_idx + 1 >= len(data):
                    uv_idx = min(uv_idx, len(data) - 2)

                if vu_order:
                    v_val = data[uv_idx]
                    u_val = data[uv_idx + 1]
                else:
                    u_val = data[uv_idx]
                    v_val = data[uv_idx + 1]

                if PixelDecoder._yuv_range == YuvRange.FULL:
                    r = int(max(0, min(255, y_val + 1.402 * (v_val - 128))))
                    g = int(max(0, min(255, y_val - 0.344 * (u_val - 128) - 0.714 * (v_val - 128))))
                    b = int(max(0, min(255, y_val + 1.772 * (u_val - 128))))
                else:
                    c = y_val - 16
                    d = u_val - 128
                    e = v_val - 128
                    r = int(max(0, min(255, 1.164 * c + 1.596 * e + 128)))
                    g = int(max(0, min(255, 1.164 * c - 0.392 * d - 0.813 * e + 128)))
                    b = int(max(0, min(255, 1.164 * c + 2.017 * d + 128)))

                img.setPixel(x, y, (r << 16) | (g << 8) | b)
        return img

    @staticmethod
    def _decode_nv24(data: bytes, width: int, height: int, vu_order: bool) -> QImage:
        """解码 NV24/NV42 (YUV 4:4:4)"""
        img = QImage(width, height, QImage.Format.Format_RGB888)
        y_size = width * height

        for y in range(height):
            for x in range(width):
                y_idx = y * width + x
                if y_idx >= len(data):
                    return img
                y_val = data[y_idx]

                uv_idx = y_size + y * width * 2 + x * 2
                if uv_idx + 1 >= len(data):
                    uv_idx = min(uv_idx, len(data) - 2)

                if vu_order:
                    v_val = data[uv_idx]
                    u_val = data[uv_idx + 1]
                else:
                    u_val = data[uv_idx]
                    v_val = data[uv_idx + 1]

                if PixelDecoder._yuv_range == YuvRange.FULL:
                    r = int(max(0, min(255, y_val + 1.402 * (v_val - 128))))
                    g = int(max(0, min(255, y_val - 0.344 * (u_val - 128) - 0.714 * (v_val - 128))))
                    b = int(max(0, min(255, y_val + 1.772 * (u_val - 128))))
                else:
                    c = y_val - 16
                    d = u_val - 128
                    e = v_val - 128
                    r = int(max(0, min(255, 1.164 * c + 1.596 * e + 128)))
                    g = int(max(0, min(255, 1.164 * c - 0.392 * d - 0.813 * e + 128)))
                    b = int(max(0, min(255, 1.164 * c + 2.017 * d + 128)))

                img.setPixel(x, y, (r << 16) | (g << 8) | b)
        return img

    @staticmethod
    def get_required_size(width: int, height: int, fmt: PixelFormat) -> int:
        """计算所需字节数"""
        size = width * height
        if fmt in [PixelFormat.RGB888, PixelFormat.BGR888]:
            return size * 3
        elif fmt in [PixelFormat.RGB565, PixelFormat.BGR565]:
            return size * 2
        elif fmt in [PixelFormat.XRGB8888, PixelFormat.XBGR8888]:
            return size * 4
        elif fmt in [PixelFormat.NV12, PixelFormat.NV21]:
            return int(size * 1.5)
        elif fmt in [PixelFormat.NV16, PixelFormat.NV61]:
            return size * 2
        elif fmt in [PixelFormat.NV24, PixelFormat.NV42]:
            return size * 3
        return size * 3


class ImageGraphicsView(QGraphicsView):
    """自定义图像视图，支持拖拽和缩放"""

    mouse_moved = pyqtSignal(int, int)
    mouse_left = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)

    def wheelEvent(self, event):
        """鼠标滚轮缩放"""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 0.85
            self.scale(factor, factor)
            event.accept()
        else:
            super().wheelEvent(event)

    def mouseMoveEvent(self, event):
        pos = self.mapToScene(event.pos())
        self.mouse_moved.emit(int(pos.x()), int(pos.y()))
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.mouse_left.emit()
        super().leaveEvent(event)


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self._file_path: Optional[str] = None
        self._file_data: Optional[bytes] = None
        self._width = 1920
        self._height = 1080
        self._pixel_format = PixelFormat.NV12
        self._zoom = 1.0

        self._setup_ui()
        self._setup_connections()

    def _setup_ui(self):
        """初始化 UI"""
        self.setWindowTitle("PixelViewer - Raw Image Viewer")
        self.setMinimumSize(800, 600)
        self.resize(1280, 720)

        # 深色主题
        self.setStyleSheet("""
            QMainWindow { background-color: #1E1E1E; }
            QWidget { background-color: #1E1E1E; color: #CCCCCC; }
            QMenuBar { background-color: #2D2D30; color: #CCCCCC; }
            QMenuBar::item:selected { background-color: #3F3F46; }
            QMenu { background-color: #2D2D30; color: #CCCCCC; }
            QMenu::item:selected { background-color: #007ACC; }
            QToolBar { background-color: #2D2D30; border: none; }
            QLabel { color: #CCCCCC; }
            QLineEdit {
                background-color: #2D2D30; color: #CCCCCC;
                border: 1px solid #5F5F63; padding: 4px;
            }
            QComboBox {
                background-color: #2D2D30; color: #CCCCCC;
                border: 1px solid #5F5F63; padding: 4px;
            }
            QPushButton {
                background-color: #3F3F46; color: #CCCCCC;
                border: 1px solid #5F5F63; padding: 6px 12px;
            }
            QPushButton:hover { background-color: #4F4F53; }
            QPushButton:pressed { background-color: #007ACC; }
            QStatusBar { background-color: #2D2D30; color: #CCCCCC; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 工具栏
        toolbar = QToolBar("Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.open_btn = QPushButton("Open")
        toolbar.addWidget(self.open_btn)
        toolbar.addSeparator()

        toolbar.addWidget(QLabel("Width:"))
        self.width_edit = QLineEdit("1920")
        self.width_edit.setFixedWidth(80)
        toolbar.addWidget(self.width_edit)

        toolbar.addWidget(QLabel("Height:"))
        self.height_edit = QLineEdit("1080")
        self.height_edit.setFixedWidth(80)
        toolbar.addWidget(self.height_edit)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        for fmt in PixelFormat:
            self.format_combo.addItem(fmt.value, fmt)
        self.format_combo.setCurrentIndex(6)
        self.format_combo.setFixedWidth(100)
        toolbar.addWidget(self.format_combo)

        toolbar.addSeparator()

        self.yuv_range_label = QLabel("Range: Auto")
        self.yuv_range_label.setStyleSheet("color: #888888;")
        toolbar.addWidget(self.yuv_range_label)

        toolbar.addSeparator()

        self.zoom_group = QButtonGroup()
        self.zoom_buttons = {}
        for zoom in ["1x", "2x", "4x", "8x", "Fit"]:
            btn = QPushButton(zoom)
            btn.setCheckable(True)
            btn.setFixedWidth(50 if zoom == "Fit" else 40)
            self.zoom_group.addButton(btn)
            self.zoom_buttons[zoom] = btn
            toolbar.addWidget(btn)

        self.reload_btn = QPushButton("Reload")
        toolbar.addWidget(self.reload_btn)

        # 图像显示区域
        self.scene = QGraphicsScene()
        self.view = ImageGraphicsView()
        self.view.setScene(self.scene)
        self.view.setBackgroundBrush(QColor("#1E1E1E"))
        main_layout.addWidget(self.view)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.file_path_label = QLabel("No file loaded")
        self.resolution_label = QLabel("")
        self.format_label = QLabel("")
        self.zoom_label = QLabel("")
        self.position_label = QLabel("")
        self.color_label = QLabel("")

        for label in [self.file_path_label, self.resolution_label, self.format_label,
                      self.zoom_label, self.position_label, self.color_label]:
            label.setStyleSheet("padding: 0 8px;")

        self.status_bar.addWidget(self.file_path_label, 1)
        self.status_bar.addWidget(self.resolution_label)
        self.status_bar.addWidget(self.format_label)
        self.status_bar.addWidget(self.zoom_label)
        self.status_bar.addWidget(self.position_label)
        self.status_bar.addWidget(self.color_label)

        # 菜单
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        open_action = QAction("Open...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_file)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = menubar.addMenu("View")
        fit_action = QAction("Fit Window", self)
        fit_action.triggered.connect(self._fit_window)
        view_menu.addAction(fit_action)
        actual_action = QAction("1:1", self)
        actual_action.triggered.connect(self._actual_size)
        view_menu.addAction(actual_action)

        help_menu = menubar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_connections(self):
        self.open_btn.clicked.connect(self._open_file)
        self.reload_btn.clicked.connect(self._reload)
        self.format_combo.currentIndexChanged.connect(self._format_changed)
        self.zoom_group.buttonClicked.connect(self._zoom_changed)
        self.width_edit.editingFinished.connect(self._resolution_changed)
        self.height_edit.editingFinished.connect(self._resolution_changed)
        self.view.mouse_moved.connect(self._mouse_moved)
        self.view.mouse_left.connect(self._mouse_left)

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "", "All Files (*.*);;YUV Files (*.yuv;*.nv12;*.nv21)"
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str):
        try:
            with open(path, "rb") as f:
                self._file_data = f.read()
            self._file_path = path
            self._update_display()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot open file:\n{str(e)}")

    def _reload(self):
        if self._file_path:
            self._load_file(self._file_path)
        elif self._file_data:
            self._update_display()

    def _format_changed(self, index: int):
        self._pixel_format = self.format_combo.currentData()
        if self._file_data:
            self._update_display()

    def _zoom_changed(self, btn: QPushButton):
        zoom_text = btn.text()
        if zoom_text == "1x":
            self._zoom = 1.0
        elif zoom_text == "2x":
            self._zoom = 2.0
        elif zoom_text == "4x":
            self._zoom = 4.0
        elif zoom_text == "8x":
            self._zoom = 8.0
        else:
            self._zoom = 0

        if self._file_data:
            self._update_display()

    def _resolution_changed(self):
        try:
            self._width = int(self.width_edit.text())
            self._height = int(self.height_edit.text())
            if self._width <= 0 or self._height <= 0:
                raise ValueError()
        except ValueError:
            QMessageBox.warning(self, "Warning", "Invalid resolution")
            return

        if self._file_data:
            self._update_display()

    def _update_display(self):
        if not self._file_data:
            return

        try:
            self._width = int(self.width_edit.text())
            self._height = int(self.height_edit.text())
        except ValueError:
            return

        required_size = PixelDecoder.get_required_size(self._width, self._height, self._pixel_format)
        if len(self._file_data) < required_size:
            self.status_bar.showMessage(f"Warning: Insufficient file size (need {required_size}, got {len(self._file_data)})", 3000)

        try:
            img = PixelDecoder.decode(self._file_data, self._width, self._height, self._pixel_format)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Decode failed:\n{str(e)}")
            return

        pixmap = QPixmap.fromImage(img)

        if self._zoom > 0:
            scaled_pixmap = pixmap.scaled(
                int(self._width * self._zoom),
                int(self._height * self._zoom),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.FastTransformation
            )
        else:
            view_size = self.view.viewport().size()
            scaled_pixmap = pixmap.scaled(
                view_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation
            )

        self.scene.clear()
        self.scene.addPixmap(scaled_pixmap)
        self.view.setSceneRect(0, 0, scaled_pixmap.width(), scaled_pixmap.height())
        self.view.setBackgroundBrush(QColor("#1E1E1E"))

        self.file_path_label.setText(self._file_path if self._file_path else "No file")
        self.resolution_label.setText(f" {self._width}x{self._height} ")
        self.format_label.setText(f" {self._pixel_format.value} ")
        self.zoom_label.setText(f" {self._zoom}x " if self._zoom > 0 else " Fit ")

        # 更新 YUV 范围标签
        yuv_range = PixelDecoder.get_yuv_range()
        self.yuv_range_label.setText(f"Range: {yuv_range.value}")

    def _fit_window(self):
        self._zoom = 0
        self.zoom_buttons["Fit"].setChecked(True)
        if self._file_data:
            self._update_display()

    def _actual_size(self):
        self._zoom = 1.0
        self.zoom_buttons["1x"].setChecked(True)
        if self._file_data:
            self._update_display()

    def _mouse_moved(self, x: int, y: int):
        if not self._file_data:
            return

        scale_x = self._width / self.scene.sceneRect().width() if self.scene.sceneRect().width() > 0 else 1
        scale_y = self._height / self.scene.sceneRect().height() if self.scene.sceneRect().height() > 0 else 1

        img_x = int(x * scale_x)
        img_y = int(y * scale_y)

        if 0 <= img_x < self._width and 0 <= img_y < self._height:
            self.position_label.setText(f" X: {img_x}, Y: {img_y} ")

            if self.scene.items():
                item = self.scene.items()[0]
                if isinstance(item, QGraphicsPixmapItem):
                    pixmap = item.pixmap()
                    if not pixmap.isNull():
                        img = pixmap.toImage()
                        if 0 <= x < pixmap.width() and 0 <= y < pixmap.height():
                            pixel = img.pixel(x, y)
                            r = (pixel >> 16) & 0xFF
                            g = (pixel >> 8) & 0xFF
                            b = pixel & 0xFF
                            self.color_label.setText(f" RGB: {r}, {g}, {b} ")
        else:
            self.position_label.setText("")
            self.color_label.setText("")

    def _mouse_left(self):
        self.position_label.setText("")
        self.color_label.setText("")

    def _show_about(self):
        numpy_status = "Enabled" if HAS_NUMPY else "Disabled (slower)"
        QMessageBox.about(
            self, "About PixelViewer",
            "PixelViewer v1.0\n\n"
            "Raw Image Viewer\n"
            "Supports RGB888, RGB565, XRGB8888\n"
            "Supports NV12, NV21, NV16, NV61, NV24, NV42\n\n"
            f"NumPy acceleration: {numpy_status}\n\n"
            "Shortcuts:\n"
            "Ctrl+O - Open file\n"
            "Ctrl+Wheel - Zoom"
        )


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("PixelViewer")
    app.setOrganizationName("PixelViewer")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()