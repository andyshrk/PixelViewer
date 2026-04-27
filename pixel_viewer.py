#!/usr/bin/env python3
"""
PixelViewer - 原始图像查看器
支持 RGB888, RGB565, XRGB8888, NV12, NV21, NV16, NV61, NV24, NV42 等格式
支持多标签页，类似 7yuv
"""

import sys
import os
import re
from enum import Enum
from typing import Optional, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QMenuBar, QMenu, QLabel, QLineEdit, QComboBox, QPushButton,
        QToolBar, QStatusBar, QFileDialog, QMessageBox,
        QButtonGroup, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
        QTabWidget, QSizePolicy
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
    def _uv_idx(y_idx: int, width: int, height: int, fmt: PixelFormat) -> int:
        """计算 UV 数据的字节偏移"""
        y_size = width * height
        y = y_idx // width
        x = y_idx % width
        if fmt in [PixelFormat.NV12, PixelFormat.NV21]:
            # UV plane: (height/2) x (width/2), interleaved UV
            uv_w = (width + 1) // 2
            uv_y = y // 2
            uv_x = x // 2
            return y_size + uv_y * uv_w * 2 + uv_x * 2
        elif fmt in [PixelFormat.NV16, PixelFormat.NV61]:
            # UV plane: height x (width/2), interleaved UV
            uv_w = (width + 1) // 2
            uv_x = x // 2
            return y_size + y * uv_w * 2 + uv_x * 2
        elif fmt in [PixelFormat.NV24, PixelFormat.NV42]:
            # UV plane: height x width, interleaved UV
            return y_size + y * width * 2 + x * 2
        return y_size + y_idx * 2

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

        for i in range(0, min(y_size, len(data)), sample_step):
            y_val = data[i]
            y_min = min(y_min, y_val)
            y_max = max(y_max, y_val)

        # 检测逻辑
        # 1. 如果有 Y=0 且 UV 接近中性 -> Full Range
        if y_min <= 4:
            for i in range(0, min(y_size, 1000), sample_step):
                if data[i] <= 4:
                    uv_idx = YuvRangeDetector._uv_idx(i, width, height, fmt)
                    if uv_idx + 1 < len(data):
                        u, v = data[uv_idx], data[uv_idx + 1]
                        if abs(u - 128) <= 20 and abs(v - 128) <= 20:
                            return YuvRange.FULL

        # 2. 如果有接近 255 的 Y 值 -> Full Range
        if y_max >= 250:
            return YuvRange.FULL

        # 3. 默认 Full Range
        return YuvRange.FULL


class PixelDecoder:
    """像素格式解码器"""
    _yuv_range = YuvRange.LIMITED

    @staticmethod
    def set_yuv_range(range_mode: YuvRange):
        """设置 YUV 色彩范围"""
        PixelDecoder._yuv_range = range_mode

    @staticmethod
    def get_yuv_range() -> YuvRange:
        """获取当前 YUV 色彩范围"""
        return PixelDecoder._yuv_range

    @staticmethod
    def decode(data: bytes, width: int, height: int, fmt: PixelFormat,
               auto_detect_range: bool = True) -> QImage:
        """解码原始数据为 QImage
        auto_detect_range: 是否在解码前自动检测 YUV 范围（手动选择 Range 时设为 False）
        """
        # 解码前自动检测 YUV 范围（仅当 auto_detect_range=True 且格式为 YUV 时）
        if auto_detect_range and fmt.name.startswith('NV'):
            PixelDecoder._yuv_range = YuvRangeDetector.detect(data, width, height, fmt)

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
                uv_size = height * uv_width * 2
                uv_data = arr[actual_y_size:actual_y_size + min(uv_actual, uv_size)]
                actual_uv_w = min(uv_width, len(uv_data) // (2 * actual_h)) if actual_h > 0 else 0
                if actual_uv_w > 0:
                    uv = uv_data[:actual_h * actual_uv_w * 2].reshape(actual_h, actual_uv_w, 2)
                    # NV16 UV plane is full-height, half-width -> repeat horizontally only
                    u = np.repeat(uv[:, :, 0], 2, axis=1)[:, :width]
                    v = np.repeat(uv[:, :, 1], 2, axis=1)[:, :width]
                else:
                    u = np.full((actual_h, width), 128, dtype=np.float32)
                    v = np.full((actual_h, width), 128, dtype=np.float32)
            else:  # NV24, NV42
                # UV plane is full-height, full-width -> no upsampling needed
                uv_data = arr[actual_y_size:]
                if actual_h > 0:
                    uv = uv_data[:actual_h * width * 2].reshape(actual_h, width, 2)
                    u = uv[:, :, 0]
                    v = uv[:, :, 1]
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
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setBackgroundBrush(QColor("#1E1E1E"))

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


class ImageTab(QWidget):
    """图像标签页"""

    def __init__(self, file_path: str, file_data: bytes, width: int, height: int,
                 pixel_format: PixelFormat, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.file_data = file_data
        self.width = width
        self.height = height
        self.pixel_format = pixel_format
        self.zoom = 1.0
        self._range_manually_set = False  # 标记用户是否手动设置过 Range

        self._setup_ui()
        self._on_zoom_changed("1x")
        self._update_display()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 工具栏
        toolbar = QWidget()
        toolbar.setStyleSheet("background-color: #2D2D30;")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 4, 8, 4)

        toolbar_layout.addWidget(QLabel("Width:"))
        self.width_edit = QLineEdit(str(self.width))
        self.width_edit.setFixedWidth(80)
        self.width_edit.editingFinished.connect(self._on_resolution_changed)
        toolbar_layout.addWidget(self.width_edit)

        toolbar_layout.addWidget(QLabel("Height:"))
        self.height_edit = QLineEdit(str(self.height))
        self.height_edit.setFixedWidth(80)
        self.height_edit.editingFinished.connect(self._on_resolution_changed)
        toolbar_layout.addWidget(self.height_edit)

        toolbar_layout.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        for fmt in PixelFormat:
            self.format_combo.addItem(fmt.value, fmt)
        idx = self.format_combo.findData(self.pixel_format)
        if idx >= 0:
            self.format_combo.setCurrentIndex(idx)
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        self.format_combo.setFixedWidth(100)
        toolbar_layout.addWidget(self.format_combo)

        toolbar_layout.addWidget(QLabel("Range:"))
        self.range_combo = QComboBox()
        self.range_combo.addItem("Full", YuvRange.FULL)
        self.range_combo.addItem("Limited", YuvRange.LIMITED)
        self.range_combo.setFixedWidth(80)
        toolbar_layout.addWidget(self.range_combo)

        toolbar_layout.addSpacing(16)

        self.zoom_group = QButtonGroup()
        for zoom in ["1x", "1/2x", "1/4x", "1/8x"]:
            btn = QPushButton(zoom)
            btn.setCheckable(True)
            btn.setFixedWidth(44)
            self.zoom_group.addButton(btn)
            btn.clicked.connect(lambda checked, z=zoom: self._on_zoom_changed(z) if checked else None)
            toolbar_layout.addWidget(btn)

        # 默认选择 1x（原始分辨率）
        for btn in self.zoom_group.buttons():
            if btn.text() == "1x":
                btn.setChecked(True)
                break

        toolbar_layout.addStretch()
        layout.addWidget(toolbar)

        # 图像显示区域
        self.scene = QGraphicsScene()
        self.view = ImageGraphicsView()
        self.view.setScene(self.scene)
        self.view.setBackgroundBrush(QColor("#1E1E1E"))
        self.view.mouse_moved.connect(self._on_mouse_moved)
        self.view.mouse_left.connect(self._on_mouse_left)
        layout.addWidget(self.view)

        # 状态栏
        self.status_bar = QWidget()
        self.status_bar.setStyleSheet("background-color: #2D2D30;")
        status_layout = QHBoxLayout(self.status_bar)
        status_layout.setContentsMargins(8, 4, 8, 4)

        self.file_label = QLabel(os.path.basename(self.file_path))
        self.res_label = QLabel("")
        self.format_label = QLabel("")
        self.zoom_label = QLabel("")
        self.pos_label = QLabel("")
        self.color_label = QLabel("")

        for lbl in [self.file_label, self.res_label, self.format_label,
                    self.zoom_label, self.pos_label, self.color_label]:
            lbl.setStyleSheet("color: #CCCCCC;")
            status_layout.addWidget(lbl)

        status_layout.addStretch()
        layout.addWidget(self.status_bar)

        # 连接 Range 信号（确保 scene 已创建后再连接）
        self.range_combo.currentIndexChanged.connect(self._on_range_changed)

    def _update_display(self):
        try:
            self.width = int(self.width_edit.text())
            self.height = int(self.height_edit.text())
        except ValueError:
            return

        required_size = PixelDecoder.get_required_size(self.width, self.height, self.pixel_format)
        if len(self.file_data) < required_size:
            pass  # 显示警告但不阻止

        try:
            # 如果用户手动设置过 Range，使用用户的选择；否则自动检测
            auto_detect = not self._range_manually_set
            img = PixelDecoder.decode(self.file_data, self.width, self.height, self.pixel_format,
                                      auto_detect_range=auto_detect)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Decode failed:\n{str(e)}")
            return

        pixmap = QPixmap.fromImage(img)

        if self.zoom > 0:
            scaled = pixmap.scaled(
                int(self.width * self.zoom),
                int(self.height * self.zoom),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.FastTransformation
            )
        else:
            view_size = self.view.viewport().size()
            if view_size.width() > 0 and view_size.height() > 0:
                # Fit 模式：填满整个窗口
                scaled = pixmap.scaled(
                    view_size,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.FastTransformation
                )
            else:
                scaled = pixmap

        self.scene.clear()
        self.scene.addPixmap(scaled)
        self.view.setSceneRect(0, 0, scaled.width(), scaled.height())

        self.res_label.setText(f" {self.width}x{self.height} ")
        self.format_label.setText(f" {self.pixel_format.value} ")
        self.zoom_label.setText(f" {self.zoom:.2f}x ")

        # 同步 Range 下拉框显示检测/设置的值
        current = self.range_combo.currentData()
        detected = PixelDecoder.get_yuv_range()
        # 只在需要时更新（避免触发 currentIndexChanged）
        if detected != current:
            idx = self.range_combo.findData(detected)
            if idx >= 0:
                self.range_combo.setCurrentIndex(idx)

    def _on_resolution_changed(self):
        self._update_display()

    def _on_format_changed(self, index):
        self.pixel_format = self.format_combo.currentData()
        self._update_display()

    def _on_range_changed(self, index):
        self._range_manually_set = True  # 标记用户手动设置过
        data = self.range_combo.currentData()
        PixelDecoder._yuv_range = data
        self._update_display()

    def _on_zoom_changed(self, zoom: str):
        if zoom == "1x":
            self.zoom = 1.0
        elif zoom == "1/2x":
            self.zoom = 0.5
        elif zoom == "1/4x":
            self.zoom = 0.25
        elif zoom == "1/8x":
            self.zoom = 0.125
        self._update_display()

    def _on_mouse_moved(self, x: int, y: int):
        scale_x = self.width / self.scene.sceneRect().width() if self.scene.sceneRect().width() > 0 else 1
        scale_y = self.height / self.scene.sceneRect().height() if self.scene.sceneRect().height() > 0 else 1

        img_x = int(x * scale_x)
        img_y = int(y * scale_y)

        if 0 <= img_x < self.width and 0 <= img_y < self.height:
            self.pos_label.setText(f" X: {img_x}, Y: {img_y} ")

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
            self.pos_label.setText("")
            self.color_label.setText("")

    def _on_mouse_left(self):
        self.pos_label.setText("")
        self.color_label.setText("")


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
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
            QTabWidget::pane { border: 0; }
            QTabBar::tab {
                background-color: #2D2D30;
                color: #CCCCCC;
                padding: 6px 12px;
                border: 1px solid #3F3F46;
            }
            QTabBar::tab:selected {
                background-color: #007ACC;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background-color: #3F3F46;
            }
            QTabBar::close-button {
                image: none;
            }
            QTabBar::close-button:hover {
                background-color: #555555;
                border-radius: 2px;
            }
        """)

        # 标签页容器
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self._close_tab)
        self.setCentralWidget(self.tab_widget)

        # 菜单
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        open_action = QAction("Open...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_file)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        close_tab_action = QAction("Close Tab", self)
        close_tab_action.setShortcut(QKeySequence.StandardKey.Close)
        close_tab_action.triggered.connect(self._close_current_tab)
        file_menu.addAction(close_tab_action)
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
        pass

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "",
            "All Files (*.*);;YUV Files (*.yuv;*.nv12;*.nv21);;Binary Files (*.bin)"
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str):
        try:
            with open(path, "rb") as f:
                file_data = f.read()

            # 从文件名解析分辨率和格式
            fname = os.path.basename(path)
            fn_width, fn_height, fn_fmt = self._parse_filename(fname)

            # 自动检测分辨率（如果文件名无法解析）
            if fn_width and fn_height:
                width, height = fn_width, fn_height
            else:
                width, height = self._auto_detect_resolution(len(file_data))

            # 显示格式选择对话框（分辨率可编辑）
            width, height, fmt, ok = self._show_format_dialog(width, height, fn_fmt)
            if not ok:
                return

            # 创建新标签页
            tab = ImageTab(path, file_data, width, height, fmt)
            name = os.path.basename(path)

            # 如果已有同名标签，关闭旧的
            for i in range(self.tab_widget.count()):
                if self.tab_widget.tabText(i) == name:
                    self.tab_widget.removeTab(i)
                    break

            index = self.tab_widget.addTab(tab, name)
            self.tab_widget.setCurrentIndex(index)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot open file:\n{str(e)}")

    def _parse_filename(self, filename: str) -> Tuple[Optional[int], Optional[int], Optional[PixelFormat]]:
        """从文件名解析分辨率和格式
        支持格式如:
          - wb_NV16_3840x2160_000.bin
          - 3840x2160_NV24_valley.bin
          - wb_NV12_960x544.bin
          - test_1920x1080_rgb888.raw
          - video3840_2160_NV12_localmain (分辨率: 3840x2160, 格式: NV12)
        """
        name = os.path.splitext(filename)[0]

        width = None
        height = None

        # 尝试匹配分辨率模式: 数字x数字 (如 3840x2160, 960x544)
        res_pattern = r'(\d+)x(\d+)'
        match = re.search(res_pattern, name, re.IGNORECASE)
        if match:
            width = int(match.group(1))
            height = int(match.group(2))

        # 尝试匹配分辨率模式: 数字_数字 (如 3840_2160)
        if width is None:
            res_pattern2 = r'(\d{3,4})_(\d{3,4})'
            match2 = re.search(res_pattern2, name)
            if match2:
                width = int(match2.group(1))
                height = int(match2.group(2))

        # 尝试匹配格式
        fmt = None
        for format_name in PixelFormat._member_names_:
            if format_name in name:
                fmt = PixelFormat[format_name]
                break

        return width, height, fmt

    def _auto_detect_resolution(self, file_size: int) -> tuple:
        """根据文件大小自动检测分辨率"""
        common_resolutions = [
            (1920, 1080), (1920, 1088),  # 1088 is common for 2K content
            (3840, 2160), (2560, 1440), (2560, 1080),
            (1280, 720), (1920, 1200), (1600, 900), (1366, 768),
            (1440, 900), (1680, 1050), (1280, 800), (1024, 768),
            (640, 480), (320, 240), (800, 600)
        ]

        for w, h in common_resolutions:
            for multiplier in [1.5, 2, 3, 4]:  # NV12, RGB888, RGB565, XRGB8888
                if w * h * int(multiplier) == file_size:
                    return w, h

        # 默认值
        return 1920, 1080

    def _show_format_dialog(self, width: int, height: int, preset_fmt: Optional[PixelFormat] = None) -> tuple:
        """显示格式选择对话框，返回 (width, height, format, ok)
        preset_fmt: 从文件名解析出的预设格式
        """
        from PyQt6.QtWidgets import QDialog, QFormLayout

        dialog = QDialog(self)
        dialog.setWindowTitle("Select Format")
        dialog.setStyleSheet("""
            QDialog { background-color: #2D2D30; }
            QLabel { color: #CCCCCC; }
            QLineEdit {
                background-color: #3C3C3C;
                color: #CCCCCC;
                border: 1px solid #555555;
                padding: 4px;
            }
            QComboBox {
                background-color: #3C3C3C;
                color: #CCCCCC;
                border: 1px solid #555555;
            }
            QPushButton {
                background-color: #007ACC;
                color: white;
                border: none;
                padding: 5px 15px;
                min-width: 70px;
            }
            QPushButton:hover {
                background-color: #005A9E;
            }
            QPushButton[accessibleName="cancel"] {
                background-color: #555555;
            }
            QPushButton[accessibleName="cancel"]:hover {
                background-color: #666666;
            }
        """)

        layout = QFormLayout(dialog)

        # 使用 QLineEdit 让用户可以编辑分辨率
        width_edit = QLineEdit()
        width_edit.setText(str(width))
        width_edit.setPlaceholderText("Width in pixels")
        height_edit = QLineEdit()
        height_edit.setText(str(height))
        height_edit.setPlaceholderText("Height in pixels")
        layout.addRow("Width:", width_edit)
        layout.addRow("Height:", height_edit)

        combo = QComboBox()
        for fmt in PixelFormat:
            combo.addItem(fmt.value, fmt)
        # 默认选中预设格式（从文件名解析），否则默认 NV12
        if preset_fmt:
            idx = combo.findData(preset_fmt)
        else:
            idx = combo.findData(PixelFormat.NV12)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        layout.addRow("Format:", combo)

        buttons = QWidget()
        btn_layout = QHBoxLayout(buttons)
        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setAccessibleName("cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addRow(buttons)

        if dialog.exec():
            try:
                w = int(width_edit.text())
                h = int(height_edit.text())
                if w > 0 and h > 0:
                    return w, h, combo.currentData(), True
                else:
                    QMessageBox.warning(self, "Invalid Resolution", "Width and Height must be positive integers.")
                    return None, None, None, False
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Width and Height must be valid integers.")
                return None, None, None, False
        return None, None, None, False

    def _close_tab(self, index: int):
        self.tab_widget.removeTab(index)

    def _close_current_tab(self):
        if self.tab_widget.currentIndex() >= 0:
            self._close_tab(self.tab_widget.currentIndex())

    def _fit_window(self):
        tab = self.tab_widget.currentWidget()
        if tab:
            for btn in tab.zoom_group.buttons():
                if btn.text() == "Fit":
                    btn.setChecked(True)
                    break
            tab.zoom = 0
            tab._update_display()

    def _actual_size(self):
        tab = self.tab_widget.currentWidget()
        if tab:
            for btn in tab.zoom_group.buttons():
                if btn.text() == "1x":
                    btn.setChecked(True)
                    break
            tab.zoom = 1.0
            tab._update_display()

    def _show_about(self):
        numpy_status = "Enabled" if HAS_NUMPY else "Disabled (slower)"
        QMessageBox.about(
            self, "About PixelViewer",
            "PixelViewer v1.0\n\n"
            "Raw Image Viewer\n"
            "Supports RGB888, RGB565, XRGB8888\n"
            "Supports NV12, NV21, NV16, NV61, NV24, NV42\n\n"
            "Features:\n"
            "- Multi-tab support (like 7yuv)\n"
            f"- NumPy acceleration: {numpy_status}\n\n"
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
