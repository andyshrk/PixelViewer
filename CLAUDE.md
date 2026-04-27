# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PixelViewer is a Windows desktop application for viewing raw YUV/RGB image files. It supports multiple pixel formats including NV12, NV21, RGB888, RGB565, etc.

## Running the Application

```bash
python pixel_viewer.py
```

Dependencies:
- **Required**: PyQt6 (`pip install PyQt6`)
- **Optional**: NumPy (`pip install numpy`) - provides faster YUV/RGB decoding

## Architecture

Single-file Python application (`pixel_viewer.py`) with the following key components:

### Core Classes

- **PixelFormat** (Enum): Defines supported pixel formats - RGB888, BGR888, RGB565, BGR565, XRGB8888, XBGR8888, NV12, NV21, NV16, NV61, NV24, NV42
- **YuvRange** (Enum): YUV color range - LIMITED (TV range 16-235) or FULL (PC range 0-255)
- **YuvRangeDetector**: Auto-detects YUV range by analyzing Y value distribution
- **PixelDecoder**: Decodes raw bytes to QImage using either NumPy or pure Python fallback
- **ImageGraphicsView**: Custom QGraphicsView with mouse tracking and Ctrl+wheel zoom
- **ImageTab**: Individual tab widget for each opened file
- **MainWindow**: Main application window with QTabWidget for multi-tab support

### Key Implementation Details

- **NV12/NV21 UV Upsampling** (line 218-219): Uses `np.repeat` twice to scale UV planes from 2:1 to 1:1
- **YUV to RGB Conversion**: Full range uses BT.709 formula; Limited range uses standard TV conversion
- **Resolution Auto-Detection** (line 933-950): Matches file size against common resolutions (1920x1080, 1920x1088, 3840x2160, etc.) with various format multipliers (1.5x for NV12, 2x for RGB888, etc.)
- **Filename Parsing** (`_parse_filename`, line 908-936): Extracts resolution and format from filename. Supports patterns like `3840x2160_NV12.bin` (x-separated), `video3840_2160_NV12.bin` (underscore-separated), and detects format names (NV12, RGB888, etc.)

### UI Theme

Dark theme with VS Code-style colors:
- Background: #1E1E1E
- Secondary: #2D2D30
- Accent: #007ACC

## Development Notes

- **YUV Range Detection**: Returns FULL by default; LIMITED only when Y values are strictly in 14-237 range
- **Manual Range Selection**: Once user manually changes Range, auto-detection is disabled for subsequent redraws (until app restart)
- **Filename Parsing**: Resolution and format are parsed from filename and pre-selected in dialog (if detected)
- **Format Dialog**: Allows editing resolution before opening (QLineEdit, not QLabel)
- **Fit Mode**: Uses `Qt.AspectRatioMode.IgnoreAspectRatio` to fill entire window

## File Structure

```
pixel_viewer.py     # Main application (single file, ~1000 lines)
SPEC.md            # Project specification document (Chinese)
test_*.py          # Various test scripts
gen_*.py           # Test data generators
```
