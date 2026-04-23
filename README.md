# PixelViewer

A Windows desktop application for viewing raw YUV/RGB image files. Supports multiple pixel formats including NV12, NV21, NV16, NV24, RGB888, RGB565, and more.

## Features

- **Multiple pixel formats**: RGB888, BGR888, RGB565, BGR565, XRGB8888, XBGR8888, NV12, NV21, NV16, NV61, NV24, NV42
- **Multi-tab support**: Open multiple files simultaneously
- **Auto YUV range detection**: Automatically detects Full or Limited range for YUV formats
- **Manual YUV range override**: Switch between Full/Limited range manually
- **Customizable resolution**: Edit width/height before opening
- **Auto resolution detection**: Matches file size against common resolutions
- **Pixel info**: Display mouse coordinates and color values (Y/R/G/B)
- **Scrollable view**: Large images show scrollbars, default to original resolution
- **Zoom controls**: 1x, 1/2x, 1/4x, 1/8x
- **Dark theme**: VS Code-style dark UI

## Installation

```bash
pip install PyQt6
```

Optional: NumPy provides faster YUV/RGB decoding.

```bash
pip install numpy
```

## Usage

```bash
python pixel_viewer.py
```

### Mouse Controls

- **Ctrl + Wheel**: Zoom in/out
- **Drag**: Pan the image (when zoomed in)
- **Hover**: Display pixel coordinates and color values in status bar

### YUV Format Notes

| Format | Chroma Sampling | UV Plane Size | UV Upsampling |
|--------|----------------|---------------|---------------|
| NV12   | 4:2:0          | (H/2) x (W/2)| 2x vertical + 2x horizontal |
| NV21   | 4:2:0 (VU)     | (H/2) x (W/2)| 2x vertical + 2x horizontal |
| NV16   | 4:2:2          | H x (W/2)    | 2x horizontal only |
| NV61   | 4:2:2 (VU)     | H x (W/2)    | 2x horizontal only |
| NV24   | 4:4:4          | H x W        | No upsampling needed |
| NV42   | 4:4:4 (VU)     | H x W        | No upsampling needed |

**YUV Range**: Most camera/sensor outputs use Full range (Y=0-255). If colors appear washed out or desaturated, try switching the Range setting between Full and Limited.

## Project Structure

```
pixel_viewer.py   # Main application (~1000 lines)
SPEC.md           # Project specification (Chinese)
CLAUDE.md         # Claude Code instructions
```

## Key Classes

- **PixelFormat** (Enum): Supported pixel formats
- **YuvRange** (Enum): YUV color range - LIMITED (TV 16-235) or FULL (PC 0-255)
- **YuvRangeDetector**: Auto-detects YUV range from data
- **PixelDecoder**: Decodes raw bytes to QImage (NumPy or pure Python fallback)
- **ImageGraphicsView**: Custom QGraphicsView with zoom and pan
- **ImageTab**: Individual tab for each opened file
- **MainWindow**: Main window with tab management
