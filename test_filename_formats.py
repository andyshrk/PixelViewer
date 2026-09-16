#!/usr/bin/env python3
"""Regression tests for supported DRM FourCC filename aliases."""

import unittest

from pixel_viewer import MainWindow, PixelFormat


class FilenameFormatsTest(unittest.TestCase):
    def test_supported_fourcc_aliases(self):
        cases = {
            "RG24": PixelFormat.RGB888,
            "BG24": PixelFormat.BGR888,
            "RG16": PixelFormat.RGB565,
            "BG16": PixelFormat.BGR565,
            "XR24": PixelFormat.XRGB8888,
            "XB24": PixelFormat.XBGR8888,
            "AR24": PixelFormat.XRGB8888,
            "AB24": PixelFormat.XBGR8888,
            "AR30": PixelFormat.AR30,
            "AB30": PixelFormat.AB30,
            "NV12": PixelFormat.NV12,
            "NV15": PixelFormat.NV15,
            "NV21": PixelFormat.NV21,
            "NV16": PixelFormat.NV16,
            "NV61": PixelFormat.NV61,
            "NV24": PixelFormat.NV24,
            "NV42": PixelFormat.NV42,
        }
        self.assertEqual(set(cases.values()), set(PixelFormat))
        for alias, expected in cases.items():
            for token in (alias, alias.lower()):
                with self.subTest(token=token):
                    self.assertEqual(
                        MainWindow._parse_filename(None, f"frame_1920x1080_{token}.raw"),
                        (1920, 1080, expected),
                    )

    def test_full_format_names(self):
        cases = {fmt.name: fmt for fmt in PixelFormat}
        cases.update({
            "ARGB8888": PixelFormat.XRGB8888,
            "ABGR8888": PixelFormat.XBGR8888,
            "ARGB2101010": PixelFormat.AR30,
            "ABGR2101010": PixelFormat.AB30,
        })
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(
                    MainWindow._parse_filename(None, f"frame_1920_1080_DRM_FORMAT_{name}.raw"),
                    (1920, 1080, expected),
                )


if __name__ == "__main__":
    unittest.main()
