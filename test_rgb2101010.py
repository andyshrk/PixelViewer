#!/usr/bin/env python3
"""Regression tests for DRM AR30 and AB30 pixel formats."""

import unittest

from pixel_viewer import HAS_NUMPY, MainWindow, PixelDecoder, PixelFormat


def pack_pixel(fmt, r, g, b, alpha=3):
    if fmt == PixelFormat.AR30:
        value = (alpha << 30) | (r << 20) | (g << 10) | b
    else:
        value = (alpha << 30) | (b << 20) | (g << 10) | r
    return value.to_bytes(4, "little")


class Rgb2101010Test(unittest.TestCase):
    def test_decode_channel_order_and_scaling(self):
        samples = [
            (1023, 0, 0),
            (0, 1023, 0),
            (0, 0, 1023),
            (341, 682, 1023),
        ]
        expected = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (85, 170, 255),
        ]
        decoders = [PixelDecoder._decode_pure]
        if HAS_NUMPY:
            decoders.append(PixelDecoder._decode_numpy)

        for fmt in [PixelFormat.AR30, PixelFormat.AB30]:
            data = b"".join(pack_pixel(fmt, *sample) for sample in samples)
            for decoder in decoders:
                with self.subTest(fmt=fmt, decoder=decoder.__name__):
                    image = decoder(data, len(samples), 1, fmt)
                    actual = []
                    for x in range(len(samples)):
                        color = image.pixelColor(x, 0)
                        actual.append((color.red(), color.green(), color.blue()))
                    self.assertEqual(actual, expected)

    def test_required_size(self):
        for fmt in [PixelFormat.AR30, PixelFormat.AB30]:
            with self.subTest(fmt=fmt):
                self.assertEqual(PixelDecoder.get_required_size(3, 2, fmt), 24)

    def test_filename_aliases(self):
        cases = {
            "frame_1920x1080_AR30.raw": PixelFormat.AR30,
            "frame_1920x1080_AB30.raw": PixelFormat.AB30,
            "frame_1920x1080_DRM_FORMAT_ARGB2101010.raw": PixelFormat.AR30,
            "frame_1920x1080_drm_format_abgr2101010.raw": PixelFormat.AB30,
        }
        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                _, _, actual = MainWindow._parse_filename(None, filename)
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
