#!/usr/bin/env python3
"""Regression tests for DRM_FORMAT_NV15."""

import unittest

from pixel_viewer import HAS_NUMPY, MainWindow, PixelDecoder, PixelFormat, YuvRange


def pack_row(samples):
    padded = list(samples)
    padded.extend([0] * ((-len(padded)) % 4))
    data = bytearray()
    for offset in range(0, len(padded), 4):
        value = 0
        for component, sample in enumerate(padded[offset:offset + 4]):
            value |= sample << (component * 10)
        data.extend(value.to_bytes(5, "little"))
    return bytes(data)


def make_nv15(y_rows, uv_rows):
    return b"".join(pack_row(row) for row in y_rows + uv_rows)


class Nv15Test(unittest.TestCase):
    def setUp(self):
        PixelDecoder.set_yuv_range(YuvRange.FULL)

    def test_decode_luma_scaling_and_packed_bit_order(self):
        data = make_nv15(
            [[0, 1023, 512, 256], [0, 1023, 512, 256]],
            [[512, 512, 512, 512]],
        )
        expected = [(0, 0, 0), (255, 255, 255), (128, 128, 128), (64, 64, 64)]

        decoders = [PixelDecoder._decode_pure]
        if HAS_NUMPY:
            decoders.append(PixelDecoder._decode_numpy)
        for decoder in decoders:
            with self.subTest(decoder=decoder.__name__):
                image = decoder(data, 4, 2, PixelFormat.NV15)
                actual = []
                for x in range(4):
                    color = image.pixelColor(x, 0)
                    actual.append((color.red(), color.green(), color.blue()))
                self.assertEqual(actual, expected)

    def test_decode_interleaved_uv_order(self):
        data = make_nv15(
            [[512, 512, 512, 512], [512, 512, 512, 512]],
            [[1023, 512, 512, 1023]],
        )

        decoders = [PixelDecoder._decode_pure]
        if HAS_NUMPY:
            decoders.append(PixelDecoder._decode_numpy)
        for decoder in decoders:
            with self.subTest(decoder=decoder.__name__):
                image = decoder(data, 4, 2, PixelFormat.NV15)
                colors = [image.pixelColor(x, 0) for x in range(4)]
                self.assertEqual((colors[0].red(), colors[0].green(), colors[0].blue()), (128, 84, 255))
                self.assertEqual((colors[2].red(), colors[2].green(), colors[2].blue()), (255, 37, 128))

    def test_required_size_and_filename_alias(self):
        self.assertEqual(PixelDecoder.get_required_size(4, 2, PixelFormat.NV15), 15)
        _, _, fmt = MainWindow._parse_filename(None, "frame_1920x1080_DRM_FORMAT_NV15.raw")
        self.assertEqual(fmt, PixelFormat.NV15)
        file_size = 1920 * 1080 * 15 // 8
        self.assertEqual(MainWindow._auto_detect_resolution(None, file_size), (1920, 1080))


if __name__ == "__main__":
    unittest.main()
