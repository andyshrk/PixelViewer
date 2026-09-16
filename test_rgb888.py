#!/usr/bin/env python3
"""Regression tests for little-endian RGB888 and BGR888 channel order."""

import unittest

from pixel_viewer import HAS_NUMPY, PixelDecoder, PixelFormat


class Rgb888Test(unittest.TestCase):
    def test_decode_channel_order(self):
        expected = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (17, 83, 201)]
        cases = [
            (PixelFormat.RGB888, bytes.fromhex("0000ff 00ff00 ff0000 c95311")),
            (PixelFormat.BGR888, bytes.fromhex("ff0000 00ff00 0000ff 1153c9")),
        ]
        decoders = [PixelDecoder._decode_pure]
        if HAS_NUMPY:
            decoders.append(PixelDecoder._decode_numpy)

        for fmt, data in cases:
            for decoder in decoders:
                with self.subTest(fmt=fmt, decoder=decoder.__name__):
                    image = decoder(data, 2, 2, fmt)
                    self.assertEqual((image.width(), image.height()), (2, 2))
                    actual = []
                    for y in range(2):
                        for x in range(2):
                            color = image.pixelColor(x, y)
                            actual.append((color.red(), color.green(), color.blue()))
                    self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
