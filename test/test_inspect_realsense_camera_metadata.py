from pathlib import Path
import sqlite3
import struct
import tempfile
import unittest

from scripts import inspect_realsense_camera_metadata as audit


def cdr(text, endian="<"):
    raw = text.encode("utf-8") + b"\x00"
    return (b"\x00\x01\x00\x00" if endian == "<" else b"\x00\x00\x00\x00") + struct.pack(endian + "I", len(raw)) + raw


class CameraMetadataTest(unittest.TestCase):
    def test_exact_cdr_both_endians_and_utf8(self):
        for endian in ("<", ">"):
            self.assertEqual(audit.decode_cdr_string(cdr("安全 camera", endian)), ("安全 camera", "none"))

    def test_reject_invalid_size_header_length_nul_encoding(self):
        good = cdr("test")
        for raw in (b"", good + b"\x00", good[:-1], b"\x00\x02\x00\x00" + good[4:],
                    good[:4] + struct.pack("<I", 999) + good[8:], cdr("a\x00b"), good[:-2] + b"\xff\x00",
                    b"x" * (audit.MAX_MESSAGE_BYTES + 1)):
            with self.subTest(raw=raw[:16]), self.assertRaises((ValueError, UnicodeDecodeError)):
                audit.decode_cdr_string(raw)

    def test_bounded_existing_zstd_and_reject_extra_data(self):
        try:
            import zstandard
        except ImportError:
            self.skipTest("Optional existing zstandard is absent; no installation")
        compressed = zstandard.ZstdCompressor().compress(cdr("camera"))
        self.assertEqual(audit.decode_cdr_string(compressed), ("camera", "zstd"))
        for raw in (compressed + b"extra", zstandard.ZstdCompressor().compress(cdr("x" * 70000))):
            with self.assertRaises(Exception):
                audit.decode_cdr_string(raw)

    def test_observed_intrinsics_scale_transform_and_invalid_fields(self):
        text = "width=640;height=480;fx=600;ppx=320;fy=601;ppy=240;model=Inverse Brown Conrady;coeffs=0,0,0,0,0"
        parsed = audit.interpret_text(text, "/Color_0/camera_info")
        self.assertEqual(parsed["K_conventional_from_named_fields"], [[600., 0., 320.], [0., 601., 240.], [0., 0., 1.]])
        self.assertEqual(audit.interpret_text("0.001000", "/option/Depth_Units/value")["value"], .001)
        transform = audit.interpret_text("rotation=1,0,0,0,1,0,0,0,1;translation=0,0,0", "/Depth_0/tf/ref_0")
        self.assertEqual(transform["transform_direction"], "unverified")
        for bad in (text + ";fx=600", text.replace("fx=600", "fx=nan"), text.replace("width=640", "width=-1")):
            with self.assertRaises(ValueError):
                audit.interpret_text(bad, "/Color_0/camera_info")

    def test_sqlite_read_only_bounded_metadata_and_no_image_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.db3"
            with sqlite3.connect(path) as c:
                c.execute("CREATE TABLE topics(id INTEGER PRIMARY KEY,name TEXT,type TEXT,serialization_format TEXT)")
                c.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY,topic_id INTEGER,timestamp INTEGER,data BLOB)")
                c.executemany("INSERT INTO topics VALUES(?,?,?,?)", [(1, "/option/Depth_Units/value", "std_msgs/msg/String", "cdr"),
                    (2, "/Color_0/image/data", "sensor_msgs/msg/Image", "cdr"), (3, "/bad/camera_info", "std_msgs/msg/String", "cdr")])
                c.executemany("INSERT INTO messages VALUES(?,?,?,?)", [(1, 1, 0, cdr("0.001")), (2, 2, 100, b"do not decode"),
                    (3, 1, 200, cdr("0.002")), (4, 2, 200, b"invalid payload"), (5, 3, 0, b"x" * (audit.MAX_MESSAGE_BYTES + 1))])
            before = path.read_bytes()
            result = audit.inspect_database(path)
            self.assertEqual(before, path.read_bytes())
            samples = result["metadata"][0]["first_and_latest_samples"]
            self.assertEqual([x["parsed"]["value"] for x in samples], [.001, .002])
            self.assertEqual(result["metadata"][1]["first_and_latest_samples"][0]["status"], "unparsed")
            self.assertFalse(result["image_timestamps"][0]["image_payload_read"])
            self.assertEqual(result["image_timestamps"][0]["message_id_timestamp_pairs"]["first_two"], [[2, 100], [4, 200]])


if __name__ == "__main__":
    unittest.main()
