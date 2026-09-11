from io import BytesIO
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zipfile

import numpy as np

from scripts import audit_realsense_hand_object as audit


class ReadonlyRealsenseAuditTest(unittest.TestCase):
    def test_inventory_identifies_add_remove_and_inplace_change(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "one.json").write_bytes(b"a")
            first = audit.inventory(root)
            (root / "one.json").write_bytes(b"longer")
            (root / "two.npz").write_bytes(b"b")
            second = audit.inventory(root)
            changes = audit.differences(first, second)
            self.assertEqual(changes["added"], ["two.npz"])
            self.assertEqual(changes["changed"], ["one.json"])
            self.assertEqual(changes["removed"], [])
            self.assertEqual(audit.differences(second, second), {"added": [], "removed": [], "changed": []})

    def test_npz_reads_headers_not_object_payloads(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "test.npz"
            np.savez(path, numeric=np.ones((4, 3, 3), dtype=np.float32), opaque=np.array([{"never": "unpickle"}], dtype=object))
            result = audit.npz_headers(path)
            members = {row["member"]: row for row in result["members"]}
            self.assertEqual(members["numeric.npy"]["shape"], [4, 3, 3])
            self.assertEqual(members["numeric.npy"]["dtype"], "float32")
            self.assertTrue(members["opaque.npy"]["has_object_dtype"])
            self.assertFalse(result["full_crc_or_array_finiteness_verified"])

    def test_npz_object_body_can_be_invalid_because_only_header_is_read(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "header-only.npz"
            buffer = BytesIO()
            np.lib.format.write_array_header_1_0(buffer, {"shape": (1,), "fortran_order": False, "descr": "|O"})
            buffer.write(b"not a pickle; must not be executed or decoded")
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("unsafe.npy", buffer.getvalue())
            result = audit.npz_headers(path)
            self.assertTrue(result["members"][0]["has_object_dtype"])

    def test_sqlite_readonly_schema_and_counts_do_not_modify_source(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "source.db3"
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE topics(id INTEGER, name TEXT, type TEXT)")
                connection.execute("INSERT INTO topics VALUES(1, '/color', 'sensor_msgs/msg/Image')")
                connection.execute("CREATE TABLE messages(topic_id INTEGER, timestamp INTEGER, data BLOB)")
                connection.execute("INSERT INTO messages VALUES(1, 100, x'0102')")
            original = path.read_bytes()
            result = audit.sqlite_summary(path)
            self.assertEqual(result["topics"][0]["name"], "/color")
            self.assertEqual({row["name"]: row["rows"] for row in result["tables"]}, {"messages": 1, "topics": 1})
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(path.with_name(path.name + "-journal").exists())


if __name__ == "__main__":
    unittest.main()
