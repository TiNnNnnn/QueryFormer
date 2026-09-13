import unittest

from model.device import resolve_device


class DeviceTest(unittest.TestCase):
    def test_explicit_cpu(self):
        self.assertEqual(resolve_device("cpu").type, "cpu")

    def test_auto_returns_supported_device(self):
        self.assertIn(resolve_device().type, {"cpu", "cuda", "mps"})


if __name__ == "__main__":
    unittest.main()
