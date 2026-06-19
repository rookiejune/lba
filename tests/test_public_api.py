import unittest

from lba import LBA, LengthBatchingAdapter


class PublicApiTest(unittest.TestCase):
    def test_short_alias_points_to_main_adapter(self) -> None:
        self.assertIs(LBA, LengthBatchingAdapter)


if __name__ == "__main__":
    unittest.main()
