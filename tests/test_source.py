import unittest

from torch.utils.data import DataLoader, IterableDataset

from lba.source import build_source_loader


class SequenceIterableDataset(IterableDataset):
    def __init__(self, samples):
        self.samples = samples

    def __iter__(self):
        yield from self.samples


class SourceLoaderTest(unittest.TestCase):
    def test_map_source_records_keep_dataset_indices(self) -> None:
        loader = DataLoader(
            [[0], [1, 1], [2, 2, 2]],
            batch_size=2,
        )

        records = next(iter(build_source_loader(loader, len)))

        self.assertEqual([record.sample for record in records], [[0], [1, 1]])
        self.assertEqual([record.length for record in records], [1, 2])
        self.assertEqual([record.index for record in records], [0, 1])

    def test_iterable_source_records_do_not_have_indices(self) -> None:
        loader = DataLoader(
            SequenceIterableDataset([[0], [1, 1]]),
            batch_size=2,
        )

        records = next(iter(build_source_loader(loader, len)))

        self.assertEqual([record.sample for record in records], [[0], [1, 1]])
        self.assertEqual([record.length for record in records], [1, 2])
        self.assertEqual([record.index for record in records], [None, None])


if __name__ == "__main__":
    unittest.main()
