import unittest

from model.device import resolve_device
from model.dataset import PlanTreeDataset
from model.model import QueryFormer


class DeviceTest(unittest.TestCase):
    def test_explicit_cpu(self):
        self.assertEqual(resolve_device("cpu").type, "cpu")

    def test_auto_returns_supported_device(self):
        self.assertIn(resolve_device().type, {"cpu", "cuda", "mps"})

    def test_plan_size_and_vocabularies_are_not_workload_constants(self):
        plan = {"Plans": [{}, {"Plans": [{}]}]}
        self.assertEqual(PlanTreeDataset.count_nodes(plan), 4)
        model = QueryFormer(tables=26, types=48, joins=96, columns=431)
        self.assertEqual(model.embbed_layer.tableEmbed.num_embeddings, 26)
        self.assertEqual(model.embbed_layer.typeEmbed.num_embeddings, 48)
        self.assertEqual(model.embbed_layer.joinEmbed.num_embeddings, 96)
        self.assertEqual(model.embbed_layer.columnEmbed.num_embeddings, 431)


if __name__ == "__main__":
    unittest.main()
