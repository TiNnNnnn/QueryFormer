import unittest

from model.device import resolve_device
from model.database_util import Encoding, formatJoin
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

    def test_postgres_predicates_are_encoded_without_workload_parser(self):
        encoding = Encoding(
            {"customer.c_birth_year": (1900, 2000)},
            {"NA": 0, "customer.c_birth_year": 1},
        )
        encoded = encoding.encode_filters(
            ["((c.c_birth_year >= 1987) AND (c.c_birth_year <= 1993) "
             "AND (c.c_state = ANY ('{CA,TX}'::bpchar[])))"],
            alias="c", table="customer",
        )
        self.assertEqual(encoded["colId"], [1, 1])
        self.assertEqual(encoded["opId"], [0, 2])
        self.assertEqual(
            formatJoin({"Hash Cond": "(orders.id = lineitem.order_id)"}),
            "lineitem.order_id = orders.id",
        )


if __name__ == "__main__":
    unittest.main()
