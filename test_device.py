import unittest
import json
import tempfile

from model.device import resolve_device
from model.database_util import Encoding, formatJoin
from model.dataset import PlanTreeDataset
from model.model import QueryFormer
from model.postgres_plan_dataset import load_pg_stats, load_plan_frames, template_family


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
        cast_number = encoding.encode_filters(
            ["(c.c_birth_year >= '1987'::numeric)"],
            alias="c", table="customer",
        )
        self.assertAlmostEqual(cast_number["val"][0], 0.87)
        self.assertEqual(
            formatJoin({"Hash Cond": "(orders.id = lineitem.order_id)"}),
            "lineitem.order_id = orders.id",
        )

    def test_native_plan_labels_and_pg_stats_are_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            plans = directory + "/plans.jsonl"
            stats = directory + "/stats.json"
            with open(plans, "w", encoding="utf-8") as output:
                for family in range(5):
                    for sample in range(2):
                        output.write(json.dumps({"kind": "plan", "record": {
                            "key": f"dsb:query{family:03d}_s{sample}",
                            "label_duration_ms": family + 1,
                            "plan": {"Plan": {"Node Type": "Result"}},
                        }}) + "\n")
            with open(stats, "w", encoding="utf-8") as output:
                json.dump([{"tablename": "customer", "attname": "id",
                            "histogram_bounds": "{1,2,3}"}], output)
            train, test = load_plan_frames(plans)
            histograms, minimums, columns = load_pg_stats(stats, 2)
        self.assertEqual(len(train) + len(test), 10)
        self.assertTrue(
            {template_family(key) for key in train["key"]}.isdisjoint(
                template_family(key) for key in test["key"]
            )
        )
        self.assertEqual(histograms.iloc[0]["bins"], [1.0, 2.0, 3.0])
        self.assertEqual(minimums["customer.id"], (1.0, 3.0))
        self.assertEqual(columns, {"NA": 0, "customer.id": 1})


if __name__ == "__main__":
    unittest.main()
