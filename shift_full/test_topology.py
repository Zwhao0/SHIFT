"""CPU-only topology planner tests; run with python3 -m unittest."""

import unittest

from shift_full.topology import (
    TABLE_4_2,
    TopologyError,
    analyze_topology,
    make_power2_layout,
    make_vendor_layout,
)


class StandardLayoutTests(unittest.TestCase):
    def test_paper_table_4_2_replica_counts_and_sources(self):
        expected = {"A": 2, "B": 2, "C": 4, "D": 8}
        for name, dimensions in TABLE_4_2.items():
            with self.subTest(layout=name):
                plan = make_power2_layout(16, cp=1, **dimensions)
                self.assertTrue(plan["valid"], plan["rejections"])
                self.assertEqual(plan["candidate_replica_count"], expected[name])
                self.assertEqual(plan["min_dp_reference"], expected[name])
                self.assertTrue(plan["checks"]["generated_min_dp_reference_matches"]["pass"])
                self.assertEqual(
                    {replica["size"] for replica in plan["replicas"]},
                    {16 // expected[name]},
                )
                self.assertTrue(plan["checks"]["role_coverage_matches"]["pass"])
                self.assertTrue(plan["checks"]["execution_groups_unbroken"]["pass"])
                self.assertTrue(plan["checks"]["recovery_sources_complete"]["pass"])
                for rank, sources in plan["recovery_sources"].items():
                    self.assertEqual(len(sources), expected[name] - 1, rank)
                    self.assertTrue(all(isinstance(source, int) for source in sources.values()))

    def test_vendor_legacy_and_node_local_pp_failure_survival(self):
        legacy = make_vendor_layout("legacy")
        local = make_vendor_layout("node_local_pp")
        self.assertTrue(legacy["valid"], legacy["rejections"])
        self.assertTrue(local["valid"], local["rejections"])
        self.assertEqual(legacy["candidate_replica_count"], 2)
        self.assertEqual(local["candidate_replica_count"], 2)
        self.assertEqual(legacy["survivable_replicas_by_node_failure"], {0: [], 1: []})
        self.assertEqual(local["survivable_replicas_by_node_failure"], {0: [1], 1: [0]})
        self.assertEqual(
            [replica["ranks"] for replica in local["replicas"]],
            [list(range(8)), list(range(8, 16))],
        )
        self.assertTrue(all(size == 0 for size in local["execution_groups_crossing_nodes"].values()))
        self.assertGreater(legacy["execution_groups_crossing_nodes"]["PP"], 0)


class ValidationTests(unittest.TestCase):
    def test_malformed_overlapping_group_is_rejected(self):
        plan = make_power2_layout(16, tp=2, pp=2, ep=4, etp=1)
        groups = plan["groups"]
        bad_tp = [[0, 1], [1, 2]] + [[rank] for rank in range(3, 16)]
        with self.assertRaisesRegex(TopologyError, "execution TP groups overlap at rank 1"):
            analyze_topology(
                16,
                groups["dense_dp"],
                groups["expert_dp"],
                {**groups["execution"], "TP": bad_tp},
            )

    def test_missing_dense_expert_source_is_reported(self):
        plan = make_power2_layout(16, tp=2, pp=2, ep=4, etp=1)
        groups = plan["groups"]
        no_cross_replica_expert_sources = [[rank] for rank in range(16)]
        result = analyze_topology(
            16,
            groups["dense_dp"],
            no_cross_replica_expert_sources,
            groups["execution"],
        )
        self.assertEqual(result["candidate_replica_count"], 2)
        self.assertFalse(result["valid"])
        self.assertFalse(result["checks"]["recovery_sources_complete"]["pass"])
        self.assertIn("missing or ambiguous dense∩expert", " ".join(result["rejections"]))

    def test_non_power_of_two_input_is_not_sent_through_min_rule(self):
        dense = [[0, 1, 2], [3, 4, 5]]
        expert = [[0, 1, 2], [3, 4, 5]]
        result = analyze_topology(
            6,
            dense,
            expert,
            {"TP": [list(range(6))]},
        )
        self.assertEqual(result["candidate_replica_count"], 1)
        self.assertEqual(result["metrics"]["dense_dp_group_sizes"], [3, 3])
        self.assertFalse(result["valid"])
        self.assertFalse(result["checks"]["recovery_sources_complete"]["pass"])
        with self.assertRaisesRegex(TopologyError, "world_size=12.*power of two"):
            make_power2_layout(12, tp=2, pp=2, ep=2, etp=1)

    def test_overlapping_execution_dependencies_return_measured_components(self):
        plan = make_power2_layout(16, tp=2, pp=2, ep=4, etp=1)
        groups = plan["groups"]
        execution = dict(groups["execution"])
        execution["FUSED"] = [list(range(16))]
        result = analyze_topology(
            16, groups["dense_dp"], groups["expert_dp"], execution
        )
        self.assertEqual(result["candidate_replica_count"], 1)
        self.assertTrue(result["checks"]["execution_groups_unbroken"]["pass"])
        self.assertFalse(result["checks"]["recovery_sources_complete"]["pass"])


if __name__ == "__main__":
    unittest.main()
