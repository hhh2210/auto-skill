from __future__ import annotations

import unittest

from auto_skill.criterion_diagnosis import (
    aggregate_bucket_diagnosis,
    classify_criterion,
    classify_criterion_buckets,
    diagnose,
    extract_criterion_scores,
    matched_keywords_for,
    per_criterion_aggregate,
    per_task_criterion_deltas,
)


def _row(
    *,
    pack_id: str,
    task_id: str,
    mode: str,
    scores: dict[str, int],
    status: str = "success",
    evaluator_kind: str = "writingbench_official_prompt_qwen_judge",
    judge_model: str = "qwen3.5-plus",
) -> dict[str, object]:
    return {
        "pack_id": pack_id,
        "task_id": task_id,
        "mode": mode,
        "status": status,
        "evaluator_kind": evaluator_kind,
        "judge_model": judge_model,
        "overall_score": sum(scores.values()) / len(scores) if scores else None,
        "scores": {
            name: [{"score": value, "reason": ""}] for name, value in scores.items()
        },
    }


class ClassifyCriterionTests(unittest.TestCase):
    def test_chinese_required_sections_keyword_hits_completeness_bucket(self) -> None:
        self.assertEqual(
            classify_criterion("教学内容完整性"),
            "required_sections_completeness",
        )

    def test_english_evidence_keyword_hits_evidence_bucket(self) -> None:
        self.assertEqual(
            classify_criterion("Academic Rigor and Evidence-Based Support"),
            "evidence_citation",
        )

    def test_depth_keyword_hits_depth_bucket(self) -> None:
        self.assertEqual(
            classify_criterion("Case Selection and Analysis Depth"),
            "depth_specificity_practical",
        )

    def test_depth_takes_precedence_over_accuracy_for_combined_names(self) -> None:
        # ``Content_Depth_and_Accuracy`` matches both keywords; we want the
        # more discriminative depth label as primary.
        self.assertEqual(
            classify_criterion("Content_Depth_and_Accuracy"),
            "depth_specificity_practical",
        )
        # The multi-label list still records the accuracy bucket so callers
        # can audit the heuristic.
        self.assertEqual(
            classify_criterion_buckets("Content_Depth_and_Accuracy"),
            ["depth_specificity_practical", "accuracy_professionalism"],
        )

    def test_pure_accuracy_name_stays_in_accuracy_bucket(self) -> None:
        self.assertEqual(
            classify_criterion("Factual Accuracy"),
            "accuracy_professionalism",
        )

    def test_unmatched_name_falls_back_to_other(self) -> None:
        self.assertEqual(classify_criterion("ZzzUnknownAxis"), "other")
        self.assertEqual(classify_criterion_buckets("ZzzUnknownAxis"), [])

    def test_matched_keywords_includes_keyword_text(self) -> None:
        matches = matched_keywords_for("Content_Depth_and_Accuracy")
        bucket_names = {bucket for bucket, _ in matches}
        self.assertIn("depth_specificity_practical", bucket_names)
        self.assertIn("accuracy_professionalism", bucket_names)
        kws = {kw for _, kw in matches}
        self.assertIn("depth", kws)
        self.assertIn("accuracy", kws)


class ExtractCriterionScoresTests(unittest.TestCase):
    def test_skips_non_success_rows(self) -> None:
        row = _row(
            pack_id="p",
            task_id="t",
            mode="m",
            scores={"a": 5},
            status="generation_incomplete",
        )
        self.assertEqual(extract_criterion_scores(row), ())

    def test_skips_missing_score_field(self) -> None:
        row = {
            "status": "success",
            "scores": {"a": [{"reason": "no score"}], "b": [{"score": 7}]},
        }
        scores = extract_criterion_scores(row)
        self.assertEqual([s.criterion for s in scores], ["b"])

    def test_keeps_only_first_judge_attempt(self) -> None:
        row = {
            "status": "success",
            "scores": {
                "a": [{"score": 4, "reason": "first"}, {"score": 9, "reason": "retry"}],
            },
        }
        scores = extract_criterion_scores(row)
        self.assertEqual(len(scores), 1)
        self.assertEqual(scores[0].score, 4.0)
        self.assertEqual(scores[0].reason, "first")


class PerTaskCriterionDeltasTests(unittest.TestCase):
    def test_pairs_only_common_tasks_with_matching_criterion_sets(self) -> None:
        rows = [
            _row(
                pack_id="p1",
                task_id="t1",
                mode="baseline",
                scores={"required completeness": 9, "depth analysis": 8},
            ),
            _row(
                pack_id="p1",
                task_id="t1",
                mode="candidate",
                scores={"required completeness": 5, "depth analysis": 6},
            ),
            _row(
                pack_id="p1",
                task_id="t2",
                mode="baseline",
                scores={"a": 7, "b": 7},
            ),
        ]
        result = per_task_criterion_deltas(
            rows, mode="candidate", baseline_mode="baseline"
        )
        self.assertEqual(len(result.deltas), 1)
        self.assertEqual(result.skipped, ())
        task = result.deltas[0]
        self.assertEqual(task.pack_id, "p1")
        self.assertEqual(task.task_id, "t1")
        self.assertAlmostEqual(task.overall_delta, -3.0)
        worst = task.worst_criterion
        self.assertIsNotNone(worst)
        self.assertEqual(worst.criterion, "required completeness")
        self.assertAlmostEqual(worst.delta, -4.0)

    def test_skips_tasks_with_mismatched_criterion_sets(self) -> None:
        rows = [
            _row(
                pack_id="p1",
                task_id="t1",
                mode="baseline",
                scores={"a": 8, "b": 8, "c": 8},
            ),
            _row(
                pack_id="p1",
                task_id="t1",
                mode="candidate",
                scores={"a": 5, "b": 5},
            ),
        ]
        result = per_task_criterion_deltas(
            rows, mode="candidate", baseline_mode="baseline"
        )
        self.assertEqual(result.deltas, ())
        self.assertEqual(len(result.skipped), 1)
        skipped = result.skipped[0]
        self.assertEqual(skipped.pack_id, "p1")
        self.assertEqual(skipped.task_id, "t1")
        self.assertEqual(skipped.baseline_only, ("c",))
        self.assertEqual(skipped.mode_only, ())

    def test_excludes_non_success_pairs(self) -> None:
        rows = [
            _row(
                pack_id="p1",
                task_id="t1",
                mode="baseline",
                scores={"a": 7, "b": 7},
                status="judge_parse_error",
            ),
            _row(
                pack_id="p1",
                task_id="t1",
                mode="candidate",
                scores={"a": 6, "b": 6},
            ),
        ]
        result = per_task_criterion_deltas(
            rows, mode="candidate", baseline_mode="baseline"
        )
        self.assertEqual(result.deltas, ())
        self.assertEqual(result.skipped, ())

    def test_duplicate_success_cells_are_reported_and_excluded(self) -> None:
        rows = [
            _row(
                pack_id="p1",
                task_id="t1",
                mode="baseline",
                scores={"a": 7, "b": 7},
            ),
            _row(
                pack_id="p1",
                task_id="t1",
                mode="candidate",
                scores={"a": 6, "b": 6},
                judge_model="qwen3.5-plus",
            ),
            _row(
                pack_id="p1",
                task_id="t1",
                mode="candidate",
                scores={"a": 8, "b": 8},
                judge_model="mimo-v2.5-pro",
            ),
        ]
        result = per_task_criterion_deltas(
            rows, mode="candidate", baseline_mode="baseline"
        )
        self.assertEqual(result.deltas, ())
        self.assertEqual(result.skipped, ())
        self.assertEqual(len(result.duplicate_cells), 1)
        duplicate = result.duplicate_cells[0]
        self.assertEqual(duplicate.mode, "candidate")
        self.assertEqual(duplicate.pack_id, "p1")
        self.assertEqual(duplicate.task_id, "t1")
        self.assertEqual(duplicate.count, 2)
        self.assertEqual(duplicate.judge_models, ("mimo-v2.5-pro", "qwen3.5-plus"))


class BucketDiagnosisTests(unittest.TestCase):
    def test_worst_pick_separates_losing_from_all_tasks(self) -> None:
        # Two tasks: t1 ours loses overall (-2.0), t2 ours wins overall (+1.5).
        # Without splitting, "depth axis" would look like a worst-pick on
        # t2 too even though ours actually won that task. We want
        # worst_pick_count_on_losing_tasks to ignore t2.
        rows = [
            _row(
                pack_id="p1",
                task_id="t1",
                mode="baseline",
                scores={"教学内容完整性": 9, "Other Axis Quality": 9},
            ),
            _row(
                pack_id="p1",
                task_id="t1",
                mode="candidate",
                scores={"教学内容完整性": 5, "Other Axis Quality": 8},
            ),
            _row(
                pack_id="p2",
                task_id="t1",
                mode="baseline",
                scores={"Analytical Depth": 6, "Other Axis Quality": 6},
            ),
            _row(
                pack_id="p2",
                task_id="t1",
                mode="candidate",
                scores={"Analytical Depth": 7, "Other Axis Quality": 9},
            ),
        ]
        pairing = per_task_criterion_deltas(
            rows, mode="candidate", baseline_mode="baseline"
        )
        diagnosis = aggregate_bucket_diagnosis(pairing.deltas)
        by_bucket = {b["bucket"]: b for b in diagnosis["buckets"]}

        completeness = by_bucket["required_sections_completeness"]
        self.assertEqual(completeness["worst_pick_count_all"], 1)
        self.assertEqual(completeness["worst_pick_count_on_losing_tasks"], 1)

        depth = by_bucket["depth_specificity_practical"]
        # On t2, "Analytical Depth" is the *least improved* axis (still
        # +1.0), so it is the worst_criterion of a winning task. It should
        # count in worst_pick_count_all but NOT on_losing_tasks.
        self.assertEqual(depth["worst_pick_count_all"], 1)
        self.assertEqual(depth["worst_pick_count_on_losing_tasks"], 0)

    def test_multilabel_worst_pick_counts_into_every_matched_bucket(self) -> None:
        rows = [
            _row(
                pack_id="p1",
                task_id="t1",
                mode="baseline",
                scores={"Content_Depth_and_Accuracy": 9, "Other Axis Quality": 9},
            ),
            _row(
                pack_id="p1",
                task_id="t1",
                mode="candidate",
                scores={"Content_Depth_and_Accuracy": 5, "Other Axis Quality": 8},
            ),
        ]
        pairing = per_task_criterion_deltas(
            rows, mode="candidate", baseline_mode="baseline"
        )
        diagnosis = aggregate_bucket_diagnosis(pairing.deltas)
        by_bucket = {b["bucket"]: b for b in diagnosis["buckets"]}
        # Single-label primary lands in depth (depth precedes accuracy).
        self.assertEqual(
            by_bucket["depth_specificity_practical"]["worst_pick_count_on_losing_tasks"],
            1,
        )
        # accuracy bucket has no single-label worst-picks but multi-label
        # records the accuracy keyword match.
        accuracy = by_bucket["accuracy_professionalism"]
        self.assertEqual(accuracy["worst_pick_count_on_losing_tasks"], 0)
        self.assertEqual(
            accuracy["worst_pick_count_on_losing_tasks_multilabel"], 1
        )


class DiagnoseEndToEndTests(unittest.TestCase):
    def test_full_diagnose_includes_top_regressions_sorted_ascending(self) -> None:
        rows = [
            _row(
                pack_id="p1",
                task_id="t1",
                mode="baseline",
                scores={"教学内容完整性": 9, "Style Voice": 8},
            ),
            _row(
                pack_id="p1",
                task_id="t1",
                mode="candidate",
                scores={"教学内容完整性": 5, "Style Voice": 7},
            ),
            _row(
                pack_id="p2",
                task_id="t2",
                mode="baseline",
                scores={"Analytical Depth": 6, "Format Structure": 7},
            ),
            _row(
                pack_id="p2",
                task_id="t2",
                mode="candidate",
                scores={"Analytical Depth": 9, "Format Structure": 8},
            ),
        ]
        result = diagnose(
            rows,
            mode="candidate",
            baseline_mode="baseline",
            top_n_regressions=2,
            top_n_improvements=2,
        )
        self.assertEqual(result["paired_overall"]["common_cells"], 2)
        self.assertEqual(result["skipped"]["skipped_mismatched_criteria_count"], 0)
        regressions = result["top_regressions"]
        self.assertEqual(regressions[0]["pack_id"], "p1")
        self.assertEqual(
            regressions[0]["worst_criterion"]["primary_bucket"],
            "required_sections_completeness",
        )
        self.assertTrue(regressions[0]["is_losing_task"])
        improvements = result["top_improvements"]
        self.assertEqual(improvements[0]["pack_id"], "p2")
        self.assertGreater(improvements[0]["overall_delta"], 0)
        self.assertTrue(improvements[0]["is_winning_task"])

    def test_diagnose_surfaces_skipped_mismatched_criteria(self) -> None:
        rows = [
            _row(
                pack_id="p1",
                task_id="t_keep",
                mode="baseline",
                scores={"a": 7, "b": 7},
            ),
            _row(
                pack_id="p1",
                task_id="t_keep",
                mode="candidate",
                scores={"a": 6, "b": 6},
            ),
            _row(
                pack_id="p1",
                task_id="t_drop",
                mode="baseline",
                scores={"a": 8, "b": 8, "c": 8},
            ),
            _row(
                pack_id="p1",
                task_id="t_drop",
                mode="candidate",
                scores={"a": 5, "b": 5},
            ),
        ]
        result = diagnose(rows, mode="candidate", baseline_mode="baseline")
        self.assertEqual(result["paired_overall"]["common_cells"], 1)
        self.assertEqual(result["skipped"]["skipped_mismatched_criteria_count"], 1)
        self.assertEqual(result["skipped"]["duplicate_success_cells_count"], 0)
        skipped = result["skipped"]["skipped_mismatched_criteria"][0]
        self.assertEqual(skipped["task_id"], "t_drop")
        self.assertEqual(skipped["baseline_only"], ["c"])

    def test_diagnose_surfaces_duplicate_success_cells(self) -> None:
        rows = [
            _row(
                pack_id="p1",
                task_id="t1",
                mode="baseline",
                scores={"a": 7, "b": 7},
            ),
            _row(
                pack_id="p1",
                task_id="t1",
                mode="candidate",
                scores={"a": 6, "b": 6},
                judge_model="qwen3.5-plus",
            ),
            _row(
                pack_id="p1",
                task_id="t1",
                mode="candidate",
                scores={"a": 8, "b": 8},
                judge_model="mimo-v2.5-pro",
            ),
        ]
        result = diagnose(rows, mode="candidate", baseline_mode="baseline")
        self.assertEqual(result["paired_overall"]["common_cells"], 0)
        self.assertEqual(result["skipped"]["duplicate_success_cells_count"], 1)
        duplicate = result["skipped"]["duplicate_success_cells"][0]
        self.assertEqual(duplicate["mode"], "candidate")
        self.assertEqual(duplicate["count"], 2)
        self.assertEqual(duplicate["judge_models"], ["mimo-v2.5-pro", "qwen3.5-plus"])

    def test_per_criterion_aggregate_orders_by_mean_delta(self) -> None:
        rows = [
            _row(
                pack_id="p1",
                task_id="t1",
                mode="baseline",
                scores={"axis_a": 8, "axis_b": 8},
            ),
            _row(
                pack_id="p1",
                task_id="t1",
                mode="candidate",
                scores={"axis_a": 4, "axis_b": 9},
            ),
        ]
        pairing = per_task_criterion_deltas(
            rows, mode="candidate", baseline_mode="baseline"
        )
        aggregate = per_criterion_aggregate(pairing.deltas)
        self.assertEqual(aggregate[0]["criterion"], "axis_a")
        self.assertAlmostEqual(aggregate[0]["mean_delta"], -4.0)
        self.assertEqual(aggregate[-1]["criterion"], "axis_b")


if __name__ == "__main__":
    unittest.main()
