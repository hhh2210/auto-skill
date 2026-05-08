from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from auto_skill.llm import ChatCompletionConfig
from auto_skill.mvp import (
    HELDOUT_GENERATION_PROMPT_VERSION,
    build_heldout_generation_prompt,
    build_presentbench_layout_plan_prompt,
)
from auto_skill.schemas import UserExample
from auto_skill.writingbench_eval import (
    average_writingbench_scores,
    build_writingbench_official_prompt,
    parse_writingbench_score,
    process_gen_field,
)
from scripts.eval.run_heldout_eval import append_checkpoint_row
from scripts.eval.run_writingbench_official_eval import (
    existing_rows_outside_expected_cells,
    has_non_success_rows,
    is_reusable_candidate_row,
    load_compatible_resume_success_rows,
    mode_needs_skill,
    mode_skill,
    row_matches_runtime,
    runtime_metadata,
    score_with_writingbench_prompt,
    skill_index_with_feature_signatures,
    summarize_rows,
    write_empty_eval_result,
)


class SequencedCompletionClient:
    def __init__(self, texts: list[str], finish_reasons: list[str] | None = None) -> None:
        self.texts = list(texts)
        self.finish_reasons = list(finish_reasons or ["stop"] * len(texts))

    def complete(self, *_args, **_kwargs):
        text = self.texts.pop(0)
        finish_reason = self.finish_reasons.pop(0)
        return type(
            "Completion",
            (),
            {
                "text": text,
                "model": "judge",
                "usage": {"total_tokens": 1},
                "finish_reason": finish_reason,
                "request_id": "req",
            },
        )()


class WritingBenchEvalTests(unittest.TestCase):
    def test_auto_skill_does_not_fallback_to_no_validation(self) -> None:
        skills = {("pack-1", "auto_skill_feature_driven_no_validation"): "Skill"}

        self.assertIsNone(mode_skill("auto_skill", skills, "pack-1"))
        self.assertEqual(mode_skill("ours_no_validation", skills, "pack-1"), "Skill")

    def test_examples_plus_modes_select_expected_skill_artifact(self) -> None:
        skills = {
            ("pack-1", "one_shot_skill_from_examples"): "One-shot Skill",
            ("pack-1", "auto_skill_feature_driven_no_validation"): "Feature Skill",
        }

        self.assertEqual(
            mode_skill("examples_plus_one_shot_skill", skills, "pack-1"),
            "One-shot Skill",
        )
        self.assertEqual(
            mode_skill("examples_plus_feature_skill", skills, "pack-1"),
            "Feature Skill",
        )
        self.assertIsNone(mode_skill("feature_signatures_only", skills, "pack-1"))
        self.assertEqual(
            mode_skill("slide_constrained_examples_plus_feature_skill", skills, "pack-1"),
            "Feature Skill",
        )
        self.assertIsNone(mode_skill("layout_plan_examples_plus_feature_skill", skills, "pack-1"))

    def test_feature_signature_modes_select_compact_feature_context(self) -> None:
        skills = skill_index_with_feature_signatures(
            [
                {
                    "pack_id": "pack-1",
                    "mode": "auto_skill_feature_driven_no_validation",
                    "skill_md": "Feature Skill",
                    "feature_reports": [
                        {
                            "content_features": {"key_variables": ["temperature"]},
                            "structure_features": {"sections": ["Methods"]},
                        }
                    ],
                    "cross_example_report": {
                        "stable_features": ["Use numbered sections"],
                        "candidate_rules": [
                            {"rule": "Keep actions concrete", "support_count": 3}
                        ],
                    },
                }
            ]
        )

        self.assertIn(
            "Abstract Feature Signatures",
            mode_skill("feature_signatures_only", skills, "pack-1") or "",
        )
        self.assertIn(
            "Keep actions concrete",
            mode_skill("examples_plus_feature_signatures", skills, "pack-1") or "",
        )
        self.assertIn(
            "Keep actions concrete",
            mode_skill("task_first_feature_signatures", skills, "pack-1") or "",
        )
        self.assertIn(
            "Task-Grounded Operational Anchors",
            mode_skill("task_first_operational_anchors", skills, "pack-1") or "",
        )

    def test_reused_candidate_bypasses_skill_requirement(self) -> None:
        reused = {"status": "success", "generation": {"text": "candidate"}}

        self.assertTrue(is_reusable_candidate_row(reused))
        self.assertFalse(mode_needs_skill("one_shot_skill_from_examples", reused))
        self.assertTrue(mode_needs_skill("one_shot_skill_from_examples", None))
        self.assertTrue(mode_needs_skill("examples_plus_feature_skill", None))
        self.assertTrue(mode_needs_skill("examples_plus_feature_signatures", None))
        self.assertTrue(mode_needs_skill("task_first_feature_signatures", None))
        self.assertTrue(mode_needs_skill("task_first_operational_anchors", None))

    def test_reused_candidate_requires_success_generation_text(self) -> None:
        self.assertFalse(is_reusable_candidate_row({"status": "success", "generation": {}}))
        self.assertFalse(
            is_reusable_candidate_row(
                {"status": "generation_incomplete", "generation": {"text": "x"}}
            )
        )

    def test_runtime_metadata_tracks_reuse_and_judge_thinking(self) -> None:
        solver = ChatCompletionConfig(
            base_url="https://example.test/v1",
            api_key="key",
            model="solver",
            enable_thinking=False,
        )
        judge = ChatCompletionConfig(
            base_url="https://example.test/v1",
            api_key="key",
            model="judge",
            enable_thinking=True,
            thinking_budget=1024,
        )

        metadata = runtime_metadata(
            config=solver,
            judge_config=judge,
            reuse_candidates_from=Path("runs/candidates.jsonl"),
            temperature=0.2,
            max_tokens=8192,
            judge_max_tokens=1024,
            max_material_chars=4000,
        )

        self.assertEqual(
            metadata,
            {
                "reused_candidates_from": "runs/candidates.jsonl",
                "judge_enable_thinking": True,
                "judge_thinking_budget": 1024,
                "judge_max_tokens": 1024,
            },
        )

    def test_runtime_metadata_tracks_solver_settings_without_reuse(self) -> None:
        solver = ChatCompletionConfig(
            base_url="https://example.test/v1",
            api_key="key",
            model="solver",
            enable_thinking=False,
            thinking_budget=256,
        )

        metadata = runtime_metadata(
            config=solver,
            judge_config=None,
            reuse_candidates_from=None,
            temperature=0.2,
            max_tokens=8192,
            judge_max_tokens=1024,
            max_material_chars=4000,
        )

        self.assertEqual(
            metadata,
            {
                "reused_candidates_from": None,
                "heldout_generation_prompt_version": HELDOUT_GENERATION_PROMPT_VERSION,
                "judge_enable_thinking": False,
                "judge_thinking_budget": 256,
                "judge_max_tokens": 1024,
                "solver_model": "solver",
                "solver_temperature": 0.2,
                "solver_max_tokens": 8192,
                "max_material_chars": 4000,
                "solver_enable_thinking": False,
                "solver_thinking_budget": 256,
            },
        )

    def test_resume_checkpoint_requires_compatible_judge_runtime(self) -> None:
        metadata = {
            "reused_candidates_from": "runs/source.jsonl",
            "judge_enable_thinking": True,
            "judge_thinking_budget": 1024,
        }
        matching = {"judge_model": "judge-a", **metadata}
        wrong_model = {"judge_model": "judge-b", **metadata}
        wrong_budget = {"judge_model": "judge-a", **metadata, "judge_thinking_budget": 256}
        missing_metadata = {"judge_model": "judge-a", "reused_candidates_from": "runs/source.jsonl"}

        self.assertTrue(
            row_matches_runtime(
                matching,
                expected_judge_model="judge-a",
                metadata=metadata,
            )
        )
        self.assertFalse(
            row_matches_runtime(
                wrong_model,
                expected_judge_model="judge-a",
                metadata=metadata,
            )
        )
        self.assertFalse(
            row_matches_runtime(
                wrong_budget,
                expected_judge_model="judge-a",
                metadata=metadata,
            )
        )
        self.assertFalse(
            row_matches_runtime(
                missing_metadata,
                expected_judge_model="judge-a",
                metadata=metadata,
            )
        )

    def test_load_resume_checkpoint_filters_incompatible_success_rows(self) -> None:
        metadata = {
            "reused_candidates_from": "runs/source.jsonl",
            "judge_enable_thinking": True,
            "judge_thinking_budget": 1024,
        }
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "rows.jsonl"
            rows = []
            append_checkpoint_row(
                out,
                rows,
                {
                    "pack_id": "pack",
                    "task_id": "task",
                    "mode": "prompt_only",
                    "status": "success",
                    "overall_score": 8,
                    "judge_model": "judge-a",
                    **metadata,
                },
            )
            append_checkpoint_row(
                out,
                rows,
                {
                    "pack_id": "pack",
                    "task_id": "task",
                    "mode": "auto_skill",
                    "status": "success",
                    "overall_score": 9,
                    "judge_model": "judge-a",
                    **metadata,
                    "judge_thinking_budget": 256,
                },
            )

            resumed = load_compatible_resume_success_rows(
                out,
                [("pack", "task", "prompt_only"), ("pack", "task", "auto_skill")],
                expected_judge_model="judge-a",
                metadata=metadata,
            )

            self.assertEqual([row["mode"] for row in resumed], ["prompt_only"])

    def test_existing_rows_outside_expected_cells_detects_prune_risk(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "rows.jsonl"
            rows = []
            append_checkpoint_row(
                out,
                rows,
                {
                    "pack_id": "pack",
                    "task_id": "task",
                    "mode": "prompt_only",
                    "status": "success",
                },
            )
            append_checkpoint_row(
                out,
                rows,
                {
                    "pack_id": "other",
                    "task_id": "task",
                    "mode": "prompt_only",
                    "status": "success",
                },
            )

            outside = existing_rows_outside_expected_cells(
                out,
                [("pack", "task", "prompt_only")],
            )

        self.assertEqual(outside, [("other", "task", "prompt_only")])

    def test_process_gen_field_removes_thinking_trace(self) -> None:
        self.assertEqual(process_gen_field("trace</think>\n\nfinal"), "final")

    def test_build_prompt_serializes_criterion(self) -> None:
        prompt = build_writingbench_official_prompt(
            template="{query}|{response}|{criteria}",
            query="Q",
            response="R",
            criteria={"name": "C"},
        )

        self.assertIn('"name": "C"', prompt)

    def test_examples_plus_skill_prompt_includes_examples_and_skill(self) -> None:
        prompt = build_heldout_generation_prompt(
            task={"task_id": "task-1", "task_input": "Write a report", "materials": []},
            mode="examples_plus_feature_skill",
            examples=[
                UserExample(
                    example_id="ex-1",
                    task_input="Example task",
                    output="Example output",
                )
            ],
            skill_md="Feature Skill",
        )

        self.assertIn("User examples:", prompt)
        self.assertIn("Example output", prompt)
        self.assertIn("Reusable skill:", prompt)
        self.assertIn("Feature Skill", prompt)

    def test_slide_constrained_examples_plus_prompt_prioritizes_current_task(self) -> None:
        prompt = build_heldout_generation_prompt(
            task={"task_id": "task-1", "task_input": "Create slides", "materials": []},
            mode="slide_constrained_examples_plus_feature_skill",
            examples=[
                UserExample(
                    example_id="ex-1",
                    task_input="Example task",
                    output="Example output",
                )
            ],
            skill_md="Feature Skill",
        )

        self.assertIn("User examples:", prompt)
        self.assertIn("Reusable skill:", prompt)
        self.assertIn("Slide-task constraint priority:", prompt)
        self.assertIn("Do not copy example slide counts", prompt)

    def test_layout_plan_prompt_extracts_current_task_constraints(self) -> None:
        prompt = build_presentbench_layout_plan_prompt(
            task={"task_id": "task-1", "task_input": "Create 12 slides", "materials": []},
            examples=[
                UserExample(
                    example_id="ex-1",
                    task_input="Example task",
                    output="Example output",
                )
            ],
            skill_md="Feature Skill",
        )

        self.assertIn("Hard constraints:", prompt)
        self.assertIn("Per-slide allocation:", prompt)
        self.assertIn("Do-not-copy-from-examples:", prompt)
        self.assertIn("Create 12 slides", prompt)

    def test_parse_score_accepts_fenced_json(self) -> None:
        parsed = parse_writingbench_score('```json\n{"score": 8, "reason": "ok"}\n```')

        self.assertEqual(parsed["score"], 8)
        self.assertNotIn("parse_error", parsed)

    def test_parse_score_rejects_non_integer_score(self) -> None:
        parsed = parse_writingbench_score('{"score": 8.5, "reason": "ok"}')

        self.assertEqual(parsed["parse_error"], "score_must_be_integer_1_to_10")

    def test_parse_score_rejects_boolean_score(self) -> None:
        parsed = parse_writingbench_score('{"score": true, "reason": "ok"}')

        self.assertEqual(parsed["parse_error"], "score_must_be_integer_1_to_10")

    def test_score_with_writingbench_prompt_retries_parse_error(self) -> None:
        status, scores, judge_calls = score_with_writingbench_prompt(
            client=SequencedCompletionClient(["bad", '{"score": 8, "reason": "ok"}']),
            system_prompt="system",
            prompt_template="{query}|{response}|{criteria}",
            query="Q",
            response="R",
            criteria=[{"name": "C"}],
            max_tokens=128,
            parse_max_attempts=2,
        )

        self.assertEqual(status, "success")
        self.assertEqual(scores["C"], [{"score": 8, "reason": "ok"}])
        self.assertEqual(
            [call["parse_error"] for call in judge_calls],
            ["no_json_object_found", None],
        )

    def test_score_with_writingbench_prompt_marks_content_filter_as_refusal(self) -> None:
        status, scores, judge_calls = score_with_writingbench_prompt(
            client=SequencedCompletionClient(
                ["The request was rejected because it was considered high risk"],
                finish_reasons=["content_filter"],
            ),
            system_prompt="system",
            prompt_template="{query}|{response}|{criteria}",
            query="Q",
            response="R",
            criteria=[{"name": "C"}],
            max_tokens=128,
            parse_max_attempts=3,
        )

        self.assertEqual(status, "judge_refusal")
        self.assertEqual(len(judge_calls), 1)
        self.assertEqual(judge_calls[0]["finish_reason"], "content_filter")
        self.assertEqual(scores["C"][0]["parse_error"], "no_json_object_found")

    def test_average_scores_matches_writingbench_mean(self) -> None:
        self.assertEqual(
            average_writingbench_scores(
                {
                    "A": [{"score": 8, "reason": "ok"}],
                    "B": [{"score": 6, "reason": "ok"}],
                }
            ),
            7.0,
        )

    def test_average_scores_ignores_boolean_score(self) -> None:
        self.assertIsNone(average_writingbench_scores({"A": [{"score": True}]}))

    def test_summary_tracks_failures(self) -> None:
        rows = [
            {
                "pack_id": "pack",
                "task_id": "t1",
                "mode": "prompt_only",
                "status": "success",
                "overall_score": 9,
            },
            {
                "pack_id": "pack",
                "task_id": "t1",
                "mode": "auto_skill",
                "status": "success",
                "overall_score": 10,
            },
            {
                "pack_id": "pack",
                "task_id": "t2",
                "mode": "auto_skill",
                "status": "missing_ours_full_skill",
            },
        ]

        summary = summarize_rows(rows)

        self.assertEqual(summary["status_counts"], {"missing_ours_full_skill": 1, "success": 2})
        self.assertEqual(summary["modes"]["prompt_only"]["mean_writingbench_score"], 9)
        self.assertEqual(summary["paired_deltas"]["auto_skill"]["mean_delta"], 1)
        self.assertEqual(
            summary["status_counts_by_mode"]["auto_skill"],
            {"missing_ours_full_skill": 1, "success": 1},
        )
        self.assertTrue(has_non_success_rows(rows))

    def test_empty_rows_have_no_success(self) -> None:
        summary = summarize_rows([])

        self.assertEqual(summary["status_counts"], {})
        self.assertEqual(summary["modes"], {})

    def test_write_empty_eval_result(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "rows.jsonl"
            summary_out = Path(tmp) / "summary.json"

            write_empty_eval_result(out, summary_out)

            self.assertEqual(out.read_text(), "")
            self.assertEqual(summarize_rows([]), __import__("json").loads(summary_out.read_text()))


if __name__ == "__main__":
    unittest.main()
