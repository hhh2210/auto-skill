"""Keyword bucket heuristics for criterion diagnostics."""

from __future__ import annotations

# Keyword buckets for the systemic-failure roll-up. Multilingual (zh+en)
# because WritingBench mixes both. Matching is case-insensitive substring on
# the normalized name. Order controls the *primary* (single-label) bucket
# returned by ``classify_criterion``; ``classify_criterion_buckets`` returns
# every match so callers can see the full label set.
#
# Ordering principle: more concrete dimensions before broader evaluative
# umbrellas. Specifically ``depth_specificity_practical`` precedes
# ``accuracy_professionalism`` so that names like
# "Content_Depth_and_Accuracy" or "Financial Accuracy and Specificity" are
# attributed to the more discriminative depth axis. Pure accuracy names
# ("Factual Accuracy", "Academic Rigor") do not contain depth keywords and
# stay in the accuracy bucket.
_KEYWORD_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "length_word_count",
        (
            "字数",
            "篇幅",
            "长度",
            "word count",
            "word_count",
            "length",
            "wordcount",
            "字符",
            "character count",
        ),
    ),
    (
        "format_structure",
        (
            "格式",
            "结构",
            "层次",
            "组织",
            "排版",
            "章节",
            "section",
            "structure",
            "format",
            "organization",
            "layout",
            "outline",
            "hierarchy",
        ),
    ),
    (
        "required_sections_completeness",
        (
            "必填",
            "必备",
            "完整",
            "齐全",
            "覆盖",
            "完备",
            "缺失",
            "全面",
            "completeness",
            "complete",
            "coverage",
            "required",
            "comprehensive",
            "essential",
            "all sections",
        ),
    ),
    (
        "evidence_citation",
        (
            "证据",
            "引用",
            "数据",
            "参考文献",
            "来源",
            "依据",
            "事实",
            "citation",
            "evidence",
            "source",
            "reference",
            "supporting",
        ),
    ),
    (
        "style_tone_voice",
        (
            "风格",
            "语气",
            "口吻",
            "笔调",
            "tone",
            "style",
            "voice",
            "register",
            "persona",
            "narrator",
        ),
    ),
    (
        "language_terminology",
        (
            "语言",
            "用词",
            "措辞",
            "术语",
            "表达",
            "language",
            "terminology",
            "wording",
            "phrasing",
            "diction",
        ),
    ),
    (
        "novelty_creativity",
        (
            "创新",
            "原创",
            "新颖",
            "独到",
            "创意",
            "novelty",
            "creativity",
            "originality",
            "innovative",
            "innovation",
        ),
    ),
    (
        "audience_persuasion",
        (
            "受众",
            "读者",
            "说服",
            "感染",
            "吸引",
            "audience",
            "persuasion",
            "engagement",
            "appeal",
            "reader",
        ),
    ),
    (
        "task_alignment",
        (
            "符合",
            "贴合",
            "对齐",
            "目标",
            "需求",
            "alignment",
            "relevance",
            "task",
            "objective",
        ),
    ),
    (
        "depth_specificity_practical",
        (
            "深度",
            "深入",
            "细节",
            "实用",
            "操作",
            "案例",
            "实证",
            "应用",
            "分析",
            "论述",
            "整合",
            "针对性",
            "具体",
            "depth",
            "deep",
            "detail",
            "practical",
            "utility",
            "case",
            "empirical",
            "application",
            "analytical",
            "analysis",
            "discussion",
            "integration",
            "specificity",
            "specific",
            "concrete",
            "implementation",
        ),
    ),
    (
        "accuracy_professionalism",
        (
            "准确",
            "专业",
            "严谨",
            "正确",
            "学术规范",
            "accuracy",
            "professional",
            "rigor",
            "correctness",
            "precise",
        ),
    ),
)


def _match_buckets(name: str) -> list[tuple[str, str]]:
    """Return all ``(bucket, matched_keyword)`` pairs that fire on ``name``."""

    normalized = (name or "").strip().lower()
    if not normalized:
        return []
    matches: list[tuple[str, str]] = []
    for bucket_name, keywords in _KEYWORD_BUCKETS:
        for kw in keywords:
            if kw in normalized:
                matches.append((bucket_name, kw))
                break
    return matches


def classify_criterion(name: str) -> str:
    """Map a free-form criterion name to a *primary* coarse bucket."""

    matches = _match_buckets(name)
    return matches[0][0] if matches else "other"


def classify_criterion_buckets(name: str) -> list[str]:
    """Return *all* matching buckets for ``name`` (multi-label)."""

    return [bucket for bucket, _ in _match_buckets(name)]


def matched_keywords_for(name: str) -> list[tuple[str, str]]:
    """Return the ``(bucket, matched_keyword)`` pairs for ``name``."""

    return _match_buckets(name)
