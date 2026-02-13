from __future__ import annotations

from mcp_servers.annotation_insights import AnnotationInsight
from mcp_servers.bid_analysis.models import ANALYSIS_DIMENSIONS, AnalysisDimension, DimensionResult


class DimensionScorer:
    """Pure scoring logic for dimensions and overall score."""

    def calculate_dimension_score(
        self,
        dimension: AnalysisDimension,
        risks: list[AnnotationInsight],
        benefits: list[AnnotationInsight],
        extracted_values: dict,
    ) -> float:
        score = 50.0

        for _insight in benefits:
            score += 10

        for insight in risks:
            if insight.risk_level == "high":
                score -= 15
            elif insight.risk_level == "medium":
                score -= 5
            else:
                score -= 2

        if dimension.risk_thresholds and extracted_values:
            thresholds = dimension.risk_thresholds

            if dimension.name == "warranty":
                max_years = max((v["max"] for v in extracted_values.values()), default=0)
                if max_years >= thresholds["excellent"][0]:
                    score += 20
                elif max_years >= thresholds["good"][0]:
                    score += 10
                elif max_years < thresholds["poor"][1]:
                    score -= 20

            elif dimension.name == "delivery":
                min_hours = min((v["min"] for v in extracted_values.values()), default=999)
                if min_hours <= thresholds["excellent"][1]:
                    score += 20
                elif min_hours <= thresholds["good"][1]:
                    score += 10
                elif min_hours > thresholds["poor"][0]:
                    score -= 20

            elif dimension.name == "training":
                days_values = [v["max"] for v in extracted_values.values()]
                total_days = sum(days_values) if days_values else 0
                if total_days >= thresholds["excellent"][0]:
                    score += 20
                elif total_days >= thresholds["good"][0]:
                    score += 10

        return max(0, min(100, score))

    @staticmethod
    def determine_dimension_risk_level(
        _dimension: AnalysisDimension,
        risks: list[AnnotationInsight],
        benefits: list[AnnotationInsight],
        _extracted_values: dict,
    ) -> str:
        high_risk_count = sum(1 for r in risks if r.risk_level == "high")

        if high_risk_count > 0:
            return "high"
        if len(risks) > len(benefits):
            return "medium"
        return "low"

    @staticmethod
    def calculate_overall_score(
        dimension_results: dict[str, DimensionResult],
    ) -> float:
        total_weight = 0.0
        weighted_sum = 0.0

        for dim_name, result in dimension_results.items():
            dim_config = ANALYSIS_DIMENSIONS.get(dim_name)
            if dim_config:
                weight = dim_config.weight
                weighted_sum += result.score * weight
                total_weight += weight

        if total_weight == 0:
            return 50.0

        return weighted_sum / total_weight

    @staticmethod
    def determine_risk_level(total_risks: int, total_benefits: int) -> str:
        ratio = total_benefits / max(1, total_risks)

        if ratio >= 2:
            return "low"
        if ratio >= 1:
            return "medium"
        return "high"
