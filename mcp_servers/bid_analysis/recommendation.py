from __future__ import annotations

from mcp_servers.bid_analysis.models import DimensionResult


class RecommendationGenerator:
    """Generate actionable recommendations from dimension results."""

    @staticmethod
    def generate(dimension_results: dict[str, DimensionResult]) -> list[str]:
        recommendations: list[str] = []

        for dim_name, result in dimension_results.items():
            if result.risk_level == "high":
                if dim_name == "warranty":
                    recommendations.append(
                        "[质保] 质保期偏短，建议确认是否可延长或增加服务承诺"
                    )
                elif dim_name == "delivery":
                    recommendations.append(
                        "[交付] 响应时间较慢，建议确认紧急情况处理流程"
                    )
                elif dim_name == "training":
                    recommendations.append("[培训] 培训时间不足，建议增加实操培训天数")
                elif dim_name == "financial":
                    recommendations.append("[商务] 付款条件较为严格，建议评估资金压力")

            if result.chunks_found == 0:
                if dim_name == "warranty":
                    recommendations.append("[质保] 文档未找到质保条款，需补充")
                elif dim_name == "delivery":
                    recommendations.append("[交付] 文档未明确交付时间，需确认")

        if not recommendations:
            recommendations.append("投标文档整体较为完整，建议按常规流程评审")

        return recommendations
