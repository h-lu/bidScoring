"""Archive utilities for E2E test outputs."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def create_timestamp_dir(base_dir: Path) -> Path:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    return output_dir


def archive_pdf_outputs(
    output_dir: Path,
    pdf_name: str,
    original_pdf: Path | None,
    mineru_output_dir: Path | None,
    annotated_pdf: Path | None,
) -> dict[str, str]:
    """Archive PDF-related outputs."""
    pdf_dir = output_dir / f"pdf_{pdf_name}"
    pdf_dir.mkdir(exist_ok=True)

    archived = {}

    if original_pdf and original_pdf.exists():
        dest = pdf_dir / "01_original.pdf"
        shutil.copy2(original_pdf, dest)
        archived["original"] = str(dest)
        logger.info(f"Archived original PDF: {dest}")

    if mineru_output_dir and mineru_output_dir.exists():
        dest = pdf_dir / "02_mineru_output"
        shutil.copytree(mineru_output_dir, dest, dirs_exist_ok=True)
        archived["mineru_output"] = str(dest)
        logger.info(f"Archived MinerU output: {dest}")

    if annotated_pdf and annotated_pdf.exists():
        dest = pdf_dir / "03_annotated.pdf"
        shutil.copy2(annotated_pdf, dest)
        archived["annotated"] = str(dest)
        logger.info(f"Archived annotated PDF: {dest}")

    return archived


def save_report(output_dir: Path, pdf_name: str, report: dict[str, Any]) -> Path:
    """Save analysis report as Markdown."""
    pdf_dir = output_dir / f"pdf_{pdf_name}"
    pdf_dir.mkdir(exist_ok=True)

    report_path = pdf_dir / "04_report.md"

    lines = [
        "# 投标分析报告",
        "",
        f"**投标人**: {report.get('bidder_name', 'Unknown')}",
        f"**项目**: {report.get('project_name', 'Unknown')}",
        f"**综合评分**: {report.get('overall_score', 0):.1f}/100",
        f"**版本ID**: {report.get('version_id', 'N/A')}",
        "",
        "## 分析维度",
        "",
    ]

    for aspect, analysis in report.get("aspects", {}).items():
        lines.extend([
            f"### {aspect.upper()}",
            f"{analysis.get('summary', '无')}",
            f"- 内容点: {analysis.get('chunks_found', 0)}",
            f"- 风险: {len(analysis.get('risks', []))}",
            f"- 优势: {len(analysis.get('benefits', []))}",
            "",
        ])

    lines.extend([
        f"## 统计",
        f"- 总风险: {report.get('total_risks', 0)}",
        f"- 总优势: {report.get('total_benefits', 0)}",
        "",
    ])

    if report.get("recommendations"):
        lines.extend(["## 建议", ""])
        for rec in report["recommendations"]:
            lines.append(f"- {rec}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved report: {report_path}")
    return report_path


def save_db_export(
    output_dir: Path,
    pdf_name: str,
    chunks: list[dict],
    content_units: list[dict],
    analysis_summary: dict[str, Any],
) -> Path:
    """Save database export."""
    pdf_dir = output_dir / f"pdf_{pdf_name}"
    export_dir = pdf_dir / "05_db_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = export_dir / "chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2, default=str)

    units_path = export_dir / "content_units.json"
    with open(units_path, "w", encoding="utf-8") as f:
        json.dump(content_units, f, ensure_ascii=False, indent=2, default=str)

    summary_path = export_dir / "analysis_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(analysis_summary, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"Saved DB export: {export_dir}")
    return export_dir


def save_manifest(
    output_dir: Path,
    execution_time: datetime,
    pdf_count: int,
    projects: list[dict[str, Any]],
) -> Path:
    """Save execution manifest."""
    manifest = {
        "execution_time": execution_time.isoformat(),
        "pdf_count": pdf_count,
        "successful": sum(1 for p in projects if p.get("status") == "success"),
        "failed": sum(1 for p in projects if p.get("status") != "success"),
        "projects": projects,
        "output_dir": str(output_dir),
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved manifest: {manifest_path}")
    return manifest_path


def save_execution_log(output_dir: Path, log_content: str) -> Path:
    """Save execution log."""
    log_path = output_dir / "execution.log"
    log_path.write_text(log_content, encoding="utf-8")
    return log_path
