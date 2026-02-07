from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .core_pages import emit_core_pages
from .profiles import ScenarioProfile


@dataclass(frozen=True)
class BuildConfig:
    page_count: int = 120
    filler_items_per_page: int = 8
    header_text_template: str = "上海市第六人民医院 共聚焦显微镜采购项目 投标文件"

    # Guardrails: keep synthetic docs close to real tender size.
    min_total_items: int = 1200
    min_ingestable_items: int = 950


class ContentBuilder:
    """Build MineRU-like content list and anchor map.

    Design points:
    - Interleave page noise (header/page_number) with page content.
    - Anchors map to indices in the full content_list (including noise items).
    """

    def __init__(
        self, profile: ScenarioProfile, *, config: BuildConfig | None = None
    ) -> None:
        self.profile = profile
        self.config = config or BuildConfig()
        self.content: list[dict[str, Any]] = []
        self.anchors: dict[str, int] = {}

        self._current_page: int | None = None
        self._page_row: int = 0

    def _begin_page(self, page_idx: int) -> None:
        self._current_page = page_idx
        self._page_row = 0

    def _bbox(self) -> list[int]:
        # Stable but page-relative rectangles; good enough for ingestion/UI.
        row = self._page_row
        top = 80 + (row % 16) * 38
        self._page_row += 1
        return [76, top, 1116, top + 30]

    def add(
        self,
        item_type: str,
        page_idx: int,
        *,
        anchor: str | None = None,
        text: str | None = None,
        text_level: int | None = None,
        list_items: list[str] | None = None,
        table_body: str | None = None,
        table_caption: list[str] | None = None,
        table_footnote: list[str] | None = None,
        image_caption: list[str] | None = None,
        image_footnote: list[str] | None = None,
        img_path: str | None = None,
        sub_type: str | None = None,
    ) -> None:
        if self._current_page != page_idx:
            self._begin_page(page_idx)

        idx = len(self.content)
        item: dict[str, Any] = {
            "type": item_type,
            "page_idx": page_idx,
            "bbox": self._bbox(),
        }

        if text is not None:
            item["text"] = text
        if text_level is not None:
            item["text_level"] = text_level
        if list_items is not None:
            item["list_items"] = list_items
        if table_body is not None:
            item["table_body"] = table_body
        if table_caption is not None:
            item["table_caption"] = table_caption
        if table_footnote is not None:
            item["table_footnote"] = table_footnote
        if image_caption is not None:
            item["image_caption"] = image_caption
        if image_footnote is not None:
            item["image_footnote"] = image_footnote
        if img_path is not None:
            item["img_path"] = img_path
        if sub_type is not None:
            item["sub_type"] = sub_type

        self.content.append(item)
        if anchor:
            if anchor in self.anchors:
                raise ValueError(f"duplicate anchor: {anchor}")
            self.anchors[anchor] = idx

    def add_page_noise(self, page_idx: int) -> None:
        self.add("header", page_idx, text=self.config.header_text_template)
        self.add("page_number", page_idx, text=str(page_idx + 1))

    def _filler_text(self, page_idx: int, i: int) -> str:
        p = self.profile
        templates = [
            (
                f"【附件】产品介绍：{p.brand_name} 系统组成包括主机、控制单元与软件套件，"
                "支持多用户权限管理。"
            ),
            "安装环境要求：恒温(20-26℃)、相对湿度(30%-70%)，配备独立接地与稳压电源。",
            "质量控制：提供出厂校准报告、到货开箱检查记录与安装调试验收记录模板。",
            "软件功能说明：提供数据导出、审计追踪、用户日志与权限分级配置。",
            "风险管理摘要：关键部件寿命管理、预防性维护建议、耗材更换提醒与告警策略。",
            "培训记录样例：签到表、考核表、问题清单与整改闭环记录（仅示例）。",
            "配置清单说明：包含必配模块与选配模块，选配项不影响招标核心性能指标。",
            "运输与包装：防震木箱包装，运输过程温湿度监测，提供到货照片记录。",
            "信息安全：离线部署可选，日志脱敏，支持本地备份与权限审计。",
            "维保流程：故障受理-远程诊断-现场修复-验收签字-归档回访。",
        ]
        return templates[(page_idx + i) % len(templates)]

    def _emit_fillers(self, page_idx: int) -> None:
        cfg = self.config
        for i in range(cfg.filler_items_per_page):
            if i == 0 and page_idx % 7 == 0:
                self.add(
                    "text",
                    page_idx,
                    text=f"附件页 {page_idx + 1}：补充说明",
                    text_level=2,
                )
                continue

            # Image pages are common in scanned tenders (stamps, certificates, etc.).
            # Keep them frequent enough to resemble real distributions.
            if i == 2 and page_idx % 3 == 0:
                self.add(
                    "image",
                    page_idx,
                    img_path=f"images/appendix_{page_idx + 1:03d}_a.png",
                    image_caption=[f"附件图 {page_idx + 1}-A：证照/盖章扫描页"],
                    image_footnote=["仅用于模拟评测数据集"],
                )
                continue

            if i == 6 and page_idx % 3 == 0:
                self.add(
                    "table",
                    page_idx,
                    table_caption=[f"附件表 {page_idx + 1}-A：配置/证照摘要"],
                    table_body=(
                        "条目 说明；出厂检验 提供；随机资料 说明书/合格证；"
                        "安装资料 安装调试记录；培训资料 签到与考核记录。"
                    ),
                    table_footnote=["本表为格式示例，最终以投标文件为准"],
                )
                continue

            if i == 7 and page_idx % 2 == 0:
                self.add(
                    "image",
                    page_idx,
                    img_path=f"images/appendix_{page_idx + 1:03d}_b.png",
                    image_caption=[f"附件图 {page_idx + 1}-1：扫描件/示意图"],
                    image_footnote=["仅用于模拟评测数据集"],
                )
                continue

            if i == 5 and page_idx % 5 == 0:
                self.add(
                    "list",
                    page_idx,
                    list_items=[
                        "交付资料：装箱单、随货资料清单、到货签收单",
                        "验收资料：安装调试记录、验收合格单、培训签到表",
                        "维护资料：保养计划、点检表、维修记录表（模板）",
                    ],
                )
                continue

            if i == 4 and page_idx % 10 == 0:
                self.add(
                    "aside_text",
                    page_idx,
                    text="提示：模板条款与投标响应条款需区分，评委以明确承诺为准。",
                )
                continue

            self.add("text", page_idx, text=self._filler_text(page_idx, i))

    @staticmethod
    def _ingestable_count(items: list[dict[str, Any]]) -> int:
        return sum(
            1
            for x in items
            if x.get("type") not in {"header", "page_number", "footer"}
        )

    def build(self) -> tuple[list[dict[str, Any]], dict[str, int]]:
        cfg = self.config

        for page_idx in range(cfg.page_count):
            self._begin_page(page_idx)
            self.add_page_noise(page_idx)
            emit_core_pages(self, page_idx)
            self._emit_fillers(page_idx)

        ingestable = self._ingestable_count(self.content)
        if len(self.content) < cfg.min_total_items:
            raise ValueError(
                f"content_list too small: {len(self.content)} < {cfg.min_total_items}"
            )
        if ingestable < cfg.min_ingestable_items:
            raise ValueError(
                f"ingestable too small: {ingestable} < {cfg.min_ingestable_items}"
            )

        required_anchors = {
            "registration_table",
            "aside_registration_consistency",
            "scoring_index_table",
            "delivery_3months",
            "service_warranty_5y",
            "service_response_2h_24h",
            "service_uptime_95",
            "service_spare_parts",
            "service_no_third_party_upgrade",
            "service_backup_machine",
            "manufacturer_response_2workdays",
            "warranty_60_month",
            "remote_diagnosis_clause",
            "parts_price_clause",
            "training_overview",
            "training_schedule_table",
            "training_target",
            "acceptance_clause",
            "payment_terms_table",
            "contract_delivery_30days",
            "contract_warranty_36_month",
            "anti_corruption_clause",
            "project_experience_table",
            "engineer_count_text",
            "bilingual_sla_clause",
            "ocr_response_clause",
        }
        missing = sorted(required_anchors - set(self.anchors))
        if missing:
            raise ValueError(f"missing required anchors: {missing}")

        return self.content, self.anchors
