from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SCENARIOS = ("A", "B", "C")


@dataclass(frozen=True)
class ScenarioProfile:
    scenario: str
    version_tag: str
    bidder_name: str
    brand_name: str
    manufacturer_name: str
    hotline: str
    service_email: str
    service_response_clause: str
    manufacturer_response_clause: str
    warranty_clause: str
    backup_clause: str


SCENARIO_PROFILES: dict[str, ScenarioProfile] = {
    "A": ScenarioProfile(
        scenario="A",
        version_tag="synthetic_bidder_A_v2",
        bidder_name="上海澄研医疗科技有限公司",
        brand_name="Lumina",
        manufacturer_name="Lumina Imaging GmbH",
        hotline="400-820-8932",
        service_email="service@lumina-med.cn",
        service_response_clause="响应时效：2小时内电话/远程响应，24小时内工程师到场。",
        manufacturer_response_clause="制造商接报后2个工作日内安排工程师上门服务，节假日顺延。",
        warranty_clause="质保期：原厂免费质保5年（60个月），自验收合格签字之日起计算。",
        backup_clause="重大故障72小时未恢复时，提供同等级备用机。",
    ),
    "B": ScenarioProfile(
        scenario="B",
        version_tag="synthetic_bidder_B_v2",
        bidder_name="苏州启衡生物仪器有限公司",
        brand_name="OptiCell",
        manufacturer_name="OptiCell Instruments AG",
        hotline="400-680-2107",
        service_email="support@opticell.cn",
        service_response_clause="服务承诺：2小时内响应，24小时内到场，必要时先远程排障。",
        manufacturer_response_clause="原厂服务中心承诺2个工作日内安排上门工程师，法定假日顺延。",
        warranty_clause="免费质量保证期5年（60个月），起算点为采购人验收合格日期。",
        backup_clause="若72小时仍未修复，供应商提供同等级替代设备保障临床使用。",
    ),
    "C": ScenarioProfile(
        scenario="C",
        version_tag="synthetic_bidder_C_v2",
        bidder_name="杭州赛泓精密医疗设备有限公司",
        brand_name="NeuroScan",
        manufacturer_name="NeuroScan Microscopy Ltd.",
        hotline="400-990-3321",
        service_email="care@neuroscan-med.com",
        service_response_clause="SLA：first response within 2 hours, on-site within 24 hours.",
        manufacturer_response_clause="Manufacturer clause: engineer visit within 2 working days after ticket accepted.",
        warranty_clause="整机质保5年（=60个月），以双方《验收合格单》日期为准。",
        backup_clause="设备72小时未恢复时，免费提供同等级备用机至故障闭环。",
    ),
}
