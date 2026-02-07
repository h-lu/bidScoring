from __future__ import annotations

from typing import Any


def emit_core_pages(builder: Any, page_idx: int) -> None:
    """Emit the anchored evidence chunks used by qrels.

    The `builder` is expected to expose:
    - builder.profile: ScenarioProfile-like object
    - builder.add(...): append a MineRU-like element
    """
    p = builder.profile

    if page_idx == 0:
        builder.add("text", 0, text="上海市第六人民医院", text_level=1)
        builder.add("text", 0, text="超分辨共聚焦显微镜采购项目", text_level=1)
        builder.add("text", 0, text="招标编号：0811-DSITC260201")
        builder.add("text", 0, text=f"投标人：{p.bidder_name}")
        builder.add(
            "image",
            0,
            img_path="images/cover_seal.png",
            image_caption=["法定代表人签章与公司公章"],
            image_footnote=["扫描件仅用于投标用途"],
        )
        return

    if page_idx == 1:
        builder.add("text", 1, text="目录", text_level=1)
        builder.add(
            "list",
            1,
            list_items=[
                "一、投标函...............................................2",
                "二、开标一览表...........................................3",
                "三、资格审查与合规文件...................................5",
                "四、技术规格响应偏离表...................................9",
                "五、与评分有关的主要内容索引表...........................15",
                "六、售后服务方案........................................16",
                "七、培训方案............................................20",
                "八、商务条款与合同响应..................................23",
                "九、类似项目业绩........................................24",
            ],
        )
        return

    if page_idx == 3:
        builder.add("text", 3, text="三、资格审查与合规文件", text_level=1)
        builder.add(
            "table",
            3,
            table_caption=["企业资质信息"],
            table_body=(
                "统一社会信用代码 91310109MA1G7XKX3A；经营范围 包含医疗器械销售、维修、技术服务；"
                "质量体系 ISO 13485:2016；医疗器械经营备案编号 沪浦药监械经营备20261234号。"
            ),
            table_footnote=["证照复印件见附件 A-1 至 A-4"],
        )
        return

    if page_idx == 4:
        builder.add(
            "table",
            4,
            anchor="registration_table",
            table_caption=["投标设备注册与管理属性"],
            table_body=(
                "设备名称 激光共聚焦显微镜；是否按医疗器械管理 否；"
                f"医疗器械注册证号/备案号 /；原产地 德国；制造商 {p.manufacturer_name}。"
            ),
            table_footnote=["按本项目技术属性，设备不纳入医疗器械注册目录"],
        )
        builder.add(
            "aside_text",
            4,
            anchor="aside_registration_consistency",
            text="提示：评委关注“是否按医疗器械管理”与“注册证号”字段一致性。",
        )
        return

    if page_idx == 5:
        builder.add("text", 5, text="五、与评分有关的主要内容索引表", text_level=1)
        builder.add(
            "table",
            5,
            anchor="scoring_index_table",
            table_caption=["评分内容索引"],
            table_body=(
                "评分项 技术参数响应 -> 第9-14页；培训方案 -> 第20-22页；"
                "售后服务方案 -> 第16-19页；类似项目业绩 -> 第24-25页；"
                "商务条款响应 -> 第23页。"
            ),
            table_footnote=["索引用于快速定位证据，不替代正文承诺"],
        )
        return

    if page_idx == 9:
        builder.add("text", 9, text="四、技术规格响应偏离表", text_level=1)
        builder.add(
            "table",
            9,
            anchor="delivery_3months",
            table_caption=["关键商务与交付条款响应"],
            table_body=(
                "条目 交货期；招标要求 合同签订后3个月内到货；投标响应 合同签订后3个月内完成交付与安装调试；"
                "响应/偏离 响应。"
            ),
            table_footnote=["交付地点：上海市第六人民医院指定科室"],
        )
        return

    if page_idx == 10:
        builder.add(
            "table",
            10,
            table_caption=["技术参数关键条款"],
            table_body=(
                "★3.1 自动对焦精度 ≤0.05um；★3.2 谱线分离能力支持6通道；"
                "★3.3 扫描速度 ≥30fps@512x512；投标响应 全部满足。"
            ),
            table_footnote=["关键参数偏离将触发扣分"],
        )
        return

    if page_idx == 16:
        builder.add("text", 16, text="六、售后服务方案", text_level=1)
        builder.add(
            "list",
            16,
            anchor="service_warranty_5y",
            list_items=[
                p.warranty_clause,
                "质保期内免费上门维护、人工费全免、常见备件免费更换。",
                "若因同一故障重复发生三次，供应商须提供专项整改报告。",
            ],
        )
        return

    if page_idx == 17:
        builder.add(
            "list",
            17,
            anchor="service_response_2h_24h",
            list_items=[
                "故障受理：7x24小时热线和工单系统。",
                p.service_response_clause,
                p.backup_clause,
            ],
        )
        builder.add(
            "text",
            17,
            anchor="service_uptime_95",
            text="质量保证期内承诺开机率不低于95%，未达标天数按比例顺延质保期。",
        )
        return

    if page_idx == 18:
        builder.add(
            "list",
            18,
            anchor="service_spare_parts",
            list_items=[
                "在上海设立备件仓库，常用易损件常备库存。",
                "关键备件（激光器模块、扫描振镜）提供安全库存预留。",
                "质保期外备件按不高于市场公开价供货。",
            ],
        )
        builder.add(
            "text",
            18,
            text=f"售后服务热线：{p.hotline}，服务邮箱：{p.service_email}。",
        )
        builder.add(
            "text",
            18,
            anchor="service_no_third_party_upgrade",
            text="本项目免费服务不包含第三方软件插件授权升级费用，但包含原厂基础版本安全补丁。",
        )
        builder.add("text", 18, anchor="service_backup_machine", text=p.backup_clause)
        return

    if page_idx == 19:
        builder.add(
            "text",
            19,
            text="制造商补充售后承诺",
            text_level=1,
        )
        builder.add(
            "text",
            19,
            anchor="manufacturer_response_2workdays",
            text=p.manufacturer_response_clause,
        )
        builder.add(
            "text",
            19,
            anchor="warranty_60_month",
            text="制造商承诺保修期为安装之日起60个月，最晚不超过发货后64个月。",
        )
        builder.add(
            "text",
            19,
            anchor="remote_diagnosis_clause",
            text="支持远程诊断与日志分析，紧急场景可先远程处置后现场闭环。",
        )
        builder.add(
            "text",
            19,
            anchor="parts_price_clause",
            text="质保期后零配件报价以公开目录价为上限，不高于市场同期均价。",
        )
        return

    if page_idx == 20:
        builder.add("text", 20, text="七、培训方案", text_level=1)
        builder.add(
            "text",
            20,
            anchor="training_overview",
            text=(
                "培训目标：使使用、管理、维护人员可独立完成日常操作、参数调整、"
                "常见故障排查与应急处理。"
            ),
        )
        return

    if page_idx == 21:
        builder.add(
            "table",
            21,
            anchor="training_schedule_table",
            table_caption=["培训计划表"],
            table_body=(
                "第1天 安装条件与安全规范、基础操作；"
                "第2天上午 参数设置与质量控制；第2天下午 故障排查与考核。"
            ),
            table_footnote=["培训结束组织实操考核并出具签到记录"],
        )
        builder.add(
            "list",
            21,
            anchor="training_target",
            list_items=[
                "培训对象：设备使用人员、科室管理员、医学工程维护人员。",
                "培训形式：现场授课 + 实机演示 + 问题答疑。",
            ],
        )
        return

    if page_idx == 23:
        builder.add("text", 23, text="八、商务条款与合同响应", text_level=1)
        builder.add(
            "text",
            23,
            anchor="acceptance_clause",
            text=(
                "货物到场后完成安装调试并经双方签署《验收合格单》视为交付完成，"
                "验收依据包括招标技术条款与厂家标准。"
            ),
        )
        builder.add(
            "table",
            23,
            anchor="payment_terms_table",
            table_caption=["付款节点"],
            table_body="预付款 30%；到货验收合格后支付60%；质保期满且无争议支付尾款10%。",
            table_footnote=["付款周期以内控审批节点为准"],
        )
        builder.add(
            "text",
            23,
            anchor="contract_delivery_30days",
            text="（合同模板条款）乙方应在合同生效后30天内完成交付。",
        )
        builder.add(
            "text",
            23,
            anchor="contract_warranty_36_month",
            text="（合同模板条款）通用设备质量保证期为36个月。",
        )
        builder.add(
            "text",
            23,
            anchor="anti_corruption_clause",
            text="双方必须遵守医疗购销廉洁要求，严禁回扣、商业贿赂及其他不正当利益输送。",
        )
        return

    if page_idx == 24:
        builder.add("text", 24, text="九、类似项目业绩", text_level=1)
        builder.add(
            "table",
            24,
            anchor="project_experience_table",
            table_caption=["近三年同类项目"],
            table_body=(
                "上海市第一人民医院 共聚焦显微镜升级项目 2024；"
                "复旦大学附属中山医院 激光扫描成像平台 2023；"
                "浙江大学医学院附属邵逸夫医院 显微成像系统维保 2022。"
            ),
            table_footnote=["均附合同首页与验收页扫描件"],
        )
        builder.add(
            "text",
            24,
            anchor="engineer_count_text",
            text="售后工程师团队共6人，其中高级工程师2人，驻华东区域现场工程师3人。",
        )
        return

    if page_idx == 25:
        builder.add("text", 25, text="附录：宣传资料（非评分项）", text_level=1)
        builder.add(
            "text",
            25,
            anchor="bilingual_sla_clause",
            text=(
                "Service SLA: first response within 2 hours, on-site support within 24 hours, "
                "excluding force majeure events."
            ),
        )
        builder.add(
            "text",
            25,
            anchor="ocr_response_clause",
            text="扫 描 件 抽 取：响 应 时 间 2 小 时 ，到 场 时 间 2 4 小 时。",
        )
        builder.add(
            "image",
            25,
            img_path="images/brochure_ai.png",
            image_caption=[f"{p.brand_name} AI-assisted microscope workflow"],
            image_footnote=["marketing material only"],
        )
        builder.add(
            "aside_text",
            25,
            text="注：宣传册中的性能描述不作为商务承诺。",
        )
        return

