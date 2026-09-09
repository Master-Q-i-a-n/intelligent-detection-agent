from intelligent_detection_agent.smart_metering import SmartMeteringService


def test_primary_risk_score_is_shared_by_detail_and_overview():
    alerts = [
        "流量异常：流量计用气期间标况瞬时流量差距过大，过滤器可能阻塞",
        "压力异常（主备管道压力波动不相似，压力传感器可能故障）",
    ]

    primary_alerts, primary_score = SmartMeteringService.primary_risk_score(alerts)
    detail_score, detail_level = SmartMeteringService._risk(alerts, [], "无表具量程信息", 0.0, 0)

    assert primary_alerts == alerts
    assert primary_score == 95
    assert detail_score == primary_score
    assert detail_level == "严重"
