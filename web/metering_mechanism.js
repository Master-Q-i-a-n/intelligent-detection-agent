window.renderMeteringMechanism = function () {
  document.querySelector('#meteringMechanism').innerHTML = `
    <div class="agent-explain-loading">
      <span class="agent-pulse"></span>
      <div><b>Agent正在解读诊断结果</b><p>正在把流量、压力、温度曲线及算法证据转换为业务说明，页面其他内容可以正常查看。</p></div>
    </div>`;
};

window.requestMeteringExplanation = async function (diagnosis) {
  try {
    const report = await api('/agent/inspect', {
      method: 'POST',
      body: JSON.stringify({
        module: 'metering',
        user_id: state.user,
        diagnosis_date: state.date,
        field_text: '',
        context: diagnosis,
      }),
    });
    const reasons = (report.evidence_chain || []).slice(0, 4);
    const actions = (report.checklist || []).slice(0, 3);
    document.querySelector('#meteringMechanism').innerHTML = `
      <div class="agent-answer">
        <div class="agent-answer-head"><span>AI</span><div><b>Agent业务解读</b><small>${report.generator || 'inspection-agent'}</small></div></div>
        <div class="agent-conclusion">${report.inspection_conclusion}</div>
        <div class="agent-answer-grid">
          <div><h4>为什么这样判断</h4><ul>${reasons.map(x => `<li>${x}</li>`).join('')}</ul></div>
          <div><h4>建议先做什么</h4><ol>${actions.map(x => `<li>${x}</li>`).join('')}</ol></div>
        </div>
        <p class="agent-boundary">${report.data_boundary || ''}</p>
      </div>`;
  } catch (error) {
    document.querySelector('#meteringMechanism').innerHTML = `<div class="mechanism-summary"><b>Agent解释暂时不可用：</b>${error.message}。下方原始曲线和诊断结果仍可正常查看。</div>`;
  }
};

window.renderMeteringSignals = function (signals, diagnosis) {
  const palette = ['#13d7d1', '#f1aa34', '#a078ff', '#ff5d58'];
  const entries = Object.entries(signals?.pipelines || {});
  const makeSeries = (key, allowed = null) => entries.filter(([pipeline]) => !allowed || allowed.has(String(pipeline))).map(([pipeline, values], index) => ({
    name: `管道${pipeline}`,
    values: values[key].map(value => value == null ? 0 : Number(value)),
    color: palette[index % palette.length],
    width: 1.5,
  }));
  lineChart(document.querySelector('#flowSignalChart'), makeSeries('flow'), {labels: signals.times, min: 0, decimals: 1});
  const pressureEvidence = (diagnosis?.details?.diagnostic_evidence?.pressure || []).filter(item => item.dissimilar);
  const flowPipelines = diagnosis?.details?.diagnostic_evidence?.flow?.[0]?.pipelines || [];
  let selectedPair = pressureEvidence.find(item => flowPipelines.length === 2 && flowPipelines.every(value => item.pipelines.includes(value)));
  if (!selectedPair && pressureEvidence.length) selectedPair = [...pressureEvidence].sort((a,b) => b.dtw_distance-a.dtw_distance)[0];
  const pressurePipelines = selectedPair ? new Set(selectedPair.pipelines.map(String)) : null;
  lineChart(document.querySelector('#pressureSignalChart'), makeSeries('pressure', pressurePipelines), {labels: signals.times, decimals: 2});
  document.querySelector('#pressureCurveNote').textContent = selectedPair
    ? `重点对比管道${selectedPair.pipelines.join('与')}：两条压力曲线的变化趋势和高频波动均不一致，是本次压力告警的主要图形证据。`
    : '当天未发现达到压力告警条件的管路对，当前显示全部可用压力曲线供参考。';
  lineChart(document.querySelector('#temperatureSignalChart'), makeSeries('temperature'), {labels: signals.times, decimals: 1});
};

const originalRenderMetering = window.renderMetering;
window.renderMetering = function (diagnosis, history) {
  originalRenderMetering(diagnosis, history);
  lineChart(document.querySelector('#volumeChart'), [
    {name: '日累计用气量', values: history.map(x => x.volume), color: '#13d7d1', width: 2.5},
  ], {labels: history.map(x => x.date), min: 0});
  lineChart(document.querySelector('#maxFlowChart'), [
    {name: '最大瞬时流量', values: history.map(x => x.max_flow), color: '#f1aa34', width: 2.5},
  ], {labels: history.map(x => x.date), min: 0, decimals: 1});
  const meterSpec = diagnosis.details?.meter_spec || {};
  if (!meterSpec.qmax) {
    document.querySelector('#meterRangeBars').innerHTML = `<div class="range-missing"><b>暂时无法判断表具量程是否适配</b>该用户的表具基础信息中缺少最大量程 <code>quantity_max</code>，因此不能计算小流量、正常量程和超量程占比。这里原先显示的0不是实际读数，现已取消。</div>`;
  }
};
