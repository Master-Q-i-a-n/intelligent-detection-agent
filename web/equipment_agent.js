window.renderEquipmentMechanism = function () {
  document.querySelector('#equipmentMechanism').innerHTML = `
    <div class="agent-explain-loading">
      <span class="agent-pulse"></span>
      <div><b>Agent正在解读设备状态</b><p>正在结合健康阶段、近7天趋势、三轴振动和模型置信度生成业务说明。</p></div>
    </div>`;
};

window.requestEquipmentExplanation = async function (equipment) {
  try {
    const report = await api('/agent/inspect', {
      method: 'POST',
      body: JSON.stringify({
        module: 'equipment',
        user_id: state.user,
        diagnosis_date: state.date,
        field_text: '',
        context: equipment,
      }),
    });
    const reasons = (report.evidence_chain || []).slice(0, 4);
    const actions = (report.checklist || []).slice(0, 3);
    document.querySelector('#equipmentMechanism').innerHTML = `
      <div class="agent-answer">
        <div class="agent-answer-head"><span>AI</span><div><b>Agent设备解读</b><small>${report.generator || 'inspection-agent'}</small></div></div>
        <div class="agent-conclusion">${report.inspection_conclusion}</div>
        <div class="agent-answer-grid">
          <div><h4>为什么这样判断</h4><ul>${reasons.map(x => `<li>${x}</li>`).join('')}</ul></div>
          <div><h4>建议先做什么</h4><ol>${actions.map(x => `<li>${x}</li>`).join('')}</ol></div>
        </div>
        <p class="agent-boundary">${report.data_boundary || ''}</p>
      </div>`;
  } catch (error) {
    document.querySelector('#equipmentMechanism').innerHTML = `<div class="mechanism-summary"><b>Agent解释暂时不可用：</b>${error.message}。下方健康趋势与三轴波形仍可正常查看。</div>`;
  }
};
