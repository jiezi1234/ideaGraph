/* 过程图只表达调用/包含关系，绝不根据时间顺序推断并行任务的依赖。 */
function processGraphData(items, label, taskStatus) {
  const records = items.slice(-500);
  const byId = new Map(records.map(item => [item.id, item]));
  const nodes = [{ id: '__task__', label: '任务过程', shape: 'ellipse', color: '#e3ebe6', level: 0 }];
  const edges = [];
  const missing = new Set();
  const parents = new Map();
  for (const item of records) {
    let parent = item.parent_id || item.parentObservationId;
    const visited = new Set([item.id]);
    let cursor = parent;
    while (cursor && byId.has(cursor)) {
      if (visited.has(cursor)) { parent = null; break; }
      visited.add(cursor);
      const ancestor = byId.get(cursor);
      cursor = ancestor.parent_id || ancestor.parentObservationId;
    }
    parents.set(item.id, parent);
  }
  function level(id) {
    let depth = 1;
    let parent = parents.get(id);
    while (parent) { depth++; parent = parents.get(parent); }
    return depth;
  }
  for (const item of records) {
    const parent = parents.get(item.id);
    if (parent && !byId.has(parent) && !missing.has(parent)) {
      missing.add(parent);
      nodes.push({ id: 'node:' + parent, label: '未载入的父阶段', color: '#eee', level: 1 });
      edges.push({ id: 'missing:' + parent, from: '__task__', to: 'node:' + parent, dashes: true });
    }
    let status = item.status || (item.level === 'ERROR' ? 'failed' : item.endTime ? 'completed' : 'unknown');
    if (status === 'running' && !['running', 'queued'].includes(taskStatus)) status = 'interrupted';
    const colors = { running: '#d9ecfa', completed: '#e0f0e5', failed: '#fbe2dc', interrupted: '#f7ead6', unknown: '#edf0f3' };
    const labels = { running: '进行中', completed: '已完成', failed: '失败', interrupted: '已中断', unknown: '未收到结束记录' };
    nodes.push({ id: 'node:' + item.id, label: label(item.name) + '\n' + labels[status],
      level: level(item.id), color: colors[status], borderWidth: status === 'running' ? 3 : 1 });
    edges.push({ id: 'edge:' + item.id, from: parent ? 'node:' + parent : '__task__', to: 'node:' + item.id });
  }
  return { nodes, edges, records, truncated: items.length > records.length };
}

class ProcessGraph {
  constructor(panel, label, fetchDetail) {
    this.panel = panel;
    this.label = label;
    this.fetchDetail = fetchDetail;
    this.local = [];
    this.remote = [];
    this.source = 'local';
    this.request = 0;
    panel.innerHTML = '<h3>任务过程 <span class="live-dot"></span></h3><p class="progress-summary" role="status"></p>' +
      '<div class="process-toolbar"><label>数据来源<select class="process-source"><option value="local">实时进度</option><option value="remote">Langfuse 调用</option></select></label>' +
      '<button type="button" class="quiet process-fit">适应画布</button><button type="button" class="quiet trace-refresh">刷新调用</button></div>' +
      '<p class="trace-state" role="status">等待同步 Langfuse…</p><p class="hint">连线表示包含 / 调用关系；蓝色进行中，绿色完成，红色失败。拖动或滚轮缩放，点击节点查看详情。</p>' +
      '<div class="process-canvas" role="img" aria-label="任务过程调用图"></div><p class="process-count hint"></p>' +
      '<label class="process-picker">选择节点查看详情<select class="process-node"><option value="">请选择节点</option></select></label>' +
      '<section class="process-inspector" hidden><div class="process-detail-heading"><h4></h4><button type="button" class="quiet io-refresh">刷新输入输出</button></div>' +
      '<p class="io-meta hint"></p><p class="io-status" role="status"></p><div class="io-grid"><section><h5>Input · 输入</h5><pre class="io-input" tabindex="0"></pre></section>' +
      '<section><h5>Output · 输出</h5><pre class="io-output" tabindex="0"></pre></section></div></section>';
    panel.querySelector('.process-source').onchange = event => {
      this.source = event.target.value;
      this.clearSelection();
      this.draw(true);
    };
    panel.querySelector('.process-node').onchange = event => this.select(event.target.value);
    panel.querySelector('.io-refresh').onclick = () => this.select(this.selected);
    panel.querySelector('.process-fit').onclick = () => this.network?.fit({ animation: false });
  }

  reset(taskId) {
    if (this.taskId === taskId) return;
    this.taskId = taskId;
    this.needsFit = true;
    this.local = []; this.remote = []; this.signature = null;
    this.localTruncated = false; this.remoteTruncated = false;
    this.panel.querySelector('.trace-state').textContent = '等待同步 Langfuse…';
    this.clearSelection();
    this.draw(true);
  }

  clearSelection() {
    this.request++;
    this.selected = null;
    this.selectedVersion = null;
    this.panel.querySelector('.process-inspector').hidden = true;
  }

  update(row) {
    this.reset(row.id);
    this.taskStatus = row.status;
    this.local = row.events || [];
    this.localTruncated = row.events_truncated;
    const current = this.local.filter(item => item.status === 'running').at(-1);
    this.panel.classList.toggle('is-running', row.status === 'running');
    this.panel.querySelector('.progress-summary').textContent = row.status === 'queued' ? '任务已排队，等待执行。' :
      row.status === 'running' ? (current ? this.label(current.name) : '正在处理，等待下一条进度…') : ({ completed: '任务已完成', failed: '任务未完成' }[row.status] || row.status);
    if (this.source === 'local') this.draw();
  }

  updateTrace(trace) {
    this.remote = trace.observations || [];
    this.remoteTruncated = trace.truncated;
    this.panel.querySelector('.trace-state').textContent = ({ disabled: '此任务未关联 Langfuse，可查看实时进度图。',
      pending: '等待 Langfuse 数据入库…', unavailable: 'Langfuse 暂不可用，保留上次调用记录。',
      ready: 'Langfuse 已同步 · ' + (trace.trace_id || '') }[trace.state] || '');
    if (this.source === 'remote') this.draw();
  }

  draw(fit = false) {
    if (fit) this.needsFit = true;
    const items = this.source === 'local' ? this.local : this.remote;
    const data = processGraphData(items, this.label, this.taskStatus);
    this.records = new Map(data.records.map(item => [item.id, item]));
    const signature = JSON.stringify([this.taskId, this.source, data.nodes, data.edges]);
    const select = this.panel.querySelector('.process-node');
    if (signature !== this.signature) {
      this.signature = signature;
      select.replaceChildren(new Option('请选择节点', ''));
      for (const item of data.records) select.add(new Option(this.label(item.name) + ' · ' + (item.started || item.startTime || '').slice(11, 19), item.id));
      if (!this.network) {
        this.nodes = new vis.DataSet(); this.edges = new vis.DataSet();
        this.network = new vis.Network(this.panel.querySelector('.process-canvas'), { nodes: this.nodes, edges: this.edges }, {
          nodes: { shape: 'box', margin: 14, widthConstraint: { maximum: 200 }, font: { size: 13, color: '#284338' } },
          edges: { arrows: 'to', color: '#9db5a8', smooth: { type: 'cubicBezier', forceDirection: 'horizontal' } },
          layout: { hierarchical: { enabled: true, direction: 'LR', sortMethod: 'directed', levelSeparation: 260, nodeSpacing: 100 } },
          physics: false, interaction: { hover: true, keyboard: { enabled: true, bindToWindow: false } },
        });
        this.network.on('selectNode', event => {
          const node = event.nodes[0];
          if (node.startsWith('node:') && this.records.has(node.slice(5))) this.select(node.slice(5));
        });
        fit = true;
      }
      const nodeIds = new Set(data.nodes.map(item => item.id));
      const edgeIds = new Set(data.edges.map(item => item.id));
      this.edges.remove(this.edges.getIds().filter(id => !edgeIds.has(id)));
      this.nodes.remove(this.nodes.getIds().filter(id => !nodeIds.has(id)));
      this.nodes.update(data.nodes); this.edges.update(data.edges);
      if (fit || this.needsFit) this.network.fit({ animation: false });
    } else if (fit) this.network?.fit({ animation: false });
    if (data.records.length) this.needsFit = false;
    this.panel.querySelector('.process-count').textContent = data.records.length ? data.records.length + ' 个过程节点' +
      ((data.truncated || (this.source === 'local' ? this.localTruncated : this.remoteTruncated)) ? ' · 仅显示部分记录，缺失父节点以灰色占位。' : '') : '暂无过程节点，等待任务执行或切换数据来源。';
    if (this.selected && this.records.has(this.selected)) {
      select.value = this.selected;
      const version = JSON.stringify(this.records.get(this.selected));
      if (version !== this.selectedVersion) this.select(this.selected);
    } else if (this.selected) this.clearSelection();
  }

  async select(id) {
    const item = this.records.get(id);
    if (!item) { this.clearSelection(); return; }
    this.selected = id;
    this.selectedVersion = JSON.stringify(item);
    const request = ++this.request;
    const inspector = this.panel.querySelector('.process-inspector');
    inspector.hidden = false;
    inspector.querySelector('h4').textContent = this.label(item.name);
    const start = item.started || item.startTime;
    const end = item.updated || item.endTime;
    const duration = start && end ? ' · ' + Math.max(0, (Date.parse(end) - Date.parse(start)) / 1000).toFixed(1) + ' 秒' : '';
    inspector.querySelector('.io-meta').textContent = (this.source === 'local' ? '实时进度' : 'Langfuse') + ' · ' + (start || '') + duration +
      (item.totalTokens != null ? ' · ' + item.totalTokens + ' tokens' : '') + ' · ' + id;
    this.panel.querySelector('.process-node').value = id;
    this.network?.selectNodes(['node:' + id]);
    inspector.querySelector('.io-status').textContent = '正在读取输入输出…';
    inspector.querySelector('.io-input').textContent = '';
    inspector.querySelector('.io-output').textContent = '';
    try {
      const detail = await this.fetchDetail(this.taskId, this.source, id);
      if (request !== this.request) return;
      for (const key of ['input', 'output']) {
        const value = detail[key];
        inspector.querySelector('.io-' + key).textContent = value == null ?
          (key === 'output' && item.status === 'running' ? '执行中，尚未产生输出。' : '此节点未记录' + (key === 'input' ? '输入' : '输出') + '。') :
          typeof value === 'string' ? value : JSON.stringify(value, null, 2);
      }
      inspector.querySelector('.io-status').textContent = [detail.input, detail.output].some(value => value?.truncated === true) ? '内容较长，已截断；preview 展示保留部分。' : '';
    } catch (error) {
      if (request === this.request) inspector.querySelector('.io-status').textContent = error.message + '，可点击刷新重试。';
    }
  }
}

if (typeof module !== 'undefined') module.exports = { processGraphData };
