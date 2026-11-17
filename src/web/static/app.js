const $ = s => document.querySelector(s);
const token = () => sessionStorage.getItem('access') || '';

// 静态资源与接口都以工作台根路径为基准：本地是 /，线上是 /research/。
const BASE = (() => {
  let path = location.pathname.replace(/\/+$/, '');
  const matched = path.match(/\/(build|idea|graph)$/);
  if (matched) path = path.slice(0, -matched[0].length);
  return path;
})();

const VIEW_LABEL = { landing: '工作空间', build: '建立文献图谱', idea: '生成研究想法', graph: '图谱浏览' };
const STATUS_LABEL = { queued: '排队中', running: '正在运行', completed: '已完成', failed: '未完成' };
const GRAPH_TYPES = {
  Paper: { label: '论文', color: '#315f46' },
  Method: { label: '方法', color: '#2f6fb0' },
  Concept: { label: '概念', color: '#b07d2f' },
  Problem: { label: '问题', color: '#b04a4a' },
  Scenario: { label: '场景', color: '#7a4ab0' },
  Domain: { label: '领域', color: '#2f9e8a' },
  Entity: { label: '实体', color: '#8a9a90' },
};

const graphs = {};
const activeTask = { build: null, idea: null };

async function api(path, options = {}) {
  const response = await fetch(BASE + path, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      Authorization: 'Bearer ' + token(),
      ...options.headers,
    },
  });
  if (response.status === 401) {
    if (!$('#login').open) $('#login').showModal();
    throw new Error('请先输入工作台访问口令');
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '请求未通过校验');
  return data;
}

function showError(id, error) {
  const el = document.getElementById(id);
  if (el) el.textContent = error.message;
}

function currentView() {
  const matched = location.pathname.replace(/\/+$/, '').match(/\/(build|idea|graph)$/);
  return matched ? matched[1] : 'landing';
}

function showView(view) {
  for (const name of ['landing', 'build', 'idea', 'graph']) {
    document.getElementById('view-' + name).hidden = name !== view;
  }
  document.querySelectorAll('[data-nav]').forEach(el => el.classList.toggle('active', el.dataset.nav === view));
  $('#crumb').textContent = VIEW_LABEL[view] || '工作空间';
}

function renderMarkdown(target, markdown) {
  const html = window.marked.parse(markdown || '', { gfm: true, breaks: true });
  target.innerHTML = window.DOMPurify.sanitize(html);
}

async function history() {
  const rows = await api('/api/tasks');
  const box = $('#history');
  box.replaceChildren();
  for (const row of rows) {
    const button = document.createElement('button');
    button.className = 'history-item';
    button.textContent = (row.payload.kind === 'build' ? '建图 · ' : '想法 · ') + row.payload.query;
    const meta = document.createElement('small');
    meta.textContent = STATUS_LABEL[row.status] + ' · ' + row.created;
    button.append(meta);
    button.onclick = () => openTask(row.payload.kind, row.id);
    box.append(button);
  }
}

function renderTask(view, row) {
  const statusEl = $('#' + view + '-status');
  const resultEl = $('#' + view + '-result');
  const downloadEl = $('#' + view + '-download');
  statusEl.textContent = STATUS_LABEL[row.status] || row.status;
  downloadEl.hidden = row.status !== 'completed';

  if (row.status === 'completed') {
    const result = row.result || {};
    activeTask[view] = row.id;
    if (result.report_markdown) {
      renderMarkdown(resultEl, result.report_markdown);
    } else {
      resultEl.innerHTML = '<p class="hint">该任务没有生成 Markdown 报告，以下为原始结果。</p><pre></pre>';
      resultEl.querySelector('pre').textContent = JSON.stringify(result, null, 2);
    }
    if (view === 'build') {
      $('#build-graph-card').hidden = false;
      renderGraph('graph-canvas-build', 'graph-stats-build', 'graph-search-build', 'graph-legend-build');
    }
  } else if (row.status === 'failed') {
    resultEl.innerHTML = '';
    const p = document.createElement('p');
    p.className = 'error';
    p.textContent = row.error || '任务未完成';
    resultEl.append(p);
  } else {
    resultEl.innerHTML = '<div class="empty"><div class="orb">◌</div><h3>正在处理中</h3><p>' +
      (row.status === 'queued' ? '任务已进入队列。' : '正在检索文献或探索图谱，通常需要 5–10 分钟。') +
      '</p></div>';
  }
}

async function openTask(view, taskId) {
  showView(view);
  try {
    const row = await api('/api/tasks/' + taskId);
    renderTask(view, row);
    if (row.status === 'queued' || row.status === 'running') follow(view, taskId);
  } catch (error) {
    showError(view + '-message', error);
  }
}

async function follow(view, taskId) {
  let stop = false;
  while (!stop) {
    await new Promise(resolve => setTimeout(resolve, 3500));
    try {
      const row = await api('/api/tasks/' + taskId);
      renderTask(view, row);
      stop = row.status === 'completed' || row.status === 'failed';
    } catch (error) {
      showError(view + '-message', error);
      stop = true;
    }
  }
}

async function submitTask(view, payload) {
  const button = $('#' + view + '-submit');
  const message = $('#' + view + '-message');
  button.disabled = true;
  message.textContent = '';
  try {
    const row = await api('/api/tasks', { method: 'POST', body: JSON.stringify(payload) });
    activeTask[view] = row.id;
    renderTask(view, { status: 'queued', payload, result: null });
    await history();
    follow(view, row.id);
  } catch (error) {
    showError(view + '-message', error);
  } finally {
    button.disabled = false;
  }
}

async function downloadReport(view) {
  const taskId = activeTask[view];
  if (!taskId) return;
  const response = await fetch(BASE + '/api/tasks/' + taskId + '/report.md', {
    headers: { Authorization: 'Bearer ' + token() },
  });
  if (!response.ok) throw new Error('报告下载失败');
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'ideagraph-' + taskId.slice(0, 8) + '.md';
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function shortLabel(label, kind) {
  const text = String(label || '');
  return kind === 'paper' && text.length > 30 ? text.slice(0, 30) + '…' : text;
}

function renderLegend(legendId, types) {
  const box = document.getElementById(legendId);
  if (!box) return;
  box.replaceChildren();
  for (const type of types) {
    const meta = GRAPH_TYPES[type] || GRAPH_TYPES.Entity;
    const item = document.createElement('span');
    item.className = 'legend-item';
    const dot = document.createElement('i');
    dot.style.background = meta.color;
    item.append(dot, document.createTextNode(meta.label));
    box.append(item);
  }
}

async function renderGraph(canvasId, statsId, searchId, legendId) {
  const container = document.getElementById(canvasId);
  if (!container) return;
  container.innerHTML = '<div class="graph-loading">正在加载图谱…</div>';
  let data;
  try {
    data = await api('/api/graph');
  } catch (error) {
    container.innerHTML = '<div class="graph-loading">图谱加载失败：' + error.message + '</div>';
    return;
  }
  container.innerHTML = '';
  if (!data.nodes.length) {
    container.innerHTML = '<div class="graph-loading">图谱还是空的，先建立文献库。</div>';
    return;
  }

  const nodes = new vis.DataSet(data.nodes.map(node => {
    const meta = GRAPH_TYPES[node.type] || GRAPH_TYPES.Entity;
    return {
      id: node.id,
      label: shortLabel(node.label, node.kind),
      title: node.label,
      group: node.type,
      shape: node.kind === 'paper' ? 'box' : 'dot',
      size: node.kind === 'paper' ? 14 : 9,
      color: { background: meta.color, border: meta.color },
      font: { color: node.kind === 'paper' ? '#22483a' : '#42584d', size: node.kind === 'paper' ? 12 : 11 },
    };
  }));
  const edges = new vis.DataSet(data.edges.map(edge => ({
    from: edge.from,
    to: edge.to,
    label: edge.label,
    arrows: 'to',
    width: edge.label === 'CITES' ? 1.4 : 1,
    color: { color: edge.label === 'CITES' ? '#b9c8bf' : '#d7e2da' },
    font: { size: 9, color: '#8a9b91', strokeWidth: 2, strokeColor: '#ffffff' },
  })));

  const network = new vis.Network(container, { nodes, edges }, {
    nodes: { borderWidth: 1 },
    edges: { smooth: { type: 'continuous' } },
    physics: {
      stabilization: { iterations: 250 },
      barnesHut: { gravitationalConstant: -9000, springLength: 150, avoidOverlap: 0.2 },
    },
    interaction: { hover: true, tooltipDelay: 200, navigationButtons: false },
  });
  graphs[canvasId] = { network, nodes };

  const stats = document.getElementById(statsId);
  if (stats) stats.textContent = data.nodes.length + ' 节点 · ' + data.edges.length + ' 关系';

  const types = [...new Set(data.nodes.map(node => node.type))];
  renderLegend(legendId, types);

  const search = document.getElementById(searchId);
  if (search && !search.dataset.bound) {
    search.dataset.bound = '1';
    search.oninput = () => {
      const term = search.value.trim().toLowerCase();
      const graph = graphs[canvasId];
      if (!graph || !term) return;
      const hit = data.nodes.find(node => String(node.label).toLowerCase().includes(term));
      if (hit) {
        graph.network.selectNodes([hit.id]);
        graph.network.focus(hit.id, { scale: 1.2, animation: true });
      }
    };
  }
}

function setupForms() {
  $('#build-form').onsubmit = event => {
    event.preventDefault();
    const form = new FormData(event.target);
    submitTask('build', {
      kind: 'build',
      query: form.get('query'),
      max_seed_papers: Number(form.get('max_seed_papers')),
    });
  };
  $('#idea-form').onsubmit = event => {
    event.preventDefault();
    const form = new FormData(event.target);
    submitTask('idea', {
      kind: 'idea',
      query: form.get('query'),
      seed_paper_id: form.get('seed_paper_id') || '',
      constraints: form.get('constraints') || '',
    });
  };
  $('#build-download').onclick = () => downloadReport('build').catch(error => showError('build-message', error));
  $('#idea-download').onclick = () => downloadReport('idea').catch(error => showError('idea-message', error));
  $('#build-graph').onclick = () => renderGraph('graph-canvas-build', 'graph-stats-build', 'graph-search-build', 'graph-legend-build');
}

$('#access').onclick = () => $('#login').showModal();
$('#login form').onsubmit = () => {
  sessionStorage.setItem('access', $('#token').value);
  $('#token').value = '';
  setTimeout(() => {
    history().catch(error => showError(currentView() + '-message', error));
    if (currentView() === 'graph') renderGraph('graph-canvas', 'graph-stats-full', 'graph-search', 'graph-legend');
  }, 0);
};

setupForms();
showView(currentView());
if (currentView() === 'graph' && token()) {
  renderGraph('graph-canvas', 'graph-stats-full', 'graph-search', 'graph-legend');
}
if (token()) {
  history().catch(() => {});
} else {
  $('#login').showModal();
}
