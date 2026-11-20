const $ = selector => document.querySelector(selector);
const token = () => sessionStorage.getItem('access') || '';
const VIEW_LABEL = { landing: '工作台概览', build: '文献建图', graph: '图谱探索', idea: '研究构思' };
const STATUS_LABEL = { queued: '排队中', running: '运行中', completed: '已完成', failed: '失败' };
const path = location.pathname.replace(/\/+$/, '');
const routeMatch = path.match(/\/(build|idea|graph)$/);
const view = routeMatch ? routeMatch[1] : 'landing';
// 子路径部署时保留工作台前缀，任务链接可直接收藏或刷新。
const BASE = routeMatch ? path.slice(0, -routeMatch[0].length) : path;
const params = new URLSearchParams(location.search);

function route(name, query = {}) {
  const suffix = new URLSearchParams(query).toString();
  return BASE + '/' + (name === 'landing' ? '' : name) + (suffix ? '?' + suffix : '');
}

const STAGES = {
  prepare_workflow: '准备模型与检索索引', build_graph: '构建文献与语义图谱', research: '探索研究问题',
  query_understanding: '理解研究问题', paper_retrieval: '检索相关论文', paper_selection: '选择起点论文',
  seed_paper_selection: '加载指定论文', related_entity_retrieval: '寻找关联实体',
  agent_generation: '探索图谱并构思方案', final_idea_generation: '生成研究方案',
  idea_evaluation: '评审研究方案', idea_comparison: '对比改进结果',
  'search:start': '开始发现论文与扩展引用', 'search:done': '论文与引用发现完成',
  'paper:node_written': '论文节点已写入', 'paper:start': '开始抽取论文实体',
  'paper:extract_done': '论文实体抽取完成', 'paper:write_done': '实体与关系已写入',
  'paper:extract_error': '论文抽取失败，继续处理其他论文', 'paper:extract_skipped': '论文内容不足，跳过抽取',
  'extract:skipped': '本次未启用实体抽取', 'citation:pending_unresolved': '部分引用等待端点论文',
};

function stageLabel(name) {
  const round = String(name).match(/^(agent_generation|graph_tool_execution|idea_evaluation)_round_(\d+)(_result)?$/);
  if (round) return ({ agent_generation: '构思与决策', graph_tool_execution: '执行图谱探索', idea_evaluation: '评审方案' }[round[1]]) +
    ' · 第 ' + round[2] + ' 轮' + (round[3] ? '结果' : '');
  return STAGES[name] || name;
}

function makeIcon(name) {
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('class', 'icon');
  svg.setAttribute('aria-hidden', 'true');
  const use = document.createElementNS(ns, 'use');
  use.setAttribute('href', document.documentElement.dataset.iconSprite + '#' + name);
  svg.append(use);
  return svg;
}

function setAccessState() {
  $('#access-state').textContent = token() ? '已保存会话口令' : '配置访问口令';
}

async function api(endpoint, options = {}, credential = token()) {
  let response;
  try {
    response = await fetch(BASE + endpoint, {
      ...options,
      headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + credential },
    });
  } catch {
    throw new Error('无法连接工作台服务，请检查网络后重试。');
  }
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const error = new Error(typeof data.detail === 'string' ? data.detail : '服务暂时无法完成请求。');
    error.status = response.status;
    if (response.status === 401 && token() === credential) {
      sessionStorage.removeItem('access');
      setAccessState();
    }
    throw error;
  }
  return endpoint.endsWith('/report.md') ? response.blob() : response.json();
}

let afterAccess;
let accessAttempt = 0;
function openAccess(callback) {
  accessAttempt++;
  afterAccess = typeof callback === 'function' ? callback : null;
  $('#access-message').textContent = '';
  if (!$('#login').open) $('#login').showModal();
  $('#token').focus();
}

function setMenu(open) {
  document.body.classList.toggle('menu-open', open);
  $('#menu').setAttribute('aria-expanded', String(open));
  $('.shell').inert = open;
  $('#menu-scrim').hidden = !open;
  if (open) $('.sidebar a').focus();
}

function setupShell() {
  for (const name of Object.keys(VIEW_LABEL)) $('#view-' + name).hidden = name !== view;
  document.title = VIEW_LABEL[view] + ' · IdeaGraph';
  const updateNav = () => {
    const active = view === 'landing' && location.hash === '#records' ? 'records' : view;
    $('#crumb').textContent = active === 'records' ? '最近任务' : VIEW_LABEL[view];
    document.querySelectorAll('[data-nav]').forEach(link => {
      link.classList.toggle('active', link.dataset.nav === active);
      if (link.dataset.nav === active) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    });
  };
  updateNav();
  window.addEventListener('hashchange', updateNav);
  $('#menu').onclick = () => setMenu(true);
  $('#close-menu').onclick = () => { setMenu(false); $('#menu').focus(); };
  $('#menu-scrim').onclick = () => { setMenu(false); $('#menu').focus(); };
  matchMedia('(min-width:861px)').addEventListener('change', event => { if (event.matches) setMenu(false); });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && document.body.classList.contains('menu-open')) {
      setMenu(false);
      $('#menu').focus();
    }
  });
  $('.sidebar').addEventListener('click', event => { if (event.target.closest('a')) setMenu(false); });
  $('#access').onclick = () => { setMenu(false); openAccess(); };
  $('#close-access').onclick = () => $('#login').close();
  $('#login').addEventListener('close', () => { $('#token').value = ''; afterAccess = null; });
  $('#today').append(makeIcon('calendar'), document.createTextNode(
    new Intl.DateTimeFormat('zh-CN', { year: 'numeric', month: 'long', day: 'numeric' }).format(new Date())));
  $('#access-form').onsubmit = async event => {
    event.preventDefault();
    const credential = $('#token').value;
    const callback = afterAccess;
    const attempt = accessAttempt;
    $('#connect').disabled = true;
    $('#access-message').textContent = '正在验证…';
    try {
      const rows = await api('/api/tasks', {}, credential);
      // 验证期间关闭弹窗即取消本次继续操作，避免取消后仍提交研究任务。
      if (!$('#login').open || attempt !== accessAttempt) return;
      sessionStorage.setItem('access', credential);
      setAccessState();
      $('#login').close();
      if (callback) callback();
      else loadPage(rows);
    } catch (error) { if (attempt === accessAttempt) $('#access-message').textContent = error.message; }
    finally { $('#connect').disabled = false; }
  };
  setAccessState();
}

function listNotice(message, actionLabel, action) {
  const line = document.createElement('div');
  line.className = 'record-empty';
  line.append(makeIcon('inbox'), document.createTextNode(message));
  if (actionLabel) {
    const button = document.createElement('button');
    button.className = 'text-action';
    button.textContent = actionLabel;
    button.onclick = action;
    line.append(button);
  }
  $('#history').replaceChildren(line);
}

function renderHistory(rows) {
  if (!rows.length) {
    listNotice('暂无研究任务。', '创建建图任务', () => { location.href = route('build'); });
    return;
  }
  const fragment = document.createDocumentFragment();
  for (const row of rows) {
    const kind = row.payload.kind;
    const link = document.createElement('a');
    link.className = 'record-row';
    link.href = route(kind, { task: row.id });
    const title = document.createElement('strong');
    title.className = 'record-name ' + (kind === 'idea' ? 'amber' : 'violet');
    const name = document.createElement('span');
    name.textContent = row.payload.query;
    link.title = row.payload.query;
    title.append(makeIcon(kind === 'idea' ? 'spark' : 'layers'), name);
    const type = document.createElement('span');
    type.className = 'record-type';
    type.textContent = VIEW_LABEL[kind];
    const status = document.createElement('span');
    status.className = 'badge';
    status.dataset.status = row.status;
    status.textContent = STATUS_LABEL[row.status] || row.status;
    const created = document.createElement('time');
    created.className = 'record-time';
    // SQLite 的 CURRENT_TIMESTAMP 是 UTC，展示时转换为浏览器所在时区。
    const iso = /^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d$/.test(row.created) ? row.created.replace(' ', 'T') + 'Z' : row.created;
    const date = new Date(iso);
    created.dateTime = iso;
    created.textContent = Number.isNaN(date.getTime()) ? row.created : date.toLocaleString('zh-CN', { year:'numeric', month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit', hour12:false });
    link.append(title,type,status,created);
    fragment.append(link);
  }
  $('#history').replaceChildren(fragment);
}

let historyRequest = 0;
async function refreshHistory(rows) {
  const request = ++historyRequest;
  if (!token()) {
    listNotice('连接工作台以查看最近的研究任务。', '连接工作台', openAccess);
    return;
  }
  if (rows) { renderHistory(rows); return; }
  listNotice('正在读取最近任务…');
  try {
    const records = await api('/api/tasks');
    if (request === historyRequest) renderHistory(records);
  } catch (error) {
    if (request === historyRequest) listNotice(error.message, token() ? '重试' : '连接工作台', token() ? () => refreshHistory() : openAccess);
  }
}

let activeRow = null;
let generation = 0;
let traceRequest = 0;
let processView;
let fullGraph;
let buildGraph;

function configureTrace() {
  const fold = $('#' + view + '-trace');
  fold.ontoggle = () => {
    if (!fold.open || !activeRow) return;
    if (!processView) {
      const panel = document.createElement('section');
      panel.className = 'progress-panel';
      fold.append(panel);
      processView = new ProcessGraph(panel, stageLabel, (taskId, source, id) =>
        api('/api/tasks/' + encodeURIComponent(taskId) + (source === 'local' ? '/events/' : '/trace/') + encodeURIComponent(id)));
      panel.querySelector('.trace-refresh').onclick = loadTrace;
    }
    processView.update(activeRow);
    processView.draw(true);
    loadTrace();
  };
}

async function loadTrace() {
  if (!activeRow || !processView || !$('#' + view + '-trace').open) return;
  const taskId = activeRow.id;
  const request = ++traceRequest;
  try {
    const trace = await api('/api/tasks/' + encodeURIComponent(taskId) + '/trace');
    if (activeRow?.id === taskId && request === traceRequest) processView.updateTrace(trace);
  } catch {
    if (activeRow?.id === taskId && request === traceRequest) processView.panel.querySelector('.trace-state').textContent = '调用详情暂时无法加载，可稍后刷新。';
  }
}

function fillForm(payload) {
  const form = $('#' + view + '-form');
  for (const field of ['query','max_seed_papers','seed_paper_id','constraints']) {
    if (field in payload && form.elements[field]) form.elements[field].value = payload[field];
  }
}

function renderTask(row) {
  const newTask = activeRow?.id !== row.id;
  activeRow = row;
  if (row.id) sessionStorage.setItem('last_' + view, row.id);
  const pending = row.status === 'queued' || row.status === 'running';
  const status = $('#' + view + '-status');
  status.dataset.status = row.status;
  status.textContent = STATUS_LABEL[row.status] || row.status;
  const completed = row.status === 'completed';
  const target = $('#' + view + '-result');
  const fold = $('#' + view + '-trace');
  fold.hidden = false;
  if (newTask) fold.open = pending;
  if (fold.open && processView) processView.update(row);
  $('#' + view + '-actions').hidden = !completed;
  if (view === 'build') {
    $('#build-graph-card').hidden = !completed;
    if (completed) {
      buildGraph.buildTaskId = row.id;
      $('#build-to-idea').href = route('idea', { from_build: row.id });
      buildGraph.load();
    }
  }
  if (completed) {
    const result = row.result || {};
    if (result.report_markdown) target.innerHTML = DOMPurify.sanitize(marked.parse(result.report_markdown, { gfm:true, breaks:true }));
    else {
      const pre = document.createElement('pre');
      pre.textContent = JSON.stringify(result, null, 2);
      target.replaceChildren(pre);
    }
  } else {
    const info = document.createElement('p');
    info.className = row.status === 'failed' ? 'error' : 'hint';
    info.textContent = row.status === 'failed' ? (row.error || '任务未完成，可修改配置后再次提交。')
      : row.status === 'queued' ? '任务已进入队列。可以离开页面，稍后从最近任务继续查看。'
      : '任务正在运行，完成后将在这里展示报告。';
    target.replaceChildren(info);
  }
}

function message(text) { $('#' + view + '-message').textContent = text; }

async function follow(taskId, version) {
  let polls = 0;
  let delay = 3500;
  while (version === generation) {
    await new Promise(resolve => setTimeout(resolve, delay));
    if (version !== generation) return;
    try {
      const row = await api('/api/tasks/' + encodeURIComponent(taskId));
      if (version !== generation) return;
      message('');
      renderTask(row);
      if (row.status === 'completed' || row.status === 'failed') {
        loadTrace();
        // Langfuse 入库晚于本地任务结束，展开的调用视图在完成后补拉一次。
        setTimeout(() => { if (version === generation) loadTrace(); }, 20000);
        return;
      }
      if (++polls % 5 === 0) loadTrace();
      delay = 3500;
    } catch (error) {
      if (version !== generation) return;
      if (error.status === 401 || error.status === 404) { message(error.message); return; }
      message(error.message + ' 正在等待重新连接。');
      delay = 10000;
    }
  }
}

async function openTask(taskId) {
  const version = ++generation;
  try {
    const row = await api('/api/tasks/' + encodeURIComponent(taskId));
    if (version !== generation) return;
    if (row.payload.kind !== view) { location.replace(route(row.payload.kind, { task: taskId })); return; }
    fillForm(row.payload);
    message('');
    renderTask(row);
    if (row.status === 'queued' || row.status === 'running') follow(row.id, version);
  } catch (error) { if (version === generation) message(error.message); }
}

function setupForm() {
  const form = $('#' + view + '-form');
  form.onsubmit = async event => {
    event.preventDefault();
    if (!token()) { openAccess(() => form.requestSubmit()); return; }
    const payload = Object.fromEntries(new FormData(form));
    payload.kind = view;
    payload.query = payload.query.trim();
    if (payload.query.length < 3) { message('请输入至少三个非空白字符作为研究主题。'); form.elements.query.focus(); return; }
    if (view === 'build') payload.max_seed_papers = Number(payload.max_seed_papers);
    const button = $('#' + view + '-submit');
    button.disabled = true;
    message('');
    try {
      const row = await api('/api/tasks', { method:'POST', body:JSON.stringify(payload) });
      const version = ++generation;
      window.history.replaceState(null, '', route(view, { task: row.id }));
      renderTask({ id:row.id, status:'queued', payload, events:[] });
      follow(row.id, version);
    } catch (error) { message(error.message); if (error.status === 401) openAccess(); }
    finally { button.disabled = false; }
  };
  $('#' + view + '-download').onclick = async () => {
    if (!activeRow) return;
    try {
      const taskId = activeRow.id;
      const blob = await api('/api/tasks/' + encodeURIComponent(taskId) + '/report.md');
      const link = document.createElement('a');
      const url = URL.createObjectURL(blob);
      link.href = url;
      link.download = 'ideagraph-' + taskId.slice(0, 8) + '.md';
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) { message(error.message); }
  };
  configureTrace();
}

async function resumeTask() {
  if (!token()) return false;
  try {
    const rows = await api('/api/tasks');
    const running = rows.find(row => row.payload.kind === view && (row.status === 'queued' || row.status === 'running'));
    const fallbackId = running ? running.id : sessionStorage.getItem('last_' + view);
    if (!fallbackId) return false;
    const row = running || rows.find(item => item.id === fallbackId);
    if (!row || row.payload.kind !== view) return false;
    window.history.replaceState(null, '', route(view, { task: row.id }));
    await openTask(row.id);
    return true;
  } catch { return false; }
}

async function loadPage(rows) {
  if (view === 'landing') { refreshHistory(rows); return; }
  if (view === 'graph') { fullGraph.load(); return; }
  const taskId = new URLSearchParams(location.search).get('task');
  if (taskId) {
    if (token()) openTask(taskId);
    else { message('连接工作台以查看此任务。'); openAccess(); }
    return;
  }
  if (await resumeTask()) return;
  if (view === 'idea') {
    const form = $('#idea-form');
    if (params.has('seed_paper_id')) form.elements.seed_paper_id.value = params.get('seed_paper_id');
    const buildId = params.get('from_build');
    if (buildId && token() && !form.elements.query.value) {
      const version = generation;
      try {
        const row = await api('/api/tasks/' + encodeURIComponent(buildId));
        if (version === generation && row.payload.kind === 'build' && !form.elements.query.value) form.elements.query.value = row.payload.query;
      } catch (error) { if (version === generation) message(error.message); }
    }
  }
}

setupShell();
if (view === 'graph') fullGraph = new GraphBrowser($('[data-graph="full"]'));
if (view === 'build') buildGraph = new GraphBrowser($('[data-graph="build"]'));
if (view === 'build' || view === 'idea') setupForm();
document.querySelectorAll('[data-icon]').forEach(slot => slot.append(makeIcon(slot.dataset.icon)));
$('#history-refresh').onclick = () => refreshHistory();
loadPage();
