const $ = s => document.querySelector(s);
let current = null, timer = null, record = null;

const labels = { queued: '排队中', running: '正在运行', completed: '已完成', failed: '未完成' };

async function api(path, options = {}) {
  const base = location.pathname.replace(/\/$/, '');
  const r = await fetch(base + path, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      Authorization: 'Bearer ' + (sessionStorage.getItem('access') || ''),
      ...options.headers
    }
  });
  if (r.status === 401) { if (!$('#login').open) $('#login').showModal(); throw Error('请先输入工作台访问口令'); }
  const data = await r.json();
  if (!r.ok) throw Error(typeof data.detail === 'string' ? data.detail : '请求未通过校验');
  return data;
}

$('#access').onclick = () => $('#login').showModal();
$('#login form').onsubmit = () => {
  sessionStorage.setItem('access', $('#token').value);
  $('#token').value = '';
  setTimeout(() => history().catch(showError), 0);
};

function showError(e) { $('#message').textContent = e.message; }

async function history() {
  const rows = await api('/api/tasks');
  $('#history').replaceChildren();
  for (const row of rows) {
    const b = document.createElement('button');
    b.className = 'history-item';
    b.textContent = row.payload.query;
    const s = document.createElement('small');
    s.textContent = labels[row.status] + ' · ' + row.created;
    b.append(s);
    b.onclick = () => select(row.id);
    $('#history').append(b);
  }
}

function section(title, text) {
  if (!text) return;
  const d = document.createElement('div');
  d.className = 'section';
  const h = document.createElement('h3');
  h.textContent = title;
  const p = document.createElement('p');
  p.textContent = typeof text === 'string' ? text : JSON.stringify(text, null, 2);
  d.append(h, p);
  $('#result').append(d);
}

function listSection(title, items) {
  if (!items || !items.length) return;
  const d = document.createElement('div');
  d.className = 'section';
  const h = document.createElement('h3');
  h.textContent = title;
  const ul = document.createElement('ul');
  for (const item of items) {
    const li = document.createElement('li');
    li.textContent = item.title || item.paper_id || String(item);
    ul.append(li);
  }
  d.append(h, ul);
  $('#result').append(d);
}

function detailsSection(title, data) {
  if (!data) return;
  const d = document.createElement('details');
  const s = document.createElement('summary');
  s.textContent = title;
  const p = document.createElement('pre');
  p.textContent = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
  d.append(s, p);
  $('#result').append(d);
}

async function select(id) {
  clearTimeout(timer);
  current = id;
  try {
    const row = await api('/api/tasks/' + id);
    if (current !== id) return;
    record = row;
    localStorage.setItem('lastTask', id);
    $('#status').textContent = labels[row.status];
    $('#result').replaceChildren();
    $('#download').hidden = row.status !== 'completed';

    if (row.status === 'completed') {
      const r = row.result || {};
      if (row.payload.kind === 'build') {
        if (!r.paper_count) {
          section('尚未写入文献', '未获取到可用论文。可能是关键词无匹配、论文缺少摘要，或 OpenAlex 限流；请调整关键词或联系管理员配置 OpenAlex API Key 后重试。');
        } else {
          section('文献库构建结果', '论文 ' + r.paper_count + ' 篇 · 引用 ' + (r.citation_count || 0) + ' 条 · 实体 ' + (r.entity_count || 0) + ' 个 · 实体关系 ' + (r.entity_relation_count || 0) + ' 条');
        }
        detailsSection('原始结果', r);
      } else {
        const idea = r.idea_output || {};
        section(idea.title || '研究方案', idea.summary);
        section('研究问题', idea.problem);
        section('方法设计', idea.method);
        section('实验计划', idea.experiment);
        section('研究思路来源', idea.idea_lineage);
        section('起点论文', idea.core_paper_title || idea.core_paper_id);
        listSection('候选论文（图谱检索命中）', r.candidate_papers);
        const initial = r.initial_idea_output;
        if (initial) {
          detailsSection('改进前的初始方案', (initial.title || '') + '\n' + (initial.summary || '') + '\n\n' + (initial.method || ''));
        }
        detailsSection('两版方案对比', r.idea_comparison);
        detailsSection('评价与改进建议', r.idea_evaluation);
        detailsSection('图谱上下文（需核查原文）', r.graph_context);
        detailsSection('Agent 工具探索记录', r.tool_trace);
      }
    } else if (row.status === 'failed') {
      section('任务未完成', row.error);
    } else {
      section('研究正在后台进行', row.status === 'queued' ? '任务已进入队列。' : '正在执行文献处理或图谱探索、方案生成与评价，请耐心等待；通常需要 5–10 分钟。');
      timer = setTimeout(() => select(id), 3500);
    }
    await history();
  } catch (e) { showError(e); }
}

$('#research').onsubmit = async e => {
  e.preventDefault();
  $('#submit').disabled = true;
  $('#message').textContent = '';
  try {
    const payload = Object.fromEntries(new FormData(e.target));
    payload.max_seed_papers = Number(payload.max_seed_papers);
    const row = await api('/api/tasks', { method: 'POST', body: JSON.stringify(payload) });
    await select(row.id);
  } catch (e) { showError(e); } finally { $('#submit').disabled = false; }
};

$('#download').onclick = () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(record, null, 2)], { type: 'application/json' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = 'ideagraph-' + current + '.json';
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};

if (sessionStorage.getItem('access')) {
  history().catch(showError);
  const id = localStorage.getItem('lastTask');
  if (id) select(id);
} else $('#login').showModal();
