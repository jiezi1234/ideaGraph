const GRAPH_TYPES = {
  Paper: ['论文', '#8066d9'], Method: ['方法', '#318d8a'], Concept: ['概念', '#4a7fb4'],
  Problem: ['问题', '#c77853'], Scenario: ['场景', '#a16b99'], Domain: ['领域', '#548d64'],
  Innovation: ['创新', '#ac813b'], Entity: ['实体', '#808397'],
};

class GraphBrowser {
  constructor(root) {
    root.append(document.getElementById('graph-template').content.cloneNode(true));
    this.root = root;
    this.canvas = root.querySelector('.graph-canvas');
    this.stats = root.querySelector('.graph-stats');
    this.selection = root.querySelector('.graph-selection');
    this.legend = root.querySelector('.legend');
    this.search = root.querySelector('input');
    this.fit = root.querySelector('.graph-fit');
    this.request = 0;
    this.matches = document.createElement('div');
    this.matches.className = 'graph-matches';
    this.canvas.before(this.matches);
    this.fit.onclick = () => this.network?.fit();
    root.querySelector('.graph-refresh').onclick = () => this.load();
    this.search.oninput = () => this.find();
    this.search.onkeydown = event => {
      if (event.key === 'Enter') this.matches.querySelector('button')?.click();
      if (event.key === 'Escape') { this.search.value = ''; this.find(); }
    };
  }

  notice(message, label, action) {
    const box = document.createElement('div');
    box.className = 'graph-loading';
    box.textContent = message;
    if (label) {
      const button = document.createElement('button');
      button.className = 'quiet';
      button.textContent = label;
      button.onclick = action;
      box.append(button);
    }
    this.canvas.replaceChildren(box);
  }

  async load() {
    const request = ++this.request;
    this.network?.destroy();
    this.network = null;
    this.records = [];
    this.search.value = '';
    this.search.disabled = this.fit.disabled = true;
    this.selection.replaceChildren();
    this.legend.replaceChildren();
    this.matches.replaceChildren();
    this.stats.textContent = '当前视图';
    if (!token()) {
      this.notice('连接工作台后即可浏览共享文献图谱。', '连接工作台', openAccess);
      return;
    }
    this.notice('正在读取文献图谱…');
    try {
      const data = await api('/api/graph');
      if (request !== this.request) return;
      this.records = data.nodes;
      const identifiers = new Set(data.nodes.map(node => node.id));
      // 节点和关系由服务端分别限量，只统计两端都在当前视图内的关系。
      const edges = data.edges.filter(edge => identifiers.has(edge.from) && identifiers.has(edge.to));
      this.stats.textContent = data.nodes.length + ' 节点 · ' + edges.length + ' 关系';
      if (!data.nodes.length) {
        this.notice('文献库中还没有节点。', '建立文献图谱', () => { location.href = route('build'); });
        return;
      }
      this.draw(edges);
    } catch (error) {
      if (request !== this.request) return;
      this.notice(error.message, token() ? '重新加载' : '连接工作台', token() ? () => this.load() : openAccess);
    }
  }

  draw(edges) {
    this.canvas.replaceChildren();
    const nodes = this.records.map(node => ({
      id: node.id, label: node.label.length > 31 ? node.label.slice(0, 30) + '…' : node.label,
      shape: node.kind === 'paper' ? 'diamond' : 'dot', size: node.kind === 'paper' ? 13 : 9,
      color: (GRAPH_TYPES[node.type] || GRAPH_TYPES.Entity)[1],
    }));
    this.network = new vis.Network(this.canvas, {
      nodes: new vis.DataSet(nodes),
      edges: new vis.DataSet(edges.map(edge => ({ ...edge, arrows: 'to' }))),
    }, {
      nodes: { font: { size: 13, color: '#454459' }, borderWidthSelected: 3 },
      edges: { color: { color: '#b9bbcd', highlight: '#907cc7', inherit: false }, font: { size: 10, color: '#67647e', strokeWidth: 3 }, smooth: false },
      physics: { stabilization: { iterations: 150 }, barnesHut: { gravitationalConstant: -7800, springLength: 145, avoidOverlap: 0.2 } },
      interaction: { hover: true, hideEdgesOnDrag: nodes.length > 300 },
    });
    this.network.on('click', event => this.inspect(this.records.find(node => node.id === event.nodes[0])));
    this.search.disabled = this.fit.disabled = false;
    this.inspect(null);
    for (const type of new Set(this.records.map(node => node.type))) {
      const [label, color] = GRAPH_TYPES[type] || [type, GRAPH_TYPES.Entity[1]];
      const item = document.createElement('span');
      item.className = 'legend-item';
      const dot = document.createElement('i');
      dot.style.backgroundColor = color;
      item.append(dot, document.createTextNode(label));
      this.legend.append(item);
    }
  }

  inspect(node) {
    this.selection.replaceChildren();
    if (!node) {
      this.selection.textContent = '点击节点查看论文摘要或实体描述，滚轮缩放画布。';
      return;
    }
    const type = document.createElement('span');
    type.className = 'badge';
    type.textContent = node.type;
    const name = document.createElement('strong');
    name.textContent = node.label;
    this.selection.append(type, name);
    const excerpt = document.createElement('p');
    const paper = node.kind === 'paper';
    excerpt.textContent = node.preview || (paper ? '该论文暂缺摘要，仅用于展示引用关系。' : '暂无实体描述。');
    if (node.preview?.length === 480) excerpt.textContent += '…';
    this.selection.append(excerpt);
    if (paper && node.preview?.trim() && node.label.length <= 300) {
      const next = document.createElement('a');
      next.className = 'btn quiet';
      const context = { seed_paper_id: node.label };
      if (this.buildTaskId) context.from_build = this.buildTaskId;
      next.href = route('idea', context);
      next.textContent = '以此论文构思';
      this.selection.append(next);
    }
  }

  find() {
    const term = this.search.value.trim().toLowerCase();
    this.matches.replaceChildren();
    if (!term || !this.network) return;
    const hits = this.records.filter(node => node.label.toLowerCase().includes(term));
    if (!hits.length) this.matches.textContent = '当前视图中没有匹配节点。';
    for (const node of hits.slice(0, 8)) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'quiet compact';
      button.textContent = node.label;
      button.onclick = () => {
        this.network.selectNodes([node.id]);
        this.network.focus(node.id, { scale: 1.15, animation: !matchMedia('(prefers-reduced-motion: reduce)').matches });
        this.inspect(node);
        this.matches.replaceChildren();
        this.search.value = node.label;
      };
      this.matches.append(button);
    }
    if (hits.length > 8) this.matches.append(document.createTextNode('另有 ' + (hits.length - 8) + ' 项，可继续输入关键词。'));
  }
}
