/* 封面是图谱结构示意，独立绘制以免在概览页请求私有文献数据。 */
function drawCover(canvas, width, height) {
  if (!width || !height) return;
  const density = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * density);
  canvas.height = Math.round(height * density);
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.scale(density, density);

  // 用倾斜的环面投影表现图谱连通性，深度排序让背面的连线隐入背景。
  const tau = Math.PI * 2;
  const columns = 80;
  const rows = 20;
  const vertices = [];
  for (let i = 0; i < columns; i++) {
    const u = tau * i / columns;
    for (let j = 0; j < rows; j++) {
      const v = tau * (j + (i % 2) * 0.45) / rows;
      const radius = 0.43 + 0.12 * Math.cos(2 * u - 0.5);
      const ring = 1.02 + radius * Math.cos(v);
      const x = ring * Math.cos(u);
      const y = ring * Math.sin(u) * 0.81;
      const z = radius * Math.sin(v) + 0.26 * Math.sin(2 * u + 0.8);
      const tiltY = y * 0.75 - z * 0.66;
      const depth = y * 0.66 + z * 0.75;
      const scale = 1 + depth * 0.11;
      vertices.push({ x: (x * 0.96 + tiltY * 0.28) * scale, y: (tiltY * 0.96 - x * 0.28) * scale, depth, i, j });
    }
  }
  const xs = vertices.map(p => p.x), ys = vertices.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const plotScale = Math.min(width * 0.84 / (maxX-minX), height * 0.80 / (maxY-minY));
  vertices.forEach(p => {
    p.x = width * 0.52 + (p.x - (minX + maxX) / 2) * plotScale;
    p.y = height * 0.49 + (p.y - (minY + maxY) / 2) * plotScale;
  });
  const at = (i, j) => vertices[(i % columns) * rows + (j % rows)];
  const color = p => {
    const mix = (1 + Math.cos(tau * p.i / columns - 0.42)) / 2;
    return [Math.round(183 - 66 * mix), Math.round(143 + 65 * mix), Math.round(254 - 14 * mix)].join(',');
  };
  ctx.lineWidth = 0.68;
  for (const p of [...vertices].sort((a,b) => a.depth - b.depth)) {
    const rgb = color(p);
    const light = Math.max(0.09, 0.45 + p.depth * 0.42);
    ctx.strokeStyle = 'rgba(' + rgb + ',' + light * 0.74 + ')';
    ctx.beginPath();
    for (const other of [at(p.i + 1,p.j), at(p.i,p.j + 1), at(p.i + 1,p.j + 1)]) {
      ctx.moveTo(p.x,p.y);
      ctx.lineTo(other.x,other.y);
    }
    ctx.stroke();
    ctx.fillStyle = 'rgba(' + rgb + ',' + Math.min(0.98,light + 0.17) + ')';
    ctx.beginPath();
    ctx.arc(p.x,p.y,p.i % 7 === 0 && p.j % 4 === 0 ? 1.9 : 0.88,0,tau);
    ctx.fill();
  }
  const labels = [
    { vertex: at(49,13), x: 0.10, y: 0.28, name: 'Paper', rgb: '202,182,255' },
    { vertex: at(2,4), x: 0.83, y: 0.17, name: 'Method', rgb: '148,224,233' },
    { vertex: at(15,7), x: 0.74, y: 0.82, name: 'Concept', rgb: '196,195,255' },
  ];
  ctx.font = '11px Consolas, monospace';
  for (const {vertex,x,y,name,rgb} of labels) {
    const tagWidth = ctx.measureText(name).width + 18;
    const tagX = Math.min(width - tagWidth - 4, width*x), tagY = height*y;
    ctx.strokeStyle = 'rgba(' + rgb + ',0.55)';
    ctx.beginPath(); ctx.moveTo(vertex.x,vertex.y); ctx.lineTo(tagX + tagWidth / 2,tagY); ctx.stroke();
    ctx.fillStyle = '#24233f'; ctx.fillRect(tagX, tagY-9, tagWidth, 19);
    ctx.strokeRect(tagX, tagY-9, tagWidth, 19);
    ctx.fillStyle = 'rgb(' + rgb + ')'; ctx.fillText(name, tagX + 9, tagY + 4);
    ctx.beginPath(); ctx.arc(vertex.x,vertex.y,2.7,0,tau); ctx.fill();
  }
}
const coverCanvas = document.getElementById('cover-graph');
new ResizeObserver(([entry]) => {
  // 只在样式已将画布移出文档流后绘制，避免高像素密度下位图反复撑大布局。
  if (getComputedStyle(coverCanvas).position !== 'absolute') return;
  drawCover(coverCanvas, entry.contentRect.width, entry.contentRect.height);
}).observe(coverCanvas.parentElement);
