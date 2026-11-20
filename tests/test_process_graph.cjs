const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processGraphData } = require('../src/web/static/process-graph.js');

test('parallel stages share parent without a false sequential edge', () => {
  const data = processGraphData([
    { id: 'root', name: 'research', status: 'running' },
    { id: 'left', parent_id: 'root', name: 'tool', status: 'completed' },
    { id: 'right', parent_id: 'root', name: 'tool', status: 'running' },
  ], name => name, 'running');
  assert.equal(data.edges.find(edge => edge.to === 'node:left').from, 'node:root');
  assert.equal(data.edges.find(edge => edge.to === 'node:right').from, 'node:root');
  assert.equal(data.nodes.find(node => node.id === 'node:left').level, data.nodes.find(node => node.id === 'node:right').level);
});

test('remote parents, missing ancestors and cycles remain renderable', () => {
  const data = processGraphData([
    { id: 'a', parentObservationId: 'missing', name: 'model', endTime: 'now' },
    { id: 'b', parentObservationId: 'c', name: 'cycle' },
    { id: 'c', parentObservationId: 'b', name: 'cycle' },
  ], name => name, 'completed');
  assert.equal(data.nodes.filter(node => node.id === 'node:missing').length, 1);
  assert.ok(data.nodes.every(node => Number.isFinite(node.level)));
  assert.match(data.nodes.find(node => node.id === 'node:b').label, /未收到结束记录/);
});

test('large traces are bounded and interrupted stages do not appear live', () => {
  const data = processGraphData(Array.from({ length: 501 }, (_, i) => ({ id: String(i), name: 'step', status: 'running' })), name => name, 'failed');
  assert.equal(data.records.length, 500);
  assert.equal(data.truncated, true);
  assert.match(data.nodes[1].label, /已中断/);
});
