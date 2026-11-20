const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { runInNewContext } = require('node:vm');

test('cover only scales its bitmap inside a sized, styled container', () => {
  const container = {};
  const canvas = { parentElement: container, width: 300, height: 150, getContext: () => null };
  let position = 'static';
  let observed;
  let resize;
  const script = readFileSync(require.resolve('../src/web/static/cover.js'), 'utf8');
  runInNewContext(script, {
    document: { getElementById: () => canvas },
    window: { devicePixelRatio: 2 },
    getComputedStyle: () => ({ position }),
    ResizeObserver: class {
      constructor(callback) { resize = callback; }
      observe(element) { observed = element; }
    },
  });
  assert.equal(observed, container);
  const size = [{ contentRect: { width: 400, height: 180 } }];
  resize(size);
  assert.deepEqual([canvas.width, canvas.height], [300, 150]);
  position = 'absolute';
  resize(size);
  resize(size);
  assert.deepEqual([canvas.width, canvas.height], [800, 360]);
});
