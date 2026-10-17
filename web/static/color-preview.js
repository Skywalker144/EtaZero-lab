// Fixed illustrative content: this preview never loads models or calls an API.
(() => {
  const $ = id => document.getElementById(id);
  const root = document.documentElement;
  const params = new URLSearchParams(location.search);
  const columns = 'ABCDEFGHJKLMNOP';
  let palette = params.get('scheme') === 'b' ? 'b' : 'a';
  let theme = params.get('theme') === 'dark' ? 'dark' : 'light';
  const moves = [[7,7],[8,7],[8,8],[6,6],[9,9],[6,7],[9,8],[7,6],[10,8],[8,6],[7,8],[6,8]];
  const candidates = [['K8',10,7,38.4,614,38.4,42.1],['J6',8,9,24.8,421,26.3,28.6],['L7',11,8,15.2,286,17.9,18.2],['H6',7,9,8.6,127,7.9,5.7],['F8',5,7,5.1,81,5.1,3.1],['K6',10,9,3.2,44,2.8,1.4],['G6',6,9,2.3,17,1.1,0.6],['J10',8,5,1.1,9,0.6,0.3]];
  function text(id, value) { $(id).textContent = value; }
  function options(id, values) { $(id).replaceChildren(...values.map(value => new Option(value, value))); }
  function applyPalette() {
    root.dataset.palette = palette;
    root.dataset.theme = theme;
    $('theme').value = theme;
    document.querySelectorAll('[data-scheme]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.scheme === palette)));
    document.querySelectorAll('[data-preview-theme]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.previewTheme === theme)));
    text('palette-description', palette === 'a' ? 'A · affectUI 方向：中性灰白 / 高对比黑白' : 'B · Claude 方向：暖纸白 / 陶土橙');
    document.querySelectorAll('[data-color-token]').forEach(code => {
      code.textContent = getComputedStyle(root).getPropertyValue(code.dataset.colorToken).trim();
    });
    const url = new URL(location.href);
    url.searchParams.set('scheme', palette); url.searchParams.set('theme', theme);
    history.replaceState(null, '', url);
  }
  document.querySelectorAll('[data-scheme]').forEach(button => button.addEventListener('click', () => { palette = button.dataset.scheme; applyPalette(); }));
  document.querySelectorAll('[data-preview-theme]').forEach(button => button.addEventListener('click', () => { theme = button.dataset.previewTheme; applyPalette(); }));
  $('theme').addEventListener('change', () => { theme = $('theme').value; applyPalette(); });
  $('theme').querySelector('[value="system"]').remove();
  options('run', ['az_g / 示例运行']); options('model', ['iteration_240 · SWA']);
  options('size', ['15 × 15']); options('rule', ['Freestyle']);
  text('version', '配色预览'); text('connection', '静态示例'); $('connection-dot').classList.add('online');
  text('model-summary', 'AlphaZero · 第 240 代 · SWA\n画布 15² · 示例模型');
  text('model-count', '示例'); $('visits').value = 1600;
  document.querySelector('[data-visits="1600"]').setAttribute('aria-pressed', 'true');
  text('new-game', '创建棋局'); text('settings-note', '配色预览 · 所有数值均为示例，棋局与模型控件仅展示外观。');
  text('rule-badge', 'Freestyle'); text('session-mode', '局面研究'); text('status', '轮到黑棋落子');
  text('move-count', '第 12 手'); text('session-info', '15 × 15 · AlphaZero · 第 240 代 · 根访问上限 1,600');
  text('history-count', '12 手'); text('rule-note', 'Freestyle · 五子或长连获胜。');
  text('analysis-move', '第 12 手 · 黑方'); text('analysis-context', '当前局面 · 示例分析');
  text('completed', '1,600'); text('seconds', '0.82 s'); text('throughput', '1,951'); text('value', '+0.324');
  text('wdl', '搜索  胜 61.2% · 和 10.0% · 负 28.8%');
  text('network-wdl', '网络  胜 56.8% · 和 12.4% · 负 30.8%');
  $('wdl-win').style.width = '61.2%'; $('wdl-draw').style.width = '10%'; $('wdl-loss').style.width = '28.8%';
  text('inference-metrics', 'NN 请求 1,568 · 推理批次 106 · 平均批量 14.8');
  text('candidate-count', '8 个候选点 · 示例数据');
  $('analysis-empty').innerHTML = '<div class="preview-empty-icon" aria-hidden="true">✧</div><strong>探索当前局面</strong><p>按 <kbd>A</kbd> 分析，查看搜索建议与网络策略。</p><small>配色预览 · 空状态示例</small>';
  $('analysis-empty').hidden = true; $('analysis-data').hidden = false;
  const board = $('board');
  let svg = '<svg viewBox="0 0 100 100" aria-hidden="true"><g stroke="var(--board-line)" stroke-width=".14">';
  for (let i=0;i<15;i++) { const p=(i+1)*6.25; svg += `<path d="M6.25 ${p}H93.75 M${p} 6.25V93.75"/>`; }
  svg += '</g><g fill="var(--board-star)">';
  for (const x of [3,7,11]) for (const y of [3,7,11]) svg += `<circle cx="${(x+1)*6.25}" cy="${(y+1)*6.25}" r=".45"/>`;
  svg += '</g><g fill="var(--board-label)" font-family="monospace" font-size="1.8" text-anchor="middle">';
  for (let i=0;i<15;i++) svg += `<text x="${(i+1)*6.25}" y="3.4">${columns[i]}</text><text x="3" y="${(i+1)*6.25+.6}">${15-i}</text>`;
  board.innerHTML = svg + '</g></svg>';
  for (let y=0;y<15;y++) for (let x=0;x<15;x++) {
    const point = document.createElement('div');
    point.className = 'board-point'; point.id = `preview-point-${x}-${y}`;
    Object.assign(point.style, {left:`${(x+1)*6.25}%`,top:`${(y+1)*6.25}%`,width:'5.8%',height:'5.8%'});
    const index = moves.findIndex(move => move[0]===x && move[1]===y);
    if (index>=0) point.innerHTML = `<span class="stone ${index%2?'white':'black'} ${index===moves.length-1?'last':''}">${index+1}</span>`;
    board.append(point);
  }
  $('numbers').addEventListener('change', () => board.classList.toggle('hide-numbers', !$('numbers').checked));
  moves.forEach(([x,y],i) => {
    const item = document.createElement('span'); item.className = `history-item ${i===11?'active':''}`;
    item.innerHTML = `<small>${i+1}</small><i class="tiny-stone ${i%2?'white':'black'}"></i>${columns[x]}${15-y}`;
    $('history').append(item);
  });
  function hover(node,x,y) {
    node.addEventListener('mouseenter', () => { $(`preview-point-${x}-${y}`).classList.add('preview-point'); text('board-coordinate', `${columns[x]}${15-y}`); });
    node.addEventListener('mouseleave', () => { $(`preview-point-${x}-${y}`).classList.remove('preview-point'); text('board-coordinate', '—'); });
  }
  candidates.forEach(([label,x,y,prior,visits,policy,selection],index) => {
    const row = document.createElement('tr');
    row.innerHTML = `<td class="${index===0?'chosen':''}">${label}${index===0?' ★':''}</td><td class="preview-percent" style="--bar:${prior}%">${prior.toFixed(1)}</td><td>${visits}</td><td class="preview-percent" style="--bar:${policy}%">${policy.toFixed(1)}</td><td class="preview-percent" style="--bar:${selection}%">${selection.toFixed(1)}</td>`;
    hover(row,x,y); $('candidates').append(row);
  });
  function heatmap(node, offset) {
    node.style.gridTemplateColumns = 'repeat(15,minmax(0,1fr))';
    for (let y=0;y<15;y++) for (let x=0;x<15;x++) {
      const value = Math.max(...candidates.map((c,i) => Math.max(0, 1-Math.hypot(x-c[1],y-c[2])/2.5)*c[3]/38.4)) * (offset? .75+.25*Math.sin(x+y+offset):1);
      const cell = document.createElement('div'); cell.className = 'heat-cell';
      cell.style.setProperty('--intensity', `${(value*100).toFixed(1)}%`);
      cell.title = `${columns[x]}${15-y} · 示例热力图`;
      const index = moves.findIndex(move => move[0]===x && move[1]===y);
      if (index>=0) cell.innerHTML = `<i class="heat-stone ${index%2?'white':'black'}"></i>`;
      hover(cell,x,y); node.append(cell);
    }
  }
  heatmap($('prior-map'),0); heatmap($('search-map'),1);
  text('prior-max','38.4'); text('search-max','42.1'); text('heatmap-position','第 12 手 · 黑方 · 示例分布');
  $('network-empty').hidden = true; $('network-data').hidden = false;
  ['普通策略','对手策略','长期 optimistic','短期 optimistic'].forEach((name,i) => {
    const section = document.createElement('section');
    section.innerHTML = `<h3 class="network-head-name">${name}</h3><div class="heat-grid"></div><p class="network-summary">固定示例 · 仅用于预览色阶</p>`;
    heatmap(section.querySelector('.heat-grid'),i); $('network-maps').append(section);
  });
  text('raw-analysis', JSON.stringify({preview:true, note:'固定示例，无真实模型推理', visits:1600, seconds:0.82, wdl:[0.612,0.1,0.288]},null,2));
  document.querySelectorAll('[data-tab]').forEach(button => button.addEventListener('click', () => {
    document.querySelectorAll('[data-tab]').forEach(tab => { const selected = tab===button; tab.setAttribute('aria-selected',String(selected)); tab.tabIndex=selected?0:-1; $(tab.dataset.tab).hidden=!selected; });
  }));
  $('preview-state').addEventListener('change', () => {
    const value = $('preview-state').value;
    document.querySelector('.position').classList.toggle('thinking',value==='thinking');
    text('status', value==='thinking'?'正在搜索 · 1.2 s':'轮到黑棋落子');
    $('analysis-empty').hidden = value!=='empty'; $('analysis-data').hidden = value==='empty';
  });
  // Keep visual-only controls readable, without suggesting real engine actions.
  $('settings').addEventListener('submit',event=>event.preventDefault());
  ['new-game','apply-settings','refresh-models','analyze','step','undo','show-analysis','first','prev','next','live'].forEach(id => {
    $(id).disabled=false; $(id).type='button'; $(id).title='外观示例 · 不执行棋局操作'; $(id).setAttribute('aria-disabled','true');
  });
  text('state-version','配色研究 · 静态示例 · 未连接引擎');
  const responsive = window.matchMedia('(max-width:1150px)');
  const collapse = () => { $('setup-details').open=!responsive.matches; };
  responsive.addEventListener('change', collapse); collapse(); applyPalette();
})();
