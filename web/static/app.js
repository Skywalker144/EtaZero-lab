const $ = id => document.getElementById(id);
const columns = 'ABCDEFGHJKLMNOPQRSTUVWXYZ';
const rules = {freestyle: 'Freestyle', standard: 'Standard', renju: 'Renju'};
const notes = {freestyle: 'Freestyle · 五子或长连获胜。', standard: 'Standard · 恰好五子获胜，长连不计胜。', renju: 'Renju · 黑棋三三、四四、长连判负，恰好五子优先；白棋五子或长连获胜。'};
let state = null, catalog = null, pending = false, online = false, notice = '';
let formKey = '', viewTurn = null, highlight = null, catalogRevision = -1;
let sortField = 'selection_weight', sortDirection = -1;
let analysisKey = null;
let historyKey = '';
const networkHeads = [
  ['普通策略', 'policy'], ['对手策略', 'opponent_policy'],
  ['长期 optimistic', 'long_optimistic_policy'], ['短期 optimistic', 'short_optimistic_policy'],
];

function coordinate(action, size) { return columns[action % size] + (size - Math.floor(action / size)); }
function percent(value, digits = 1) { return (value * 100).toFixed(digits); }
function samePosition(game, analysis) {
  return game && analysis && game.board_size === analysis.board_size && game.turn === analysis.turn &&
    game.board.every((value, i) => value === analysis.board[i]);
}
function connection(value) {
  online = value;
  $('connection').textContent = value ? '引擎已连接' : '断线 · 重连中';
  $('connection-dot').classList.toggle('online', value);
}
async function api(path, body) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch(path, {signal: controller.signal, ...(body ? {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
    } : {})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `请求失败 (${response.status})`);
    return result;
  } finally { clearTimeout(timeout); }
}
function accept(next) {
  if (state && next.instance === state.instance && next.version < state.version) return;
  if (!state || next.instance !== state.instance || next.game_id !== state.game_id ||
      JSON.stringify(next.game?.moves) !== JSON.stringify(state.game?.moves)) {
    viewTurn = null;
    highlight = null;
  }
  state = next;
  render();
}
async function command(operation, values = {}) {
  if (!state || pending || state.busy || !online) return;
  pending = true;
  notice = '';
  render();
  try { accept(await api(`/api/${operation}`, {version: state.version, ...values})); }
  catch (error) {
    notice = error.message;
    try { accept(await api('/api/state')); } catch { connection(false); }
  } finally { pending = false; render(); }
}
function displayedGame() {
  const game = state?.game;
  if (!game || viewTurn === null) return game;
  const moves = game.moves.slice(0, viewTurn);
  const board = Array(game.board_size ** 2).fill(0);
  moves.forEach((action, i) => { board[action] = i % 2 === 0 ? 1 : -1; });
  return {...game, board, moves, turn: viewTurn, player: viewTurn % 2 === 0 ? 1 : -1};
}
function navigate(turn) {
  if (!state?.game) return;
  const value = Math.max(0, Math.min(state.game.turn, turn));
  viewTurn = value === state.game.turn ? null : value;
  highlight = null;
  render();
}
function renderBoard(game, disabled) {
  const size = game?.board_size || Number($('size').value) || catalog.default_size;
  const step = 100 / (size + 1);
  const end = step * size;
  let drawing = `<svg viewBox="0 0 100 100" aria-hidden="true"><g stroke="var(--board-line)" stroke-width=".12">`;
  for (let i = 1; i <= size; i++) {
    const p = i * step;
    drawing += `<path d="M${step},${p}H${end} M${p},${step}V${end}"/>`;
  }
  drawing += '</g><g fill="var(--board-label)" font-family="system-ui" font-size="1.55" text-anchor="middle">';
  for (let i = 0; i < size; i++) {
    drawing += `<text x="${(i + 1) * step}" y="${100 - step * .37}">${columns[i]}</text><text x="${step * .38}" y="${(i + 1) * step + .55}">${size - i}</text>`;
  }
  drawing += '</g><g fill="var(--board-star)">';
  if (size % 2 === 1) {
    const stars = size >= 11 ? [3, (size + 1) / 2, size - 2] : [(size + 1) / 2];
    for (const x of stars) for (const y of stars) {
      if (x === y || x + y === size + 1) drawing += `<circle cx="${x * step}" cy="${y * step}" r=".36"/>`;
    }
  }
  drawing += '</g></svg>';
  const numbers = new Map((game?.moves || []).map((action, i) => [action, i + 1]));
  for (let action = 0; action < size * size; action++) {
    const stone = game?.board[action] || 0;
    const number = numbers.get(action);
    const last = number && number === game.turn;
    const color = stone === 1 ? 'black' : 'white';
    const label = coordinate(action, size) + (stone ? `，${stone === 1 ? '黑' : '白'}棋，第 ${number} 手` : '，空位');
    drawing += `<button class="board-point" data-action="${action}" aria-label="${label}" style="left:${(action % size + 1) * step}%;top:${(Math.floor(action / size) + 1) * step}%;width:${step * .96}%;height:${step * .96}%" ${disabled || stone ? 'disabled' : ''}>${stone ? `<span class="stone ${color} ${last ? 'last' : ''}">${number}</span>` : ''}</button>`;
  }
  $('board').innerHTML = drawing;
  const analysis = state.analysis;
  if (analysis && samePosition(game, analysis) && $('overlay').value !== 'none') {
    const field = $('overlay').value;
    const maximum = Math.max(0, ...analysis.candidates.map(c => c[field]));
    for (const candidate of analysis.candidates) {
      if (candidate[field] <= 0) continue;
      const cell = $('board').querySelector(`[data-action="${candidate.action}"]`);
      const value = document.createElement('span');
      value.className = 'overlay-value';
      value.style.background = `hsl(${160 - 125 * candidate[field] / maximum} 55% 70% / .8)`;
      value.textContent = size > 15 ? '' : (candidate[field] * 100).toFixed(1);
      cell?.append(value);
    }
  }
  if (highlight !== null) $('board').querySelector(`[data-action="${highlight}"]`)?.classList.add('highlight');
  $('board').classList.toggle('white-turn', game?.player === -1);
  $('board').classList.toggle('hide-numbers', !$('numbers').checked);
}

function renderHeatmap(id, field, maximumId, analysis) {
  const size = analysis.board_size;
  const candidates = new Map(analysis.candidates.map(candidate => [candidate.action, candidate]));
  const maximum = Math.max(0, ...analysis.candidates.map(candidate => candidate[field]));
  let html = '<span></span>';
  for (let x = 0; x < size; x++) html += `<span class="heat-axis">${columns[x]}</span>`;
  for (let y = 0; y < size; y++) {
    html += `<span class="heat-axis">${size - y}</span>`;
    for (let x = 0; x < size; x++) {
      const action = y * size + x;
      const stone = analysis.board[action];
      const candidate = candidates.get(action);
      const value = candidate ? candidate[field] : 0;
      const ratio = maximum > 0 ? value / maximum : 0;
      const color = value > 0 ? `hsl(${160 - 125 * ratio} 55% ${94 - 38 * ratio}%)` : '#edf0e8';
      const detail = `${coordinate(action, size)} · ` + (stone ? `${stone === 1 ? '黑' : '白'}棋` : !candidate ? '不可选点' :
        `网络先验 ${(candidate.network_prior * 100).toFixed(3)}% · 访问 ${candidate.visits} 次（${(candidate.visit_policy * 100).toFixed(3)}%） · 选择权重 ${(candidate.selection_weight * 100).toFixed(3)}%`);
      const percent = value === 0 ? '·' : value < .001 ? '&lt;0.1' : (value * 100).toFixed(1);
      html += `<div class="heat-cell ${action === analysis.action ? 'chosen-point' : ''}" data-action="${action}" data-value="${value}" data-detail="${detail}" role="img" aria-label="${detail}" title="${detail}" tabindex="0" style="background:${color}">${stone ? `<i class="heat-stone ${stone === 1 ? 'black' : 'white'}"></i>` : `<span>${percent}</span>`}</div>`;
    }
  }
  const map = $(id);
  map.style.gridTemplateColumns = `16px repeat(${size}, minmax(0, 1fr))`;
  map.style.gridTemplateRows = `16px repeat(${size}, minmax(0, 1fr))`;
  map.classList.toggle('dense', size > 15);
  map.classList.toggle('compact', size >= 11);
  map.innerHTML = html;
  $(maximumId).textContent = `${(maximum * 100).toFixed(2)}%`;
}


function renderHeatmaps() {
  const analysis = state.analysis;
  $('heatmap-empty').hidden = Boolean(analysis);
  $('heatmap-data').hidden = !analysis;
  if (!analysis) {
    $('prior-map').replaceChildren();
    $('search-map').replaceChildren();
    return;
  }
  $('heatmap-position').textContent = `第 ${analysis.turn + 1} 手落子前 · ${analysis.player === 1 ? '黑' : '白'}方视角`;
  $('heatmap-detail').textContent = '悬停 / 聚焦查看数值；描边为搜索建议点。';
  renderHeatmap('prior-map', 'network_prior', 'prior-max', analysis);
  renderHeatmap('search-map', $('search-map-kind').value, 'search-max', analysis);
}
function renderNetworkMaps() {
  const analysis = state?.analysis;
  const planes = analysis?.network_planes;
  $('network-empty').hidden = Boolean(planes);
  $('network-data').hidden = !planes;
  $('network-maps').replaceChildren();
  if (!planes) return;
  const size = analysis.board_size;
  const kind = $('network-kind').value;
  const probabilities = kind === 'probabilities';
  const common = $('network-common-scale').checked;
  const all = planes.heads.flatMap(head => head[kind]);
  const minimum = probabilities ? 0 : Math.min(...all);
  const maximum = Math.max(...all);
  const format = value => probabilities ? `${(value * 100).toFixed(2)}%` : value.toFixed(3);
  $('network-position').textContent = `第 ${analysis.turn + 1} 手落子前 · 输入执${analysis.player === 1 ? '黑' : '白'} · ${planes.precision}`;
  $('network-detail').textContent = '悬停 / 聚焦查看各点概率与 logits；点击定位到搜索前局面。';
  planes.heads.forEach((head, index) => {
    const values = head[kind];
    const low = common ? minimum : probabilities ? 0 : Math.min(...values);
    const high = common ? maximum : Math.max(...values);
    const [label, name] = networkHeads.find(([, name]) => name === head.name);
    let html = '<span></span>';
    for (let x = 0; x < size; x++) html += `<span class="heat-axis">${columns[x]}</span>`;
    for (let y = 0; y < size; y++) {
      html += `<span class="heat-axis">${size - y}</span>`;
      for (let x = 0; x < size; x++) {
        const action = y * size + x, value = values[action], stone = analysis.board[action];
        const ratio = high > low ? (value - low) / (high - low) : 0;
        const color = `hsl(${160 - 125 * ratio} 55% ${94 - 38 * ratio}%)`;
        const detail = `${label} · ${coordinate(action, size)} · 概率 ${(head.probabilities[action] * 100).toFixed(4)}% · logit ${head.logits[action].toFixed(5)}` +
          (stone ? ` · 已有${stone === 1 ? '黑' : '白'}棋（仍参与网络归一化）` : '');
        const number = probabilities ? (value * 100).toFixed(1) : value.toFixed(1);
        html += `<div class="heat-cell network-cell" data-action="${action}" data-value="${value}" data-detail="${detail}" role="img" aria-label="${detail}" title="${detail}" tabindex="0" style="background:${color}">${stone ? `<i class="heat-stone ${stone === 1 ? 'black' : 'white'}"></i>` : `<span>${number}</span>`}</div>`;
      }
    }
    let best = 0, entropy = 0, occupied = 0;
    head.probabilities.forEach((p, action) => {
      if (p > head.probabilities[best]) best = action;
      if (p > 0) entropy -= p * Math.log(p);
      if (analysis.board[action]) occupied += p;
    });
    const section = document.createElement('section');
    section.dataset.head = name;
    section.innerHTML = `<div class="heatmap-heading"><h3>${label}</h3><span>${index + 1} / ${networkHeads.length}</span></div><div class="network-head-name mono">${name}</div>` +
      `<div class="heat-grid ${size > 15 ? 'dense' : ''} ${size >= 11 ? 'compact' : ''}" role="group" aria-label="${label}热力图" style="grid-template-columns:16px repeat(${size},minmax(0,1fr));grid-template-rows:16px repeat(${size},minmax(0,1fr))">${html}</div>` +
      `<div class="heat-legend"><span>${format(low)}</span><i></i><span>${format(high)}</span></div>` +
      `<p class="network-summary">最高 ${coordinate(best, size)} ${(head.probabilities[best] * 100).toFixed(2)}%<br>熵 ${entropy.toFixed(2)} nats · 已落子质量 ${(occupied * 100).toFixed(2)}%</p>`;
    $('network-maps').append(section);
  });
}
function renderCandidates() {
  const analysis = state?.analysis;
  $('candidates').replaceChildren();
  if (!analysis) return;
  const candidates = analysis.candidates.filter(c => !$('visited-only').checked || c.visits > 0)
    .sort((a, b) => sortDirection * (a[sortField] - b[sortField]) || a.action - b.action);
  $('candidate-count').textContent = `${candidates.length} / ${analysis.candidates.length} 个候选点`;
  const labels = {action: '点位', network_prior: '先验 %', visits: '访问', visit_policy: '访问 %', selection_weight: '选择 %'};
  document.querySelectorAll('[data-sort]').forEach(button => {
    button.textContent = labels[button.dataset.sort] + (sortField === button.dataset.sort ? (sortDirection === -1 ? ' ↓' : ' ↑') : '');
    button.parentElement.setAttribute('aria-sort', sortField === button.dataset.sort ? (sortDirection === -1 ? 'descending' : 'ascending') : 'none');
  });
  const fragment = document.createDocumentFragment();
  for (const c of candidates) {
    const row = document.createElement('tr');
    row.dataset.action = c.action;
    row.classList.toggle('selected', highlight === c.action);
    row.innerHTML = `<td><button data-candidate="${c.action}" class="${c.action === analysis.action ? 'chosen' : ''}" title="在棋盘定位 ${coordinate(c.action, analysis.board_size)}">${coordinate(c.action, analysis.board_size)}${c.action === analysis.action ? ' ★' : ''}</button></td><td>${percent(c.network_prior, 2)}</td><td>${c.visits}</td><td>${percent(c.visit_policy, 2)}</td><td>${percent(c.selection_weight, 2)}</td>`;
    fragment.append(row);
  }
  $('candidates').append(fragment);
}
function renderAnalysis() {
  const a = state.analysis;
  $('analysis-empty').hidden = Boolean(a);
  $('analysis-data').hidden = !a;
  $('analysis-move').textContent = a ? `第 ${a.turn + 1} 手前 · ${a.player === 1 ? '黑' : '白'}方` : '—';
  if (!a) { analysisKey = null; renderHeatmaps(); renderNetworkMaps(); return; }
  const updating = state.busy && state.phase === 'thinking';
  $('analysis-context').textContent = updating ? '正在更新 · 当前显示上次分析' :
    state.error ? '操作失败 · 当前显示上次分析' :
    samePosition(state.game, a) ? '当前局面分析' : '历史分析 · 当前局面尚未分析';
  $('analysis-context').classList.toggle('updating', updating);
  const key = JSON.stringify(a);
  if (key !== analysisKey) {
    $('seconds').textContent = `${a.seconds.toFixed(3)}s`;
    $('completed').textContent = a.completed_visits.toLocaleString();
    $('throughput').textContent = a.seconds > 0 ? Math.round(a.completed_visits / a.seconds).toLocaleString() : '—';
    $('value-label').textContent = `${a.player === 1 ? '黑' : '白'}方视角 · 搜索 W−L`;
    $('value').textContent = `${a.root_value >= 0 ? '+' : ''}${a.root_value.toFixed(4)}`;
    $('wdl').textContent = `搜索 W / D / L   ${a.wdl.map(v => percent(v) + '%').join(' / ')}`;
    $('network-wdl').textContent = `网络 W / D / L   ${a.network_wdl.map(v => percent(v) + '%').join(' / ')}`;
    ['wdl-win', 'wdl-draw', 'wdl-loss'].forEach((id, i) => { $(id).style.width = `${a.wdl[i] * 100}%`; });
    $('inference-metrics').textContent = `NN 请求 ${a.requests} · 批次 ${a.batches} · 平均批量 ${a.batches ? (a.requests / a.batches).toFixed(2) : '—'}`;
    $('raw-analysis').textContent = JSON.stringify(a, null, 2);
    renderHeatmaps();
    renderNetworkMaps();
    analysisKey = key;
  }
  renderCandidates();
}
function render() {
  if (!state || !catalog) return;
  const game = state.game;
  const busy = pending || state.busy || !online;
  const key = game ? [state.game_id, state.model, game.board_size, state.rule, state.human, state.visits, state.mode].join('|') : '';
  if (key !== formKey) {
    if (game) {
      populateRuns(catalog.models.find(m => m.id === state.model)?.run);
      populateModels(state.model);
      $('size').value = game.board_size;
      $('rule').value = state.rule;
      $('visits').value = state.visits;
      $('mode').value = state.mode;
      $('opening').value = state.opening_kind || 'empty';
      document.querySelector(`input[name="human"][value="${state.human}"]`).checked = true;
      updateSizes();
    }
    formKey = key;
  }
  for (const input of $('settings').elements) input.disabled = busy;
  updateSizes();
  $('new-game').disabled = busy || !$('model').value;
  $('apply-settings').disabled = busy || !game;
  $('refresh-models').disabled = busy;
  const rule = game ? state.rule : $('rule').value;
  $('rule-badge').textContent = rules[rule] || '—';
  $('rule-note').textContent = notes[rule] || '';
  $('move-count').textContent = `第 ${game?.turn || 0} 手`;
  $('new-game').textContent = game ? '按配置重开棋局' : '创建棋局';
  $('session-mode').textContent = game ? (state.mode === 'manual' ? '手动双方' : `人类执${state.human === 1 ? '黑' : '白'}`) : '未创建';
  const activeModel = catalog.models.find(m => m.id === state.model);
  $('session-info').textContent = game ? `${activeModel?.label || state.model} · ${state.visits}v` : '模型常驻 · 每次搜索从新根开始';
  $('session-info').title = game ? state.model : '';
  const opening = game?.opening;
  $('opening-info').hidden = !opening;
  $('opening-info').textContent = opening ? `平衡开局 ${opening.moves.length} 手 · 尝试 ${opening.attempts} 次 · ${opening.value_player === 1 ? '黑' : '白'}方网络 W−L ${opening.value >= 0 ? '+' : ''}${opening.value.toFixed(4)} · ${opening.seconds.toFixed(2)}s` : '';
  $('opening-info').title = opening ? `开局种子：${opening.seed}` : '';
  $('state-version').textContent = `单会话 · 状态 v${state.version} · ${state.analysis_config?.device || selectedRun()?.analysis_config.device || catalog.device}`;
  $('undo').disabled = busy || viewTurn !== null || !game?.moves.length || (state.mode === 'play' && !game.moves.some((_, i) => i >= (opening?.moves.length || 0) && (i % 2 === 0 ? 1 : -1) === state.human));
  $('analyze').disabled = $('step').disabled = busy || !game || game.finished || viewTurn !== null;
  $('retry').hidden = busy || state.mode !== 'play' || !game || game.finished || game.player === state.human;
  $('error').textContent = notice || state.error || '';
  $('error').hidden = !$('error').textContent;
  const displayed = displayedGame();
  renderBoard(displayed, busy || !game || game.finished || viewTurn !== null || (state.mode === 'play' && game.player !== state.human));
  const overlayMismatch = $('overlay').value !== 'none' && !samePosition(displayed, state.analysis);
  $('view-note').hidden = viewTurn === null && !overlayMismatch;
  $('view-note').textContent = viewTurn !== null ? `回看第 ${viewTurn} / ${game.turn} 手 · 只读；点击“从此处继续”回退并进入手动研究。` : '当前局面尚无匹配分析，叠加暂不显示。按 A 分析，或点击“定位局面”查看上次搜索。';
  renderStatus();
  const history = $('history');
  const scrollTop = history.scrollTop;
  history.replaceChildren();
  (game?.moves || []).forEach((action, i) => {
    const item = document.createElement('button');
    item.className = 'history-item';
    item.classList.toggle('active', i + 1 === (viewTurn ?? game.turn));
    item.dataset.turn = i + 1;
    item.title = `查看第 ${i + 1} 手后的局面${i < (opening?.moves.length || 0) ? ' · 开局生成' : ''}`;
    item.innerHTML = `<small>${i + 1}</small><i class="tiny-stone ${i % 2 === 0 ? 'black' : 'white'}"></i>${coordinate(action, game.board_size)}`;
    history.append(item);
  });
  history.scrollTop = scrollTop;
  const nextHistoryKey = `${state.game_id}:${game?.turn}:${viewTurn}`;
  if (historyKey !== nextHistoryKey) {
    const active = history.querySelector('.active');
    if (active) {
      const row = active.getBoundingClientRect(), container = history.getBoundingClientRect();
      if (row.bottom > container.bottom) history.scrollTop += row.bottom - container.bottom;
      else if (row.top < container.top) history.scrollTop += row.top - container.top;
    }
    historyKey = nextHistoryKey;
  }
  $('history-count').textContent = `${viewTurn ?? game?.turn ?? 0} / ${game?.turn || 0} 手`;
  $('timeline').max = game?.turn || 0;
  $('timeline').value = viewTurn ?? game?.turn ?? 0;
  $('timeline').style.setProperty('--range-progress', `${game?.turn ? Number($('timeline').value) / game.turn * 100 : 0}%`);
  $('timeline').disabled = !game?.turn;
  $('first').disabled = $('prev').disabled = !game || (viewTurn ?? game.turn) === 0;
  $('next').disabled = $('live').disabled = viewTurn === null;
  $('branch').hidden = viewTurn === null;
  $('branch').disabled = busy;
  const analysis_config = {...(state.analysis_config || selectedRun()?.analysis_config || catalog.analysis_config), visits: Number($('visits').value), reuse_tree: false};
  if (game) analysis_config.visits = state.visits;
  if (game) { analysis_config.board_size = game.board_size; analysis_config.rule = state.rule; }
  if (analysis_config.inference_precision === 'auto') analysis_config.inference_precision = analysis_config.device.startsWith('cuda:') ? 'float16' : 'float32';
  $('engine-config').textContent = JSON.stringify(analysis_config, null, 2);
  $('opening-config').textContent = JSON.stringify({parameters: state.opening_config || selectedRun()?.opening,
    ...(opening ? {generated: opening} : {})}, null, 2);
  updateEngineInfo();
  renderSettingsNote();
  renderAnalysis();
}
function renderSettingsNote() {
  document.querySelectorAll('[data-visits]').forEach(button => {
    button.setAttribute('aria-pressed', String(Number(button.dataset.visits) === Number($('visits').value)));
  });
  const edited = state?.game && (Number($('visits').value) !== state.visits || $('mode').value !== state.mode ||
    Number(document.querySelector('input[name="human"]:checked').value) !== state.human);
  $('apply-settings').classList.toggle('settings-dirty', Boolean(edited));
  $('settings-note').textContent = edited ? '模式 / 执子 / 预算尚未应用；分析使用当前会话配置。模型、棋盘、棋规和开局需重开生效。' : '模型、棋盘、棋规和开局在创建棋局时生效。';
}
function renderStatus() {
  if (!state) return;
  const game = state.game;
  let text = '选择模型，创建棋局';
  if (!online) text = '等待服务连接';
  else if (state.busy || pending) {
    const elapsed = state.started_at ? Math.max(0, Date.now() / 1000 - state.started_at).toFixed(1) : '0.0';
    text = `${state.phase === 'loading' ? '加载中' : state.phase === 'opening' ? '生成平衡开局' : '处理中'} · ${elapsed}s`;
  } else if (state.error) text = '操作失败 · 查看错误信息后重试';
  else if (game?.finished) text = game.winner === 0 ? '本局和棋' : `${game.winner === 1 ? '黑棋' : '白棋'}获胜`;
  else if (game) text = state.mode === 'manual' ? `${game.player === 1 ? '黑' : '白'}方落子 · 手动研究` : game.player === state.human ? '轮到你落子' : '等待 AI 落子';
  if (game?.finished && game.reason === 2) text += ' · 黑棋禁手';
  $('status').textContent = text;
  $('status').parentElement.classList.toggle('thinking', state.busy || pending);
}
function selectedRun() { return catalog?.runs.find(run => run.id === $('run').value); }
function populateRuns(selected = $('run').value || catalog.default_run) {
  $('run').replaceChildren(...catalog.runs.map(run => new Option(`${run.label} · ${run.algorithm === 'muzero' ? 'MuZero' : 'AlphaZero'}${run.models.length ? '' : ' · 暂无权重'}`, run.id)));
  if (catalog.runs.some(run => run.id === selected)) $('run').value = selected;
  else if (catalog.default_run) $('run').value = catalog.default_run;
}
function populateModels(selected = $('model').value) {
  const run = selectedRun();
  const models = (run?.models || []).map(id => catalog.models.find(m => m.id === id));
  const options = models.map(m => {
    const duplicate = models.some(other => other.id !== m.id && other.iteration === m.iteration);
    const detail = duplicate ? ` · ${m.manifest.id}` : '';
    return new Option(`第 ${m.iteration} 代${m.id === run.current_model ? ' · 当前发布' : ''}${detail}`, m.id);
  });
  $('model').replaceChildren(...(options.length ? options : [new Option('暂无已发布权重', '')]));
  $('model').value = models.some(m => m.id === selected) ? selected : run?.default_model || '';
  $('model-count').textContent = `${models.length} 代`;
  updateSizes();
}
function updateSizes() {
  const model = catalog.models.find(m => m.id === $('model').value);
  for (const option of $('size').options) option.disabled = !model || Number(option.value) > model.canvas;
  if (model && Number($('size').value) > model.canvas) $('size').value = model.canvas;
  $('new-game').disabled = !model || !online || pending || Boolean(state?.busy);
  $('model-summary').textContent = model ? `${model.manifest.algorithm === 'muzero' ? 'MuZero' : 'AlphaZero'} · 第 ${model.iteration} 代 · ${model.manifest.weights || '—'} · 画布 ${model.canvas}² · 累计 ${(model.manifest.checkpoint.total_steps ?? 0).toLocaleString()} 步` : '此配置目录暂无已发布权重';
  $('model-info').textContent = model ? JSON.stringify({file: model.path, ...model.manifest}, null, 2) : '';
}
async function loadCatalog() {
  const selected = $('model').value;
  const run = $('run').value;
  const size = $('size').value;
  const first = !catalog;
  catalog = await api('/api/catalog');
  $('size').replaceChildren();
  for (let value = 5; value <= Math.max(catalog.default_size, ...catalog.models.map(m => m.canvas)); value++) $('size').add(new Option(`${value} × ${value}`, value));
  $('size').value = size || catalog.default_size;
  if (first) {
    $('visits').value = catalog.default_visits;
    $('rule').replaceChildren(...catalog.rules.map(rule => new Option(rules[rule], rule)));
    $('rule').value = catalog.default_rule;
  }
  populateRuns(run || catalog.models.find(m => m.id === state?.model)?.run || catalog.default_run);
  populateModels(selected);
  if (first) applyRunDefaults();
  $('version').textContent = catalog.version.replace('EtaZero_', '');
  updateEngineInfo();
}
function updateEngineInfo() {
  const c = state?.analysis_config || selectedRun()?.analysis_config || catalog.analysis_config;
  $('engine-info').textContent = `${c.device} · ${c.search_threads} search threads · VL ${c.virtual_loss}`;
}
function applyRunDefaults() {
  const c = selectedRun()?.analysis_config || catalog.analysis_config;
  $('size').value = c.board_size;
  $('rule').value = c.rule;
  $('visits').value = c.visits;
  updateSizes();
}
function sessionSettings() {
  return {human: Number(document.querySelector('input[name="human"]:checked').value), visits: Number($('visits').value), mode: $('mode').value};
}
$('settings').addEventListener('submit', event => {
  event.preventDefault();
  command('new', {model: $('model').value, size: Number($('size').value), rule: $('rule').value, opening: $('opening').value, ...sessionSettings()});
});
$('settings').addEventListener('input', renderSettingsNote);
$('apply-settings').addEventListener('click', () => {
  if (!$('visits').reportValidity()) return;
  command('configure', sessionSettings());
});
$('board').addEventListener('click', event => {
  const button = event.target.closest('button[data-action]');
  if (button && !button.disabled) command('play', {action: Number(button.dataset.action)});
});
for (const name of ['undo', 'retry', 'analyze', 'step']) $(name).addEventListener('click', () => command(name));
$('refresh-models').addEventListener('click', () => command('refresh'));
$('branch').addEventListener('click', () => { if (viewTurn !== null) command('branch', {turn: viewTurn}); });
$('numbers').addEventListener('change', () => { $('board').classList.toggle('hide-numbers', !$('numbers').checked); savePreferences(); });
$('search-map-kind').addEventListener('change', renderHeatmaps);
$('network-kind').addEventListener('change', renderNetworkMaps);
$('network-common-scale').addEventListener('change', renderNetworkMaps);
$('visited-only').addEventListener('change', renderCandidates);
$('model').addEventListener('change', updateSizes);
$('run').addEventListener('change', () => { populateModels(); applyRunDefaults(); updateEngineInfo(); render(); });
for (const id of ['rule', 'size']) $(id).addEventListener('change', () => { if (!state?.game) render(); });
document.querySelectorAll('[data-visits]').forEach(button => button.addEventListener('click', () => { $('visits').value = button.dataset.visits; renderSettingsNote(); }));
$('first').addEventListener('click', () => navigate(0));
$('prev').addEventListener('click', () => navigate((viewTurn ?? state.game.turn) - 1));
$('next').addEventListener('click', () => navigate((viewTurn ?? state.game.turn) + 1));
$('live').addEventListener('click', () => navigate(state.game.turn));
$('timeline').addEventListener('input', () => navigate(Number($('timeline').value)));
$('history').addEventListener('click', event => { const button = event.target.closest('[data-turn]'); if (button) navigate(Number(button.dataset.turn)); });
$('show-analysis').addEventListener('click', () => { if (state.analysis) navigate(state.analysis.turn); });
$('overlay').addEventListener('change', () => {
  if ($('overlay').value !== 'none' && state?.analysis) navigate(state.analysis.turn);
  else render();
});
function selectCandidate(action) {
  if (!state?.analysis) return;
  navigate(state.analysis.turn);
  highlight = action;
  render();
}
$('candidates').addEventListener('click', event => { const button = event.target.closest('[data-candidate]'); if (button) selectCandidate(Number(button.dataset.candidate)); });
for (const eventName of ['mouseover', 'focusin']) {
  $('board').addEventListener(eventName, event => {
    const point = event.target.closest('[data-action]');
    if (point) $('board-coordinate').textContent = coordinate(Number(point.dataset.action), displayedGame()?.board_size || Number($('size').value));
  });
  $('heatmap-data').addEventListener(eventName, event => {
    const cell = event.target.closest('[data-detail]');
    if (cell) $('heatmap-detail').textContent = cell.dataset.detail;
  });
  $('network-data').addEventListener(eventName, event => {
    const cell = event.target.closest('[data-detail]');
    if (cell) $('network-detail').textContent = cell.dataset.detail;
  });
}
$('network-data').addEventListener('click', event => { const cell = event.target.closest('[data-action]'); if (cell) selectCandidate(Number(cell.dataset.action)); });
$('heatmap-data').addEventListener('click', event => { const cell = event.target.closest('[data-action]'); if (cell) selectCandidate(Number(cell.dataset.action)); });
document.querySelectorAll('[data-sort]').forEach(button => button.addEventListener('click', () => {
  sortDirection = sortField === button.dataset.sort ? -sortDirection : button.dataset.sort === 'action' ? 1 : -1;
  sortField = button.dataset.sort;
  renderCandidates();
}));
const tabs = [...document.querySelectorAll('[data-tab]')];
function selectTab(button) {
  tabs.forEach(tab => {
    const selected = tab === button;
    tab.setAttribute('aria-selected', String(selected));
    tab.tabIndex = selected ? 0 : -1;
    $(tab.dataset.tab).hidden = !selected;
  });
}
tabs.forEach((button, index) => {
  button.addEventListener('click', () => selectTab(button));
  button.addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault(); event.stopPropagation();
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    selectTab(tabs[next]); tabs[next].focus();
  });
});
document.addEventListener('keydown', event => {
  if (event.ctrlKey || event.metaKey || event.altKey || event.repeat || event.target.closest('input,select,textarea,button,[contenteditable="true"]')) return;
  const shortcuts = {a: 'analyze', s: 'step', u: 'undo', ArrowLeft: 'prev', ArrowRight: 'next', Home: 'first', End: 'live'};
  const id = shortcuts[event.key] || shortcuts[event.key.toLowerCase()];
  if (id && !$(id).disabled && !$(id).hidden) { event.preventDefault(); $(id).click(); }
});
function savePreferences() {
  try { localStorage.setItem('etazero-ui', JSON.stringify({theme: document.documentElement.dataset.themePreference, numbers: $('numbers').checked})); } catch { /* Browser storage is optional. */ }
}
try {
  const preferences = JSON.parse(localStorage.getItem('etazero-ui') || '{}');
  $('numbers').checked = preferences.numbers !== false;
} catch { /* Use defaults when storage is unavailable. */ }
setInterval(renderStatus, 200);
const compactLayout = window.matchMedia('(max-width: 1150px)');
function updateSetupLayout() { $('setup-details').open = !compactLayout.matches; }
compactLayout.addEventListener('change', updateSetupLayout);
updateSetupLayout();
async function start() {
  for (;;) {
    try { await loadCatalog(); break; }
    catch (error) { connection(false); $('error').textContent = error.message; $('error').hidden = false; await new Promise(resolve => setTimeout(resolve, 1500)); }
  }
  for (;;) {
    try {
      const next = await api(`/api/state${state && online ? `?since=${state.version}` : ''}`);
      const changed = !online || !state || next.instance !== state.instance || next.version !== state.version;
      if (!state || next.instance !== state.instance || next.catalog_revision !== catalogRevision) {
        await loadCatalog();
        catalogRevision = next.catalog_revision;
      }
      connection(true);
      if (changed) accept(next);
    } catch (error) {
      connection(false); render();
      await new Promise(resolve => setTimeout(resolve, 1500));
    }
  }
}
start();
