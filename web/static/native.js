const $ = id => document.getElementById(id);
const columns = 'ABCDEFGHJKLMNOPQRSTUVWXYZ';
const gameNames = {
  go: '围棋',
  freestyle: 'Freestyle',
  standard: 'Standard',
  renju: 'Renju',
  hex: 'Hex',
  xiangqi: '中国象棋',
  chinese: '围棋 · 中国规则',
  japanese: '围棋 · 日本规则',
  aga: '围棋 · AGA',
  new_zealand: '围棋 · 新西兰规则',
  'tromp-taylor': '围棋 · Tromp–Taylor',
};
const ruleNames = {
  default: '使用模型配置',
  freestyle: '五子棋 · Freestyle',
  standard: '五子棋 · Standard',
  renju: '五子棋 · Renju',
  hex: 'Hex',
  go: '围棋 · 配置棋规',
  xiangqi: '中国象棋',
  chinese: '围棋 · 中国规则',
  japanese: '围棋 · 日本规则',
  aga: '围棋 · AGA',
  new_zealand: '围棋 · 新西兰规则',
  'tromp-taylor': '围棋 · Tromp–Taylor',
};
const pieces = {
  1: ['', '車', '仕', '炮', '兵', '馬', '相', '帥'],
  '-1': ['', '車', '士', '砲', '卒', '馬', '象', '將'],
};
const phaseNames = {
  idle: '就绪',
  loading: '加载中',
  opening: '生成平衡开局中',
  working: '更新局面',
  diagnosing: '读取网络输出',
  thinking: 'AI 思考中',
  analyzing: '持续分析',
  stopping: '正在停止',
  error: '操作失败',
};

let state = null, catalog = null, pending = false, online = false, notice = '';
let viewTurn = null, selectedFrom = null, highlight = null, previewMove = null, catalogRevision = -1;
let loadingCatalog = false, boardKey = '', historyKey = '';
let setupInitialized = false, setupRevision = 0, setupSent = 0, setupFlight = null, openingQueued = false, syncingSetup = false;
let boardSizes = null, setupSyncKey = '';
let lastAnimatedKey = '';

const esc = text => String(text).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const number = (v, digits = 3) => Number.isFinite(v) ? v.toFixed(digits) : '—';
const percent = v => Number.isFinite(v) ? (100 * v).toFixed(1) : '—';
const side = (player, game = state?.game?.game) => game === 'xiangqi' ? (player === 'B' ? '红方' : '黑方') : (player === 'B' ? '黑方' : '白方');
const sideStoneClass = (player, game = state?.game?.game) => game === 'xiangqi' ? (player === 'B' ? 'red' : 'black') : (player === 'B' ? 'black' : 'white');
const coordinate = (x, y, g) => g.game === 'xiangqi' ? `${'abcdefghi'[x]}${y}` : `${columns[x]}${g.boardYSize - y}`;
const over = () => Boolean(state?.game?.isGameFinished || state?.resignation);
const interactive = () => online && !pending && !state?.busy;
const isMuZero = () => state?.models?.[0]?.isMuZero ??
  Boolean(catalog?.runs.find(run => run.models.some(m => m.id === state?.selected_model?.id))
    ?.algorithm?.toLowerCase().includes('muzero'));

function compactVisits(v) {
  if (!Number.isFinite(v)) return '—';
  if (v >= 1e6) return (v / 1e6).toFixed(1) + 'm';
  if (v >= 10000) return (v / 1000).toFixed(0) + 'k';
  if (v >= 1000) return (v / 1000).toFixed(1) + 'k';
  return String(v);
}

function formatGoRules(r) {
  if (!r || typeof r !== 'object') return '原生围棋规则。';
  const parts = [];
  if (r.komi !== undefined) parts.push(`贴目 ${r.komi}`);
  if (r.scoring) parts.push({AREA: '面积计分', TERRITORY: '数目计分'}[r.scoring] || r.scoring);
  if (r.ko) parts.push({SIMPLE: '简单劫', POSITIONAL: '位置超级劫', SITUATIONAL: '局面超级劫'}[r.ko] || r.ko);
  if (typeof r.suicide === 'boolean') parts.push(r.suicide ? '允许自杀' : '禁止自杀');
  if (r.tax && r.tax !== 'NONE') parts.push(`税目 ${r.tax}`);
  if (r.hasButton) parts.push('Button');
  return `原生棋规：${parts.join(' · ')}`;
}

function samePosition(a, b) {
  return a && b && a.game === b.game && a.boardXSize === b.boardXSize && a.boardYSize === b.boardYSize &&
    a.turnNumber === b.turnNumber && a.currentPlayer === b.currentPlayer && a.board.every((v, i) => v === b.board[i]) && a.fen === b.fen;
}

function displayedGame() {
  if (!state?.game) return null;
  return viewTurn === null ? state.game : {...state.history[viewTurn], moves: state.game.moves.slice(0, viewTurn)};
}

function canPlay() {
  return interactive() && state?.connected && state.game && !over() && viewTurn === null &&
    (state.mode === 'manual' || state.game.currentPlayer === state.human);
}

function savePreferences() {
  try {
    localStorage.setItem('etazero-ui', JSON.stringify({theme: $('theme').value, numbers: $('numbers').checked}));
  } catch {}
}

async function api(path, body) {
  const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch(path, {
      signal: controller.signal,
      ...(body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)}),
    });
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
    return value;
  } finally {
    clearTimeout(timer);
  }
}

function accept(next) {
  if (state && next.instance === state.instance && next.version < state.version) return;
  const previous = state;
  if (!state || next.instance !== state.instance || next.game_id !== state.game_id ||
      JSON.stringify(next.game?.moves) !== JSON.stringify(state.game?.moves)) {
    viewTurn = null;
    selectedFrom = null;
    highlight = null;
    previewMove = null;
  }
  state = next;
  if (setupFlight?.accepted && !state.busy) {
    const flight = setupFlight;
    setupFlight = null;
    if (state.error && flight.revision === setupRevision && state.ui_setup) hydrateSetup(state.ui_setup);
    if (state.ui_setup) setupSyncKey = JSON.stringify(state.ui_setup);
  }
  if (catalogRevision !== state.catalog_revision) loadCatalog();
  if (setupInitialized && !setupFlight && setupRevision === setupSent && state.ui_setup &&
      setupSyncKey !== JSON.stringify(state.ui_setup)) hydrateSetup(state.ui_setup);
  if (setupInitialized && !setupFlight && setupRevision !== setupSent && previous?.ui_setup && state.ui_setup &&
      previous.game_id !== state.game_id && selectedModel()?.id === state.ui_setup.model) {
    // A position import can finish while a mode or budget change is queued.
    hydratePosition(state.ui_setup, previous.ui_setup);
    updateSetup();
  }
  render();
  queueMicrotask(drainSetup);
}

async function command(operation, payload = {}) {
  if (!state || pending || !online || (state.busy && operation !== 'stop')) return;
  pending = true;
  notice = '';
  render();
  try {
    accept(await api('/api/' + operation, {version: state.version, game_id: state.game_id, ...payload}));
  } catch (error) {
    notice = error.message;
    try { accept(await api('/api/state')); } catch {}
  } finally {
    pending = false;
    render();
    queueMicrotask(drainSetup);
  }
}

function selectedRun() { return catalog?.runs.find(run => run.id === $('run').value); }
function selectedModel() { return selectedRun()?.models.find(model => model.id === $('model').value); }

function populateModels(preferred) {
  const run = selectedRun();
  $('model').replaceChildren();
  for (const model of run?.models || []) {
    const label = model.iteration === undefined ? model.path.split('/').at(-1) :
      `Iteration ${model.iteration} · ${model.training_seconds == null ? '训练时间未知' : `训练 ${(model.training_seconds / 3600).toFixed(2)} h`}`;
    $('model').add(new Option(label, model.id));
  }
  if (run?.models.some(m => m.id === preferred)) $('model').value = preferred;
  else if (run?.current_model) $('model').value = run.current_model;
  if (!run?.models.length) $('model').add(new Option('暂无发布模型', ''));
  updateRules();
}

async function loadCatalog() {
  if (loadingCatalog) return;
  loadingCatalog = true;
  const revision = state?.catalog_revision ?? 0;
  try {
    const incoming = await api('/api/catalog');
    const previousRun = $('run').value, previousModel = $('model').value;
    catalog = incoming;
    catalogRevision = revision;
    $('version').textContent = catalog.version;
    $('run').replaceChildren();
    for (const run of catalog.runs) $('run').add(new Option(run.label, run.id));
    const preferred = previousModel || state?.selected_model?.id || catalog.default_model;
    const run = catalog.runs.find(r => r.id === previousRun) ||
      catalog.runs.find(r => r.models.some(m => m.id === preferred)) ||
      catalog.runs.find(r => r.models.length && r.config) ||
      catalog.runs[0];
    if (run) $('run').value = run.id;
    populateModels(preferred);
    if (!setupInitialized) {
      setupInitialized = true;
      if (state?.ui_setup) hydrateSetup(state.ui_setup);
      else if (state?.game) {
        $('mode').value = state.mode;
        $('human').value = state.human;
        if ([...$('rules').options].some(o => o.value === state.game.game)) $('rules').value = state.game.game;
        setBoardSize(boardSizeText([state.game.boardXSize, state.game.boardYSize]), true);
        const params = state.parameters || {}, cap = 1125899906842624;
        const kind = params.maxPlayouts < cap ? 'visits' : params.maxVisits < cap ? 'total' : params.maxTime < 1e12 ? 'time' : 'visits';
        $('budget-type').value = kind;
        $('budget-type').dataset.previous = kind;
        $('budget').value = kind === 'visits' && params.maxPlayouts < cap ? params.maxPlayouts :
          kind === 'total' ? params.maxVisits : kind === 'time' ? params.maxTime : 500;
        $('ponder').checked = params.ponderingEnabled === true || params.ponderingEnabled === 'true';
        updateSetup();
      } else requestSetup();
    } else if (state?.ui_setup && !setupFlight && setupRevision === setupSent &&
               selectedModel()?.id === state.ui_setup.model) hydrateSetup(state.ui_setup);
  } catch (error) {
    notice = error.message;
    render();
  } finally {
    loadingCatalog = false;
    queueMicrotask(drainSetup);
  }
}

function updateRules() {
  const game = selectedModel()?.game || selectedRun()?.game;
  const choices = ['freestyle', 'standard', 'renju'].includes(game) ? ['freestyle', 'standard', 'renju'] :
    game === 'go' ? ['go', 'chinese', 'japanese', 'aga', 'new_zealand', 'tromp-taylor'] :
    game && ruleNames[game] ? [game] :
    ['default', 'freestyle', 'standard', 'renju', 'hex', 'go', 'chinese', 'japanese', 'aga', 'new_zealand', 'tromp-taylor', 'xiangqi'];
  const previous = $('rules').value;
  const activeRule = state?.ui_setup?.rules;
  if (state?.game?.game === 'go' && (!game || game === 'go') && activeRule && !choices.includes(activeRule))
    choices.push(activeRule);
  $('rules').replaceChildren(...choices.map(value => new Option(ruleNames[value] || formatGoRules(state.game.rules), value)));
  $('rules').value = choices.includes(previous) ? previous : choices.includes(game) ? game : choices[0];
  updateBoardSizes();
  updateSetup();
}

function boardSizeText([width, height]) {
  return width === height ? String(width) : `${width}×${height}`;
}

function setBoardSize(size, includeCurrent = false) {
  if (boardSizes?.length) {
    if (includeCurrent && size && !boardSizes.includes(size)) {
      boardSizes.push(size);
      $('board-size-range').max = String(boardSizes.length - 1);
    }
    let index = boardSizes.indexOf(size);
    if (index < 0) index = boardSizes.indexOf(String(selectedRun()?.canvas));
    if (index < 0) index = boardSizes.length - 1;
    $('board-size-range').value = String(index);
  } else $('board-size').value = size || '';
}

function updateBoardSizes() {
  const old = selectedBoardSize();
  const sizes = selectedRun()?.trained_board_sizes;
  boardSizes = Array.isArray(sizes) && sizes.length ? sizes.map(String) : null;
  $('board-size-range').hidden = !boardSizes;
  $('board-size').hidden = Boolean(boardSizes);
  if (boardSizes) {
    $('board-size-range').min = '0';
    $('board-size-range').max = String(boardSizes.length - 1);
    const game = selectedModel()?.game || selectedRun()?.game;
    const keepCurrent = state?.ui_setup?.size && (!game || game === state.game.game) &&
      old === boardSizeText(state.ui_setup.size);
    setBoardSize(old, keepCurrent);
    $('board-size-note').textContent = '可选尺寸来自训练配置。';
  } else {
    $('board-size').value = old == null ? '' : String(old);
    $('board-size-value').textContent = '';
    $('board-size-note').textContent = Array.isArray(selectedRun()?.trained_board_sizes)
      ? '训练配置没有方形尺寸，可手动输入。' : '训练尺寸未知，可手动输入。';
  }
}

function selectedBoardSize() {
  return boardSizes?.length ? boardSizes[Number($('board-size-range').value)] :
    $('board-size').value.trim();
}

function renderBudgetPresets() {
  const type = $('budget-type').value;
  const current = Number($('budget').value);
  const presets = type === 'time' ? [1, 3, 10, 30] : [100, 500, 1600, 5000];
  const container = $('budget-presets');
  if (!container) return;
  container.innerHTML = presets.map(v =>
    `<button type="button" data-preset="${v}" aria-pressed="${current === v}">${type === 'time' ? `${v}s` : v}</button>`
  ).join('');
}

function updateSetup() {
  const play = $('mode').value === 'play';
  const rule = $('rules').value;
  const xi = rule === 'xiangqi';
  const go = ['go', 'chinese', 'japanese', 'aga', 'new_zealand', 'tromp-taylor'].includes(rule) ||
    (rule === state?.ui_setup?.rules && state?.game?.game === 'go');
  const type = $('budget-type').value;
  $('play-settings').hidden = !play;
  $('budget').disabled = !play;
  $('ponder-note').hidden = !$('ponder').checked;
  $('board-size').hidden = xi || Boolean(boardSizes);
  $('board-size-range').hidden = xi || !boardSizes;
  $('board-size').disabled = xi;
  $('board-size-range').disabled = xi || boardSizes?.length === 1;
  if (xi) {
    $('board-size-value').textContent = '9 × 10';
    $('board-size-note').textContent = '中国象棋使用固定棋盘。';
  } else if (!boardSizes) {
    $('board-size-value').textContent = '';
    $('board-size-note').textContent = Array.isArray(selectedRun()?.trained_board_sizes)
      ? '训练配置没有方形尺寸，可手动输入。' : '训练尺寸未知，可手动输入。';
  } else {
    const size = selectedBoardSize();
    $('board-size-value').textContent = size.includes('×') ? size.replace('×', ' × ') : `${size} × ${size}`;
    $('board-size-note').textContent = selectedRun().trained_board_sizes.includes(Number(size))
      ? '可选尺寸来自训练配置。' : '当前尺寸来自已载入的局面。';
  }
  $('budget-label').textContent = type === 'time' ? '思考时间（秒）' : type === 'total' ? '总 visits' : '新增 visits';
  $('budget').min = type === 'time' ? '0.001' : '1';
  $('budget').step = type === 'time' ? 'any' : '1';
  renderBudgetPresets();
  if ($('import-fen-field')) $('import-fen-field').hidden = !xi;
  if ($('import-komi-field')) $('import-komi-field').hidden = !go;
  $('opening-hint').textContent = ['freestyle', 'standard', 'renju', 'hex'].includes(rule)
    ? '可按当前设置重新开局或生成平衡开局。' : '平衡开局仅支持五子棋和 Hex。';
  render();
}

function settings() {
  const play = $('mode').value === 'play', type = $('budget-type').value, value = Number($('budget').value);
  if (play && (!$('budget').value || !Number.isFinite(value) || value <= 0 || (type !== 'time' && !Number.isSafeInteger(value))))
    throw new Error(type === 'time' ? '思考时间必须大于 0 秒' : 'visits 必须是正整数');
  const cap = 1125899906842624;
  return {
    mode: $('mode').value,
    human: $('human').value,
    budget_type: type,
    budget_value: Number.isFinite(value) && value > 0 ? value : 500,
    ponder: $('ponder').checked,
    parameters: play ? {
      ponderingEnabled: $('ponder').checked,
      maxVisits: type === 'total' ? value : cap,
      maxPlayouts: type === 'visits' ? value : cap,
      maxTime: type === 'time' ? value : 1e20,
    } : {ponderingEnabled: false},
  };
}

function setupPayload(opening = false) {
  const rule = $('rules').value, text = rule === 'xiangqi' ? '' : selectedBoardSize();
  let size = null;
  if (text) {
    const match = /^(\d+)(?:\s*[x×]\s*(\d+))?$/i.exec(text);
    if (match) size = [Number(match[1]), Number(match[2] || match[1])];
    if (!size || size.some(n => !Number.isSafeInteger(n) || n < 2))
      throw new Error('棋盘大小请输入大于 1 的边长，或宽 × 高');
  }
  const result = {
    model: $('model').value,
    ...settings(),
    size,
    rules: ['go', 'default'].includes(rule) ? null : rule,
  };
  if (opening) result.opening = {};
  return result;
}

function newGamePayload(extra = {}) {
  const base = setupPayload(false);
  const payload = {
    model: base.model,
    mode: base.mode,
    human: base.human,
    parameters: base.parameters,
  };
  if (!extra.sgf) {
    if (base.size !== null) payload.size = base.size;
    if (base.rules !== null) payload.rules = base.rules;
  }
  return {...payload, ...extra};
}

function hydratePosition(setup, previous = null) {
  const defaultRule = selectedRun()?.game === 'go' ? 'go' : 'default';
  const rule = setup.rules || defaultRule;
  if (!previous || $('rules').value === (previous.rules || defaultRule)) {
    if (![...$('rules').options].some(option => option.value === rule) && state.game?.game === 'go')
      $('rules').add(new Option(ruleNames[rule] || formatGoRules(state.game.rules), rule));
    if ([...$('rules').options].some(option => option.value === rule)) $('rules').value = rule;
  }
  if (setup.size && (!previous || selectedBoardSize() === (previous.size ? boardSizeText(previous.size) : '')))
    setBoardSize(boardSizeText(setup.size), true);
}

function hydrateSetup(setup) {
  if (!catalog || !setup) return;
  syncingSetup = true;
  const run = catalog.runs.find(r => r.models.some(m => m.id === setup.model));
  if (run) { $('run').value = run.id; populateModels(setup.model); }
  $('mode').value = setup.mode;
  $('human').value = setup.human;
  $('budget-type').value = setup.budget_type;
  $('budget-type').dataset.previous = setup.budget_type;
  $('budget').value = setup.budget_value;
  $('ponder').checked = setup.ponder;
  hydratePosition(setup);
  setupSyncKey = JSON.stringify(setup);
  updateSetup();
  syncingSetup = false;
}

function requestSetup(opening = false) {
  if (syncingSetup) return;
  if (!opening && !setupFlight && setupRevision === setupSent && state?.ui_setup) {
    try {
      const candidate = setupPayload();
      const keys = ['model', 'size', 'rules', 'mode', 'human', 'budget_type', 'budget_value', 'ponder'];
      if (keys.every(key => JSON.stringify(candidate[key]) === JSON.stringify(state.ui_setup[key]))) return;
    } catch {}
  }
  setupRevision++;
  if (opening) openingQueued = true;
  queueMicrotask(drainSetup);
}

async function drainSetup() {
  if (!setupInitialized || !state || !online || !selectedModel() || !selectedRun()?.config || pending || setupRevision === setupSent) return;
  if (setupFlight) {
    if (state.busy && ['thinking', 'opening'].includes(state.phase) && !state.stop_requested) command('stop');
    return;
  }
  if (state.busy) {
    if (['thinking', 'opening'].includes(state.phase) && !state.stop_requested) command('stop');
    return;
  }
  let payload;
  try { payload = setupPayload(openingQueued); }
  catch (error) { notice = error.message; setupSent = setupRevision; render(); return; }
  const flight = {revision: setupRevision, accepted: false};
  setupFlight = flight;
  setupSent = setupRevision;
  openingQueued = false;
  pending = true;
  notice = '';
  render();
  try {
    const next = await api('/api/apply', {version: state.version, game_id: state.game_id, ...payload});
    flight.accepted = true;
    accept(next);
  } catch (error) {
    setupFlight = null;
    notice = error.message;
    try { accept(await api('/api/state')); } catch {}
    if (flight.revision === setupRevision && state?.ui_setup) hydrateSetup(state.ui_setup);
  } finally {
    pending = false;
    render();
    queueMicrotask(drainSetup);
  }
}

function view(turn) {
  if (!state?.game) return;
  turn = Math.max(0, Math.min(state.history.length - 1, turn));
  viewTurn = turn === state.history.length - 1 ? null : turn;
  selectedFrom = null;
  highlight = null;
  previewMove = null;
  render();
}

function candidates() {
  const report = state?.analysis?.report;
  if (!report) return [];
  const minimum = Number($('candidate-min').value) || 0;
  const count = Math.max(1, Number($('candidate-top').value) || 10);
  return [...report.moveInfos].sort((a, b) => a.order - b.order)
    .filter(row => row.visits >= report.rootInfo.visits * minimum / 100)
    .slice(0, count);
}

function renderCandidateDetail(row, fallbackText = '') {
  const el = $('candidate-detail');
  if (!row) {
    el.textContent = fallbackText || (isMuZero() ? '悬停或点击候选着法，查看 MuZero 的潜空间主变化（PV）。' :
      '悬停或点击候选着法，在棋盘预览主变化（PV）。');
    return;
  }
  const pv = row.pv || [];
  el.innerHTML =
    `<div class="pv-summary">` +
    `<span><b>${esc(row.move)}</b></span>` +
    `<span>得分 <b>${percent(row.winrate)}%</b></span>` +
    `<span>访问 <b>${row.visits}</b></span>` +
    `<span>边访问 ${row.edgeVisits ?? '—'}</span>` +
    `<span>先验 ${percent(row.prior)}%</span>` +
    `<span>D ${percent(row.drawValue)}%</span>` +
    `<span>NR ${percent(row.noResultValue)}%</span>` +
    `<span>LCB ${number(row.lcb)}</span>` +
    `<span>效用 ${number(row.utility)}</span>` +
    `<span>选点值 ${number(row.playSelectionValue)}</span>` +
    `</div>` +
    `<div class="pv-line"><span class="pv-tag">${isMuZero() ? '潜空间 PV' : 'PV'}</span> ` +
    (pv.length ? pv.map(m => `<span class="pv-move">${esc(m)}</span>`).join(' ') : '<span class="muted">—</span>') +
    `</div>`;
}

function selectCandidate(move) {
  if (!state?.analysis) return;
  const ply = state.analysis.ply;
  viewTurn = ply === state.history.length - 1 ? null : ply;
  highlight = move;
  selectedFrom = null;
  const row = state.analysis.report?.moveInfos.find(r => r.move === move);
  renderCandidateDetail(row, move);
  renderBoard();
  renderHistory();
  renderAnalysisSelection();
}

function renderAnalysisSelection() {
  $('candidates').querySelectorAll('tr').forEach(tr => {
    tr.classList.toggle('selected', tr.dataset.move === highlight);
  });
}

function setPreview(move) {
  if (previewMove === move) return;
  previewMove = move;
  const row = move ? state?.analysis?.report?.moveInfos.find(r => r.move === move) : null;
  if (row) renderCandidateDetail(row);
  else if (highlight) renderCandidateDetail(state?.analysis?.report?.moveInfos.find(r => r.move === highlight), highlight);
  renderBoard();
}

function starPointsFor(nx, ny) {
  if (nx !== ny || nx < 5) return [];
  const mid = (nx - 1) / 2;
  const pts = [];
  if (nx === 5 || nx === 7) return [[mid, mid]];
  if (nx === 9) return [[2, 2], [6, 2], [4, 4], [2, 6], [6, 6]];
  const edge = nx >= 13 ? 3 : 2;
  const coords = nx % 2 === 1 ? [edge, mid, nx - 1 - edge] : [edge, nx - 1 - edge];
  for (const x of coords) {
    for (const y of coords) {
      if (nx === 19 || x === y || x + y === nx - 1 || (x === mid && y === mid)) pts.push([x, y]);
    }
  }
  return pts;
}

function xiangqiCrosshair(x, y, point) {
  const [cx, cy] = point(x, y);
  const g = 1.0, l = 2.0;
  let d = '';
  if (x > 0) {
    d += `M${cx - g - l},${cy - g}H${cx - g}V${cy - g - l}`;
    d += `M${cx - g - l},${cy + g}H${cx - g}V${cy + g + l}`;
  }
  if (x < 8) {
    d += `M${cx + g + l},${cy - g}H${cx + g}V${cy - g - l}`;
    d += `M${cx + g + l},${cy + g}H${cx + g}V${cy + g + l}`;
  }
  return `<path d="${d}"/>`;
}

function animateXiangqiMove(g) {
  if (g.game !== 'xiangqi' || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const moves = g.moves || [];
  const key = `${state.game_id}:${moves.length}:${moves.at(-1)?.[1] || ''}`;
  if (key === lastAnimatedKey) return;
  const prevKey = lastAnimatedKey;
  lastAnimatedKey = key;
  if (!prevKey || !prevKey.startsWith(`${state.game_id}:`) || !moves.length) return;
  const prevLen = Number(prevKey.split(':')[1]);
  if (moves.length !== prevLen + 1) return;
  const lastMove = moves.at(-1)?.[1];
  if (!lastMove || !/^[a-i][0-9][a-i][0-9]$/.test(lastMove)) return;
  const fromX = lastMove.charCodeAt(0) - 97, fromY = Number(lastMove[1]);
  const toX = lastMove.charCodeAt(2) - 97, toY = Number(lastMove[3]);
  const piece = $('board').querySelector(`[data-point="${lastMove.slice(2)}"] .xiangqi-piece`);
  if (!piece) return;
  const dx = (fromX - toX) * 10;
  const dy = (toY - fromY) * 10;
  piece.animate([
    {transform: `translate(${dx}cqw, ${dy}cqw) scale(1.06)`},
    {transform: 'translate(0, 0) scale(1)'},
  ], {duration: 210, easing: 'cubic-bezier(0.22, 0.68, 0.3, 1)'});
}

function renderBoard() {
  const g = displayedGame();
  if (!g) return;
  const active = canPlay(), xi = g.game === 'xiangqi', hex = g.game === 'hex', nx = g.boardXSize, ny = g.boardYSize;
  const matches = samePosition(g, state.analysis?.position);
  const overlayMode = $('overlay').value;
  const list = matches && overlayMode !== 'none' ? candidates() : [];
  const activeFocus = matches ? (previewMove || highlight) : null;
  const key = JSON.stringify([
    state.game_id, g, active, viewTurn, selectedFrom, highlight, previewMove,
    list, overlayMode, $('numbers').checked, isMuZero(),
  ]);
  if (key === boardKey) return;
  boardKey = key;

  const unit = xi ? 10 : hex ? 88 / (1.5 * (nx - 1) + 2.8) : 100 / (nx + 1);
  const radius = unit / Math.sqrt(3);
  const height = xi ? 110 : hex ? Math.sqrt(3) / 2 * (ny - 1) * unit + 3.8 * radius + 10 : (ny + 1) * unit;
  const ox = (100 - 1.5 * (nx - 1) * unit) / 2;
  const oy = (height - Math.sqrt(3) / 2 * (ny - 1) * unit) / 2;
  const point = (x, y) => xi ? [10 + x * 10, 100 - y * 10] :
    hex ? [ox + (x + y / 2) * unit, oy + y * unit * Math.sqrt(3) / 2] :
    [(x + 1) * unit, (y + 1) * unit];

  $('board').classList.toggle('hex-board', hex);
  $('board').classList.toggle('xiangqi-board', xi);
  $('board').classList.toggle('white-turn', g.currentPlayer === 'W');
  $('board').classList.toggle('hide-numbers', !$('numbers').checked);
  $('board').style.aspectRatio = `100 / ${height}`;
  $('board').style.setProperty('--candidate-unit', `${unit}cqw`);
  $('board').style.setProperty('--hex-unit', `${unit}cqw`);
  $('board').setAttribute('aria-label', `${gameNames[g.game]}棋盘`);

  let html = `<svg viewBox="0 0 100 ${height}" aria-hidden="true">`;
  if (xi) {
    html += `<defs>
      <marker id="xq-arrow-def" viewBox="0 0 10 10" refX="7.5" refY="5" markerWidth="4.6" markerHeight="4.6" orient="auto-start-reverse"><path d="M1,1.5 L8.5,5 L1,8.5 z" fill="#2b7a8c"/></marker>
      <marker id="xq-arrow-best" viewBox="0 0 10 10" refX="7.5" refY="5" markerWidth="4.8" markerHeight="4.8" orient="auto-start-reverse"><path d="M1,1.5 L8.5,5 L1,8.5 z" fill="#1e8a5e"/></marker>
      <marker id="xq-arrow-focus" viewBox="0 0 10 10" refX="7.5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M1,1.5 L8.5,5 L1,8.5 z" fill="#d96b27"/></marker>
    </defs>`;
    html += `<g fill="none" stroke="var(--board-line)">`;
    html += `<rect x="8.6" y="8.6" width="82.8" height="92.8" stroke-width=".42" rx=".4"/>`;
    html += `<g stroke-width=".2">`;
    for (let y = 0; y < 10; y++) html += `<path d="M10,${10 + y * 10}H90"/>`;
    for (let x = 0; x < 9; x++) {
      html += x === 0 || x === 8
        ? `<path d="M${10 + x * 10},10V100"/>`
        : `<path d="M${10 + x * 10},10V50 M${10 + x * 10},60V100"/>`;
    }
    html += '<path d="M40,10L60,30 M60,10L40,30 M40,80L60,100 M60,80L40,100"/>';
    for (const [cx, cy] of [[1, 2], [7, 2], [1, 7], [7, 7], [0, 3], [2, 3], [4, 3], [6, 3], [8, 3], [0, 6], [2, 6], [4, 6], [6, 6], [8, 6]]) {
      html += xiangqiCrosshair(cx, cy, point);
    }
    html += `</g></g><g fill="var(--board-label)" font-family="'Noto Serif SC','Songti SC',serif" font-size="4.1" font-weight="600" text-anchor="middle" opacity=".88">`;
    html += `<text x="30" y="56.5">楚 河</text><text x="70" y="56.5">漢 界</text></g>`;
    html += `<g fill="var(--board-label)" font-family="ui-monospace,monospace" font-size="1.9" font-weight="500" text-anchor="middle" opacity=".85">`;
    for (let x = 0; x < 9; x++) html += `<text x="${10 + x * 10}" y="106.8">${'abcdefghi'[x]}</text>`;
    for (let y = 0; y < 10; y++) html += `<text x="4.3" y="${100 - y * 10 + 0.6}">${y}</text>`;
    html += `</g>`;
  } else if (hex) {
    const vertex = (x, y, i) => {
      const [cx, cy] = point(x, y), angle = (30 + i * 60) * Math.PI / 180;
      return [cx + radius * Math.cos(angle), cy + radius * Math.sin(angle)];
    };
    const edges = [[], [], [], []];
    for (let x = 0; x < nx; x++) edges[0].push(...[3, 4, 5].map(v => vertex(x, 0, v)));
    for (let y = 0; y < ny; y++) edges[1].push(...[5, 0].map(v => vertex(nx - 1, y, v)));
    for (let x = nx - 1; x >= 0; x--) edges[2].push(...[0, 1, 2].map(v => vertex(x, ny - 1, v)));
    for (let y = ny - 1; y >= 0; y--) edges[3].push(...[2, 3].map(v => vertex(0, y, v)));
    const sides = edges.map(edge => edge.filter((p, i) => !i || Math.hypot(p[0] - edge[i - 1][0], p[1] - edge[i - 1][1]) > 1e-8));
    const contour = sides.flatMap(s => s.slice(0, -1));
    const outer = contour.map((p, i) => {
      const prev = contour[(i + contour.length - 1) % contour.length];
      const next = contour[(i + 1) % contour.length];
      const norm = (a, b) => {
        const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
        return [(b[1] - a[1]) / len, (a[0] - b[0]) / len];
      };
      const a = norm(prev, p), b = norm(p, next);
      const sc = unit * 0.12 / (1 + a[0] * b[0] + a[1] * b[1]);
      return [p[0] + (a[0] + b[0]) * sc, p[1] + (a[1] + b[1]) * sc];
    });
    html += `<g stroke="var(--hex-rail-outline)" stroke-width=".16" stroke-linejoin="round">`;
    let offset = 0;
    sides.forEach((s, i) => {
      const outside = s.map((_, j) => outer[(offset + j) % outer.length]);
      html += `<path class="hex-goal" d="M${s.concat(outside.reverse()).join('L')}Z" fill="${i % 2 ? '#faf9f5' : '#2b313a'}"/>`;
      offset += s.length - 1;
    });
    html += `</g><g class="hex-cells" stroke="var(--hex-line)" stroke-width=".16" stroke-linejoin="round">`;
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const verts = Array.from({length: 6}, (_, i) => vertex(x, y, i).join(',')).join(' ');
        html += `<polygon points="${verts}" fill="var(--hex-cell)"/>`;
      }
    }
    html += `</g><g fill="var(--muted)" font-family="ui-monospace,monospace" font-size="${Math.min(2.1, unit * 0.29)}" text-anchor="middle" dominant-baseline="central">`;
    for (let x = 0; x < nx; x++) {
      const [tx, ty] = point(x, 0), [bx, by] = point(x, ny - 1);
      html += `<text x="${tx}" y="${ty - 1.55 * radius}">${columns[x]}</text>`;
      html += `<text x="${bx}" y="${by + 1.55 * radius}">${columns[x]}</text>`;
    }
    for (let y = 0; y < ny; y++) {
      const [lx, ly] = point(0, y), [rx, ry] = point(nx - 1, y);
      html += `<text x="${lx - 0.98 * unit}" y="${ly}">${ny - y}</text>`;
      html += `<text x="${rx + 0.98 * unit}" y="${ry}">${ny - y}</text>`;
    }
    html += `</g>`;
  } else {
    html += `<g fill="none" stroke="var(--board-line)">`;
    html += `<rect x="${unit}" y="${unit}" width="${(nx - 1) * unit}" height="${(ny - 1) * unit}" stroke-width=".32"/>`;
    html += `<g stroke-width=".18">`;
    for (let x = 1; x < nx - 1; x++) html += `<path d="M${(x + 1) * unit},${unit}V${ny * unit}"/>`;
    for (let y = 1; y < ny - 1; y++) html += `<path d="M${unit},${(y + 1) * unit}H${nx * unit}"/>`;
    html += `</g></g><g fill="var(--board-star)">`;
    for (const [sx, sy] of starPointsFor(nx, ny)) {
      const [px, py] = point(sx, sy);
      html += `<circle cx="${px}" cy="${py}" r="${Math.max(0.38, Math.min(0.75, unit * 0.085))}"/>`;
    }
    html += `</g><g fill="var(--board-label)" font-family="ui-monospace,monospace" font-size="${Math.min(2.1, Math.max(1.3, unit * 0.26))}" text-anchor="middle">`;
    for (let x = 0; x < nx; x++) {
      const [px, py] = point(x, ny - 1);
      html += `<text x="${px}" y="${py + unit * 0.72}">${columns[x]}</text>`;
    }
    for (let y = 0; y < ny; y++) {
      const [px, py] = point(0, y);
      html += `<text x="${px - unit * 0.68}" y="${py + 0.55}">${ny - y}</text>`;
    }
    html += `</g>`;
  }

  // Xiangqi candidate arrows: layered back-to-front with crisp arrowheads & focus isolation
  if (xi && matches) {
    const maxV = Math.max(1, ...list.map(r => r.visits));
    const arrowRows = [...list];
    if (activeFocus && !arrowRows.some(r => r.move === activeFocus)) {
      const found = state.analysis.report?.moveInfos.find(r => r.move === activeFocus);
      arrowRows.push(found || {move: activeFocus, order: -1, visits: maxV, winrate: NaN});
    }
    // Sort so lowest-ranked draw first, #1 and focused move always draw on top
    arrowRows.sort((a, b) => {
      if (a.move === activeFocus) return 1;
      if (b.move === activeFocus) return -1;
      return b.order - a.order;
    });
    for (const row of arrowRows) {
      const move = row.move;
      if (!/^[a-i][0-9][a-i][0-9]$/.test(move)) continue;
      const [x1, y1] = point(move.charCodeAt(0) - 97, Number(move[1]));
      const [x2, y2] = point(move.charCodeAt(2) - 97, Number(move[3]));
      const dx = x2 - x1, dy = y2 - y1, dist = Math.hypot(dx, dy) || 1;
      const isFocus = move === activeFocus;
      const isBest = row.order === 0;
      const startTrim = Math.min(3.5, dist * 0.32);
      const endTrim = (isFocus || isBest) ? Math.min(4.5, dist * 0.44) : Math.min(3.2, dist * 0.32);
      const sx = x1 + dx / dist * startTrim, sy = y1 + dy / dist * startTrim;
      const ex = x2 - dx / dist * endTrim, ey = y2 - dy / dist * endTrim;
      const ratio = Math.max(0.15, (row.visits || 0) / maxV);
      const dimmed = Boolean(activeFocus) && !isFocus;
      const opacity = isFocus ? 0.96 : dimmed ? 0.16 : (isBest ? 0.9 : 0.28 + 0.45 * ratio);
      const width = isFocus ? 0.95 : (isBest ? 0.85 : 0.4 + 0.35 * ratio);
      const stroke = isFocus ? '#d96b27' : isBest ? '#1e8a5e' : '#2b7a8c';
      const marker = isFocus ? 'url(#xq-arrow-focus)' : isBest ? 'url(#xq-arrow-best)' : 'url(#xq-arrow-def)';
      html += `<g opacity="${opacity.toFixed(2)}" fill="none" stroke="${stroke}">`;
      html += `<line x1="${sx.toFixed(2)}" y1="${sy.toFixed(2)}" x2="${ex.toFixed(2)}" y2="${ey.toFixed(2)}" stroke-width="${width.toFixed(2)}" stroke-linecap="round" marker-end="${marker}"/>`;
      if (isFocus || isBest) {
        html += `<circle cx="${x2}" cy="${y2}" r="4.1" stroke-width=".55" stroke-dasharray="${isFocus ? 'none' : '1.4 1'}"/>`;
      }
      html += `</g>`;
    }
  }
  html += '</svg>';

  // PV ghost preview map (only for non-Xiangqi, non-MuZero real-board games)
  const pvGhosts = new Map();
  if (!xi && matches && previewMove && !isMuZero()) {
    const pvRow = state.analysis?.report?.moveInfos.find(r => r.move === previewMove);
    if (pvRow?.pv?.length) {
      let turnPlayer = g.currentPlayer;
      pvRow.pv.slice(0, 6).forEach((m, idx) => {
        if (m && m !== 'pass' && !pvGhosts.has(m)) {
          pvGhosts.set(m, {step: idx + 1, color: turnPlayer === 'B' ? 'black' : 'white'});
        }
        turnPlayer = turnPlayer === 'B' ? 'W' : 'B';
      });
    }
  }

  const moveList = g.moves || [];
  const lastMoveStr = moveList.at(-1)?.[1] || '';
  const lastXiFrom = xi && /^[a-i][0-9][a-i][0-9]$/.test(lastMoveStr) ? lastMoveStr.slice(0, 2) : '';
  const lastXiTo = xi && /^[a-i][0-9][a-i][0-9]$/.test(lastMoveStr) ? lastMoveStr.slice(2) : '';
  const numbers = new Map(moveList.map(([player, move], i) => [move, i + 1]));
  const byMove = new Map(list.map(row => [row.move, row]));
  const maximum = Math.max(1, ...list.map(row => overlayMode === 'prior' ? row.prior : row.visits));
  const destinations = xi && selectedFrom ? state.game.legalMoves.filter(m => m.slice(0, 2) === selectedFrom) : [];

  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) {
      const index = y * nx + x;
      const move = coordinate(x, y, g);
      const pc = (xi ? g.pieces : g.board)[index];
      const [px, py] = point(x, y);
      const target = destinations.some(m => m.slice(2) === move);
      const row = byMove.get(move);
      const ghost = !pc ? pvGhosts.get(move) : null;
      const disabled = !active || (!xi && !state.game.legalMoves.includes(move));
      const classes = ['board-point'];
      if (selectedFrom === move) classes.push('selected-piece');
      if (target) {
        classes.push('legal-destination');
        if (pc) classes.push('has-piece');
      }
      if (highlight === move) classes.push('highlight');
      if (previewMove === move && !ghost) classes.push('preview-point');
      if (xi && move === lastXiFrom) classes.push('last-origin');
      if (xi && move === lastXiTo) classes.push('last-dest');

      html += `<button class="${classes.join(' ')}" data-point="${move}" aria-label="${move}${pc ? ' ' + (xi ? pieces[Math.sign(pc)][Math.abs(pc)] : pc > 0 ? '黑棋' : '白棋') : ' 空位'}" style="left:${px}%;top:${py / height * 100}%;width:${unit * 0.96}%;height:${unit * 0.96 / height * 100}%" ${disabled ? 'disabled' : ''}>`;
      if (pc) {
        html += xi
          ? `<span class="xiangqi-piece ${pc > 0 ? 'red-piece' : 'black-piece'}">${pieces[Math.sign(pc)][Math.abs(pc)]}</span>`
          : `<span class="stone ${pc > 0 ? 'black' : 'white'} ${numbers.get(move) === moveList.length ? 'last' : ''}">${numbers.get(move) || ''}</span>`;
      } else if (ghost) {
        html += `<span class="pv-stone ${ghost.color}">${ghost.step}</span>`;
      } else if (row) {
        if (overlayMode === 'candidates') {
          const isBest = row.order === 0;
          html += `<span class="candidate-marker ${isBest ? 'best' : ''}" style="--visit-intensity:${(row.visits / maximum * 82).toFixed(1)}%"><strong>${percent(row.winrate)}</strong><small>${compactVisits(row.visits)}</small><sup>${row.order + 1}</sup></span>`;
        } else {
          html += `<span class="overlay-value" style="--intensity:${(row[overlayMode] / maximum * 100).toFixed(1)}%">${overlayMode === 'prior' ? percent(row.prior) : compactVisits(row.visits)}</span>`;
        }
      }
      html += '</button>';
    }
  }
  $('board').innerHTML = html;
  if (xi) animateXiangqiMove(g);
}

function renderHistory() {
  const moves = state?.game?.moves || [];
  const turn = viewTurn ?? moves.length;
  const max = moves.length;
  $('timeline').max = max;
  $('timeline').value = turn;
  $('timeline').disabled = max === 0;
  $('timeline').style.setProperty('--range-progress', `${max ? (turn / max * 100) : 0}%`);
  $('history-count').textContent = viewTurn === null ? `${max} 手` : `${turn} / ${max} 手`;
  const viewing = viewTurn !== null;
  $('view-banner').hidden = !viewing;
  $('view-note').hidden = !viewing;
  $('view-note').textContent = `回看第 ${turn} 手后的原生局面（只读）`;
  $('branch').hidden = !viewing;
  $('branch').disabled = !interactive();
  $('first').disabled = $('prev').disabled = turn === 0;
  $('next').disabled = $('live').disabled = !viewing;
  const key = JSON.stringify([moves, turn, state?.game?.game]);
  if (key === historyKey) return;
  historyKey = key;
  const game = state?.game?.game;
  $('history').innerHTML = moves.map(([player, move], i) => {
    const active = turn === i + 1;
    const dot = sideStoneClass(player, game);
    return `<button type="button" data-turn="${i + 1}" class="history-item ${active ? 'active selected' : ''}">` +
      `<small>${i + 1}.</small><i class="tiny-stone ${dot}"></i><span>${esc(move)}</span></button>`;
  }).join('');
  const activeBtn = $('history').querySelector('.active');
  activeBtn?.scrollIntoView({block: 'nearest', inline: 'nearest'});
}

function renderPolicy() {
  if (!$('policy-panel').open || !state?.analysis?.report) return;
  const {report, position: g} = state.analysis, source = $('policy-source').value, values = report[source];
  $('policy-map').hidden = g.game === 'xiangqi';
  $('policy-map').replaceChildren();
  $('policy-list').replaceChildren();
  if (!values) {
    $('policy-note').textContent = `当前原生报告没有 ${source}；HumanSL 策略需要加载可用的次级模型。`;
    return;
  }
  $('policy-note').textContent = `来源 ${source}；负值表示被屏蔽的动作。五子棋与 Hex 的末尾 pass 槽不可落子。`;
  if (g.game === 'xiangqi') {
    const select = document.createElement('select');
    select.setAttribute('aria-label', '策略起点');
    for (let y = 0; y < 10; y++) for (let x = 0; x < 9; x++) select.add(new Option(`${'abcdefghi'[x]}${y}`, `${x},${y}`));
    const saved = $('policy-list').dataset.from || '0,0';
    select.value = saved;
    const rows = document.createElement('div');
    rows.className = 'native-policy-list';
    const draw = () => {
      $('policy-list').dataset.from = select.value;
      rows.replaceChildren();
      const entries = [];
      for (const [i, m] of catalog.xiangqi_moves.entries()) {
        if (`${m[0]},${m[1]}` === select.value) {
          entries.push({move: `${'abcdefghi'[m[0]]}${m[1]}${'abcdefghi'[m[2]]}${m[3]}`, v: values[i]});
        }
      }
      const max = Math.max(0, ...entries.map(e => e.v));
      for (const {move, v} of entries) {
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = `${move} · ${v < 0 ? '—' : percent(v) + '%'}`;
        button.title = `${move}: ${v}`;
        if (v >= 0 && max > 0) button.style.setProperty('--bar', `${(v / max * 100).toFixed(1)}%`);
        button.onclick = () => selectCandidate(move);
        button.onpointerenter = () => setPreview(move);
        button.onpointerleave = () => setPreview(null);
        rows.append(button);
      }
    };
    select.onchange = draw;
    $('policy-list').append(select, rows);
    draw();
  } else {
    const max = Math.max(0, ...values), nx = g.boardXSize, ny = g.boardYSize;
    $('policy-map').style.gridTemplateColumns = `repeat(${nx}, 1fr)`;
    $('policy-map').style.aspectRatio = `${nx} / ${ny}`;
    $('policy-map').classList.toggle('dense', nx > 15);
    $('policy-map').innerHTML = values.slice(0, nx * ny).map((v, i) => {
      const move = coordinate(i % nx, Math.floor(i / nx), g);
      const masked = v < 0;
      const intensity = !masked && max ? (v / max * 100).toFixed(1) : '0';
      return `<button type="button" class="heat-cell ${masked ? 'is-masked' : ''}" data-candidate="${move}" title="${move}: ${v}" style="--intensity:${intensity}%"><span>${masked ? '—' : percent(v)}</span></button>`;
    }).join('');
    if (values.length > nx * ny) {
      const passVal = values[nx * ny];
      $('policy-list').innerHTML = `<span class="pass-pill">pass · ${passVal < 0 ? '已屏蔽' : percent(passVal) + '%'}</span>`;
    }
  }
}

function renderAnalysis() {
  const analysis = state?.analysis, report = analysis?.report;
  $('analysis-empty').hidden = Boolean(report);
  $('analysis-data').hidden = !report;
  const statusEl = $('analysis-status');
  if (!report) {
    statusEl.textContent = '—';
    statusEl.classList.remove('is-running');
    return;
  }
  const root = report.rootInfo;
  const perspective = side(analysis.position.analysisPerspective, analysis.position.game);
  statusEl.textContent = {running: '分析中', complete: '已完成', stopped: '已停止', error: '中断'}[analysis.status] || '—';
  statusEl.classList.toggle('is-running', analysis.status === 'running');
  $('analysis-context').textContent = `会话第 ${analysis.ply} 手后 · ${side(analysis.position.currentPlayer, analysis.position.game)}行棋`;
  $('completed').textContent = Number.isFinite(root.visits) ? root.visits.toLocaleString() : '—';
  $('winrate').textContent = percent(root.winrate) + '%';
  const diff = root.winValue - root.lossValue;
  const valEl = $('value');
  valEl.textContent = Number.isFinite(diff) ? (diff > 0.0005 ? '+' : '') + diff.toFixed(3) : '—';
  valEl.classList.toggle('is-positive', diff > 0.02);
  valEl.classList.toggle('is-negative', diff < -0.02);

  $('perspective').textContent = `报告视角：${perspective}；棋盘角标沿用原生候选顺序。`;
  const wdl = [root.winValue, root.drawValue, root.lossValue, root.noResultValue];
  ['win', 'draw', 'loss', 'no-result'].forEach((key, i) => {
    $('wdl-' + key).style.width = `${100 * (wdl[i] || 0)}%`;
  });
  $('wdl').innerHTML =
    `<span class="wdl-legend">` +
    `<span class="wdl-item"><i class="wdl-key win"></i>W ${percent(wdl[0])}%</span>` +
    `<span class="wdl-item"><i class="wdl-key draw"></i>D ${percent(wdl[1])}%</span>` +
    `<span class="wdl-item"><i class="wdl-key loss"></i>L ${percent(wdl[2])}%</span>` +
    `<span class="wdl-item"><i class="wdl-key no-result"></i>NR ${percent(wdl[3])}%</span>` +
    `</span>`;
  $('network-wdl').textContent = `网络原始 · 得分 ${percent(root.rawWinrate)}% · D ${percent(root.rawDrawProb)}% · NR ${percent(root.rawNoResultProb)}%`;

  const field = $('sort').value;
  document.querySelectorAll('[data-sort-col]').forEach(btn => {
    btn.dataset.activeSort = String(btn.dataset.sortCol === field);
  });
  const rows = report.moveInfos.filter(row => !$('visited-only').checked || row.visits > 0)
    .sort((a, b) => field === 'order' ? a.order - b.order : b[field] - a[field]);
  const maxVisits = Math.max(1, ...rows.map(r => r.visits || 0));
  const maxPrior = Math.max(1e-9, ...rows.map(r => r.prior || 0));
  $('candidate-count').textContent = `${rows.length} 个原生候选`;
  $('candidates').innerHTML = rows.map(row => {
    const chosen = analysis.move === row.move;
    const visitPct = Math.min(100, (row.visits / maxVisits) * 100).toFixed(1);
    const priorPct = Math.min(100, ((row.prior || 0) / maxPrior) * 100).toFixed(1);
    return `<tr data-move="${esc(row.move)}" class="${highlight === row.move ? 'selected' : ''}">` +
      `<td><button type="button" data-candidate="${esc(row.move)}" class="${chosen ? 'chosen' : ''}">${esc(row.move)}${chosen ? '<span class="move-star">★</span>' : ''}</button></td>` +
      `<td class="winrate-cell">${percent(row.winrate)}</td>` +
      `<td class="visit-bar" style="--bar:${visitPct}%">${row.visits}</td>` +
      `<td>${row.edgeVisits ?? '—'}</td>` +
      `<td class="probability" style="--bar:${priorPct}%">${percent(row.prior)}</td>` +
      `<td>${percent(row.drawValue)}</td>` +
      `<td>${percent(row.noResultValue)}</td>` +
      `<td>${number(row.lcb)}</td>` +
      `<td>${number(row.utility)}</td>` +
      `<td>${number(row.playSelectionValue)}</td>` +
      `</tr>`;
  }).join('');
  $('raw-analysis').textContent = JSON.stringify(report, null, 2);
  if (highlight) {
    const selectedRow = report.moveInfos.find(r => r.move === highlight);
    if (selectedRow) renderCandidateDetail(selectedRow);
  }
  renderPolicy();
}

function renderRawNN() {
  const allowed = state?.game?.game === 'xiangqi' ? [0, 2] : state?.game?.game === 'hex' ? [0, 3] : [0, 1, 2, 3, 4, 5, 6, 7];
  const sym = $('nn-symmetry');
  if (sym.dataset.choices !== allowed.join(',')) {
    const previous = Number(sym.value);
    sym.replaceChildren();
    allowed.forEach(n => sym.add(new Option(String(n), String(n))));
    if (allowed.includes(previous)) sym.value = String(previous);
    sym.dataset.choices = allowed.join(',');
  }
  $('nn-optimism').disabled = $('nn-source').value === 'human';
  $('diagnose').disabled = !interactive() || !state?.connected || !state?.game || over() || viewTurn !== null;
  const data = state?.raw_nn;
  $('nn-empty').hidden = Boolean(data);
  $('nn-data').hidden = !data;
  if (!data || !$('nn-panel').open) return;
  const g = data.position, report = data.report, values = report.values;
  $('nn-context').textContent = `会话第 ${data.ply} 手后 · ${side(g.currentPlayer, g.game)}行棋` +
    (samePosition(displayedGame(), g) ? '' : ' · 与当前展示局面不同');
  $('nn-command').textContent = `来源：${data.command} · ${data.source === 'human' ? '次级模型' : '主模型'}`;
  $('nn-values').textContent = `${side('W', g.game)}视角 W / D / L / NR · ` +
    [values.whiteWin, values.draw, values.whiteLoss, values.noResult].map(v => percent(v) + '%').join(' / ');
  $('nn-text').textContent = data.text;
  $('nn-map').hidden = g.game === 'xiangqi';
  $('nn-map').replaceChildren();
  $('nn-list').replaceChildren();
  if (g.game === 'xiangqi') {
    const select = document.createElement('select');
    select.setAttribute('aria-label', '网络诊断起点');
    const origins = [...new Set(report.policy.map(row => row.move.slice(0, 2)))];
    origins.forEach(move => select.add(new Option(move, move)));
    if (origins.includes($('nn-list').dataset.from)) select.value = $('nn-list').dataset.from;
    const rows = document.createElement('div');
    rows.className = 'native-policy-list';
    const draw = () => {
      $('nn-list').dataset.from = select.value;
      rows.replaceChildren();
      const filtered = report.policy.filter(row => row.move.startsWith(select.value));
      const max = Math.max(0, ...filtered.map(r => r.probability ?? 0));
      for (const row of filtered) {
        const cell = document.createElement('span');
        cell.className = 'nn-entry';
        cell.dataset.move = row.move;
        cell.textContent = `${row.move} · ${row.probability === null ? '—' : percent(row.probability) + '%'}`;
        cell.title = `${row.move}: ${row.probability === null ? 'NAN' : row.probability}`;
        if (row.probability !== null && max > 0) cell.style.setProperty('--bar', `${(row.probability / max * 100).toFixed(1)}%`);
        rows.append(cell);
      }
    };
    select.onchange = draw;
    $('nn-list').append(select, rows);
    draw();
  } else {
    const nx = g.boardXSize, ny = g.boardYSize, max = Math.max(0, ...report.policy.map(row => row.probability ?? 0));
    $('nn-map').style.gridTemplateColumns = `repeat(${nx}, 1fr)`;
    $('nn-map').style.aspectRatio = `${nx} / ${ny}`;
    $('nn-map').classList.toggle('dense', nx > 15);
    for (const row of report.policy.slice(0, nx * ny)) {
      const cell = document.createElement('span');
      const masked = row.probability === null;
      cell.className = `heat-cell ${masked ? 'is-masked' : ''}`;
      cell.dataset.move = row.move;
      cell.textContent = masked ? '—' : percent(row.probability);
      cell.title = `${row.move}: ${masked ? 'NAN' : row.probability}`;
      cell.style.setProperty('--intensity', `${max ? (row.probability ?? 0) / max * 100 : 0}%`);
      $('nn-map').append(cell);
    }
    const pass = report.policy.at(-1).probability;
    $('nn-list').innerHTML = `<span class="pass-pill">pass · ${pass === null ? '已屏蔽' : percent(pass) + '%'}</span>`;
  }
}

function renderModelMeta(run, model) {
  const el = $('model-meta');
  if (!el) return;
  const tags = [];
  const game = model?.game || run?.game;
  if (game && gameNames[game]) tags.push(`<span class="meta-chip highlight">${esc(gameNames[game])}</span>`);
  if (run?.algorithm) tags.push(`<span class="meta-chip">${esc(run.algorithm)}</span>`);
  if (model?.iteration !== undefined) tags.push(`<span class="meta-chip">Iter ${esc(model.iteration)}</span>`);
  if (model?.accepted === false) tags.push(`<span class="meta-chip">未通过门控</span>`);
  if (run?.canvas && game !== 'xiangqi') tags.push(`<span class="meta-chip">画布 ${run.canvas}×${run.canvas}</span>`);
  el.innerHTML = tags.join('');
  el.hidden = tags.length === 0;
}

function render() {
  $('connection').textContent = online ? '工作台已连接' : '断线 · 重连中';
  $('connection-dot').classList.toggle('online', online);
  $('connection-dot').title = online ? '工作台已连接' : '断线 · 重连中';
  const error = notice || state?.error || '';
  $('error').hidden = !error;
  $('error').textContent = error;

  const run = selectedRun(), model = selectedModel();
  renderModelMeta(run, model);
  const summaryMsg = !run?.config ? '所选目录缺少原生 gtp.cfg，无法创建棋局。' : model ? '' : '暂无可用模型';
  $('model-summary').textContent = summaryMsg;
  $('model-summary').hidden = !summaryMsg;
  $('catalog-errors').textContent = (catalog?.errors || []).map(item => `${item.run}: ${item.error}`).join('\n');

  $('refresh-models').disabled = !interactive();
  const canStart = online && Boolean(model) && Boolean(run?.config) && !['opening', 'stopping'].includes(state?.phase);
  if ($('new-game')) $('new-game').disabled = !interactive() || !model || !run?.config;
  if ($('import-apply')) $('import-apply').disabled = !interactive() || !model || !run?.config;
  $('balanced-opening').disabled = !canStart || !['freestyle', 'standard', 'renju', 'hex'].includes($('rules').value);

  const g = state?.game, available = interactive() && state?.connected && g;
  const analyzing = Boolean(state?.analysis_enabled);
  $('analyze').disabled = $('step').disabled = !available || over() || viewTurn !== null;
  $('analyze-label').textContent = analyzing ? '暂停分析' : '开启分析';
  $('analyze').classList.toggle('analyzing-active', analyzing);
  $('analyze').setAttribute('aria-pressed', String(analyzing));

  const stoppable = ['thinking', 'analyzing', 'opening', 'stopping'].includes(state?.phase);
  $('stop').hidden = !stoppable;
  $('stop').disabled = pending || state?.stop_requested;
  $('stop-label').textContent = state?.phase === 'opening' ? '停止生成' : '停止';

  const undoable = g && (state.resignation || (state.mode === 'manual' ? g.moves.length : g.moves.some(([p]) => p === state.human)));
  $('undo').disabled = !available || !undoable;
  $('pass').hidden = g?.game !== 'go';
  $('pass').disabled = !canPlay() || !g?.legalMoves.includes('pass');
  $('retry').hidden = !available || over() || state.mode !== 'play' || g.currentPlayer === state.human;

  $('opening-info').hidden = !state?.opening;
  $('opening-info').textContent = state?.opening ? `初始平衡开局 · ${state.opening.moves.length} 手 · 原生 KataGomo 开局算法` : '';
  $('state-version').textContent = `单会话 · 状态 v${state?.version ?? '—'}`;
  $('candidate-controls').hidden = $('overlay').value === 'none';

  const isActiveWork = Boolean(state?.busy || ['thinking', 'analyzing', 'opening', 'loading', 'diagnosing', 'stopping'].includes(state?.phase));
  $('status-bar').classList.toggle('thinking', isActiveWork);
  $('status-dot').hidden = !isActiveWork && Boolean(g && !over());

  if (g) {
    $('rule-badge').textContent = gameNames[g.game];
    $('session-mode').textContent = state.mode === 'manual' ? '手动双方' : '人机对弈';
    $('turn-stone').hidden = isActiveWork || over();
    $('turn-stone').className = `tiny-stone ${sideStoneClass(g.currentPlayer, g.game)}`;
    $('status').textContent = state.phase === 'opening'
      ? '生成平衡开局中 · 成功后替换当前棋局'
      : over()
        ? `对局结束 · ${state.resignation?.result || g.result}`
        : `${phaseNames[state.phase]} · ${side(g.currentPlayer)}行棋`;
    $('move-count').textContent = `原生第 ${g.turnNumber} 手`;
    $('session-info').textContent = `${g.boardXSize}×${g.boardYSize} · ${state.parameters?.useGumbel ? 'Gumbel' : 'PUCT'}${isMuZero() ? ' · MuZero' : ''} · ${state.connected ? '引擎在线' : '引擎断开'}`;
    $('rule-note').textContent = g.game === 'hex'
      ? '黑方连接上下边，白方连接左右边。'
      : g.game === 'xiangqi'
        ? '先选择棋子，再选择原生合法落点；红方在下。'
        : g.game === 'renju'
          ? '黑方禁手按原生规则落子判负。'
          : g.game === 'go'
            ? formatGoRules(g.rules)
            : '合法性与终局由原生棋规裁定。';
  } else {
    $('turn-stone').hidden = true;
    $('status').textContent = state?.busy ? phaseNames[state.phase] : '选择模型后自动开始';
  }
  renderBoard();
  renderHistory();
  renderAnalysis();
  renderRawNN();
}

// Form & Setup Events
$('settings').onsubmit = event => event.preventDefault();
$('new-game').onclick = () => {
  try {
    const payload = newGamePayload();
    command('new', payload);
  } catch (error) {
    notice = error.message;
    render();
  }
};
$('balanced-opening').onclick = () => requestSetup(true);
$('run').onchange = () => { populateModels(); requestSetup(); };
$('model').onchange = () => { updateRules(); requestSetup(); };
$('rules').onchange = () => { updateSetup(); requestSetup(); };
$('mode').onchange = () => { updateSetup(); requestSetup(); };
$('human').onchange = () => requestSetup();
$('ponder').onchange = () => { updateSetup(); requestSetup(); };
$('budget-type').onchange = () => {
  const type = $('budget-type').value, previous = $('budget-type').dataset.previous || 'visits';
  if ((type === 'time') !== (previous === 'time')) $('budget').value = type === 'time' ? '5' : '500';
  $('budget-type').dataset.previous = type;
  updateSetup();
  requestSetup();
};
$('budget').onchange = () => { renderBudgetPresets(); requestSetup(); };
$('budget-presets').onclick = event => {
  const btn = event.target.closest('[data-preset]');
  if (!btn) return;
  $('budget').value = btn.dataset.preset;
  renderBudgetPresets();
  requestSetup();
};
$('board-size').onchange = () => requestSetup();
for (const id of ['budget', 'board-size']) {
  $(id).addEventListener('keydown', event => {
    if (event.key === 'Enter') { event.preventDefault(); $(id).blur(); requestSetup(); }
  });
}
$('board-size-range').oninput = () => updateSetup();
$('board-size-range').onchange = () => requestSetup();
$('refresh-models').onclick = () => command('refresh');

// SGF / FEN / Komi Import
$('import-sgf-file').onchange = async event => {
  const file = event.target.files?.[0];
  if (!file) return;
  $('import-file-name').textContent = file.name;
  $('import-sgf').value = await file.text();
};
$('import-apply').onclick = () => {
  try {
    const sgf = $('import-sgf').value.trim();
    const sgfTurnRaw = $('import-sgf-turn').value.trim();
    const fen = $('import-fen').value.trim();
    const komiRaw = $('import-komi').value.trim();
    const extra = {};
    if (sgf) {
      extra.sgf = sgf;
      if (sgfTurnRaw !== '') extra.sgf_turn = Number(sgfTurnRaw);
    } else {
      if (fen && !$('import-fen-field').hidden) extra.fen = fen;
      if (komiRaw !== '' && !$('import-komi-field').hidden) extra.komi = Number(komiRaw);
    }
    command('new', newGamePayload(extra));
  } catch (error) {
    notice = error.message;
    render();
  }
};

// Action & Replay Controls
$('analyze').onclick = () => command(state?.analysis_enabled ? 'stop' : 'analyze');
for (const name of ['step', 'stop', 'undo', 'retry']) $(name).onclick = () => command(name);
$('pass').onclick = () => command('play', {move: 'pass'});
$('branch').onclick = () => command('branch', {turn: viewTurn});
$('timeline').oninput = () => view(Number($('timeline').value));
$('first').onclick = () => view(0);
$('prev').onclick = () => view((viewTurn ?? state?.game?.moves.length ?? 0) - 1);
$('next').onclick = () => view((viewTurn ?? 0) + 1);
$('live').onclick = () => view(state?.game?.moves.length ?? 0);
$('history').onclick = event => {
  const row = event.target.closest('[data-turn]');
  if (row) view(Number(row.dataset.turn));
};
$('show-analysis').onclick = () => view(state.analysis.ply);
$('show-nn').onclick = () => view(state.raw_nn.ply);
$('copy-raw').onclick = async () => {
  try {
    await navigator.clipboard.writeText($('raw-analysis').textContent);
    const btn = $('copy-raw');
    btn.textContent = '已复制 ✓';
    setTimeout(() => { btn.textContent = '复制 JSON'; }, 1400);
  } catch {}
};

// Raw NN Diagnostics
$('diagnose').onclick = () => {
  const source = $('nn-source').value;
  if (source === 'main' && !$('nn-optimism').reportValidity()) return;
  command('diagnose', {
    source,
    symmetry: Number($('nn-symmetry').value),
    ...(source === 'main' ? {optimism: Number($('nn-optimism').value)} : {}),
  });
};
$('nn-panel').ontoggle = renderRawNN;
$('nn-source').onchange = renderRawNN;

// Board Interaction
$('board').onclick = event => {
  const cell = event.target.closest('[data-point]');
  if (!cell || !canPlay()) return;
  const move = cell.dataset.point;
  if (state.game.game !== 'xiangqi') return command('play', {move});
  const chosen = selectedFrom + move;
  if (selectedFrom && state.game.legalMoves.includes(chosen)) {
    selectedFrom = null;
    command('play', {move: chosen});
  } else {
    selectedFrom = state.game.legalMoves.some(m => m.startsWith(move)) ? move : null;
    renderBoard();
  }
};
$('board').onpointerover = event => {
  const cell = event.target.closest('[data-point]');
  if (cell) {
    $('board-coordinate').textContent = cell.dataset.point;
    const row = state?.analysis?.report?.moveInfos.find(r => r.move === cell.dataset.point);
    if (row && samePosition(displayedGame(), state?.analysis?.position)) setPreview(row.move);
  }
};
$('board').onpointerleave = () => {
  $('board-coordinate').textContent = '—';
  if (previewMove) setPreview(null);
};

// Candidate Table & Policy Interaction
document.querySelectorAll('[data-sort-col]').forEach(btn => {
  btn.onclick = () => {
    $('sort').value = btn.dataset.sortCol;
    renderAnalysis();
  };
});
for (const id of ['candidates', 'policy-map']) {
  $(id).onclick = event => {
    const cell = event.target.closest('[data-candidate]');
    if (cell) selectCandidate(cell.dataset.candidate);
  };
  for (const name of ['pointerover', 'focusin']) {
    $(id).addEventListener(name, event => {
      const cell = event.target.closest('[data-candidate]');
      const tr = event.target.closest('tr[data-move]');
      const move = cell?.dataset.candidate || tr?.dataset.move;
      if (move) setPreview(move);
    });
  }
  $(id).addEventListener('pointerleave', () => setPreview(null));
  $(id).addEventListener('focusout', event => {
    if (!$(id).contains(event.relatedTarget)) setPreview(null);
  });
}

for (const id of ['overlay', 'candidate-top', 'candidate-min', 'sort', 'visited-only']) $(id).onchange = render;
$('numbers').onchange = () => { savePreferences(); renderBoard(); };
$('policy-panel').ontoggle = renderPolicy;
$('policy-source').onchange = renderPolicy;

// Keyboard Shortcuts
document.addEventListener('keydown', event => {
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  if (event.key === 'Escape') {
    if (!$('stop').disabled && !$('stop').hidden) { event.preventDefault(); $('stop').click(); }
    return;
  }
  if (event.target.closest('input,select,textarea')) return;
  const actions = {a: 'analyze', s: 'step', u: 'undo', ArrowLeft: 'prev', ArrowRight: 'next', Home: 'first', End: 'live'};
  const id = actions[event.key] || actions[event.key.toLowerCase()];
  if (id) {
    event.preventDefault();
    if (!$(id).disabled && !$(id).hidden) $(id).click();
  }
});

// Preferences & Responsive Setup Collapse
try {
  const saved = JSON.parse(localStorage.getItem('etazero-ui') || '{}');
  if (typeof saved.numbers === 'boolean') $('numbers').checked = saved.numbers;
} catch {}
const compactMedia = matchMedia('(max-width: 1150px)');
const syncDetails = () => { $('setup-details').open = !compactMedia.matches; };
syncDetails();
compactMedia.addEventListener('change', syncDetails);

// Long-poll State Loop
(async () => {
  while (true) {
    try {
      const next = await api('/api/state' + (state ? `?since=${state.version}` : ''));
      online = true;
      accept(next);
    } catch (error) {
      online = false;
      render();
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }
})();
