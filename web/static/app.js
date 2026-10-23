const $ = id => document.getElementById(id);
const columns = 'ABCDEFGHJKLMNOPQRSTUVWXYZ';
const rules = {freestyle: 'Freestyle', standard: 'Standard', renju: 'Renju', hex: 'Hex', go: '围棋', xiangqi: '中国象棋'};
const notes = {go: '围棋 · 提子、劫争与终局计分按所选规则；连续停一手结束当前阶段。', xiangqi: '中国象棋 · 先选棋子，再选落点；无合法着法判负，重复局面按长将长捉规则裁定。', hex: 'Hex · 黑棋连接上下两边，白棋连接左右两边；按六邻接实际连通判胜。', freestyle: 'Freestyle · 五子或长连获胜。', standard: 'Standard · 恰好五子获胜，长连不计胜。', renju: 'Renju · 黑棋三三、四四、长连判负，恰好五子优先；白棋五子或长连获胜。'};
const xiangqiNames = {1: ['', '車', '仕', '炮', '兵', '馬', '相', '帥'], '-1': ['', '車', '士', '砲', '卒', '馬', '象', '將']};
let state = null, catalog = null, pending = false, online = false, notice = '';
let formKey = '', viewTurn = null, highlight = null, catalogRevision = -1;
let sortField = 'selection_weight', sortDirection = -1;
let analysisKey = null;
let networkKey = null, boardMarkup = '';
let historyKey = '';
let selectedFrom = null;
let xiangqiPosition = null;
let xiangqiAnimation = null;
const networkHeads = [
  ['普通策略', 'policy'], ['对手策略', 'opponent_policy'],
  ['长期 optimistic', 'long_optimistic_policy'], ['短期 optimistic', 'short_optimistic_policy'],
];
function sideName(player, rule = state?.rule) {
  return rule === 'xiangqi' ? (player === 1 ? '红' : '黑') : (player === 1 ? '黑' : '白');
}
function sideClass(player, rule = state?.rule) {
  return rule === 'xiangqi' ? (player === 1 ? 'red' : 'black') : (player === 1 ? 'black' : 'white');
}

function coordinate(action, size) {
  if (state?.rule === 'xiangqi' && catalog?.xiangqi_moves?.[action]) {
    const [x,y,u,v] = catalog.xiangqi_moves[action]; return `${'abcdefghi'[x]}${y}–${'abcdefghi'[u]}${v}`;
  }
  return action === size*size ? '停一手' : columns[action % size] + (size - Math.floor(action / size));
}
function percent(value, digits = 1) { return Number.isFinite(value) ? (value * 100).toFixed(digits) : '—'; }
function samePosition(game, analysis) {
  return game && analysis && game.board_size === analysis.board_size && game.turn === analysis.turn && game.player === analysis.player && game.rule === analysis.rule &&
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
    selectedFrom = null;
  }
  if (next.analysis_progress) next = {...next, analysis: next.analysis_progress};
  state = next;
  render();
}
async function command(operation, values = {}) {
  if (!state || pending || (state.busy && operation !== 'stop') || !online) return;
  pending = true;
  notice = '';
  render();
  try { accept(await api(`/api/${operation}`, {version: state.version, game_id: state.game_id, ...values})); }
  catch (error) {
    notice = error.message;
    try { accept(await api('/api/state')); } catch { connection(false); }
  } finally { pending = false; render(); }
}
function displayedGame() {
  const game = state?.game;
  if (!game || viewTurn === null) return game;
  const moves = game.moves.slice(0, viewTurn);
  if (game.boards?.length) return {...game, board: game.boards[viewTurn], moves, turn: viewTurn, player: game.players[viewTurn]};
  const board = Array(game.board_stride * game.board_height).fill(0);
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
function boardCandidates(analysis) {
  if (!analysis) return [];
  const top = Math.max(1, Math.min(625, Number($('candidate-top').value) || 10));
  const minimum = Math.max(0, Number($('candidate-min').value) || 0) / 100;
  return analysis.candidates.filter(c => c.visits > 0 && c.visit_policy >= minimum)
    .sort((a, b) => b.visits - a.visits || a.action - b.action).slice(0, top)
    .sort((a, b) => a.rank - b.rank).map((c, i) => ({...c, rank: i + 1}));
}
function candidateDetail(c, analysis) {
  const value = v => Number.isFinite(v) ? v.toFixed(4) : '—';
  const boardRank = boardCandidates(analysis).find(point => point.action === c.action)?.rank;
  return `${coordinate(c.action, analysis.board_size)} · 棋盘排名 ${boardRank || '—'} · 得分率 ${percent(c.winrate)}% · W/D/L ${c.wdl ? c.wdl.map(p => percent(p)).join(' / ') : '—'} · 访问 ${c.visits} (${percent(c.visit_policy)}%) · 选择 ${percent(c.selection_weight)}% · 原始权重 ${value(c.raw_selection_weight)} · 效用 ${value(c.utility)} · LCB ${value(c.lcb)} · 半径 ${value(c.radius)} · 改进策略 ${percent(c.improved_policy)}% · PV ${(c.pv || []).map(a => coordinate(a, analysis.board_size)).join(' ')}`;
}
function compactVisits(visits) {
  if (visits >= 1e6) return (visits / 1e6).toFixed(1) + 'm';
  if (visits >= 1000) return (visits / 1000).toFixed(1) + 'k';
  return String(visits);
}
function cancelXiangqiAnimation() {
  if (!xiangqiAnimation) return;
  cancelAnimationFrame(xiangqiAnimation.frame);
  xiangqiAnimation.animations.forEach(animation => animation.cancel());
  xiangqiAnimation = null;
}
function renderXiangqiPieces(game) {
  const layer = $('board').querySelector('.xiangqi-pieces');
  const key = `${state.instance}:${state.game_id}`;
  const previous = xiangqiPosition?.key === key ? xiangqiPosition.game : null;
  const live = viewTurn === null;
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const unchanged = previous && game && previous.turn === game.turn &&
    previous.moves.every((action, i) => action === game.moves[i]) &&
    previous.board.every((piece, i) => piece === game.board[i]);
  const liveForward = previous && game && live && xiangqiPosition.live && game.turn > previous.turn &&
    previous.moves.every((action, i) => action === game.moves[i]);
  if ((unchanged && (!xiangqiAnimation || !reduced)) || (xiangqiAnimation && liveForward && !reduced)) {
    xiangqiPosition.live = live;
    return;
  }

  cancelXiangqiAnimation();
  // Polling may deliver the human move and AI reply together. Play each saved
  // board in order; history jumps still show their requested position directly.
  if (liveForward && !reduced && game.turn > previous.turn + 1) {
    const turn = previous.turn + 1;
    game = {...game, board: game.boards[turn], moves: game.moves.slice(0, turn), turn, player: game.players[turn]};
  }
  xiangqiPosition = {key, game, live};
  layer.replaceChildren();
  if (!game) return;
  const stride = game.board_stride;
  function pieceAt(square, piece) {
    const slot = document.createElement('span');
    slot.className = 'xiangqi-piece-slot';
    slot.style.left = `${10 + (square % stride) * 10}%`;
    slot.style.top = `${(100 - Math.floor(square / stride) * 10) / 110 * 100}%`;
    slot.innerHTML = `<span class="xiangqi-piece ${piece > 0 ? 'red-piece' : 'black-piece'}">${xiangqiNames[Math.sign(piece)][Math.abs(piece)]}</span>`;
    layer.append(slot);
    return slot;
  }
  const pieces = new Map();
  game.board.forEach((piece, square) => {
    if (piece) pieces.set(square, pieceAt(square, piece));
  });
  if (!previous || reduced) return;
  const forward = game.turn === previous.turn + 1 && previous.moves.every((action, i) => action === game.moves[i]);
  const backward = game.turn === previous.turn - 1 && game.moves.every((action, i) => action === previous.moves[i]);
  if (!forward && !backward) return;
  const move = catalog.xiangqi_moves[(forward ? game : previous).moves.at(-1)];
  const [x, y, u, v] = forward ? move : [move[2], move[3], move[0], move[1]];
  const destination = v * stride + u, moving = pieces.get(destination);
  if (!moving || previous.board[y * stride + x] !== game.board[destination]) return;
  const timing = {duration: 240, easing: 'cubic-bezier(.22,.68,.3,1)'};
  moving.classList.add('moving-piece');
  // Board units are square, so both axes use the board's width during resizing.
  const movement = moving.animate([
    {transform: `translate(calc(-50% + ${(x - u) * 10}cqw), calc(-50% + ${(v - y) * 10}cqw))`},
    {transform: 'translate(-50%, -50%)'},
  ], timing);
  const playback = {animations: [movement], frame: null};
  let captured = null;
  if (forward && previous.board[destination]) {
    captured = pieceAt(destination, previous.board[destination]);
    captured.classList.add('captured-piece');
    playback.animations.push(captured.animate([{opacity: 1}, {opacity: 1, offset: .65}, {opacity: 0}], timing));
  }
  xiangqiAnimation = playback;
  playback.animations.forEach(animation => { animation.pause(); animation.currentTime = 0; });
  // AI replies also rebuild the analysis panels. Start after that render so
  // their work cannot consume the movement's visible duration.
  playback.frame = requestAnimationFrame(() => playback.animations.forEach(animation => animation.play()));
  movement.onfinish = () => {
    if (xiangqiAnimation !== playback) return;
    captured?.remove();
    moving.classList.remove('moving-piece');
    xiangqiAnimation = null;
    render();
  };
}
function renderXiangqiBoard(game, disabled) {
  $('board').classList.remove('hex-board'); $('board').classList.add('xiangqi-board');
  $('board').style.aspectRatio='10 / 11'; $('board').setAttribute('aria-label','中国象棋棋盘，红方在下');
  if (!$('board').querySelector('.xiangqi-squares')) {
    cancelXiangqiAnimation();
    $('board').innerHTML = '<div class="xiangqi-squares"></div><div class="xiangqi-pieces" aria-hidden="true"></div>';
    boardMarkup = '';
    xiangqiPosition = null;
  }
  const stride=game?.board_stride || 10;
  const point=(x,y)=>[10+x*10,100-y*10];
  let html='<svg viewBox="0 0 100 110" aria-hidden="true"><g stroke="var(--board-line)" stroke-width=".18" fill="none">';
  for(let y=0;y<10;y++) html+=`<path d="M10,${10+y*10}H90"/>`;
  for(let x=0;x<9;x++) html+= x===0||x===8 ? `<path d="M${10+x*10},10V100"/>` : `<path d="M${10+x*10},10V50 M${10+x*10},60V100"/>`;
  html+='<path d="M40,10L60,30 M60,10L40,30 M40,80L60,100 M60,80L40,100"/></g><g font-size="4" text-anchor="middle" fill="var(--board-label)"><text x="30" y="56.5">楚 河</text><text x="70" y="56.5">漢 界</text></g>';
  const analysis=state.analysis;
  if(samePosition(game,analysis)) {
    const actions=new Set($('overlay').value==='none'?[]:boardCandidates(analysis).map(c=>c.action));
    // Explicit selection remains visible even outside the candidate filters.
    if(highlight!==null){actions.delete(highlight);actions.add(highlight);}
    for(const action of actions) {
      const [x,y,u,v]=catalog.xiangqi_moves[action], [ax,ay]=point(x,y), [bx,by]=point(u,v);
      html+=`<g data-candidate="${action}" stroke="${action===highlight?'#d76025':'#347f94'}" fill="none" opacity=".75"><path d="M${ax},${ay}L${bx},${by}" stroke-width=".6"/><circle cx="${bx}" cy="${by}" r="3.9"/></g>`;
    }
  }
  html+='</svg>';
  const targets=new Set((game?.legal_actions||[]).filter(a=>{const m=catalog.xiangqi_moves[a];return m[1]*stride+m[0]===selectedFrom;}).map(a=>{const m=catalog.xiangqi_moves[a];return m[3]*stride+m[2];}));
  for(let y=0;y<10;y++) for(let x=0;x<9;x++) {
    const a=y*stride+x, pc=game?.board[a]||0, [px,py]=point(x,y), name=pc?xiangqiNames[Math.sign(pc)][Math.abs(pc)]:'';
    html+=`<button class="board-point ${a===selectedFrom?'selected-piece':''} ${targets.has(a)?'legal-destination':''}" data-square="${a}" aria-label="${'abcdefghi'[x]}${y} ${name||'空位'}" style="left:${px}%;top:${py/110*100}%;width:9%;height:${9/110*100}%" ${disabled?'disabled':''}></button>`;
  }
  if(boardMarkup!==html){$('board').querySelector('.xiangqi-squares').innerHTML=html;boardMarkup=html;}
  renderXiangqiPieces(game);
}
function renderXiangqiNetwork(analysis, planes) {
  const stride = analysis.board_stride;
  const kind = $('network-kind').value, probabilities = kind === 'probabilities';
  $('network-position').textContent=`第 ${analysis.turn+1} 手前 · ${analysis.player===1?'红':'黑'}方 · ${planes.precision}`;
  $('network-detail').textContent='2086 个走法按起点筛选；概率在完整动作空间归一化，含当前非法走法。';
  const chooser=document.createElement('select'); chooser.setAttribute('aria-label','策略走法起点');
  for(let y=0;y<10;y++)for(let x=0;x<9;x++)chooser.add(new Option(`${'abcdefghi'[x]}${y}`,y*stride+x));
  chooser.value=selectedFrom ?? 0; $('network-maps').append(chooser);
  const content=document.createElement('div');content.className='xiangqi-policy-heads';$('network-maps').append(content);
  const draw=()=>{content.replaceChildren();for(const head of planes.heads){
    const section=document.createElement('section'), source=Number(chooser.value);
    const moves=catalog.xiangqi_moves.map((m,a)=>({m,a})).filter(({m})=>m[1]*stride+m[0]===source).sort((a,b)=>head.probabilities[b.a]-head.probabilities[a.a]);
    section.innerHTML=`<h3>${networkHeads.find(([,name])=>name===head.name)?.[0]||head.name}</h3><div class="xiangqi-policy-list">`+moves.map(({a})=>`<button data-action="${a}" title="logit ${head.logits[a].toFixed(5)} · 概率 ${percent(head.probabilities[a],3)}%">${coordinate(a,10)} · ${probabilities ? percent(head.probabilities[a],3)+'%' : head.logits[a].toFixed(5)}</button>`).join('')+'</div>';content.append(section);
  }};chooser.addEventListener('change',draw);draw();
}
function renderBoard(game, disabled) {
  if ((game ? state.rule : $('rule').value) === 'xiangqi') { renderXiangqiBoard(game, disabled); return; }
  cancelXiangqiAnimation();
  xiangqiPosition = null;
  $('board').classList.remove('xiangqi-board');
  const size = game?.board_size || Number($('size').value) || catalog.default_size;
  const hex = game ? state.rule === 'hex' : $('rule').value === 'hex';
  // Keep SVG units isotropic; HTML hit targets use the same geometry.
  const step = hex ? 88 / (1.5 * (size - 1) + 2.8) : 100 / (size + 1);
  const radius = step / Math.sqrt(3);
  const height = hex ? Math.sqrt(3) / 2 * (size - 1) * step + 3.8 * radius + 10 : 100;
  const originX = (100 - 1.5 * (size - 1) * step) / 2;
  const originY = (height - Math.sqrt(3) / 2 * (size - 1) * step) / 2;
  const point = (x, y) => hex ? [originX + (x + y / 2) * step, originY + y * step * Math.sqrt(3) / 2] : [(x + 1) * step, (y + 1) * step];
  const end = step * size;
  $('board').classList.toggle('hex-board', hex);
  $('board').style.aspectRatio = `100 / ${height}`;
  $('board').style.setProperty('--hex-unit', `${step}cqw`);
  $('board').style.setProperty('--candidate-unit', `${step}cqw`);
  $('board').setAttribute('aria-label', hex ? 'Hex 棋盘，黑棋连接上下边，白棋连接左右边' : (game ? state.rule : $('rule').value) === 'go' ? '围棋棋盘' : '五子棋棋盘');
  let drawing = `<svg viewBox="0 0 100 ${height}" aria-hidden="true"><g stroke="var(--board-line)" stroke-width=".12">`;
  if (hex) {
    const vertex = (x, y, i) => {
      const [cx, cy] = point(x, y), angle = (30 + i * 60) * Math.PI / 180;
      return [cx + radius * Math.cos(angle), cy + radius * Math.sin(angle)];
    };
    // Follow the exposed cell edges so goal rails never cross playable cells.
    const edges = [[], [], [], []];
    for (let i = 0; i < size; i++) {
      edges[0].push(...[3, 4, 5].map(v => vertex(i, 0, v)));
      edges[1].push(...[5, 0].map(v => vertex(size - 1, i, v)));
      edges[2].push(...[0, 1, 2].map(v => vertex(size - 1 - i, size - 1, v)));
      edges[3].push(...[2, 3].map(v => vertex(0, size - 1 - i, v)));
    }
    // One offset contour gives adjacent goal bands the exact same corner seam.
    // Independent round-capped strokes overlap at the four colour transitions.
    const sides = edges.map(edge => edge.filter((p, i) => !i || Math.hypot(p[0] - edge[i - 1][0], p[1] - edge[i - 1][1]) > 1e-8));
    const contour = sides.flatMap(side => side.slice(0, -1));
    const outer = contour.map((p, i) => {
      const previous = contour[(i + contour.length - 1) % contour.length];
      const next = contour[(i + 1) % contour.length];
      const normal = (a, b) => {
        const length = Math.hypot(b[0] - a[0], b[1] - a[1]);
        return [(b[1] - a[1]) / length, (a[0] - b[0]) / length];
      };
      const a = normal(previous, p), b = normal(p, next);
      const scale = step * .11 / (1 + a[0] * b[0] + a[1] * b[1]);
      return [p[0] + (a[0] + b[0]) * scale, p[1] + (a[1] + b[1]) * scale];
    });
    drawing += '</g><g stroke="var(--hex-rail-outline)" stroke-width=".07" stroke-linejoin="round">';
    let offset = 0;
    sides.forEach((side, i) => {
      const outside = side.map((_, j) => outer[(offset + j) % outer.length]);
      drawing += `<path class="hex-goal" d="M${side.concat(outside.reverse()).join('L')}Z" fill="${i % 2 ? '#faf9f5' : '#343b43'}"/>`;
      offset += side.length - 1;
    });
    drawing += '</g><g class="hex-cells" stroke="var(--hex-line)" stroke-width=".13" stroke-linejoin="round">';
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      const vertices = Array.from({length: 6}, (_, i) => vertex(x, y, i).join(',')).join(' ');
      drawing += `<polygon points="${vertices}" fill="var(--hex-cell)"/>`;
    }
    drawing += `</g><g class="hex-coordinates" fill="var(--muted)" font-family="ui-monospace,monospace" font-size="${Math.min(2.1, step * .30)}" text-anchor="middle" dominant-baseline="central">`;
    for (let i = 0; i < size; i++) {
      for (const [x, y, dx, dy, label] of [
        [i, 0, 0, -1.55 * radius, columns[i]],
        [i, size - 1, 0, 1.55 * radius, columns[i]],
        [0, i, -.98 * step, 0, size - i],
        [size - 1, i, .98 * step, 0, size - i],
      ]) {
        const [cx, cy] = point(x, y);
        drawing += `<text x="${cx + dx}" y="${cy + dy}">${label}</text>`;
      }
    }
    drawing += '</g></svg>';
  } else {
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
  }
  const numbers = new Map((game?.moves || []).map((action, i) => [action, i + 1]));
  for (let action = 0; action < size * size; action++) {
    const stone = game?.board[action] || 0;
    const number = numbers.get(action);
    const last = number && number === game.turn;
    const color = stone === 1 ? 'black' : 'white';
    const [px, py] = point(action % size, Math.floor(action / size));
    const label = coordinate(action, size) + (stone ? `，${stone === 1 ? '黑' : '白'}棋，第 ${number} 手` : '，空位');
    drawing += `<button class="board-point" data-action="${action}" aria-label="${label}" style="left:${px}%;top:${py / height * 100}%;width:${step * .96}%;height:${step * .96 / height * 100}%" ${disabled || stone ? 'disabled' : ''}>${stone ? `<span class="stone ${color} ${last ? 'last' : ''}">${number}</span>` : ''}</button>`;
  }
  if (boardMarkup !== drawing) {
    $('board').innerHTML = drawing;
    boardMarkup = drawing;
  }
  $('board').querySelectorAll('.candidate-marker,.overlay-value').forEach(node => {
    node.parentElement.setAttribute('aria-label', `${coordinate(Number(node.parentElement.dataset.action), size)}，空位`);
    node.remove();
  });
  $('board').querySelectorAll('.highlight').forEach(node => node.classList.remove('highlight'));
  const analysis = state.analysis;
  if (analysis && samePosition(game, analysis) && $('overlay').value !== 'none') {
    const field = $('overlay').value;
    const list = field === 'candidates' ? boardCandidates(analysis) : analysis.candidates;
    const maximum = Math.max(0, ...list.map(c => field === 'candidates' ? c.visits : (c[field] || 0)));
    for (const candidate of list) {
      if (field !== 'candidates' && !(candidate[field] > 0)) continue;
      const cell = $('board').querySelector(`[data-action="${candidate.action}"]`);
      const value = document.createElement('span');
      value.className = field === 'candidates' ? 'candidate-marker' : 'overlay-value';
      if (field === 'candidates') {
        value.style.setProperty('--visit-intensity', `${maximum ? candidate.visits / maximum * 80 : 0}%`);
        value.innerHTML = `<strong>${Number.isFinite(candidate.winrate) ? Math.round(candidate.winrate * 100) : '—'}</strong><small>${compactVisits(candidate.visits)}</small><sup>${candidate.rank}</sup>`;
        if (candidate.rank === 1) value.classList.add('best');
        value.title = candidateDetail(candidate, analysis);
        cell?.setAttribute('aria-label', value.title);
      } else {
        value.style.setProperty('--intensity', `${candidate[field] / maximum * 100}%`);
        value.textContent = size > 15 ? '' : percent(candidate[field]);
      }
      cell?.append(value);
    }
  }
  if (highlight !== null) $('board').querySelector(`[data-action="${highlight}"]`)?.classList.add('highlight');
  $('board').classList.toggle('white-turn', game?.player === -1);
  $('board').classList.toggle('hide-numbers', !$('numbers').checked);
}

function renderHeatmap(id, field, maximumId, analysis) {
  if (analysis.rule === 5) {
    const sorted = [...analysis.candidates].sort((a,b)=>(b[field]||0)-(a[field]||0));
    $(id).style.gridTemplateColumns = 'repeat(3, minmax(0,1fr))'; $(id).style.gridTemplateRows = 'none';
    $(id).innerHTML = sorted.map(c=>`<button data-action="${c.action}" title="${candidateDetail(c,analysis)}">${coordinate(c.action,10)} ${percent(c[field])}%</button>`).join('');
    $(maximumId).textContent = `${percent(sorted[0]?.[field])}%`; return;
  }
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
      const missing = candidate && candidate[field] == null;
      const value = candidate && !missing ? candidate[field] : 0;
      const ratio = maximum > 0 ? value / maximum : 0;
      const detail = `${coordinate(action, size)} · ` + (stone ? `${stone === 1 ? '黑' : '白'}棋` : !candidate ? '不可选点' :
        candidateDetail(candidate, analysis));
      const percent = missing ? '—' : value === 0 ? '·' : value < .001 ? '&lt;0.1' : (value * 100).toFixed(1);
      html += `<div class="heat-cell ${action === analysis.action ? 'chosen-point' : ''}" data-action="${action}" data-value="${value}" data-detail="${detail}" role="img" aria-label="${detail}" title="${detail}" tabindex="0" style="--intensity:${ratio * 100}%">${stone ? `<i class="heat-stone ${stone === 1 ? 'black' : 'white'}"></i>` : `<span>${percent}</span>`}</div>`;
    }
  }
  const map = $(id);
  if (analysis.rule === 4) {
    const pass = candidates.get(size*size);
    html += `<button data-action="${size*size}" style="grid-column:1 / -1">停一手 · ${percent(pass?.[field])}%</button>`;
  }
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
  $('heatmap-position').textContent = `第 ${analysis.turn + 1} 手落子前 · ${sideName(analysis.player)}方视角`;
  $('heatmap-detail').textContent = '悬停 / 聚焦查看数值；描边为搜索建议点。';
  renderHeatmap('prior-map', 'network_prior', 'prior-max', analysis);
  renderHeatmap('search-map', $('search-map-kind').value, 'search-max', analysis);
}
function renderNetworkMaps() {
  const analysis = state?.analysis;
  const planes = analysis?.network_planes;
  $('network-empty').hidden = Boolean(planes);
  $('network-data').hidden = !planes;
  const key = JSON.stringify([analysis?.rule, analysis?.player, analysis?.board, planes, $('network-kind').value, $('network-common-scale').checked]);
  if (networkKey === key) return;
  networkKey = key;
  $('network-maps').replaceChildren();
  $('network-maps').classList.toggle('xiangqi-network', analysis?.rule === 5);
  if (!planes) return;
  if (analysis.rule === 5) { renderXiangqiNetwork(analysis, planes); return; }
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
        const detail = `${label} · ${coordinate(action, size)} · 概率 ${(head.probabilities[action] * 100).toFixed(4)}% · logit ${head.logits[action].toFixed(5)}` +
          (stone ? ` · 已有${stone === 1 ? '黑' : '白'}棋（仍参与网络归一化）` : '');
        const number = probabilities ? (value * 100).toFixed(1) : value.toFixed(1);
        html += `<div class="heat-cell network-cell" data-action="${action}" data-value="${value}" data-detail="${detail}" role="img" aria-label="${detail}" title="${detail}" tabindex="0" style="--intensity:${ratio * 100}%">${stone ? `<i class="heat-stone ${stone === 1 ? 'black' : 'white'}"></i>` : `<span>${number}</span>`}</div>`;
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
    if (analysis.rule === 4) { const pass=document.createElement('p'); pass.textContent=`停一手：${format(values[size*size])}`; section.append(pass); }
    $('network-maps').append(section);
  });
}
function renderCandidates() {
  const analysis = state?.analysis;
  $('candidates').replaceChildren();
  if (!analysis) return;
  const ranked = new Map(boardCandidates(analysis).map(c => [c.action, c.rank]));
  const visible = new Set(ranked.keys());
  const candidates = analysis.candidates.filter(c => (!$('visited-only').checked || c.visits > 0) && (!$('board-candidates-only').checked || visible.has(c.action)))
    .sort((a, b) => {
      const x = a[sortField], y = b[sortField];
      if (x == null || y == null) return x == null ? (y == null ? a.action - b.action : 1) : -1;
      return sortDirection * (x - y) || a.action - b.action;
    });
  $('candidate-count').textContent = `${candidates.length} / ${analysis.candidates.length} 个候选点`;
  const labels = {action: '点位', network_prior: '先验 %', visits: '访问', visit_policy: '访问 %', selection_weight: '选择 %', winrate: '得分 %', draw_rate: '和棋 %', lcb: 'LCB', raw_selection_weight: '原始权重', improved_policy: '改进策略 %'};
  document.querySelectorAll('[data-sort]').forEach(button => {
    button.textContent = labels[button.dataset.sort] + (sortField === button.dataset.sort ? (sortDirection === -1 ? ' ↓' : ' ↑') : '');
    button.parentElement.setAttribute('aria-sort', sortField === button.dataset.sort ? (sortDirection === -1 ? 'descending' : 'ascending') : 'none');
  });
  const fragment = document.createDocumentFragment();
  const probabilityCell = (value, field) => `<td data-field="${field || ''}" class="probability" style="--bar:${(value || 0) * 100}%">${percent(value, 2)}</td>`;
  for (const c of candidates) {
    const row = document.createElement('tr');
    row.dataset.action = c.action;
    row.title = candidateDetail(c, analysis);
    row.classList.toggle('selected', highlight === c.action);
    row.innerHTML = `<td><button data-candidate="${c.action}" class="${c.action === analysis.action ? 'chosen' : ''}" title="在棋盘定位 ${coordinate(c.action, analysis.board_size)}">${coordinate(c.action, analysis.board_size)}${c.action === analysis.action ? ' ★' : ''}<small>${ranked.has(c.action) ? ' #' + ranked.get(c.action) : ''}</small></button></td>${probabilityCell(c.winrate, 'winrate')}<td>${c.visits}</td>${probabilityCell(c.selection_weight)}${probabilityCell(c.network_prior, 'network_prior')}${probabilityCell(c.visit_policy)}${probabilityCell(c.draw_rate)}<td>${Number.isFinite(c.lcb) ? c.lcb.toFixed(3) : '—'}</td><td>${Number.isFinite(c.raw_selection_weight) ? c.raw_selection_weight.toFixed(2) : '—'}</td>${probabilityCell(c.improved_policy)}`;
    fragment.append(row);
  }
  $('candidates').append(fragment);
}
function renderWdl(id, label, values) {
  const names = ['W', 'D', 'L'], classes = ['win', 'draw', 'loss'];
  $(id).classList.add('wdl-row');
  $(id).innerHTML = `<span>${label}</span>` + values.map((value, i) =>
    `<span><i class="wdl-key ${classes[i]}" aria-hidden="true"></i>${names[i]} ${percent(value)}%</span>`).join('');
}
function renderAnalysis() {
  const a = state.analysis;
  $('analysis-empty').hidden = Boolean(a);
  $('analysis-data').hidden = !a;
  $('analysis-move').textContent = a ? `第 ${a.turn + 1} 手前 · ${sideName(a.player)}方` : '—';
  if (!a) {
    analysisKey = null;
    $('candidate-detail').textContent = '悬停候选点查看 W/D/L、LCB 和主变化。';
    renderHeatmaps(); renderNetworkMaps(); return;
  }
  const updating = ['thinking', 'analyzing', 'stopping'].includes(state.phase);
  $('analysis-context').textContent = updating ? (a.final ? '正在更新 · 当前显示上次分析' : `${state.phase === 'stopping' ? '正在停止' : '实时分析'} · ${a.root_search_algo.toUpperCase()}`) :
    state.error ? '操作失败 · 当前显示上次分析' :
    samePosition(state.game, a) ? `当前局面分析 · ${state.analysis_enabled ? '已开启' : '已暂停'}` : '历史分析 · 当前局面尚未分析';
  $('analysis-context').classList.toggle('updating', updating);
  const key = JSON.stringify(a);
  if (key !== analysisKey) {
    $('candidate-detail').textContent = '悬停候选点查看 W/D/L、LCB 和主变化。';
    $('seconds').textContent = `${a.seconds.toFixed(3)}s`;
    $('completed').textContent = a.completed_visits.toLocaleString();
    $('throughput').textContent = a.seconds > 0 ? Math.round(a.completed_visits / a.seconds).toLocaleString() : '—';
    $('value-label').textContent = `${sideName(a.player)}方视角 · 搜索 W−L`;
    $('value').textContent = `${a.root_value >= 0 ? '+' : ''}${a.root_value.toFixed(4)}`;
    renderWdl('wdl', '搜索', a.wdl);
    renderWdl('network-wdl', '网络', a.network_wdl);
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
      if (state.rule === 'go') setGoOptions(game);
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
  $('pass').hidden = rule !== 'go';
  $('pass').disabled = busy || !game || game.finished || viewTurn !== null || (state.mode==='play' && game.player!==state.human);
  $('go-options').hidden = $('rule').value !== 'go';
  $('xiangqi-options').hidden = $('rule').value !== 'xiangqi';
  $('move-count').textContent = `第 ${game?.turn || 0} 手`;
  $('new-game').textContent = game ? '按配置重开棋局' : '创建棋局';
  $('session-mode').textContent = game ? (state.mode === 'manual' ? '手动双方' : `人类执${sideName(state.human)}`) : '未创建';
  const activeModel = catalog.models.find(m => m.id === state.model);
  $('session-info').textContent = game ? `${activeModel?.label || state.model} · AI 落子 ${state.visits}v · 分析${state.analysis_enabled ? '开启' : '暂停'}` : '模型常驻 · 分析可随时开启或暂停';
  $('session-info').title = game ? state.model : '';
  const opening = game?.opening;
  $('opening-info').hidden = !opening;
  $('opening-info').textContent = opening ? `平衡开局 ${opening.moves.length} 手 · 尝试 ${opening.attempts} 次 · ${opening.value_player === 1 ? '黑' : '白'}方网络 W−L ${opening.value >= 0 ? '+' : ''}${opening.value.toFixed(4)} · ${opening.seconds.toFixed(2)}s` : '';
  $('opening-info').title = opening ? `开局种子：${opening.seed}` : '';
  $('state-version').textContent = `单会话 · 状态 v${state.version} · ${state.analysis_config?.device || selectedRun()?.analysis_config.device || catalog.device}`;
  $('undo').disabled = busy || viewTurn !== null || !game?.moves.length || (state.mode === 'play' && !game.moves.some((_, i) => i >= (opening?.moves.length || 0) && (game.players?.[i] ?? (i % 2 === 0 ? 1 : -1)) === state.human));
  $('stop').hidden = !state.busy || !['thinking', 'stopping'].includes(state.phase);
  $('stop').disabled = pending || state.stop_requested || !online;
  $('analyze').innerHTML = `${state.analysis_enabled ? '暂停分析' : '开启分析'} <kbd>A</kbd>`;
  $('analyze').setAttribute('aria-pressed', String(Boolean(state.analysis_enabled)));
  $('analyze').title = state.analysis_enabled ? '暂停持续分析，保持暂停直到手动开启 (A)' : '持续分析当前局面，不落子；落子与悔棋后自动跟随新局面 (A)';
  $('analyze').disabled = busy || !game || (game.finished && !state.analysis_enabled) || (viewTurn !== null && !state.analysis_enabled);
  $('step').disabled = busy || !game || game.finished || viewTurn !== null;
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
    item.innerHTML = `<small>${i + 1}</small><i class="tiny-stone ${sideClass(game.players?.[i] ?? (i % 2 === 0 ? 1 : -1))}"></i>${coordinate(action, game.board_size)}`;
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
  if (game && state.rule === 'go') { analysis_config.go_rules = game.go_rules; analysis_config.komi = game.komi; }
  if (analysis_config.inference_precision === 'auto') analysis_config.inference_precision = analysis_config.device.startsWith('cuda:') ? 'float16' : 'float32';
  $('engine-config').textContent = JSON.stringify(analysis_config, null, 2);
  $('opening-config').textContent = JSON.stringify({parameters: state.opening_config || ((game ? state.rule : $('rule').value)==='hex' ? selectedRun()?.hex_opening : selectedRun()?.opening),
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
  $('settings-note').textContent = edited ? '模式 / 执子 / AI 预算尚未应用；模型、棋盘、棋规和开局需重开生效。' : '模型、棋盘、棋规和开局在创建棋局时生效。AI 访问上限不限制持续分析。';
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
  else if (game?.finished) text = catalog.reasons[game.reason]==='no_result' ? '本局无结果' : game.winner === 0 ? '本局和棋' : `${state.rule==='xiangqi' ? (game.winner===1?'红方':'黑方') : (game.winner===1?'黑棋':'白棋')}获胜`;
  else if (state.phase === 'analyzing') text = `${sideName(game.player)}方持续分析 · ${(state.analysis_progress?.completed_visits || 0).toLocaleString()} 访问 · 可落子 / 悔棋`;
  else if (game) text = state.mode === 'manual' ? `${sideName(game.player)}方落子 · 分析已暂停` : game.player === state.human ? '轮到你落子' : '等待 AI 落子';
  if (game?.finished && catalog.reasons[game.reason] === 'forbidden') text += ' · 黑棋禁手';
  $('status').textContent = text;
  $('status').parentElement.classList.toggle('thinking', state.busy || pending || state.phase === 'analyzing');
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
  const family = model?.manifest.game || 'gomoku';
  for (const option of $('rule').options) option.disabled = ['go','xiangqi'].includes(family) ? option.value!==family : ['go','xiangqi'].includes(option.value);
  if ($('rule').selectedOptions[0]?.disabled) $('rule').value = ['go','xiangqi'].includes(family) ? family : 'freestyle';
  document.querySelectorAll('input[name="human"]').forEach(input => {
    const player = Number(input.value), rule = $('rule').value;
    input.nextElementSibling.innerHTML = `<i class="tiny-stone ${sideClass(player, rule)}"></i> ${sideName(player, rule)}棋`;
  });
  for (const option of $('size').options) {
    option.disabled = !model || Number(option.value) > model.canvas || (family==='xiangqi' && Number(option.value)!==10);
    option.textContent = family==='xiangqi' && Number(option.value)===10 ? '9 × 10' : `${option.value} × ${option.value}`;
  }
  $('opening').querySelector('option[value="empty"]').textContent = family==='xiangqi' ? '标准开局' : '空盘';
  if (family==='xiangqi') $('size').value=10;
  const balanced=$('opening').querySelector('option[value="balanced"]');
  if (balanced) balanced.disabled=['go','xiangqi'].includes(family);
  if (balanced?.disabled) $('opening').value='empty';
  if (model && Number($('size').value) > model.canvas) $('size').value = model.canvas;
  $('new-game').disabled = !model || !online || pending || Boolean(state?.busy);
  const summary = $('model-summary');
  summary.replaceChildren();
  if (model) {
    const tags = [model.manifest.algorithm === 'muzero' ? 'MuZero' : 'AlphaZero', `第 ${model.iteration} 代`, model.manifest.weights || '—', `画布 ${model.canvas}²`];
    for (const text of tags) {
      const tag = document.createElement('span');
      tag.className = 'model-tag';
      tag.textContent = text;
      summary.append(tag, ' ');
    }
    const steps = document.createElement('span');
    steps.className = 'model-steps mono';
    steps.textContent = `累计 ${(model.manifest.checkpoint.total_steps ?? 0).toLocaleString()} 步`;
    summary.append(steps);
  } else summary.textContent = '此配置目录暂无已发布权重';
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
function setGoOptions(config) {
  $('komi').value = config.komi;
  const select = $('go-rules');
  if (![...select.options].some(option => option.value === config.go_rules)) {
    select.add(new Option(config.go_rules, config.go_rules));
  }
  select.value = config.go_rules;
}
function applyRunDefaults() {
  const c = selectedRun()?.analysis_config || catalog.analysis_config;
  $('size').value = c.board_size;
  $('rule').value = c.rule;
  $('visits').value = c.visits;
  if (c.rule === 'go') setGoOptions(c);
  updateSizes();
}
function sessionSettings() {
  return {human: Number(document.querySelector('input[name="human"]:checked').value), visits: Number($('visits').value), mode: $('mode').value};
}
$('settings').addEventListener('submit', event => {
  event.preventDefault();
  command('new', {model: $('model').value, size: Number($('size').value), rule: $('rule').value, opening: $('opening').value, ...($('rule').value==='go'?{komi:Number($('komi').value),go_rules:$('go-rules').value}:{}), ...($('rule').value==='xiangqi'?{fen:$('fen').value}:{}), ...sessionSettings()});
});
$('settings').addEventListener('input', renderSettingsNote);
$('apply-settings').addEventListener('click', () => {
  if (!$('visits').reportValidity()) return;
  command('configure', sessionSettings());
});
$('pass').addEventListener('click', () => command('play', {action:state.game.board_size**2}));
$('board').addEventListener('click', event => {
  const square=event.target.closest('button[data-square]');
  if (square && !square.disabled) {
    if (xiangqiAnimation) return;
    const point=Number(square.dataset.square), game=state.game;
    const action=(game.legal_actions||[]).find(a=>{const [x,y,u,v]=catalog.xiangqi_moves[a];return y*game.board_stride+x===selectedFrom && v*game.board_stride+u===point;});
    if (action!==undefined) { selectedFrom=null; command('play',{action}); }
    else { selectedFrom=game.board[point]*game.player>0?point:null; render(); }
    return;
  }
  const button = event.target.closest('button[data-action]');
  if (button && !button.disabled) command('play', {action: Number(button.dataset.action)});
});
for (const name of ['undo', 'retry', 'step', 'stop']) $(name).addEventListener('click', () => command(name));
$('analyze').addEventListener('click', () => command(state.analysis_enabled ? 'stop' : 'analyze'));
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
for (const id of ['candidate-top', 'candidate-min', 'board-candidates-only']) $(id).addEventListener('input', () => render());
$('candidates').addEventListener('click', event => { const button = event.target.closest('[data-candidate]'); if (button) selectCandidate(Number(button.dataset.candidate)); });
function clearPointPreview() {
  $('board').querySelectorAll('.preview-point').forEach(point => point.classList.remove('preview-point'));
  $('board-coordinate').textContent = '—';
}
for (const id of ['candidates', 'heatmap-data', 'network-data']) {
  const container = $(id);
  for (const eventName of ['mouseover', 'focusin']) container.addEventListener(eventName, event => {
    const point = event.target.closest('[data-action], [data-candidate]');
    clearPointPreview();
    // A previous search can refer to a different board; preview only matching positions.
    if (!point || !samePosition(displayedGame(), state?.analysis)) return;
    const action = Number(point.dataset.action ?? point.dataset.candidate);
    if (state.rule === 'xiangqi') {
      const [x,y,u,v] = catalog.xiangqi_moves[action], stride = state.analysis.board_stride;
      for (const square of [y*stride+x, v*stride+u])
        $('board').querySelector(`[data-square="${square}"]`)?.classList.add('preview-point');
    } else $('board').querySelector(`[data-action="${action}"]`)?.classList.add('preview-point');
    $('board-coordinate').textContent = coordinate(action, state.analysis.board_size);
    const candidate = state.analysis.candidates.find(c => c.action === action);
    if (candidate) $('candidate-detail').textContent = candidateDetail(candidate, state.analysis);
  });
  for (const eventName of ['mouseout', 'focusout']) container.addEventListener(eventName, event => {
    if (!container.contains(event.relatedTarget)) clearPointPreview();
  });
}
for (const eventName of ['mouseover', 'focusin']) {
  $('board').addEventListener(eventName, event => {
    const point = event.target.closest('[data-action], [data-candidate]');
    if (point) {
      const action = Number(point.dataset.action ?? point.dataset.candidate);
      $('board-coordinate').textContent = coordinate(action, displayedGame()?.board_size || Number($('size').value));
      const c = state?.analysis?.candidates.find(c => c.action === action);
      if (c && samePosition(displayedGame(), state.analysis)) $('candidate-detail').textContent = candidateDetail(c, state.analysis);
    }
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
  if (event.ctrlKey || event.metaKey || event.altKey || event.repeat) return;
  if (event.key === 'Escape' && state?.phase === 'analyzing' && !pending && online) {
    event.preventDefault(); command('stop'); return;
  }
  if (event.target.closest('input,select,textarea,[contenteditable="true"]')) return;
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
