const $ = id => document.getElementById(id);
const columns = 'ABCDEFGHJKLMNOPQRSTUVWXYZ';
const gameNames = {go:'围棋',freestyle:'Freestyle',standard:'Standard',renju:'Renju',hex:'Hex',xiangqi:'中国象棋',chinese:'围棋 · 中国规则',japanese:'围棋 · 日本规则',aga:'围棋 · AGA',new_zealand:'围棋 · 新西兰规则', 'tromp-taylor':'围棋 · Tromp–Taylor'};
const pieces = {1:['','車','仕','炮','兵','馬','相','帥'],'-1':['','車','士','砲','卒','馬','象','將']};
const phaseNames = {idle:'就绪',loading:'加载中',opening:'生成平衡开局中',working:'更新局面',diagnosing:'读取网络输出',thinking:'AI 思考中',analyzing:'持续分析',stopping:'正在停止',error:'操作失败'};
let state = null, catalog = null, pending = false, online = false, notice = '';
let viewTurn = null, selectedFrom = null, highlight = null, catalogRevision = -1;
let loadingCatalog = false, boardKey = '', historyKey = '';
let setupInitialized=false, setupRevision=0, setupSent=0, setupFlight=null, openingQueued=false, syncingSetup=false;
let boardSizes=null, setupSyncKey='';
const esc = text => String(text).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = (v, digits=3) => Number.isFinite(v) ? v.toFixed(digits) : '—';
const percent = v => Number.isFinite(v) ? (100*v).toFixed(1) : '—';
const side = (player, game=state?.game?.game) => game==='xiangqi' ? (player==='B'?'红方':'黑方') : (player==='B'?'黑方':'白方');
const coordinate = (x,y,g) => g.game==='xiangqi' ? `${'abcdefghi'[x]}${y}` : `${columns[x]}${g.boardYSize-y}`;
const over = () => Boolean(state?.game?.isGameFinished || state?.resignation);
const interactive = () => online && !pending && !state?.busy;
function samePosition(a,b) {
  return a && b && a.game===b.game && a.boardXSize===b.boardXSize && a.boardYSize===b.boardYSize &&
    a.turnNumber===b.turnNumber && a.currentPlayer===b.currentPlayer && a.board.every((v,i)=>v===b.board[i]) && a.fen===b.fen;
}
function displayedGame() {
  if (!state?.game) return null;
  return viewTurn===null ? state.game : {...state.history[viewTurn], moves:state.game.moves.slice(0,viewTurn)};
}
function canPlay() {
  return interactive() && state?.connected && state.game && !over() && viewTurn===null &&
    (state.mode==='manual' || state.game.currentPlayer===state.human);
}
function savePreferences() {
  try { localStorage.setItem('etazero-ui', JSON.stringify({theme:$('theme').value,numbers:$('numbers').checked})); } catch {}
}
async function api(path, body) {
  const controller = new AbortController(), timer = setTimeout(()=>controller.abort(),30000);
  try {
    const response = await fetch(path,{signal:controller.signal,...(body===undefined?{}:
      {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})});
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
    return value;
  } finally { clearTimeout(timer); }
}
function accept(next) {
  if (state && next.instance===state.instance && next.version<state.version) return;
  if (!state || next.instance!==state.instance || next.game_id!==state.game_id ||
      JSON.stringify(next.game?.moves)!==JSON.stringify(state.game?.moves)) {
    viewTurn=null; selectedFrom=null; highlight=null;
  }
  state=next;
  if (setupFlight?.accepted && !state.busy) {
    const flight=setupFlight; setupFlight=null;
    if (state.error && flight.revision===setupRevision && state.ui_setup) hydrateSetup(state.ui_setup);
    if (state.ui_setup) setupSyncKey=JSON.stringify(state.ui_setup);
  }
  if (catalogRevision!==state.catalog_revision) loadCatalog();
  if (setupInitialized && !setupFlight && setupRevision===setupSent && state.ui_setup &&
      setupSyncKey!==JSON.stringify(state.ui_setup)) hydrateSetup(state.ui_setup);
  render();
  queueMicrotask(drainSetup);
}
async function command(operation, payload={}) {
  if (!state || pending || !online || (state.busy && operation!=='stop')) return;
  pending=true; notice=''; render();
  try { accept(await api('/api/'+operation,{version:state.version,game_id:state.game_id,...payload})); }
  catch(error) { notice=error.message; try {accept(await api('/api/state'));} catch {} }
  finally {pending=false;render();queueMicrotask(drainSetup);}
}
function selectedRun() {return catalog?.runs.find(run=>run.id===$('run').value);}
function selectedModel() {return selectedRun()?.models.find(model=>model.id===$('model').value);}
function populateModels(preferred) {
  const run=selectedRun(); $('model').replaceChildren();
  for (const model of run?.models || []) {
    const label=model.iteration===undefined ? model.path.split('/').at(-1) :
      `Iteration ${model.iteration} · ${model.training_seconds==null?'训练时间未知':`训练 ${(model.training_seconds/3600).toFixed(2)} h`}`;
    $('model').add(new Option(label,model.id));
  }
  if (run?.models.some(m=>m.id===preferred)) $('model').value=preferred;
  else if (run?.current_model) $('model').value=run.current_model;
  if (!run?.models.length) $('model').add(new Option('暂无发布模型',''));
  updateRules();
}
async function loadCatalog() {
  if (loadingCatalog) return;
  loadingCatalog=true;
  const revision=state?.catalog_revision ?? 0;
  try {
    const incoming=await api('/api/catalog');
    const previousRun=$('run').value, previousModel=$('model').value;
    catalog=incoming; catalogRevision=revision;
    $('version').textContent=catalog.version; $('run').replaceChildren();
    for (const run of catalog.runs) $('run').add(new Option(run.label,run.id));
    const preferred=previousModel || state?.selected_model?.id || catalog.default_model;
    const run=catalog.runs.find(r=>r.id===previousRun) || catalog.runs.find(r=>r.models.some(m=>m.id===preferred)) ||
      catalog.runs.find(r=>r.models.length && r.config) || catalog.runs[0];
    if (run) $('run').value=run.id;
    populateModels(preferred);
    if (!setupInitialized) {
      setupInitialized=true;
      if (state?.ui_setup) hydrateSetup(state.ui_setup);
      else if (state?.game) {
        $('mode').value=state.mode; $('human').value=state.human;
        if ([...$('rules').options].some(o=>o.value===state.game.game)) $('rules').value=state.game.game;
        setBoardSize(state.game.boardXSize);
        const params=state.parameters || {}, cap=1125899906842624;
        const kind=params.maxPlayouts<cap?'visits':params.maxVisits<cap?'total':params.maxTime<1e12?'time':'visits';
        $('budget-type').value=kind; $('budget-type').dataset.previous=kind;
        $('budget').value=kind==='visits' && params.maxPlayouts<cap?params.maxPlayouts:
          kind==='total'?params.maxVisits:kind==='time'?params.maxTime:500;
        $('ponder').checked=params.ponderingEnabled===true || params.ponderingEnabled==='true';
        updateSetup();
      } else requestSetup();
    }
  } catch(error) {notice=error.message;render();}
  finally {loadingCatalog=false;queueMicrotask(drainSetup);}
}
const ruleNames = {default:'使用模型配置',freestyle:'五子棋 · Freestyle',standard:'五子棋 · Standard',renju:'五子棋 · Renju',hex:'Hex',go:'围棋 · 配置棋规',xiangqi:'中国象棋',chinese:'围棋 · 中国规则',japanese:'围棋 · 日本规则',aga:'围棋 · AGA',new_zealand:'围棋 · 新西兰规则', 'tromp-taylor':'围棋 · Tromp–Taylor'};
function updateRules() {
  const game=selectedModel()?.game || selectedRun()?.game;
  const choices=['freestyle','standard','renju'].includes(game) ? ['freestyle','standard','renju'] :
    game==='go' ? ['go','chinese','japanese','aga','new_zealand','tromp-taylor'] :
    game && ruleNames[game] ? [game] : ['default','freestyle','standard','renju','hex','go','chinese','japanese','aga','new_zealand','tromp-taylor','xiangqi'];
  const previous=$('rules').value;
  $('rules').replaceChildren(...choices.map(value=>new Option(ruleNames[value],value)));
  $('rules').value=choices.includes(previous)?previous:choices.includes(game)?game:choices[0];
  updateBoardSizes(); updateSetup();
}
function setBoardSize(size) {
  if (boardSizes?.length) {
    let index=boardSizes.indexOf(Number(size));
    if(index<0) index=boardSizes.indexOf(selectedRun()?.canvas);
    if(index<0) index=boardSizes.length-1;
    $('board-size-range').value=String(index);
    $('board-size-value').textContent=`${boardSizes[index]} × ${boardSizes[index]}`;
  } else $('board-size').value=size==null?'':String(size);
}
function updateBoardSizes() {
  const old=selectedBoardSize();
  const sizes=selectedRun()?.trained_board_sizes;
  boardSizes=Array.isArray(sizes) && sizes.length ? sizes : null;
  $('board-size-range').hidden=!boardSizes;
  $('board-size').hidden=Boolean(boardSizes);
  if(boardSizes) {
    $('board-size-range').min='0'; $('board-size-range').max=String(boardSizes.length-1);
    setBoardSize(old);
    $('board-size-note').textContent='可选尺寸来自训练配置。';
  } else {
    $('board-size').value=old==null?'':String(old);
    $('board-size-value').textContent='';
    $('board-size-note').textContent=Array.isArray(selectedRun()?.trained_board_sizes)?'训练配置没有方形尺寸，可手动输入。':'训练尺寸未知，可手动输入。';
  }
}
function selectedBoardSize() {
  return boardSizes?.length ? boardSizes[Number($('board-size-range').value)] :
    $('board-size').value==='' ? null : Number($('board-size').value);
}
function updateSetup() {
  const play=$('mode').value==='play', xi=$('rules').value==='xiangqi', type=$('budget-type').value;
  $('play-settings').hidden=!play; $('budget').disabled=!play;
  $('ponder-note').hidden=!$('ponder').checked;
  $('board-size').hidden=xi || Boolean(boardSizes); $('board-size-range').hidden=xi || !boardSizes;
  $('board-size').disabled=xi; $('board-size-range').disabled=xi || boardSizes?.length===1;
  if(xi){$('board-size-value').textContent='9 × 10';$('board-size-note').textContent='中国象棋使用固定棋盘。';}
  else if(!boardSizes) {$('board-size-value').textContent='';$('board-size-note').textContent=Array.isArray(selectedRun()?.trained_board_sizes)?'训练配置没有方形尺寸，可手动输入。':'训练尺寸未知，可手动输入。';}
  else if(boardSizes) {$('board-size-value').textContent=`${selectedBoardSize()} × ${selectedBoardSize()}`;$('board-size-note').textContent='可选尺寸来自训练配置。';}
  $('budget-label').textContent=type==='time'?'思考时间（秒）':type==='total'?'总 visits':'新增 visits';
  $('budget').min=type==='time'?'0.001':'1';$('budget').step=type==='time'?'any':'1';
  $('opening-hint').textContent=['freestyle','standard','renju','hex'].includes($('rules').value)
    ? '按当前设置生成平衡开局。' : '平衡开局支持五子棋和 Hex。';
  render();
}
function settings() {
  const play=$('mode').value==='play', type=$('budget-type').value, value=Number($('budget').value);
  if(play && (!$('budget').value || !Number.isFinite(value) || value<=0 || (type!=='time' && !Number.isSafeInteger(value))))
    throw new Error(type==='time'?'思考时间必须大于 0 秒':'visits 必须是正整数');
  const cap=1125899906842624;
  return {mode:$('mode').value,human:$('human').value, budget_type:type,
    budget_value:Number.isFinite(value)&&value>0?value:500, ponder:$('ponder').checked,
    parameters:play ? {ponderingEnabled:$('ponder').checked,maxVisits:type==='total'?value:cap,
      maxPlayouts:type==='visits'?value:cap,maxTime:type==='time'?value:1e20} : {ponderingEnabled:false}};
}
function setupPayload(opening=false) {
  const rule=$('rules').value, size=rule==='xiangqi'?null:selectedBoardSize();
  if(size!==null && (!Number.isSafeInteger(size)||size<2))throw new Error('棋盘大小必须是大于 1 的整数');
  const result={model:$('model').value,...settings(),size:size===null?null:[size,size],
    rules:['go','default'].includes(rule)?null:rule};
  if(opening)result.opening={};
  return result;
}
function hydrateSetup(setup) {
  if(!catalog || !setup)return;
  syncingSetup=true;
  const run=catalog.runs.find(r=>r.models.some(m=>m.id===setup.model));
  if(run) {$('run').value=run.id;populateModels(setup.model);}
  $('mode').value=setup.mode; $('human').value=setup.human;
  $('budget-type').value=setup.budget_type; $('budget-type').dataset.previous=setup.budget_type; $('budget').value=setup.budget_value;
  $('ponder').checked=setup.ponder;
  const rule=setup.rules || (selectedRun()?.game==='go'?'go':'default');
  if([...$('rules').options].some(option=>option.value===rule))$('rules').value=rule;
  if(setup.size)setBoardSize(setup.size[0]);
  setupSyncKey=JSON.stringify(setup); updateSetup();syncingSetup=false;
}
function requestSetup(opening=false) {
  if(syncingSetup)return;
  if(!opening && !setupFlight && setupRevision===setupSent && state?.ui_setup) {
    try {
      const candidate=setupPayload();
      const keys=['model','size','rules','mode','human','budget_type','budget_value','ponder'];
      if(keys.every(key=>JSON.stringify(candidate[key])===JSON.stringify(state.ui_setup[key])))return;
    } catch {}
  }
  setupRevision++; if(opening)openingQueued=true;
  queueMicrotask(drainSetup);
}
async function drainSetup() {
  if(!setupInitialized || !state || !online || !selectedModel() || !selectedRun()?.config || pending || setupRevision===setupSent)return;
  if(setupFlight) {
    if(state.busy && ['thinking','opening'].includes(state.phase) && !state.stop_requested)command('stop');
    return;
  }
  if(state.busy) {
    if(['thinking','opening'].includes(state.phase) && !state.stop_requested)command('stop');
    return;
  }
  let payload;
  try {payload=setupPayload(openingQueued);}
  catch(error) {notice=error.message;setupSent=setupRevision;render();return;}
  const flight={revision:setupRevision,accepted:false};setupFlight=flight;setupSent=setupRevision;openingQueued=false;
  pending=true;notice='';render();
  try {
    const next=await api('/api/apply',{version:state.version,game_id:state.game_id,...payload});
    flight.accepted=true;accept(next);
  } catch(error) {
    setupFlight=null;notice=error.message;
    try {accept(await api('/api/state'));} catch {}
    if(flight.revision===setupRevision && state?.ui_setup)hydrateSetup(state.ui_setup);
  } finally {pending=false;render();queueMicrotask(drainSetup);}
}
function view(turn) {
  if (!state?.game) return;
  turn=Math.max(0,Math.min(state.history.length-1,turn));
  viewTurn=turn===state.history.length-1 ? null : turn;
  selectedFrom=null;highlight=null;render();
}
function candidates() {
  const report=state?.analysis?.report;
  if (!report) return [];
  const minimum=Number($('candidate-min').value)||0, count=Math.max(1,Number($('candidate-top').value)||10);
  return [...report.moveInfos].sort((a,b)=>a.order-b.order)
    .filter(row=>row.visits>=report.rootInfo.visits*minimum/100).slice(0,count);
}
function describe(row) {
  const pv=(row.pv||[]).join(' ');
  return `${row.move} · 得分 ${percent(row.winrate)}% · 访问 ${row.visits} · 边访问 ${row.edgeVisits ?? '—'} · `+
    `先验 ${percent(row.prior)}% · D ${percent(row.drawValue)}% · NR ${percent(row.noResultValue)}% · `+
    `LCB ${number(row.lcb)} · utilityLCB ${number(row.utilityLcb)} · 选点值 ${number(row.playSelectionValue)} · PV ${pv}`;
}
function selectCandidate(move) {
  if (!state?.analysis) return;
  const ply=state.analysis.ply;
  viewTurn=ply===state.history.length-1?null:ply; highlight=move;selectedFrom=null;
  const row=state.analysis.report?.moveInfos.find(row=>row.move===move);
  $('candidate-detail').textContent=row?describe(row):move;
  renderBoard();
  renderHistory();
}
function renderBoard() {
  const g=displayedGame(); if (!g) return;
  const active=canPlay(), xi=g.game==='xiangqi', hex=g.game==='hex', nx=g.boardXSize, ny=g.boardYSize;
  const matches=samePosition(g,state.analysis?.position), list=matches && $('overlay').value!=='none'?candidates():[];
  const key=JSON.stringify([state.game_id,g,active,viewTurn,selectedFrom,highlight,list,$('overlay').value,$('numbers').checked]);
  if (key===boardKey) return; boardKey=key;
  const unit=xi?10:hex?88/(1.5*(nx-1)+2.8):100/(nx+1), radius=unit/Math.sqrt(3);
  const height=xi?110:hex?Math.sqrt(3)/2*(ny-1)*unit+3.8*radius+10:(ny+1)*unit;
  const ox=(100-1.5*(nx-1)*unit)/2, oy=(height-Math.sqrt(3)/2*(ny-1)*unit)/2;
  const point=(x,y)=>xi?[10+x*10,100-y*10]:hex?[ox+(x+y/2)*unit,oy+y*unit*Math.sqrt(3)/2]:[(x+1)*unit,(y+1)*unit];
  $('board').classList.toggle('hex-board',hex);$('board').classList.toggle('xiangqi-board',xi);
  $('board').classList.toggle('white-turn',g.currentPlayer==='W');$('board').classList.toggle('hide-numbers',!$('numbers').checked);
  $('board').style.aspectRatio=`100 / ${height}`;
  $('board').style.setProperty('--candidate-unit',`${unit}cqw`);$('board').style.setProperty('--hex-unit',`${unit}cqw`);
  $('board').setAttribute('aria-label',`${gameNames[g.game]}棋盘`);
  let html=`<svg viewBox="0 0 100 ${height}" aria-hidden="true"><g fill="none" stroke="var(--board-line)" stroke-width=".18">`;
  if (xi) {
    for(let y=0;y<10;y++) html+=`<path d="M10,${10+y*10}H90"/>`;
    for(let x=0;x<9;x++) html+=x===0||x===8?`<path d="M${10+x*10},10V100"/>`:`<path d="M${10+x*10},10V50 M${10+x*10},60V100"/>`;
    html+='<path d="M40,10L60,30 M60,10L40,30 M40,80L60,100 M60,80L40,100"/></g><g fill="var(--board-label)" font-size="4" text-anchor="middle"><text x="30" y="56.5">楚 河</text><text x="70" y="56.5">漢 界</text>';
  } else if (hex) {
    for(let y=0;y<ny;y++) for(let x=0;x<nx;x++) {
      const [cx,cy]=point(x,y), vertices=Array.from({length:6},(_,i)=>{
        const angle=(30+i*60)*Math.PI/180;return [cx+radius*Math.cos(angle),cy+radius*Math.sin(angle)];
      });
      html+=`<polygon points="${vertices.map(v=>v.join(',')).join(' ')}" fill="var(--hex-cell)" stroke="var(--hex-line)"/>`;
      for(const [edge,color] of [[y===0?[3,4]:[], '#343b43'],[y===ny-1?[0,1]:[], '#343b43'],[x===0?[2,3]:[], '#faf9f5'],[x===nx-1?[5,0]:[], '#faf9f5']])
        for(const i of edge) html+=`<path d="M${vertices[i]}L${vertices[(i+1)%6]}" stroke="${color}" stroke-width="${unit*.08}"/>`;
    }
  } else {
    for(let x=0;x<nx;x++) html+=`<path d="M${(x+1)*unit},${unit}V${ny*unit}"/>`;
    for(let y=0;y<ny;y++) html+=`<path d="M${unit},${(y+1)*unit}H${nx*unit}"/>`;
  }
  html+='</g><g fill="var(--board-label)" font-size="1.8" text-anchor="middle">';
  if (!xi) {
    for(let x=0;x<nx;x++){const [px,py]=point(x,ny-1);html+=`<text x="${px}" y="${py+unit*.75}">${columns[x]}</text>`;}
    for(let y=0;y<ny;y++){const [px,py]=point(0,y);html+=`<text x="${px-unit*.75}" y="${py+.6}">${ny-y}</text>`;}
  }
  html+='</g>';
  if (xi && matches) {
    const arrows=new Set(list.map(row=>row.move));if(highlight) arrows.add(highlight);
    for(const move of arrows) if (/^[a-i][0-9][a-i][0-9]$/.test(move)) {
      const [x,y]=point(move.charCodeAt(0)-97,Number(move[1])),[u,v]=point(move.charCodeAt(2)-97,Number(move[3]));
      html+=`<g stroke="${move===highlight?'#d76025':'#347f94'}" fill="none"><path d="M${x},${y}L${u},${v}" stroke-width=".65"/><circle cx="${u}" cy="${v}" r="3.8"/></g>`;
    }
  }
  html+='</svg>';
  const numbers=new Map((g.moves||[]).map(([player,move],i)=>[move,i+1]));
  const byMove=new Map(list.map(row=>[row.move,row]));
  const maximum=Math.max(1,...list.map(row=>$('overlay').value==='prior'?row.prior:row.visits));
  for(let y=0;y<ny;y++)for(let x=0;x<nx;x++) {
    const index=y*nx+x, move=coordinate(x,y,g), pc=(xi?g.pieces:g.board)[index], [px,py]=point(x,y);
    const destinations=xi && selectedFrom ? state.game.legalMoves.filter(m=>m.slice(0,2)===selectedFrom):[];
    const target=destinations.some(m=>m.slice(2)===move), row=byMove.get(move);
    const disabled=!active || (!xi && !state.game.legalMoves.includes(move));
    html+=`<button class="board-point ${selectedFrom===move?'selected-piece':''} ${target?'legal-destination':''} ${highlight===move?'highlight':''}" data-point="${move}" aria-label="${move}${pc?' '+(xi?pieces[Math.sign(pc)][Math.abs(pc)]:pc>0?'黑棋':'白棋'):' 空位'}" style="left:${px}%;top:${py/height*100}%;width:${unit*.96}%;height:${unit*.96/height*100}%" ${disabled?'disabled':''}>`;
    if (pc) html+=xi?`<span class="xiangqi-piece ${pc>0?'red-piece':'black-piece'}">${pieces[Math.sign(pc)][Math.abs(pc)]}</span>`:
      `<span class="stone ${pc>0?'black':'white'} ${numbers.get(move)===(g.moves||[]).length?'last':''}">${numbers.get(move)||''}</span>`;
    else if(row) {
      const field=$('overlay').value;
      html+=field==='candidates'?`<span class="candidate-marker" style="--visit-intensity:${row.visits/maximum*80}%"><strong>${percent(row.winrate)}</strong><small>${row.visits}</small><sup>${row.order+1}</sup></span>`:
        `<span class="overlay-value" style="--intensity:${row[field]/maximum*100}%">${field==='prior'?percent(row.prior):row.visits}</span>`;
    }
    html+='</button>';
  }
  $('board').innerHTML=html;
}
function renderHistory() {
  const moves=state?.game?.moves||[], turn=viewTurn??moves.length;
  $('timeline').max=moves.length;$('timeline').value=turn;
  $('history-count').textContent=`${moves.length} 手`;
  $('view-note').hidden=viewTurn===null;$('view-note').textContent=`回看会话第 ${turn} 手后的原生局面`;
  $('branch').hidden=viewTurn===null;$('branch').disabled=!interactive();
  $('first').disabled=$('prev').disabled=turn===0;
  $('next').disabled=$('live').disabled=viewTurn===null;
  const key=JSON.stringify([moves,turn,state?.game?.game]);
  if(key===historyKey)return;historyKey=key;
  $('history').innerHTML=moves.map(([player,move],i)=>`<button data-turn="${i+1}" class="${turn===i+1?'selected':''}">${i+1}. ${esc(side(player))} ${esc(move)}</button>`).join('');
}
function renderPolicy() {
  if (!$('policy-panel').open || !state?.analysis?.report) return;
  const {report,position:g}=state.analysis, source=$('policy-source').value, values=report[source];
  $('policy-map').hidden=g.game==='xiangqi';$('policy-map').replaceChildren();$('policy-list').replaceChildren();
  if(!values) {
    $('policy-note').textContent=`当前原生报告没有 ${source}；HumanSL 策略需要加载可用的次级模型。`;
    return;
  }
  $('policy-note').textContent=`来源 ${source}；负值表示被屏蔽的动作。五子棋与 Hex 的末尾 pass 槽不可落子。`;
  if(g.game==='xiangqi') {
    const select=document.createElement('select');select.setAttribute('aria-label','策略起点');
    for(let y=0;y<10;y++)for(let x=0;x<9;x++)select.add(new Option(`${'abcdefghi'[x]}${y}`,`${x},${y}`));
    const saved=$('policy-list').dataset.from || '0,0';select.value=saved;
    const rows=document.createElement('div');rows.className='native-policy-list';
    const draw=()=>{
      $('policy-list').dataset.from=select.value;
      rows.replaceChildren();
      for(const [i,m] of catalog.xiangqi_moves.entries()) if(`${m[0]},${m[1]}`===select.value) {
        const move=`${'abcdefghi'[m[0]]}${m[1]}${'abcdefghi'[m[2]]}${m[3]}`, button=document.createElement('button');
        button.textContent=`${move} · ${values[i]<0?'—':percent(values[i])+'%'}`;
        button.title=`${move}: ${values[i]}`;
        button.onclick=()=>selectCandidate(move);rows.append(button);
      }
    };
    select.onchange=draw;$('policy-list').append(select,rows);draw();
  } else {
    const max=Math.max(0,...values),nx=g.boardXSize,ny=g.boardYSize;
    $('policy-map').style.gridTemplateColumns=`repeat(${nx},1fr)`;
    $('policy-map').style.aspectRatio=`${nx} / ${ny}`;
    $('policy-map').innerHTML=values.slice(0,nx*ny).map((v,i)=>{
      const move=coordinate(i%nx,Math.floor(i/nx),g);
      return `<button class="heat-cell" data-candidate="${move}" title="${move}: ${v}" style="--intensity:${v>=0&&max?v/max*100:0}%">${v<0?'—':percent(v)}</button>`;
    }).join('');
    if(values.length>nx*ny) $('policy-list').textContent=`pass · ${values[nx*ny]<0?'已屏蔽':percent(values[nx*ny])+'%'}`;
  }
}
function renderAnalysis() {
  const analysis=state?.analysis,report=analysis?.report;
  $('analysis-empty').hidden=Boolean(report);$('analysis-data').hidden=!report;
  if(!report)return;
  const root=report.rootInfo,perspective=side(analysis.position.analysisPerspective,analysis.position.game);
  $('analysis-status').textContent={running:'分析中',complete:'已完成',stopped:'已停止',error:'中断'}[analysis.status];
  $('analysis-context').textContent=`会话第 ${analysis.ply} 手后 · ${side(analysis.position.currentPlayer,analysis.position.game)}行棋`;
  $('completed').textContent=root.visits;$('winrate').textContent=percent(root.winrate)+'%';
  $('value').textContent=number(root.winValue-root.lossValue);
  $('perspective').textContent=`报告视角：${perspective}；棋盘角标沿用原生候选顺序。`;
  const wdl=[root.winValue,root.drawValue,root.lossValue,root.noResultValue];
  ['win','draw','loss','no-result'].forEach((key,i)=>{$('wdl-'+key).style.width=`${100*wdl[i]}%`;});
  $('wdl').textContent='W / D / L / NR · '+wdl.map(v=>percent(v)+'%').join(' / ');
  $('network-wdl').textContent=`网络得分 ${percent(root.rawWinrate)}% · D ${percent(root.rawDrawProb)}% · NR ${percent(root.rawNoResultProb)}%`;
  const field=$('sort').value;
  const rows=report.moveInfos.filter(row=>!$('visited-only').checked || row.visits>0)
    .sort((a,b)=>field==='order'?a.order-b.order:b[field]-a[field]);
  $('candidate-count').textContent=`${rows.length} 个原生候选`;
  $('candidates').innerHTML=rows.map(row=>`<tr><td><button data-candidate="${esc(row.move)}">${esc(row.move)}${analysis.move===row.move?' ★':''}</button></td>`+
    `<td>${percent(row.winrate)}</td><td>${row.visits}</td><td>${row.edgeVisits??'—'}</td><td>${percent(row.prior)}</td><td>${percent(row.drawValue)}</td><td>${percent(row.noResultValue)}</td><td>${number(row.lcb)}</td><td>${number(row.utility)}</td><td>${number(row.playSelectionValue)}</td></tr>`).join('');
  $('raw-analysis').textContent=JSON.stringify(report,null,2);renderPolicy();
}
function renderRawNN() {
  const allowed=state?.game?.game==='xiangqi'?[0,2]:state?.game?.game==='hex'?[0,3]:[0,1,2,3,4,5,6,7];
  const sym=$('nn-symmetry');
  if(sym.dataset.choices!==allowed.join(',')) {
    const previous=Number(sym.value);sym.replaceChildren();
    allowed.forEach(n=>sym.add(new Option(String(n),String(n))));
    if(allowed.includes(previous))sym.value=String(previous);
    sym.dataset.choices=allowed.join(',');
  }
  $('nn-optimism').disabled=$('nn-source').value==='human';
  $('diagnose').disabled=!interactive()||!state?.connected||!state?.game||over()||viewTurn!==null;
  const data=state?.raw_nn;
  $('nn-empty').hidden=Boolean(data);$('nn-data').hidden=!data;
  if(!data || !$('nn-panel').open)return;
  const g=data.position,report=data.report,values=report.values;
  $('nn-context').textContent=`会话第 ${data.ply} 手后 · ${side(g.currentPlayer,g.game)}行棋`+
    (samePosition(displayedGame(),g)?'':' · 与当前展示局面不同');
  $('nn-command').textContent=`来源：${data.command} · ${data.source==='human'?'次级模型':'主模型'}`;
  $('nn-values').textContent=`${side('W',g.game)}视角 W / D / L / NR · `+
    [values.whiteWin,values.draw,values.whiteLoss,values.noResult].map(v=>percent(v)+'%').join(' / ');
  $('nn-text').textContent=data.text;
  $('nn-map').hidden=g.game==='xiangqi';$('nn-map').replaceChildren();$('nn-list').replaceChildren();
  if(g.game==='xiangqi') {
    const select=document.createElement('select');select.setAttribute('aria-label','网络诊断起点');
    const origins=[...new Set(report.policy.map(row=>row.move.slice(0,2)))];
    origins.forEach(move=>select.add(new Option(move,move)));
    if(origins.includes($('nn-list').dataset.from))select.value=$('nn-list').dataset.from;
    const rows=document.createElement('div');rows.className='native-policy-list';
    const draw=()=>{
      $('nn-list').dataset.from=select.value;
      rows.replaceChildren();
      for(const row of report.policy.filter(row=>row.move.startsWith(select.value))) {
        const cell=document.createElement('span');cell.className='nn-entry';cell.dataset.move=row.move;
        cell.textContent=`${row.move} · ${row.probability===null?'—':percent(row.probability)+'%'}`;
        cell.title=`${row.move}: ${row.probability===null?'NAN':row.probability}`;rows.append(cell);
      }
    };
    select.onchange=draw;$('nn-list').append(select,rows);draw();
  } else {
    const nx=g.boardXSize,ny=g.boardYSize,max=Math.max(0,...report.policy.map(row=>row.probability??0));
    $('nn-map').style.gridTemplateColumns=`repeat(${nx},1fr)`;$('nn-map').style.aspectRatio=`${nx} / ${ny}`;
    for(const row of report.policy.slice(0,nx*ny)) {
      const cell=document.createElement('span');cell.className='heat-cell';cell.dataset.move=row.move;
      cell.textContent=row.probability===null?'—':percent(row.probability);
      cell.title=`${row.move}: ${row.probability===null?'NAN':row.probability}`;
      cell.style.setProperty('--intensity',`${max?(row.probability??0)/max*100:0}%`);$('nn-map').append(cell);
    }
    const pass=report.policy.at(-1).probability;
    $('nn-list').textContent=`pass · ${pass===null?'已屏蔽':percent(pass)+'%'}`;
  }
}
function render() {
  $('connection').textContent=online?'工作台已连接':'断线 · 重连中';$('connection-dot').classList.toggle('online',online);
  const error=notice || state?.error || '';$('error').hidden=!error;$('error').textContent=error;
  const run=selectedRun(),model=selectedModel();
  $('model-summary').textContent=!run?.config ? '所选目录缺少原生 gtp.cfg，无法创建棋局。' : model ? '' : '暂无可用模型';
  $('model-summary').hidden=!$('model-summary').textContent;
  $('catalog-errors').textContent=(catalog?.errors||[]).map(item=>`${item.run}: ${item.error}`).join('\n');
  $('refresh-models').disabled=!interactive();
  $('balanced-opening').disabled=!online || !model || !run?.config || ['opening','stopping'].includes(state?.phase) || !['freestyle','standard','renju','hex'].includes($('rules').value);
  const g=state?.game,available=interactive()&&state?.connected&&g;
  $('analyze').disabled=$('step').disabled=!available || over() || viewTurn!==null;
  $('analyze').textContent=state?.analysis_enabled?'暂停分析 (A)':'开启分析 (A)';
  $('analyze').setAttribute('aria-pressed',Boolean(state?.analysis_enabled));
  $('stop').hidden=!['thinking','analyzing','opening','stopping'].includes(state?.phase);$('stop').disabled=pending||state?.stop_requested;
  $('stop').textContent=state?.phase==='opening'?'停止生成':'停止';
  const undoable=g && (state.resignation || (state.mode==='manual'?g.moves.length:g.moves.some(([p])=>p===state.human)));
  $('undo').disabled=!available || !undoable;
  $('pass').hidden=g?.game!=='go';$('pass').disabled=!canPlay()||!g?.legalMoves.includes('pass');
  $('retry').hidden=!available || over() || state.mode!=='play' || g.currentPlayer===state.human;
  $('opening-info').hidden=!state?.opening;
  $('opening-info').textContent=state?.opening?`初始平衡开局 · ${state.opening.moves.length} 手 · 来源：原生 KataGomo 开局算法` : '';
  $('state-version').textContent=`单会话 · 状态 ${state?.version??'—'}`;
  if(g) {
    $('rule-badge').textContent=gameNames[g.game];$('session-mode').textContent=state.mode==='manual'?'手动双方':'人机对弈';
    $('status').textContent=state.phase==='opening'?'生成平衡开局中 · 成功后替换当前棋局':over()?`对局结束 · ${state.resignation?.result || g.result}`:`${phaseNames[state.phase]} · ${side(g.currentPlayer)}行棋`;
    $('move-count').textContent=`原生第 ${g.turnNumber} 手`;
    $('session-info').textContent=`${g.boardXSize} × ${g.boardYSize} · ${state.parameters?.useGumbel?'Gumbel':'PUCT'} · ${state.connected?'引擎已连接':'引擎已断开'}`;
    $('rule-note').textContent=g.game==='hex'?'黑方连接上下边，白方连接左右边。':g.game==='xiangqi'?'先选择棋子，再选择原生合法落点；红方在下。':g.game==='renju'?'黑方禁手按原生规则落子判负。':g.game==='go'?`原生棋规：${JSON.stringify(g.rules)}`:'合法性与终局由原生棋规裁定。';
  } else $('status').textContent=state?.busy?phaseNames[state.phase]:'选择模型后自动开始';
  renderBoard();renderHistory();renderAnalysis();renderRawNN();
}
$('settings').onsubmit=event=>event.preventDefault();
$('balanced-opening').onclick=()=>requestSetup(true);
$('run').onchange=()=>{populateModels();requestSetup();};
$('model').onchange=()=>{updateRules();requestSetup();};
$('rules').onchange=()=>{updateSetup();requestSetup();};
$('mode').onchange=()=>{updateSetup();requestSetup();};
$('human').onchange=()=>requestSetup();
$('ponder').onchange=()=>{updateSetup();requestSetup();};
$('budget-type').onchange=()=>{
  const type=$('budget-type').value, previous=$('budget-type').dataset.previous || 'visits';
  if((type==='time') !== (previous==='time'))$('budget').value=type==='time'?'5':'500';
  $('budget-type').dataset.previous=type;updateSetup();requestSetup();
};
$('budget').onchange=()=>requestSetup();
$('board-size').onchange=()=>requestSetup();
for(const id of ['budget','board-size'])$(id).addEventListener('keydown',event=>{
  if(event.key==='Enter'){event.preventDefault();$(id).blur();requestSetup();}
});
$('board-size-range').oninput=()=>updateSetup();
$('board-size-range').onchange=()=>requestSetup();
$('refresh-models').onclick=()=>command('refresh');
$('analyze').onclick=()=>command(state?.analysis_enabled?'stop':'analyze');
for(const name of ['step','stop','undo','retry'])$(name).onclick=()=>command(name);
$('pass').onclick=()=>command('play',{move:'pass'});
$('branch').onclick=()=>command('branch',{turn:viewTurn});
$('timeline').oninput=()=>view(Number($('timeline').value));
$('first').onclick=()=>view(0);$('prev').onclick=()=>view((viewTurn??state?.game?.moves.length??0)-1);
$('next').onclick=()=>view((viewTurn??0)+1);$('live').onclick=()=>view(state?.game?.moves.length??0);
$('history').onclick=event=>{const row=event.target.closest('[data-turn]');if(row)view(Number(row.dataset.turn));};
$('show-analysis').onclick=()=>{view(state.analysis.ply);};
$('show-nn').onclick=()=>{view(state.raw_nn.ply);};
$('diagnose').onclick=()=>{
  const source=$('nn-source').value;
  if(source==='main' && !$('nn-optimism').reportValidity())return;
  command('diagnose',{source,symmetry:Number($('nn-symmetry').value),
    ...(source==='main'?{optimism:Number($('nn-optimism').value)}:{})});
};
$('nn-panel').ontoggle=renderRawNN;$('nn-source').onchange=renderRawNN;
$('board').onclick=event=>{
  const cell=event.target.closest('[data-point]');if(!cell||!canPlay())return;
  const move=cell.dataset.point;
  if(state.game.game!=='xiangqi')return command('play',{move});
  const chosen=selectedFrom+move;
  if(selectedFrom && state.game.legalMoves.includes(chosen)){selectedFrom=null;command('play',{move:chosen});}
  else {selectedFrom=state.game.legalMoves.some(m=>m.startsWith(move))?move:null;renderBoard();}
};
$('board').onpointerover=event=>{const cell=event.target.closest('[data-point]');if(cell)$('board-coordinate').textContent=cell.dataset.point;};
for(const id of ['candidates','policy-map']){
  $(id).onclick=event=>{const cell=event.target.closest('[data-candidate]');if(cell)selectCandidate(cell.dataset.candidate);};
  for(const name of ['pointerover','focusin'])$(id).addEventListener(name,event=>{
    const cell=event.target.closest('[data-candidate]'),row=state?.analysis?.report?.moveInfos.find(r=>r.move===cell?.dataset.candidate);
    if(row)$('candidate-detail').textContent=describe(row);
  });
}
for(const id of ['overlay','candidate-top','candidate-min','sort','visited-only'])$(id).onchange=render;
$('numbers').onchange=()=>{savePreferences();renderBoard();};$('policy-panel').ontoggle=renderPolicy;
$('policy-source').onchange=renderPolicy;
document.addEventListener('keydown',event=>{
  if(event.target.closest('input,select,textarea')||event.ctrlKey||event.metaKey||event.altKey)return;
  const actions={a:'analyze',s:'step',u:'undo',Escape:'stop',ArrowLeft:'prev',ArrowRight:'next',Home:'first',End:'live'};
  const id=actions[event.key]||actions[event.key.toLowerCase()];if(id){event.preventDefault();if(!$(id).disabled&&!$(id).hidden)$(id).click();}
});
try{const saved=JSON.parse(localStorage.getItem('etazero-ui')||'{}');if(typeof saved.numbers==='boolean')$('numbers').checked=saved.numbers;}catch{}
if(matchMedia('(max-width:680px)').matches)$('setup-details').open=false;
(async()=>{
  while(true){
    try{const next=await api('/api/state'+(state?`?since=${state.version}`:''));online=true;accept(next);}
    catch(error){online=false;render();await new Promise(resolve=>setTimeout(resolve,1000));}
  }
})();
