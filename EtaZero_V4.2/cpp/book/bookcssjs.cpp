#include "../book/book.h"

const std::string Book::BOOK_CSS = R"%%(


body {
  font-family: BlinkMacSystemFont,-apple-system,"Segoe UI",Roboto,Oxygen,Ubuntu,Cantarell,"Fira Sans","Droid Sans","Helvetica Neue",Helvetica,Arial,sans-serif;
  font-weight:500;
}

a:link { text-decoration: none; color:#485fc7 }
a:visited { text-decoration: none; color:rgb(85, 26, 139) }
a:hover { text-decoration: none; color:#000000 }
a:active { text-decoration: none; color:#ff0055 }

svg {
  font-family:sans-serif;
  font-weight:400;
  max-width:100%;
  height:auto;
}

.moveTableContainer { overflow-x:auto; }
.jointControls { margin:0.75em 0; }
.jointControls select { margin-left:0.5em; font:inherit; }
.xiangqi-piece text { font-family:"Noto Serif CJK SC",serif; font-weight:bold; pointer-events:none; }
.xiangqi-piece[data-selectable="true"] { cursor:pointer; }
.xiangqi-piece.selected circle { stroke:#2676c7; stroke-width:0.09; }

h1 {
  margin-top:10px;
  margin-bottom:10px;
}

.backLink {
  margin-top:6px;
  margin-bottom:6px;
}

.moveTable {
  display: table;
  border-style: solid;
  border-width: 1px;
  border-collapse: collapse;
}

.moveTableHeader {
  display: table-row;
  font-weight: bold;
  /*
  border-style: solid;
  border-width: 1px;
  border-color: black;
  */
}

.moveTableRow {
  display: table-row;
  /*
  border-style: solid;
  border-width: 1px;
  border-color: black;
  */
}

.moveTableRowLinked:link { text-decoration: none; color: rgb(0, 0, 238) }
.moveTableRowLinked:visited { text-decoration: none; color: rgb(85, 26, 139) }
.moveTableRowLinked:hover { text-decoration: none; }
.moveTableRowLinked:active { text-decoration: none; color:#ff0000 }

.moveTableCell {
  display: table-cell;
  padding: 10px;
  text-decoration:none;
}

.moveTableRow:hover {
  outline-style: solid;
  outline-width: 1px;
  outline-color: black;
}
.moveTableRow.moveHovered {
  outline-style: solid;
  outline-width: 1px;
  outline-color: black;
}

#whoToPlay {
  padding:10px;
}

.stoneShadow {
  display: block;
  opacity: 0.001;
}
.stoneShadow.tableHovered {
  display: block;
  opacity: 0.3;
}
.stoneShadowUnhoverable {
  display: block;
  opacity: 0.001;
}
.stoneShadowUnhoverable.tableHovered {
  display: block;
  opacity: 0.3;
}
.stoneShadow:hover {
  display: block;
  opacity: 0.3;
}

.legend {
  padding-top:10px;
}
.legend ul {
  margin-top:0.5em;
}

)%%";

const std::string Book::BOOK_JS1 = R"%%(

let url = new URL(window.location.href);
let sym = Number(url.searchParams.get("symmetry"));
if(!allowedSymmetries.includes(sym))
  sym = 0;

const badnessColors = [
  [0.00, [100,255,245]],
  [0.12, [120,235,130]],
  [0.30, [205,235,60]],
  [0.70, [255,100,0]],
  [1.00, [200,0,0]],
  [2.00, [50,0,0]],
];

function rgba(values,alpha) {
  return "rgba(" + values.join(",") + "," + alpha + ")";
}

function clamp(x,x0,x1) {
  return Math.min(Math.max(x,x0),x1);
}

function getBadnessColor(bestWinLossValue, winLossDiff, sqrtPolicyDiff, alpha) {
  winLossDiff = (nextPla == 1 ? 1 : -1) * winLossDiff;
  let x = winLossDiff*0.8 - 0.05 * sqrtPolicyDiff;
  let losingness = Math.max(0.0, (nextPla == 1 ? 1 : -1) * 0.5 * bestWinLossValue);
  x += losingness * 0.6 + (x * 1.25 * losingness);

  for(let i = 0; i<badnessColors.length; i++) {
    [x1,c1] = badnessColors[i];
    if(x < x1) {
      if(i <= 0)
        return rgba(c1,alpha);
      [x0,c0] = badnessColors[i-1];
      interp = (x-x0)/(x1-x0);
      return rgba([c0[0] + (c1[0]-c0[0])*interp, c0[1] + (c1[1]-c0[1])*interp, c0[2] + (c1[2]-c0[2])*interp],alpha);
    }
  }
  return rgba(badnessColors[badnessColors.length-1][1],alpha);
}

function getBadnessColorOfMoveIdx(idx, alpha) {
  let moveData = moves[idx];
  let winLossDiff = moveData["wl"] - moves[0]["wl"];
  let sqrtPolicyDiff = Math.sqrt(moveData["p"]) - Math.sqrt(moves[0]["p"]);
  let moveValueColor = getBadnessColor(moves[0]["wl"], winLossDiff, sqrtPolicyDiff, alpha);
  return moveValueColor;
}

function getSymPos(pos) {
  let y = Math.floor(pos / bSizeX);
  let x = pos % bSizeX;
  if(sym & 1)
    y = bSizeY-1-y;
  if(sym & 2)
    x = bSizeX-1-x;
  if(sym >= 4 && bSizeX == bSizeY) {
    let tmp = x;
    x = y;
    y = tmp;
  }
  return x + y*bSizeX;
}
function getInvSymPos(pos) {
  let y = Math.floor(pos / bSizeX);
  let x = pos % bSizeX;
  if(sym >= 4 && bSizeX == bSizeY) {
    let tmp = x;
    x = y;
    y = tmp;
  }
  if(sym & 1)
    y = bSizeY-1-y;
  if(sym & 2)
    x = bSizeX-1-x;
  return x + y*bSizeX;
}

function compose(sym1,sym2) {
  if(sym1 & 0x4)
    sym2 = (sym2 & 0x4) | ((sym2 & 0x2) >> 1) | ((sym2 & 0x1) << 1);
  return sym1 ^ sym2;
}

function getLinkForPos(pos) {
  if(!(pos in links))
    return null;
  let linkPath = links[pos];
  if(linkPath.length == 0)
    return null;
  // This is the symmetry we need to add as a GET parameter in the URL for the linked position.
  let symmetryToAlign = linkSyms[pos];
  // Except we need to composite it with our current symmetry too.
  symmetryToAlign = compose(symmetryToAlign, sym);
  return linkPath + "?symmetry=" + symmetryToAlign;
}

let body = document.getElementsByTagName("body")[0];
const coordChars = "ABCDEFGHJKLMNOPQRSTUVWXYZ";

let hoverShadowEltsByMove = {};
let hoverTableEltsByMove = {};
const gameNames = {go:"Go",freestyle:"Freestyle",standard:"Standard",renju:"Renju",hex:"Hex",xiangqi:"Xiangqi"};
const firstPlayerName = game == "xiangqi" ? "Red" : "Black";
const secondPlayerName = game == "xiangqi" ? "Black" : "White";

function boardPoint(x,y) {
  if(game == "xiangqi") return [x,bSizeY-1-y];
  if(game == "hex") return [x+y/2,y*Math.sqrt(3)/2];
  return [x,y];
}

function jointMoveString(action, displayed = false) {
  const coords = [];
  for(let offset of [0,2]) {
    let pos = action[offset] + action[offset+1]*bSizeX;
    if(displayed) pos = getSymPos(pos);
    coords.push("abcdefghi"[pos%bSizeX] + Math.floor(pos/bSizeX));
  }
  return coords.join("");
}

let selectedFrom = null;
let hoveredMove = null;
const jointCandidates = [];
const jointPieces = [];
const jointOrigins = new Set();
function updateJointCandidates() {
  for(const candidate of jointCandidates)
    candidate.element.style.display = candidate.from == selectedFrom || candidate.index == hoveredMove ? "" : "none";
  for(const piece of jointPieces)
    piece.element.classList.toggle("selected",piece.pos == selectedFrom);
}
function selectFrom(pos) {
  selectedFrom = pos;
  const select = document.getElementById("fromPiece");
  if(select) select.value = String(pos);
  updateJointCandidates();
}

{
  let title = document.createElement("h1");
  title.appendChild(document.createTextNode("KataGo Opening Book " + (game == "go" ? "" : gameNames[game] + " ") + bSizeX + " x " + bSizeY));
  title.id = "title";
  body.appendChild(title);
}
{
  let link = document.createElement("div");
  link.classList.add("backLink");
  let innerHtml = '';
  innerHtml += '<a href="'+rulesLink+'">'+rulesLabel+'</a> <br/> ';
  innerHtml += '<a href="/">Back to home page</a> <br/> ';
  innerHtml += '<a href="../root/root.html">Back to root position</a>';
  if(pLink != '')
    innerHtml += '&emsp;<a href="'+pLink+'?symmetry='+compose(pSym,sym)+'">Canonical parent</a>';
  link.innerHTML = innerHtml;
  body.appendChild(link);
}

if(game == "xiangqi") {
  const origins = jointOrigins;
  for(const move of moves)
    if(move.v > 0 && move.actions)
      for(const action of move.actions) origins.add(action[0]+action[1]*bSizeX);
  if(origins.size > 0) {
    const label = document.createElement("label");
    label.classList.add("jointControls");
    label.style.display = "block";
    label.textContent = "Select a piece, then a numbered destination: ";
    const select = document.createElement("select");
    select.id = "fromPiece";
    for(const pos of origins) {
      const option = document.createElement("option");
      const displayed = getSymPos(pos);
      option.value = String(pos);
      option.textContent = "abcdefghi"[displayed%bSizeX] + Math.floor(displayed/bSizeX);
      select.appendChild(option);
    }
    select.addEventListener("change",() => selectFrom(Number(select.value)));
    label.appendChild(select);
    body.appendChild(label);
    selectedFrom = Number(select.value);
  }
}

)%%";
const std::string Book::BOOK_JS2 = R"%%(

let svgNS = "http://www.w3.org/2000/svg";
{
  const edgeScaleSize = Math.sqrt((bSizeX+1)*(bSizeY+1));
  const pixelsPerTile = 50.0 / Math.sqrt(edgeScaleSize / 8);
  const borderTiles = 1.1;
  const strokeWidth = 0.05;
  const stoneRadius = 0.5;
  const stoneInnerRadius = 0.45;
  const stoneBlackFill = "#222";
  const stoneWhiteFill = "#FFF";
  const markerFontSize = 0.7;
  const coordFontSize = 0.4;
  const coordSpacing = 0.75;
  const backgroundColor = "#DCB35C";

  const boardWidth = game == "hex" ? bSizeX-1+(bSizeY-1)/2 : bSizeX-1;
  const boardHeight = game == "hex" ? (bSizeY-1)*Math.sqrt(3)/2 : bSizeY-1;
  let boardSvg = document.createElementNS(svgNS, "svg");
  boardSvg.id = "bookBoard";
  boardSvg.setAttribute("aria-label",gameNames[game]+" board");
  boardSvg.setAttribute("width", pixelsPerTile * (2 * borderTiles + boardWidth));
  boardSvg.setAttribute("height", pixelsPerTile * (2 * borderTiles + boardHeight));
  boardSvg.setAttribute("viewBox", "-" + borderTiles + " -" + borderTiles + " " + (2 * borderTiles + boardWidth) + " " + (2 * borderTiles + boardHeight));

  // Draw background color
  let background = document.createElementNS(svgNS, "rect");
  background.setAttribute("style", "fill:"+backgroundColor);
  background.setAttribute("width", 2 * borderTiles + boardWidth);
  background.setAttribute("height", 2 * borderTiles + boardHeight);
  background.setAttribute("x", -borderTiles);
  background.setAttribute("y", -borderTiles);
  boardSvg.appendChild(background);
  if(game == "xiangqi") {
    const defs = document.createElementNS(svgNS,"defs");
    const arrow = document.createElementNS(svgNS,"marker");
    arrow.id = "moveArrow";
    arrow.setAttribute("viewBox","0 0 10 10"); arrow.setAttribute("refX",9); arrow.setAttribute("refY",5);
    arrow.setAttribute("markerWidth",4); arrow.setAttribute("markerHeight",4); arrow.setAttribute("orient","auto");
    const tip = document.createElementNS(svgNS,"path");
    tip.setAttribute("d","M0,0L10,5L0,10z"); tip.setAttribute("fill","#2676c7");
    arrow.appendChild(tip); defs.appendChild(arrow); boardSvg.appendChild(defs);
  }

  if(game == "hex") {
    const radius = 1/Math.sqrt(3);
    for(let y = 0; y < bSizeY; y++) for(let x = 0; x < bSizeX; x++) {
      const [cx,cy] = boardPoint(x,y);
      const vertices = Array.from({length:6},(_,i) => {
        const angle = (30+60*i)*Math.PI/180;
        return [cx+radius*Math.cos(angle),cy+radius*Math.sin(angle)];
      });
      const cell = document.createElementNS(svgNS,"polygon");
      cell.classList.add("hex-cell");
      cell.setAttribute("points",vertices.map(v=>v.join(",")).join(" "));
      cell.setAttribute("fill",backgroundColor);
      cell.setAttribute("stroke","#705627");
      cell.setAttribute("stroke-width",strokeWidth);
      boardSvg.appendChild(cell);
      for(const [edges,color,player] of [
        [y == 0 ? [3,4] : [],stoneBlackFill,"B"], [y == bSizeY-1 ? [0,1] : [],stoneBlackFill,"B"],
        [x == 0 ? [2,3] : [],stoneWhiteFill,"W"], [x == bSizeX-1 ? [5,0] : [],stoneWhiteFill,"W"]
      ]) for(const edge of edges) {
        const path = document.createElementNS(svgNS,"path");
        path.classList.add("hex-goal");
        path.setAttribute("data-player",player);
        path.setAttribute("d","M"+vertices[edge]+"L"+vertices[(edge+1)%6]);
        path.setAttribute("stroke",color);
        path.setAttribute("stroke-width",0.10);
        boardSvg.appendChild(path);
      }
    }
  }
  else {
    // Draw border of board
    let border = document.createElementNS(svgNS, "rect");
    border.setAttribute("width", bSizeX-1);
    border.setAttribute("height", bSizeY-1);
    border.setAttribute("x", 0);
    border.setAttribute("y", 0);
    border.setAttribute("stroke","black");
    border.setAttribute("stroke-width",strokeWidth);
    border.setAttribute("fill","none");
    boardSvg.appendChild(border);

    // Draw internal gridlines of board
    for(let y = 1; y < bSizeY-1; y++) {
      let stroke = document.createElementNS(svgNS, "path");
      stroke.setAttribute("stroke","black");
      stroke.setAttribute("stroke-width",strokeWidth);
      stroke.setAttribute("fill","none");
      stroke.setAttribute("d","M0,"+y+"h"+(bSizeX-1));
      boardSvg.appendChild(stroke);
    }
    for(let x = 1; x < bSizeX-1; x++) {
      let stroke = document.createElementNS(svgNS, "path");
      stroke.setAttribute("stroke","black");
      stroke.setAttribute("stroke-width",strokeWidth);
      stroke.setAttribute("fill","none");
      stroke.setAttribute("d",game == "xiangqi" ? "M"+x+",0v4 M"+x+",5v4" : "M"+x+",0v"+(bSizeY-1));
      boardSvg.appendChild(stroke);
    }

    if(game == "xiangqi") {
      const palace = document.createElementNS(svgNS,"path");
      palace.setAttribute("d","M3,0L5,2 M5,0L3,2 M3,7L5,9 M5,7L3,9");
      palace.setAttribute("stroke","black");
      palace.setAttribute("stroke-width",strokeWidth);
      boardSvg.appendChild(palace);
      for(const [x,text] of [[2,"楚 河"],[6,"漢 界"]]) {
        const river = document.createElementNS(svgNS,"text");
        river.textContent = text;
        river.setAttribute("x",x); river.setAttribute("y",4.55);
        river.setAttribute("text-anchor","middle"); river.setAttribute("font-size",0.45);
        boardSvg.appendChild(river);
      }
    }
  }

  if(game != "hex" && game != "xiangqi") {
  // Draw star points
  var xStars = [];
  var yStars = [];
  if(bSizeX < 9)
    xStars = [];
  else if(bSizeX == 9)
    xStars = [2,bSizeX-3];
  else if(bSizeX % 2 == 0)
    xStars = [3,bSizeX-4];
  else
    xStars = [3,(bSizeX-1)/2,bSizeX-4];

  if(bSizeY < 9)
    yStars = [];
  else if(bSizeY == 9)
    yStars = [2,bSizeY-3];
  else if(bSizeY % 2 == 0)
    yStars = [3,bSizeY-4];
  else
    yStars = [3,(bSizeY-1)/2,bSizeY-4];

  let stars = [];
  for(let y = 0; y<yStars.length; y++) {
    for(let x = 0; x<xStars.length; x++) {
      stars.push([xStars[x],yStars[y]]);
    }
  }
  if(bSizeY == 9 && bSizeX == 9) {
    stars.push([4,4]);
  }

  for(const point of stars) {
    let dot = document.createElementNS(svgNS, "path");
    dot.setAttribute("stroke","black");
    dot.setAttribute("stroke-width",0.2);
    dot.setAttribute("stroke-linecap","round");
    dot.setAttribute("d","M"+point[0]+","+point[1]+"l0,0");
    boardSvg.appendChild(dot);
  }

  }

  // Draw stones
  for(let y = 0; y<bSizeY; y++) {
    for(let x = 0; x<bSizeX; x++) {
      let pos = y * bSizeX + x;
      // We need to draw the stones with the given symmetry applied, so for looking up
      // the stone of a position we need to use the inverse symmetry.
      let invSymPos = getInvSymPos(pos);
      if(board[invSymPos] == 1 || board[invSymPos] == 2) {
        const [screenX,screenY] = boardPoint(x,y);
        const pieceGroup = document.createElementNS(svgNS,"g");
        pieceGroup.classList.add(game == "xiangqi" ? "xiangqi-piece" : "stone");
        pieceGroup.setAttribute("data-pos",invSymPos);
        // Layer stone color on top of black circle so that we have precise control over the border
        // being a certain radius.
        let stone = document.createElementNS(svgNS, "circle");
        let stoneBorder = document.createElementNS(svgNS, "circle");
        stone.setAttribute("cx",screenX);
        stone.setAttribute("cy",screenY);
        stone.setAttribute("r",stoneInnerRadius);
        stoneBorder.setAttribute("cx",screenX);
        stoneBorder.setAttribute("cy",screenY);
        stoneBorder.setAttribute("r",stoneRadius);
        stone.setAttribute("stroke","none");
        stoneBorder.setAttribute("stroke","none");
        stoneBorder.setAttribute("fill","black");
        if(board[invSymPos] == 1)
          stone.setAttribute("fill",stoneBlackFill);
        else
          stone.setAttribute("fill",stoneWhiteFill);
        pieceGroup.appendChild(stoneBorder);
        pieceGroup.appendChild(stone);
        if(game == "xiangqi") {
          const piece = pieces[invSymPos];
          const symbols = piece > 0 ? ["","車","仕","炮","兵","馬","相","帥"] : ["","車","士","砲","卒","馬","象","將"];
          const color = piece > 0 ? "#aa2020" : "#222";
          stone.setAttribute("fill","#fff0cc");
          stoneBorder.setAttribute("fill",color);
          const character = document.createElementNS(svgNS,"text");
          character.textContent = symbols[Math.abs(piece)];
          character.setAttribute("x",screenX); character.setAttribute("y",screenY);
          character.setAttribute("font-size",0.65); character.setAttribute("fill",color);
          character.setAttribute("text-anchor","middle"); character.setAttribute("dominant-baseline","central");
          pieceGroup.appendChild(character);
          pieceGroup.setAttribute("data-piece",piece);
          pieceGroup.setAttribute("data-selectable",jointOrigins.has(invSymPos));
          if(jointOrigins.has(invSymPos)) {
            pieceGroup.addEventListener("click",() => selectFrom(invSymPos));
            jointPieces.push({element:pieceGroup,pos:invSymPos});
          }
        }
        boardSvg.appendChild(pieceGroup);
      }
    }
  }

  // Draw move labels on board
  for(let i = 0; i<moves.length; i++) {
    let moveData = moves[i];
    if(moveData["move"] == "other" || moveData["move"] == "pass")
      continue;
    if(moveData["v"] <= 0.0)
      continue;
    const variants = game == "xiangqi" ? moveData.actions : moveData.xy;
    for(let j = 0; j<variants.length; j++) {
      const action = game == "xiangqi" ? variants[j] : null;
      let xy = action ? action.slice(2) : variants[j];
      let x = xy[0];
      let y = xy[1];
      let pos = y * bSizeX + x;
      let symPos = getSymPos(pos);
      let [symX,symY] = boardPoint(symPos % bSizeX,Math.floor(symPos / bSizeX));
      let moveLayer = boardSvg;
      if(action) {
        moveLayer = document.createElementNS(svgNS,"g");
        moveLayer.classList.add("jointCandidate");
        moveLayer.setAttribute("data-action",jointMoveString(action));
        jointCandidates.push({element:moveLayer,index:i,from:action[0]+action[1]*bSizeX});
        boardSvg.appendChild(moveLayer);
      }
      // Keep the captured piece readable underneath a Xiangqi move marker.
      const labelX = symX + (action ? 0.34 : 0);
      const labelY = symY - (action ? 0.34 : 0);
      const labelScale = action ? 0.5 : 1;

      // Background-colored circle to mask out the gridlines so that text isn't fighting
      // it for contrast.
      // let lineMask = document.createElementNS(svgNS, "circle");
      // lineMask.setAttribute("cx",symX);
      // lineMask.setAttribute("cy",symY);
      // lineMask.setAttribute("r",stoneRadius);
      // lineMask.setAttribute("stroke","none");
      // lineMask.setAttribute("fill",backgroundColor);
      // boardSvg.appendChild(lineMask);

      // Colored circle based on move value.
      let moveValueGroup = document.createElementNS(svgNS, "g");
      moveValueGroup.setAttribute("opacity",0.85);
      let moveValueColor = getBadnessColorOfMoveIdx(i,1.0);

      let moveValueBigCircle = document.createElementNS(svgNS, "circle");
      moveValueBigCircle.setAttribute("cx",labelX);
      moveValueBigCircle.setAttribute("cy",labelY);
      moveValueBigCircle.setAttribute("r",labelScale*stoneRadius);
      moveValueBigCircle.setAttribute("stroke","none");
      moveValueBigCircle.setAttribute("fill",moveValueColor);
      moveValueGroup.appendChild(moveValueBigCircle);

      let moveValueCircleBorder = document.createElementNS(svgNS, "circle");
      moveValueCircleBorder.setAttribute("cx",labelX);
      moveValueCircleBorder.setAttribute("cy",labelY);
      moveValueCircleBorder.setAttribute("r",labelScale*stoneRadius);
      moveValueCircleBorder.setAttribute("stroke","none");
      moveValueCircleBorder.setAttribute("fill",nextPla == 1 ? "white" : "white");
      moveValueCircleBorder.setAttribute("opacity",0.5);
      moveValueGroup.appendChild(moveValueCircleBorder);

      let moveValueCircle = document.createElementNS(svgNS, "circle");
      moveValueCircle.setAttribute("cx",labelX);
      moveValueCircle.setAttribute("cy",labelY);
      moveValueCircle.setAttribute("r",labelScale*stoneInnerRadius);
      moveValueCircle.setAttribute("stroke","none");
      moveValueCircle.setAttribute("opacity",0.95);
      moveValueCircle.setAttribute("fill",moveValueColor);
      moveValueGroup.appendChild(moveValueCircle);

      moveLayer.appendChild(moveValueGroup);

      // Text for marker, centered.
      let marker = document.createElementNS(svgNS, "text");
      marker.textContent = ""+(i+1);
      marker.setAttribute("x",labelX);
      marker.setAttribute("y",labelY);
      marker.setAttribute("font-size",labelScale*markerFontSize);
      marker.setAttribute("dominant-baseline","central");
      marker.setAttribute("text-anchor","middle");
      marker.setAttribute("fill",nextPla == 1 ? "black" : "white");
      moveLayer.appendChild(marker);

      let linkForPos = getLinkForPos(action ? jointMoveString(action) : pos);

      // Group for hover shadow
      let shadowGroup = document.createElementNS(svgNS, "g");
      shadowGroup.setAttribute("opacity",0.65);
      shadowGroup.setAttribute("moveX",symX);
      shadowGroup.setAttribute("moveY",symY);
      if(linkForPos)
        shadowGroup.classList.add("stoneShadow");
      else
        shadowGroup.classList.add("stoneShadowUnhoverable");
      let stoneShadow = document.createElementNS(svgNS, "circle");
      stoneShadow.setAttribute("cx",symX);
      stoneShadow.setAttribute("cy",symY);
      stoneShadow.setAttribute("r",stoneRadius);
      stoneShadow.setAttribute("stroke","none");
      if(nextPla == 1)
        stoneShadow.setAttribute("fill",stoneBlackFill);
      else
        stoneShadow.setAttribute("fill",stoneWhiteFill);
      shadowGroup.appendChild(stoneShadow);
      if(action) {
        const fromPos = getSymPos(action[0]+action[1]*bSizeX);
        const [fromX,fromY] = boardPoint(fromPos%bSizeX,Math.floor(fromPos/bSizeX));
        const arrow = document.createElementNS(svgNS,"path");
        arrow.classList.add("joint-arrow");
        arrow.setAttribute("d","M"+fromX+","+fromY+"L"+symX+","+symY);
        arrow.setAttribute("stroke","#2676c7"); arrow.setAttribute("stroke-width",0.14);
        arrow.setAttribute("marker-end","url(#moveArrow)"); arrow.setAttribute("pointer-events","none");
        shadowGroup.appendChild(arrow);
      }

      shadowGroup.setAttribute("x",symX);
      shadowGroup.setAttribute("y",symY);
      if(j == 0)
        hoverShadowEltsByMove[i] = shadowGroup;

      shadowGroup.addEventListener("mouseover",(event) => {
        for(let j = 0; j<moves.length; j++) {
          if(j in hoverTableEltsByMove)
            hoverTableEltsByMove[j].classList.remove("moveHovered");
        }
        if(i in hoverTableEltsByMove)
          hoverTableEltsByMove[i].classList.add("moveHovered");
      });
      shadowGroup.addEventListener("mouseout",(event) => {
        if(i in hoverTableEltsByMove)
          hoverTableEltsByMove[i].classList.remove("moveHovered");
      });

      if(linkForPos) {
        let shadowGroupLink = document.createElementNS(svgNS, "a");
        shadowGroupLink.setAttribute("href",linkForPos);
        shadowGroupLink.appendChild(shadowGroup);
        moveLayer.appendChild(shadowGroupLink);
      }
      else {
        moveLayer.appendChild(shadowGroup);
      }
    }
  }

  // Labels use displayed coordinates; the native action keys remain canonical.
  function coordinateLabel(text,x,y) {
    const label = document.createElementNS(svgNS,"text");
    label.textContent = text;
    label.setAttribute("x",x); label.setAttribute("y",y);
    label.setAttribute("font-size",coordFontSize);
    label.setAttribute("dominant-baseline","central"); label.setAttribute("text-anchor","middle");
    label.setAttribute("fill","black"); boardSvg.appendChild(label);
  }
  for(let y = 0; y < bSizeY; y++) {
    const left = boardPoint(0,y), right = boardPoint(bSizeX-1,y);
    const text = String(game == "xiangqi" ? y : bSizeY-y);
    coordinateLabel(text,left[0]-coordSpacing,left[1]);
    coordinateLabel(text,right[0]+coordSpacing,right[1]);
  }
  for(let x = 0; x < bSizeX; x++) {
    const top = boardPoint(x,game == "xiangqi" ? bSizeY-1 : 0);
    const bottom = boardPoint(x,game == "xiangqi" ? 0 : bSizeY-1);
    const text = (game == "xiangqi" ? "abcdefghi" : coordChars)[x];
    coordinateLabel(text,top[0],top[1]-coordSpacing);
    coordinateLabel(text,bottom[0],bottom[1]+coordSpacing);
  }
  updateJointCandidates();

  body.appendChild(boardSvg);
}

)%%";
const std::string Book::BOOK_JS3 = R"%%(

{
  let whoToPlay = document.createElement("div");
  const status = outcome == null ? (nextPla == 1 ? firstPlayerName : secondPlayerName) + " to play!" :
    outcome == "D" ? "Draw" : outcome == "NR" ? "No result" : (outcome == "B" ? firstPlayerName : secondPlayerName) + " wins";
  whoToPlay.appendChild(document.createTextNode(status));
  whoToPlay.id = "whoToPlay";
  body.appendChild(whoToPlay);
}

function textCell(text) {
  let cell = document.createElement("div");
  cell.classList.add("moveTableCell");
  cell.appendChild(document.createTextNode(text));
  return cell;
}

// Generate move table. Emulate table using divs, so that we can have more precise control via css
// such as linking a whole row.
{
  let table = document.createElement("div");
  table.classList.add("moveTable");

  let headerRow = document.createElement("div");
  headerRow.classList.add("moveTableHeader");
  headerRow.appendChild(textCell("Index"));
  headerRow.appendChild(textCell("Move"));
  headerRow.appendChild(textCell(game == "xiangqi" ? "RWin%" : "BWin%"));
  if(devMode) {
    headerRow.appendChild(textCell("Win%LCB"));
    headerRow.appendChild(textCell("Win%UCB"));
    headerRow.appendChild(textCell("Prior%"));
    // headerRow.appendChild(textCell("Wgt"));
    headerRow.appendChild(textCell("Visits"));
    headerRow.appendChild(textCell("AVisits"));
    headerRow.appendChild(textCell("Cost"));
    headerRow.appendChild(textCell("TCost"));
    headerRow.appendChild(textCell("CostWLPV"));
    headerRow.appendChild(textCell("BigWLC"));
  }
  else {
    headerRow.appendChild(textCell("Win% Uncertainty"));
    headerRow.appendChild(textCell("Prior%"));
    headerRow.appendChild(textCell("Visits"));
    headerRow.appendChild(textCell("AVisits"));
  }
  table.appendChild(headerRow);

  for(let i = 0; i<moves.length; i++) {
    let moveData = moves[i];
    if(moveData["v"] <= 0.0)
      continue;

    let linkForPos = null;
    if(moveData.actions) {
      linkForPos = getLinkForPos(jointMoveString(moveData.actions[0]));
    }
    else if(moveData["xy"]) {
      let xy = moveData["xy"][0];
      let x = xy[0];
      let y = xy[1];
      let pos = y * bSizeX + x;
      linkForPos = getLinkForPos(pos);
    }
    else if(moveData["move"] && moveData["move"] == "pass") {
      let pos = bSizeY * bSizeX;
      linkForPos = getLinkForPos(pos);
    }

    let dataRow = null;
    if(linkForPos) {
      dataRow = document.createElement("a");
      dataRow.setAttribute("href",linkForPos);
      dataRow.classList.add("moveTableRowLinked");
    }
    else {
      dataRow = document.createElement("span");
      dataRow.classList.add("moveTableRowLinked");
    }

    dataRow.classList.add("moveTableRow");
    dataRow.setAttribute("role","row");
    dataRow.setAttribute("data-move-index",i);
    dataRow.addEventListener("mouseover",(event) => {
      hoveredMove = i;
      updateJointCandidates();
      for(let j = 0; j<moves.length; j++) {
        if(j in hoverShadowEltsByMove)
          hoverShadowEltsByMove[j].classList.remove("tableHovered");
      }
      if(i in hoverShadowEltsByMove)
        hoverShadowEltsByMove[i].classList.add("tableHovered");
    });
    dataRow.addEventListener("mouseout",(event) => {
      hoveredMove = null;
      updateJointCandidates();
      if(i in hoverShadowEltsByMove)
        hoverShadowEltsByMove[i].classList.remove("tableHovered");
    });

    dataRow.appendChild(textCell(i+1));
    if(moveData.actions)
      dataRow.appendChild(textCell(jointMoveString(moveData.actions[0],true)));
    else if(moveData["move"])
      dataRow.appendChild(textCell(moveData["move"]));
    else if(moveData["xy"]) {
      let xy = moveData["xy"][0];
      let x = xy[0];
      let y = xy[1];
      let pos = y * bSizeX + x;
      let symPos = getSymPos(pos);
      let symX = symPos % bSizeX;
      let symY = Math.floor(symPos / bSizeX);
      let coord = coordChars[symX] + "" + (bSizeY-symY);
      dataRow.appendChild(textCell(coord));
    }
    else
      dataRow.appendChild(textCell(""));

    dataRow.appendChild(textCell((100.0 * (0.5*(1.0-moveData["wl"]))).toFixed(1)+"%"));
    if(devMode) {
      dataRow.appendChild(textCell((100.0 * (0.5*(1.0-moveData["wlUCB"]))).toFixed(1)+"%"));
      dataRow.appendChild(textCell((100.0 * (0.5*(1.0-moveData["wlLCB"]))).toFixed(1)+"%"));
      dataRow.appendChild(textCell((100.0 * moveData["p"]).toFixed(2)+"%"));
      // dataRow.appendChild(textCell(Math.round(moveData["w"]).toLocaleString()));
      dataRow.appendChild(textCell(Math.round(moveData["v"]).toLocaleString()));
      dataRow.appendChild(textCell(Math.round(moveData["av"]).toLocaleString()));
      dataRow.appendChild(textCell(moveData["cost"].toFixed(3)));
      dataRow.appendChild(textCell(moveData["costRoot"].toFixed(3)));
      dataRow.appendChild(textCell(moveData["costWLPV"].toFixed(3)));
      dataRow.appendChild(textCell(moveData["bigWLC"].toFixed(3)));
    }
    else {
      if(moveData["wlRad"] <= 0)
        dataRow.appendChild(textCell("-"));
      else
        dataRow.appendChild(textCell((100.0 * 0.5 * moveData["wlRad"]).toFixed(1)+"%"));
      dataRow.appendChild(textCell((100.0 * moveData["p"]).toFixed(2)+"%"));
      dataRow.appendChild(textCell(Math.round(moveData["v"]).toLocaleString()));
      dataRow.appendChild(textCell(Math.round(moveData["av"]).toLocaleString()));
    }

    dataRow.style.background = getBadnessColorOfMoveIdx(i,0.35);

    table.appendChild(dataRow);
    hoverTableEltsByMove[i] = dataRow;
  }

  const tableContainer = document.createElement("div");
  tableContainer.classList.add("moveTableContainer");
  tableContainer.appendChild(table);
  body.appendChild(tableContainer);
}

function textItem(label,text) {
  let item = document.createElement("li");
  item.classList.add("legendItem");
  let labelSpan = document.createElement("span");
  let textSpan = document.createElement("span");
  labelSpan.appendChild(document.createTextNode(label + ": "));
  textSpan.appendChild(document.createTextNode(text));
  textSpan.classList.add("legendItemText");
  item.appendChild(labelSpan);
  item.appendChild(textSpan);
  return item;
}

{
  let legend = document.createElement("div");
  legend.classList.add("legend");
  legend.appendChild(document.createTextNode("Explanation of metrics:"));
  let legendList = document.createElement("ul");
  legendList.appendChild(textItem("Index","Order of preference of moves. Trades off between multiple considerations (e.g. win% vs uncertainty vs prior). Other orderings may be better depending on your goal. For example if you care only about likely win-loss-draw optimality, ignore this ordering and pick based on the appropriate metrics directly."));
  legendList.appendChild(textItem(firstPlayerName+" Win%","Minimax of MCTS result value from "+firstPlayerName+"'s perspective: win probability plus half of draw and no-result probability."));
  legendList.appendChild(textItem("Win% Uncertainty","Measure of uncertainty in Win%. Does NOT correspond to any standard well-defined statistical metric, this is purely a heuristic indicator. Browse the book to get a feel for its scaling and what it means."));
  legendList.appendChild(textItem("Prior%","Raw policy prior of neural net"));
  legendList.appendChild(textItem("Visits","Total number of visits, multi-counting transpositions (i.e., number of visits to produce this book if there were no transposition handling)."));
  legendList.appendChild(textItem("AVisits","Adjusted number of visits, still multi-counting transpositions, but downweighting visits on a less-preferred move that got vastly more visits than the top move."));
  if(game == "hex")
    legendList.appendChild(textItem("Hex goals","Black connects the top and bottom edges; White connects the left and right edges."));
  legend.appendChild(legendList);
  body.appendChild(legend);
}

)%%";
