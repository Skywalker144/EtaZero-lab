"""Native book exports and real Chromium navigation across game/symmetry domains."""
import math
from pathlib import Path
import subprocess
from urllib.parse import urljoin, urlparse

import pytest
from playwright.sync_api import sync_playwright

from test_book import BINARY, GAMES
from test_booktoposes import create_books
from test_genbook import configure, run as generate


@pytest.fixture(scope='module')
def books(tmp_path_factory):
    return create_books(tmp_path_factory.mktemp('book-html') / 'books', BINARY)


def export(book, output, game, *, dev=False, minimum=0):
    config = configure(output.with_suffix('.cfg'), game,
        boardSizeX=9 if game == 'xiangqi' else 7, boardSizeY=10 if game == 'xiangqi' else 7)
    result = subprocess.run([str(BINARY), 'writebook', '-config', str(config), '-book-file', str(book),
        '-html-dir', str(output), '-html-min-visits', str(minimum), *(['-html-dev-mode'] if dev else [])],
        capture_output=True, text=True, timeout=60)
    text = result.stdout + result.stderr
    output.with_suffix('.log').write_text(text)
    assert result.returncode == 0 and 'Failed to get board history' not in text, text
    return output / 'root/root.html'


@pytest.fixture(scope='module')
def browser():
    with sync_playwright() as api:
        browser = api.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def page(browser):
    page = browser.new_page(viewport={'width': 1280, 'height': 1200})
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    yield page
    page.close()
    assert not errors, errors


def open_page(page, root, symmetry=0):
    page.goto(root.as_uri() + f'?symmetry={symmetry}')
    page.locator('#bookBoard').wait_for()


def snapshot(page):
    return page.evaluate('''() => ({game, sym, width:bSizeX, height:bSizeY, board,
        pieces:game === 'xiangqi' ? pieces : null, nextPla, outcome, moves, links, linkSyms, pLink, pSym})''')


def transform(x, y, width, height, symmetry):
    if symmetry & 1:
        y = height-1-y
    if symmetry & 2:
        x = width-1-x
    if symmetry & 4:
        x, y = y, x
    return x, y


def displayed(data, field='board'):
    result = [0] * len(data[field])
    for pos, value in enumerate(data[field]):
        x, y = transform(pos % data['width'], pos // data['width'], data['width'], data['height'], data['sym'])
        result[x+y*data['width']] = value
    return result


def expected_after(data, move):
    values = displayed(data, 'pieces' if data['game'] == 'xiangqi' else 'board')
    if data['game'] == 'xiangqi':
        x, y, u, v = move['actions'][0]
        x, y = transform(x, y, data['width'], data['height'], data['sym'])
        u, v = transform(u, v, data['width'], data['height'], data['sym'])
        values[u+v*data['width']] = values[x+y*data['width']]
        values[x+y*data['width']] = 0
    else:
        x, y = transform(*move['xy'][0], data['width'], data['height'], data['sym'])
        values[x+y*data['width']] = data['nextPla']
    return values


@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('dev', [False, True])
def test_board_links_metrics_and_parent(books, tmp_path, page, game, dev):
    root = export(books / f'{game}.katabook', tmp_path / 'html', game, dev=dev)
    symmetry = 2 if game == 'xiangqi' else 3 if game == 'hex' else 5
    open_page(page, root, symmetry)
    first = snapshot(page)
    assert first['game'] == game and first['sym'] == symmetry
    assert page.locator('#whoToPlay').inner_text() == ('Red' if game == 'xiangqi' else 'Black')+' to play!'
    cells = page.locator('.moveTableRow').first.locator('.moveTableCell').all_text_contents()
    assert cells[2] == '43.8%'  # Every fixture node has white W-L = 1/8.
    assert ('Cost' in page.locator('.moveTableHeader').inner_text()) == dev
    if game == 'xiangqi':
        assert page.locator('.xiangqi-piece').count() == 32
        assert page.locator('.xiangqi-piece[data-piece="7"] text').text_content() == '帥'
        assert page.locator('.xiangqi-piece[data-piece="-7"] text').text_content() == '將'
        assert 'RWin%' in page.locator('.moveTableHeader').inner_text()
    elif game == 'hex':
        assert page.locator('.hex-cell').count() == 49
        assert page.locator('.hex-goal[data-player="B"]').count() == 28
        assert page.locator('.hex-goal[data-player="W"]').count() == 28
        assert 'Black connects the top and bottom' in page.locator('body').inner_text()
        centers = []
        for polygon in page.locator('.hex-cell').all():
            points = [[float(v) for v in pair.split(',')] for pair in polygon.get_attribute('points').split()]
            assert len(points) == 6
            centers.append([sum(p[axis] for p in points)/6 for axis in (0, 1)])
        center = centers[3+3*7]
        touching = {i for i, other in enumerate(centers) if math.dist(center, other) == pytest.approx(1)}
        assert touching == {2+3*7, 4+3*7, 3+2*7, 3+4*7, 4+2*7, 2+4*7}
    else:
        assert page.locator('.hex-cell, .xiangqi-piece').count() == 0

    expected = expected_after(first, first['moves'][0])
    first_row = page.locator('.moveTableRowLinked').first
    first_row.hover()
    assert page.locator('.stoneShadow.tableHovered').count() == 1
    first_row.click()
    page.wait_for_function('nextPla === 2')
    child = snapshot(page)
    assert displayed(child, 'pieces' if game == 'xiangqi' else 'board') == expected
    assert set(child['linkSyms'].values()) <= ({0, 2} if game == 'xiangqi' else {0, 3} if game == 'hex' else set(range(8)))
    page.get_by_role('link', name='Canonical parent', exact=True).click()
    page.wait_for_function('nextPla === 1')
    parent = snapshot(page)
    assert displayed(parent, 'pieces' if game == 'xiangqi' else 'board') == displayed(first, 'pieces' if game == 'xiangqi' else 'board')
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
    if not dev and game in ('hex', 'xiangqi'):
        page.screenshot(path=f'/tmp/etazero-v4-book-{game}.png', full_page=True)


@pytest.mark.parametrize('game,bad', [('hex', 2), ('xiangqi', 3), ('go', 'nan')])
def test_invalid_display_symmetry_uses_identity(books, tmp_path, page, game, bad):
    root = export(books / f'{game}.katabook', tmp_path / 'html', game)
    open_page(page, root, bad)
    assert snapshot(page)['sym'] == 0


@pytest.mark.parametrize('game', GAMES)
def test_symmetric_variant_board_click(books, tmp_path, page, game):
    root = export(books / f'{game}.katabook', tmp_path / 'html', game)
    open_page(page, root, 2 if game == 'xiangqi' else 3 if game == 'hex' else 6)
    data = snapshot(page)
    field = 'actions' if game == 'xiangqi' else 'xy'
    move = next(m for m in data['moves'] if len(m.get(field, [])) > 1)
    variant = move[field][-1]
    expected = expected_after(data, {field: [variant]})
    if game == 'xiangqi':
        x,y,u,v = variant
        page.locator(f'.xiangqi-piece[data-pos="{x+y*9}"]').click()
        action = 'abcdefghi'[x]+str(y)+'abcdefghi'[u]+str(v)
        page.locator(f'.jointCandidate[data-action="{action}"] a circle').click()
    else:
        x,y = transform(*variant, data['width'], data['height'], data['sym'])
        if game == 'hex':
            x,y = x+y/2, y*math.sqrt(3)/2
        targets = [element for element in page.locator('.stoneShadow').all()
                   if float(element.get_attribute('moveX')) == pytest.approx(x) and
                   float(element.get_attribute('moveY')) == pytest.approx(y)]
        assert len(targets) == 1
        targets[0].locator('circle').click()
    page.wait_for_function('nextPla === 2')
    assert displayed(snapshot(page), 'pieces' if game == 'xiangqi' else 'board') == expected


@pytest.mark.parametrize('game', ['go', 'hex', 'xiangqi'])
def test_visit_threshold_has_no_dangling_move_links(books, tmp_path, page, game):
    root = export(books / f'{game}.katabook', tmp_path / 'html', game, minimum=11)
    pages = list(root.parent.parent.glob('*/*.html'))
    assert len(pages) > 1
    for path in pages:
        open_page(page, path)
        data = snapshot(page)
        for link in data['links'].values():
            assert Path(urlparse(urljoin(page.url, link)).path).is_file()
        if data['pLink']:
            assert Path(urlparse(urljoin(page.url, data['pLink'])).path).is_file()
        if game != 'go':
            assert all(move.get('move') != 'pass' for move in data['moves'])


@pytest.mark.parametrize('name,game,result', [
    ('freestyle-terminal', 'freestyle', 'Black wins'), ('standard-terminal', 'standard', 'Black wins'),
    ('renju-terminal', 'renju', 'Black wins'), ('hex-terminal', 'hex', 'Black wins'),
    ('xiangqi-history', 'xiangqi', 'Draw'), ('xiangqi-check', 'xiangqi', 'Black wins'),
    ('xiangqi-chase', 'xiangqi', 'Black wins'), ('xiangqi-limit', 'xiangqi', 'Draw'),
])
def test_terminal_history_pages(books, tmp_path, page, name, game, result):
    root = export(books / f'{name}.katabook', tmp_path / 'html', game)
    found = []
    for path in root.parent.parent.glob('*/*.html'):
        open_page(page, path)
        data = snapshot(page)
        if data['outcome'] is not None:
            assert page.locator('#whoToPlay').inner_text() == result
            assert not data['links']
            found.append(path)
    assert found


def test_xiangqi_capture_click_and_mirrored_coordinates(tmp_path, page):
    config = configure(tmp_path / 'book.cfg', 'xiangqi')
    sgf = tmp_path / 'capture.sgf'
    sgf.write_text('(;GM[7]SZ[9:10]RU[xiangqi];B[b2b9];W[a9b9])')
    book = tmp_path / 'book.katabook'
    generate(config, book, iterations=0, extra=['-trace-sgf-file', sgf])
    root = export(book, tmp_path / 'html', 'xiangqi')
    open_page(page, root, 2)
    for ply in range(2):
        data = snapshot(page)
        index = next(i for i,m in enumerate(data['moves']) if m.get('actions') and
                     data['pieces'][m['actions'][0][2]+m['actions'][0][3]*9] != 0)
        move = data['moves'][index]
        x,y,u,v = move['actions'][0]
        display_from = transform(x,y,9,10,data['sym'])
        display_to = transform(u,v,9,10,data['sym'])
        label = ''.join('abcdefghi'[a]+str(b) for a,b in (display_from, display_to))
        row = page.locator(f'[data-move-index="{index}"]')
        assert row.locator('.moveTableCell').nth(1).inner_text() == label
        page.locator(f'.xiangqi-piece[data-pos="{x+y*9}"]').click()
        assert page.locator('#fromPiece').input_value() == str(x+y*9)
        action = 'abcdefghi'[x]+str(y)+'abcdefghi'[u]+str(v)
        candidate = page.locator(f'.jointCandidate[data-action="{action}"]')
        assert candidate.is_visible()
        expected = expected_after(data, move)
        candidate.locator('a circle').click()
        page.wait_for_function(f'nextPla === {2 if ply == 0 else 1}')
        assert displayed(snapshot(page), 'pieces') == expected
        assert page.locator('.xiangqi-piece').count() == 31-ply
    # The DOM uses the actual piece types and the displayed, mirrored geometry.
    data = snapshot(page)
    for group in page.locator('.xiangqi-piece').all():
        pos = int(group.get_attribute('data-pos'))
        x,y = transform(pos%9,pos//9,9,10,data['sym'])
        circle = group.locator('circle').first
        assert float(circle.get_attribute('cx')) == x and float(circle.get_attribute('cy')) == 9-y


def test_go_pass_link_keeps_board(tmp_path, page):
    config = configure(tmp_path / 'book.cfg', 'go')
    sgf = tmp_path / 'pass.sgf'
    sgf.write_text('(;GM[1]SZ[5]RU[Tromp-Taylor]KM[0];B[aa];W[])')
    book = tmp_path / 'book.katabook'
    generate(config, book, iterations=0, extra=['-trace-sgf-file', sgf])
    root = export(book, tmp_path / 'html', 'go')
    for path in root.parent.parent.glob('*/*.html'):
        open_page(page, path)
        before = snapshot(page)
        found = [i for i,m in enumerate(before['moves']) if m.get('move') == 'pass']
        if found:
            break
    assert found
    index = found[0]
    page.locator(f'[data-move-index="{index}"]').click()
    page.wait_for_function('nextPla === 1')
    assert displayed(snapshot(page)) == displayed(before)


def test_rectangular_go_display_preserves_dimensions(tmp_path, page):
    config = configure(tmp_path / 'book.cfg', 'go', boardSizeX=4, boardSizeY=5)
    book = tmp_path / 'book.katabook'
    generate(config, book, iterations=1)
    root = export(book, tmp_path / 'html', 'go')
    open_page(page, root, 4)
    data = snapshot(page)
    assert (data['width'], data['height'], data['sym']) == (4, 5, 0)
    assert page.evaluate('allowedSymmetries') == [0, 1, 2, 3]
    open_page(page, root, 3)
    data = snapshot(page)
    move_index = next(i for i, move in enumerate(data['moves']) if move.get('xy'))
    expected = expected_after(data, data['moves'][move_index])
    page.locator(f'[data-move-index="{move_index}"]').click()
    page.wait_for_function('nextPla === 2')
    assert displayed(snapshot(page)) == expected
