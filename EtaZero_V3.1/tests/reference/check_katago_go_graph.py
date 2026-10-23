"""Compare Go graph identities against the pinned, independently built KataGo."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
ORACLE = r'''
#include "game/graphhash.h"
#include <iostream>
#include <random>
#include <sstream>
Loc location(int action,int size) {
  return action==size*size?Board::PASS_LOC:Location::getLoc(action%size,action/size,size);
}
int main(int argc,char**) {
  Board::initHash();
  if(argc>1) {
    std::mt19937 rng(619);
    for(const char* rules:{"chinese","chinese-ogs","japanese","tromp-taylor","new-zealand","aga"})
      for(int size:{5,9,19})for(int bound:{3,11,50}) {
        double komi=bound==3?-.5:bound==11?7.5:0;
        Board board(size,size);BoardHistory hist(board,P_BLACK,Rules::parseRulesWithoutKomi(rules,komi),0,BoardHistoryModes(true,true));
        std::vector<int> moves;
        for(int turn=0;turn<240;++turn) {
          if(turn<10 || turn%9==0 || hist.isGameFinished) {
            std::cout<<size<<' '<<size+1<<' '<<bound<<' '<<rules<<' '<<komi<<' '<<moves.size();
            for(int a:moves)std::cout<<' '<<a;std::cout<<'\n';
          }
          if(hist.isGameFinished)break;
          std::vector<int> legal;
          for(int a=0;a<=size*size;++a)if(hist.isLegal(board,location(a,size),hist.presumedNextMovePla))legal.push_back(a);
          int a=turn%29==0?size*size:legal[rng()%legal.size()];
          hist.makeBoardMoveAssumeLegal(board,location(a,size),hist.presumedNextMovePla,nullptr);moves.push_back(a);
        }
      }
    return 0;
  }
  std::string line;
  while(std::getline(std::cin,line)) {
    std::istringstream input(line);int size,canvas,bound,count,action;std::string rules;double komi;
    input>>size>>canvas>>bound>>rules>>komi>>count;
    Board board(size,size);BoardHistory hist(board,P_BLACK,Rules::parseRulesWithoutKomi(rules,komi),0,BoardHistoryModes(true,true));
    Hash128 hash=GraphHash::getGraphHashFromScratch(hist,P_BLACK,bound,.5);
    for(int i=0;i<count;++i) {
      input>>action;Loc loc=location(action,size);
      if(hist.isGameFinished || !hist.isLegal(board,loc,hist.presumedNextMovePla))throw std::runtime_error("Illegal fixture");
      hist.makeBoardMoveAssumeLegal(board,loc,hist.presumedNextMovePla,nullptr);
      hash=GraphHash::getGraphHash(hash,hist,hist.presumedNextMovePla,bound,.5);
    }
    if(hash!=GraphHash::getGraphHashFromScratch(hist,hist.presumedNextMovePla,bound,.5))throw std::runtime_error("Reference replay differs");
    std::cout<<hash.toString()<<'\n';
  }
}
'''


def main():
    manifest = json.loads((ROOT / 'reference_sources.json').read_text())['KataGo']
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path(manifest['root']))
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/go_graph_test')
    args = parser.parse_args()
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.source, text=True).strip() == manifest['commit']
    for name in ('cpp/game/graphhash.cpp', 'cpp/game/graphhash.h'):
        assert hashlib.sha256((args.source / name).read_bytes()).hexdigest() == manifest['sha256'][name]
    sources = ['game/graphhash.cpp', 'game/board.cpp', 'game/boardhistory.cpp', 'game/rules.cpp',
               'core/global.cpp', 'core/hash.cpp', 'core/rand.cpp', 'core/sha2.cpp', 'core/md5.cpp',
               'core/bsearch.cpp', 'core/test.cpp', 'core/timer.cpp']
    with tempfile.TemporaryDirectory(prefix='etazero_go_graph_reference_') as directory:
        path = Path(directory)
        (path / 'oracle.cpp').write_text(ORACLE)
        subprocess.run(['c++', '-std=c++17', '-O2', '-DCOMPILE_MAX_BOARD_LEN=25', '-pthread',
                        '-I' + str(args.source / 'cpp'), str(path / 'oracle.cpp'),
                        *(str(args.source / 'cpp' / s) for s in sources), '-o', str(path / 'oracle')], check=True)
        queries = subprocess.check_output([str(path / 'oracle'), '--generate'], text=True).splitlines()
        # Ko expiry, positional superko, pass/encore transitions and move orders.
        for rules, moves in [('chinese', [5,6,1,10,7,12,24,16,11,25,4]), ('japanese', [25]*6),
                             ('tromp-taylor', [0,5,24,6,23,2,1,25]),
                             ('chinese', [0,24,2]), ('chinese', [2,24,0])]:
            for bound in (3, 11, 50):
                queries += [f'5 7 {bound} {rules} 7.5 {i} ' + ' '.join(map(str, moves[:i])) for i in range(len(moves)+1)]
        payload = '\n'.join(queries) + '\n'
        expected = subprocess.check_output([str(path / 'oracle')], input=payload, text=True).splitlines()
        actual = subprocess.check_output([str(args.binary), '--hashes'], input=payload, text=True).splitlines()
        assert len(actual) == len(expected) == len(queries)
        for query, want, got in zip(queries, expected, actual):
            assert want == got, f'Go graph mismatch: {query}\nKataGo={want}\nEtaZero={got}'
    print(json.dumps(dict(status='passed', source_commit=manifest['commit'], positions=len(queries)), indent=2))


if __name__ == '__main__':
    main()
