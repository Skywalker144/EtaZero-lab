"""Compare Go game-layer keys with the pinned, unmodified KataGo SVB table."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
ORACLE = r'''
#include "search/subtreevaluebiastable.cpp"
#include "game/boardhistory.h"
#include <iostream>
#include <random>
#include <sstream>

Loc location(int action,int size) {
  return action==size*size?Board::PASS_LOC:Location::getLoc(action%size,action/size,size);
}
int main(int argc,char**) {
  Board::initHash();SubtreeValueBiasTable table(1);
  if(argc>1) {
    std::mt19937 rng(619);
    for(const char* rules:{"chinese","japanese","tromp-taylor","new-zealand","aga"})
      for(int size:{5,9,19,25})for(int trial=0;trial<2;++trial) {
        Board board(size,size);BoardHistory hist(board,P_BLACK,Rules::parseRulesWithoutKomi(rules,7.5),0,BoardHistoryModes(true,true));
        std::vector<int> moves;
        for(int turn=0;turn<300;++turn) {
          if(turn<20 || turn%7==0) {
            std::cout<<size<<' '<<size+(size<25?trial:0)<<' '<<rules<<' '<<moves.size();
            for(int a:moves)std::cout<<' '<<a;std::cout<<'\n';
          }
          if(hist.isGameFinished)break;
          std::vector<int> legal;
          for(int a=0;a<=size*size;++a)if(hist.isLegal(board,location(a,size),hist.presumedNextMovePla))legal.push_back(a);
          int a=turn%31==0?size*size:legal[rng()%legal.size()];
          hist.makeBoardMoveAssumeLegal(board,location(a,size),hist.presumedNextMovePla,nullptr);moves.push_back(a);
        }
      }
    return 0;
  }
  std::string query,keys;size_t positions=0,patterns=0,atari=0,ko=0,previous_pass=0;
  while(std::getline(std::cin,query) && std::getline(std::cin,keys)) {
    std::istringstream input(query),actual(keys);int size,canvas,count,action;std::string rules;
    input>>size>>canvas>>rules>>count;
    Board board(size,size);BoardHistory hist(board,P_BLACK,Rules::parseRulesWithoutKomi(rules,7.5),0,BoardHistoryModes(true,true));
    for(int i=0;i<count;++i) {
      input>>action;Loc loc=location(action,size);
      if(hist.isGameFinished || !hist.isLegal(board,loc,hist.presumedNextMovePla))throw std::runtime_error("Illegal reference fixture");
      hist.makeBoardMoveAssumeLegal(board,loc,hist.presumedNextMovePla,nullptr);
    }
    for(int a=0;a<size*size;++a) {
      Loc loc=location(a,size);
      if(hist.isGameFinished || hist.moveHistory.empty() || !hist.isLegal(board,loc,hist.presumedNextMovePla))continue;
      uint64_t pattern,moves;
      if(!(actual>>action>>pattern>>moves) || action!=a)throw std::runtime_error("Wrong SVB eligibility/action");
      // Translate the compact representation into the source's random basis.
      // Expected entry is produced by the actual SubtreeValueBiasTable::get.
      Hash128 hash=ZOBRIST_MOVE_LOCS[moves&65535][0]^ZOBRIST_MOVE_LOCS[(moves>>16)&65535][1];
      hash^=patternHasher.zobristPla[(moves>>32)&3];
      for(int i=0;i<25;++i) {
        int color=(pattern>>(2*(24-i)))&3;
        if(color!=C_WALL)hash^=patternHasher.zobristLocalPattern[color*25+i];
        if((moves>>(34+i))&1)hash^=patternHasher.zobristAtari[i];
      }
      int ban=pattern>>50;if(ban)hash^=ZOBRIST_KO_BAN[ban];
      auto expected=table.get(hist.presumedNextMovePla,hist.moveHistory.back().loc,loc,board);
      auto found=table.entries[0].find(hash);
      if(found==table.entries[0].end() || found->second!=expected)
        throw std::runtime_error("Go SVB differs from KataGo at position "+std::to_string(positions)+" action "+std::to_string(a));
      ++patterns;atari+=bool(moves>>34);ko+=bool(ban);previous_pass+=hist.moveHistory.back().loc==Board::PASS_LOC;
    }
    if(actual>>action)throw std::runtime_error("Unexpected SVB key (first move, terminal or selected pass)");
    ++positions;
  }
  if(!atari || !ko || !previous_pass)throw std::runtime_error("Missing atari/ko/pass coverage");
  std::cout<<"{\"positions\":"<<positions<<",\"patterns\":"<<patterns<<",\"atari\":"<<atari
    <<",\"ko\":"<<ko<<",\"previous_pass\":"<<previous_pass<<"}\n";
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('/home/sky/RL/SkyZero/KataGo'))
    parser.add_argument('--binary', type=Path, default=ROOT/'build/subtree_value_bias_test')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'reference_sources.json').read_text())['KataGo']
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.source, text=True).strip() == manifest['commit']
    for path in ('cpp/search/subtreevaluebiastable.cpp', 'cpp/search/localpattern.cpp', 'cpp/search/search.cpp'):
        assert hashlib.sha256((args.source/path).read_bytes()).hexdigest() == manifest['sha256'][path]
    sources = ['game/board.cpp', 'game/boardhistory.cpp', 'game/rules.cpp', 'neuralnet/nninputs.cpp',
               'core/global.cpp', 'core/hash.cpp', 'core/rand.cpp', 'core/sha2.cpp', 'core/md5.cpp',
               'core/bsearch.cpp', 'core/test.cpp', 'core/timer.cpp', 'search/localpattern.cpp', 'search/mutexpool.cpp']
    with tempfile.TemporaryDirectory(prefix='etazero_go_bias_reference_') as directory:
        path = Path(directory)
        (path/'oracle.cpp').write_text(ORACLE)
        subprocess.run(['c++', '-std=c++17', '-O2', '-DCOMPILE_MAX_BOARD_LEN=25', '-pthread',
                        '-I'+str(args.source/'cpp'), str(path/'oracle.cpp'),
                        *(str(args.source/'cpp'/s) for s in sources), '-o', str(path/'oracle')], check=True)
        queries = subprocess.check_output([str(path/'oracle'), '--generate'], text=True).splitlines()
        # Deterministic simple-ko, expired-ko, positional-superko and encore transitions.
        for rules, moves in [('chinese', [5,6,1,10,7,12,24,16,11,25,4]), ('japanese', [25]*6),
                             ('tromp-taylor', [0,5,24,6,23,2,1,25])]:
            queries += [f'5 7 {rules} {i} '+ ' '.join(map(str,moves[:i])) for i in range(len(moves)+1)]
        actual = subprocess.check_output([str(args.binary), '--patterns'], input='\n'.join(queries)+'\n', text=True).splitlines()
        assert len(actual) == len(queries)
        report = subprocess.check_output([str(path/'oracle')], input=''.join(q+'\n'+a+'\n' for q,a in zip(queries,actual)), text=True)
    print(json.dumps(dict(status='passed', source_commit=manifest['commit'], **json.loads(report)), indent=2))


if __name__ == '__main__':
    main()
