"""Compare all hint/PCR/reduced/PDA limit branches to the fixed source function body."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import tempfile
from check_katago_graph import body
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('/home/sky/RL/SkyZero/KataGo'))
    p.add_argument('--binary',type=Path,default=ROOT/'build/sampling_test');p.add_argument('--output',type=Path);a=p.parse_args()
    manifest=json.loads((ROOT/'reference_sources.json').read_text())['KataGo'];source=a.source/'cpp/program/play.cpp'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.source,text=True).strip()==manifest['commit']
    checksum=hashlib.sha256(source.read_bytes()).hexdigest();assert checksum==manifest['sha256']['cpp/program/play.cpp']
    limits_function=body(source.read_text(),'static SearchLimitsThisMove getSearchLimitsThisMove(')
    limits_body=limits_function[limits_function.index('{')+1:-1]
    oracle=r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
#include <stdexcept>
using std::vector;using std::pow;using std::round;using std::ceil;using Player=int;using Loc=int;int getOpp(int p){return -p;}
constexpr int C_EMPTY=0;using StringError=std::runtime_error;
#define testAssert(x) assert(x)
struct Board{static constexpr int NULL_LOC=-1;int pos_hash=1;};
struct BoardHistory{vector<int> moveHistory;};
struct Params{int64_t maxVisits=400,maxPlayouts=600;};
struct Search{Params searchParams;BoardHistory hist;Board board;
 const BoardHistory& getRootHist()const{return hist;}const Board& getRootBoard()const{return board;}};
struct Rand{bool choice;double probability=0;bool nextBool(double p){probability=p;return choice;}};
struct PlaySettings{
 double cheapSearchProb=.6;int cheapSearchVisits=70;float cheapSearchTargetWeight=0;
 bool reduceVisits=true;int reducedVisitsMin=350,reduceVisitsThresholdLookback=3;
 double reduceVisitsThreshold=.9;float reducedVisitsWeight=.1f;
};
struct OtherGameProperties{int hintLoc=-1,hintTurn=-1,hintPosHash=1;bool isHintFork=false;
 double playoutDoublingAdvantage=0;Player playoutDoublingAdvantagePla=0;};
struct SearchLimitsThisMove{
 bool doAlterVisitsPlayouts;int64_t numAlterVisits,numAlterPlayouts;
 bool clearBotBeforeSearchThisMove,removeRootNoise;float targetWeight;bool isCheapSearch;
 double playoutDoublingAdvantage;Player playoutDoublingAdvantagePla;int hintLoc;
};
SearchLimitsThisMove getSearchLimitsThisMove(const Search* toMoveBot,Player pla,const PlaySettings& playSettings,Rand& gameRand,
 const vector<double>& historicalMctsWinLossValues,size_t numHistoricalMctsValuesToUse,bool clearBotBeforeSearch,
 const OtherGameProperties& otherGameProps,bool forceFullSearch){BODY}
int main(){double d;int adv,pla,current,turn,hint,exact,hint_fork,force,cheap,count,clear;
 std::cout.precision(17);
 while(std::cin>>d>>adv>>pla>>current>>turn>>hint>>exact>>hint_fork>>force>>cheap>>clear>>count){
  vector<double> history(count);for(auto& q:history)std::cin>>q;
  Search search;search.hist.moveHistory.resize(current);search.board.pos_hash=exact?1:2;
  OtherGameProperties properties{hint,turn,1,bool(hint_fork),d,adv};PlaySettings config;Rand rng{bool(cheap)};
  auto limits=getSearchLimitsThisMove(&search,pla,config,rng,history,history.size(),clear,properties,force);
  std::cout<<limits.numAlterVisits<<' '<<limits.numAlterPlayouts<<' '<<limits.targetWeight<<' '<<limits.isCheapSearch<<' '
           <<(limits.clearBotBeforeSearchThisMove || limits.hintLoc>=0)<<' '<<limits.removeRootNoise<<' '<<limits.hintLoc<<' '<<rng.probability<<'\n';
 }}
'''.replace('BODY',limits_body)
    contexts=[(0,-1,-1,0,0),(2,2,12,1,0),(2,2,12,0,0),(7,2,12,0,0),(8,2,12,0,0),
              (2,2,-1,0,1),(7,2,-1,0,1),(8,2,-1,0,1)]
    rows=[]
    for d,adv,pla,context,force,cheap,clear,history in itertools.product((0,1,3),(-1,1),(-1,1),contexts,(0,1),(0,1),(0,1),
                                                                     ((),(.8,.8,.8),(.95,.96,.97),(-1,-1,-1))):
        current,turn,hint,exact,hint_fork=context
        rows.append(' '.join(map(str,(d,adv,pla,current,turn,hint,exact,hint_fork,force,cheap,clear,len(history),*history))))
    with tempfile.TemporaryDirectory(prefix='etazero_fork_reference_') as temp:
        path=Path(temp);(path/'limits.cpp').write_text(oracle)
        subprocess.run(['c++','-std=c++17','-O2',str(path/'limits.cpp'),'-o',str(path/'limits')],check=True)
        query='\n'.join(rows)+'\n'
        expected=subprocess.check_output([str(path/'limits')],input=query,text=True).splitlines()
        actual=subprocess.check_output([str(a.binary),'--limits'],input=query,text=True).splitlines()
        assert len(actual)==len(expected)==len(rows)
        for i,(one,two) in enumerate(zip(actual,expected)):
            av,bv=list(map(float,one.split())),list(map(float,two.split()));assert len(av)==len(bv)==8
            # Source float targetWeight vs double internal frequency, both float32 on disk.
            assert all(math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-14) if j!=2 else abs(x-y)<6e-8
                       for j,(x,y) in enumerate(zip(av,bv))), (i,rows[i],one,two)
    result={'status':'passed','cases':len(rows),'source_commit':manifest['commit'],'source_sha256':checksum,
            'scope':'Complete fixed-source limits body: exact hint/hash mismatch, six-turn boundary, forceFull, PCR/reduced priority, PDA both players, explicit finite playouts; effective hint clear from runBotWithLimits',
            'differences':['Source float targetWeight vs EtaZero double internal frequency (float32 disk), abs tolerance 6e-8 only on weight; all budgets/flags/probabilities use 1e-12/1e-14.',
                           'Pure NOVC hint input and W-L fork ranking; Go SGF/score/komi/seki excluded; source RNG/scheduling equivalence not claimed.']}
    if a.output:assert not a.output.exists();a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
