"""Execute fixed source PDA and value-surprise bodies with NOVC scalar shims."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import subprocess
import tempfile
from check_katago_graph import body
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('/home/sky/RL/SkyZero/KataGo'))
    p.add_argument('--binary',type=Path,default=ROOT/'build/sampling_test');p.add_argument('--output',type=Path);a=p.parse_args()
    manifest=json.loads((ROOT/'reference_sources.json').read_text())['KataGo']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.source,text=True).strip()==manifest['commit']
    paths=('cpp/program/play.cpp','cpp/neuralnet/nninputs.cpp','cpp/search/searchnnhelpers.cpp')
    hashes={name:hashlib.sha256((a.source/name).read_bytes()).hexdigest() for name in paths}
    assert all(hashes[name]==manifest['sha256'][name] for name in paths)
    play,inputs,nn=((a.source/name).read_text() for name in paths)
    budget=body(play,'  if(otherGameProps.playoutDoublingAdvantage != 0.0 && otherGameProps.playoutDoublingAdvantagePla != C_EMPTY)')
    sign=body(nn,'  if(searchParams.playoutDoublingAdvantage != 0)')
    globals_block=body(inputs[inputs.index('//Parameter 15 is used'):],'  if(nnInputParams.playoutDoublingAdvantage != 0)')
    kl=body(play,'static double valueSurpriseKL(')
    surprise=body(play,'static void computeValueSurpriseByTurn(')
    source=r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
using std::vector;using std::pow;using std::log;
#define testAssert(x) assert(x)
using Player=int;constexpr int C_EMPTY=0;
int getOpp(int p){return -p;}using StringError=std::runtime_error;
struct Params {double playoutDoublingAdvantage;};
struct Inputs {double playoutDoublingAdvantage=0;};
struct Condition {
 Params searchParams;int advantaged;
 int getPlayoutDoublingAdvantagePla(){return advantaged;}
 double input(int pla){Inputs nnInputParams;SIGN return nnInputParams.playoutDoublingAdvantage;}
};
void budget(double d,int advantaged,int pla,int n,int cheap){
 struct Props {double playoutDoublingAdvantage;int playoutDoublingAdvantagePla;}otherGameProps{d,advantaged};
 int64_t numAlterVisits=n,numAlterPlayouts=n;
 double playoutDoublingAdvantage=0;int playoutDoublingAdvantagePla=0;
 bool doAlterVisitsPlayouts=false,clearBotBeforeSearchThisMove=!cheap;
 BUDGET
 Inputs nnInputParams{Condition{{d},advantaged}.input(pla)};float rowGlobal[17]={0};GLOBALS
 std::cout<<numAlterVisits<<' '<<numAlterPlayouts<<' '<<clearBotBeforeSearchThisMove<<' '<<rowGlobal[15]<<' '<<rowGlobal[16]<<'\n';
}
struct ValueTargets {double win,loss,noResult;};
struct ReportedSearchValues {double winValue,lossValue,noResultValue;};
KL
SURPRISE
int main(int argc,char**){std::cout.precision(17);
 if(argc==1){double d;int adv,pla,n,cheap;
  while(std::cin>>d>>adv>>pla>>n>>cheap)budget(d,adv,pla,cheap?std::max(5,n/2):n,cheap);
 }else{int direct,size,winner,count;
  while(std::cin>>direct>>size>>winner>>count){vector<ValueTargets> searched;vector<ReportedSearchValues> raw;
   for(int i=0;i<count;++i){int p;double nw,nd,nl,sw,sd,sl;std::cin>>p>>nw>>nd>>nl>>sw>>sd>>sl;
    searched.push_back({p==1?sw:sl,p==1?sl:sw,sd});raw.push_back({p==1?nw:nl,p==1?nl:nw,nd});}
   searched.push_back({double(winner==1),double(winner==-1),double(winner==0)});
   vector<double> out;computeValueSurpriseByTurn(out,searched,raw,size*size,direct);
   for(double v:out)std::cout<<v<<' ';std::cout<<'\n';}
 }}
'''
    for key,text in [('SIGN',sign),('BUDGET',budget),('GLOBALS',globals_block),('KL',kl),('SURPRISE',surprise)]:source=source.replace(key,text)
    budgets=[]
    for d,adv,pla,n,cheap in itertools.product((0,.001,.5,1,2,3,math.log2(100)),(-1,1),(-1,1),(80,160,400,1000),(0,1)):
        cap=max(5,n//2) if cheap else n
        if cap*2/(1+2**d)<4.5:continue
        budgets.append(f'{d:.17g} {adv} {pla} {n} {cheap}')
    rng=random.Random(707);surprises=[]
    for direct,size,winner,count in itertools.product((0,1),(5,11,15),(-1,0,1),(1,2,6)):
        for repeat in range(4):
            entries=[]
            for i in range(count):
                raw=[rng.choice((0,.01,.25,.5,1)) for _ in range(3)]
                if not sum(raw):raw[1]=1
                raw=[x/sum(raw) for x in raw]
                searched=[rng.random() for _ in range(3)];searched=[x/sum(searched) for x in searched]
                entries.append(' '.join(map(str,(1 if i%2==0 else -1,*raw,*searched))))
            surprises.append(f'{direct} {size} {winner} {count}\n'+'\n'.join(entries))
    with tempfile.TemporaryDirectory(prefix='etazero_sampling_reference_') as temp:
        path=Path(temp);(path/'oracle.cpp').write_text(source)
        subprocess.run(['c++','-std=c++17','-O2',str(path/'oracle.cpp'),'-o',str(path/'oracle')],check=True)
        for name,rows,native_arg,ref_arg in [('pda',budgets,'--oracle',None),('value_surprise',surprises,'--surprise','--surprise')]:
            query='\n'.join(rows)+'\n'
            expected=subprocess.check_output([str(path/'oracle')]+([ref_arg] if ref_arg else []),input=query,text=True).splitlines()
            actual=subprocess.check_output([str(a.binary),native_arg],input=query,text=True).splitlines()
            assert len(actual)==len(expected)==len(rows)
            for i,(one,two) in enumerate(zip(actual,expected)):
                av,bv=list(map(float,one.split())),list(map(float,two.split()));assert len(av)==len(bv)
                assert all(math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-14) for x,y in zip(av,bv)),(name,i,rows[i],one,two)
    result={'status':'passed','cases':{'pda_budget_input':len(budgets),'value_surprise':len(surprises)},'source_commit':manifest['commit'],'source_sha256':hashes,
            'scope':'PDA signed input, explicit finite visits/playouts, direct/smoothed fixed-perspective WDL; no Go komi/handicap/score; EtaZero int32 max is unlimited; no source RNG/scheduling equivalence claim'}
    if a.output:assert not a.output.exists();a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
