"""Compare fixed-source surprise redistribution to the actual production core library."""
import argparse
import hashlib
import itertools
import json
import math
import re
from pathlib import Path
import subprocess
import tempfile
from check_katago_graph import body
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('/home/sky/RL/SkyZero/KataGo'))
    p.add_argument('--build',type=Path,default=ROOT/'build');p.add_argument('--output',type=Path);a=p.parse_args()
    manifest=json.loads((ROOT/'reference_sources.json').read_text())['KataGo'];source=a.source/'cpp/program/play.cpp'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.source,text=True).strip()==manifest['commit']
    checksum=hashlib.sha256(source.read_bytes()).hexdigest();assert checksum==manifest['sha256']['cpp/program/play.cpp']
    code=source.read_text();weights=body(code,'    if(playSettings.policySurpriseDataWeight > 0 || playSettings.valueSurpriseDataWeight > 0)')
    kl=body(code,'static double valueSurpriseKL(')
    oracle=r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
using std::vector;using std::log;
#define testAssert(x) assert(x)
struct ReportedSearchValues{double winValue,lossValue,noResultValue;};
__KL_BODY__
struct Reanalysis{bool wasReanalyzed;};
struct Data{vector<float> targetWeightByTurn;vector<double> policySurpriseByTurn;vector<Reanalysis> reanalysisByTurn;};
struct PlaySettings{double policySurpriseDataWeight,valueSurpriseDataWeight;bool useReanalyze;};
int main(){double alpha,beta;int reanalyze,count;std::cout.precision(17);
 while(std::cin>>alpha>>beta>>reanalyze>>count){Data data;Data* gameData=&data;PlaySettings playSettings{alpha,beta,bool(reanalyze)};
  vector<bool> wasCheapSearchByTurn;vector<double> valueSurpriseByTurn;
  for(int i=0;i<count;++i){float weight;double surprise,nw,nd,nl,sw,sd,sl;int cheap,redone;
   std::cin>>weight>>surprise>>cheap>>redone>>nw>>nd>>nl>>sw>>sd>>sl;
   data.targetWeightByTurn.push_back(weight);data.policySurpriseByTurn.push_back(surprise);data.reanalysisByTurn.push_back({bool(redone)});
   wasCheapSearchByTurn.push_back(cheap);valueSurpriseByTurn.push_back(std::max(0.,std::min(1.,valueSurpriseKL(sw,sl,sd,{nw,nl,nd}))));}
  WEIGHTS
  for(float w:data.targetWeightByTurn)std::cout<<w<<' ';std::cout<<'\n';
 }}
'''.replace('__KL_BODY__',kl).replace('WEIGHTS',weights)
    native=r'''
#include "etazero/record.h"
#include <iostream>
using namespace etazero;
int main(){double alpha,beta;int reanalyze,count;std::cout.precision(17);
 while(std::cin>>alpha>>beta>>reanalyze>>count){FinishedGame game{};game.size=5;std::mt19937_64 rng(7);
  for(int i=0;i<count;++i){Step step{};float weight;int cheap,redone;step.player=1;
   std::cin>>weight>>step.policy_surprise>>cheap>>redone;
   for(auto& x:step.network_wdl)std::cin>>x;for(auto& x:step.search_wdl)std::cin>>x;
   step.target_weight=weight;step.cheap_search=cheap;step.reanalyzed=redone;game.steps.push_back(step);}
  apply_training_weights(game,alpha,beta,rng,true,reanalyze);
  for(const auto& step:game.steps)std::cout<<float(step.target_weight)<<' ';std::cout<<'\n';
 }}
'''
    # Zero/all-low surprise, total<1, PCR excess, un-reanalyzed cheap exclusion,
    # reanalyzed reduced excess, both mixtures and actual KL clipping.
    templates=[[(1,0,0,0),(1,0,0,0)],[(.1,1,0,0),(.1,20,1,1)],
               [(1,1,0,0),(1,1,0,0),(0,20,1,0)],[(1,1,0,0),(.1,20,0,0)],
               [(1,1,0,0),(.1,20,1,1)],[(1,0,0,0),(0,0,1,0)],
               [(1,0,0,0),(1,5,0,0),(.1,20,0,0),(0,20,1,0)]]
    pairs=[((.5,0,.5),(.5,0,.5)),((.999,0,.001),(1,0,0)),((.9,0,.1),(1,0,0)),((.01,0,.99),(1,0,0))]
    rows=[]
    for alpha,beta,reanalyze,template,pair in itertools.product((0,.5), (0,.1,.5),(0,1),templates,pairs):
        row=f'{alpha} {beta} {reanalyze} {len(template)}\n'
        row+='\n'.join(' '.join(map(str,(*step,*pair[0],*pair[1]))) for step in template);rows.append(row)
    library=a.build/'libetazero_core.a'
    zlib=re.search(r'^ZLIB_LIBRARY_RELEASE:FILEPATH=(.+)$',(a.build/'CMakeCache.txt').read_text(),re.M).group(1)
    assert Path(zlib).is_file()
    with tempfile.TemporaryDirectory(prefix='etazero_weight_reference_') as temp:
        path=Path(temp);(path/'oracle.cpp').write_text(oracle);(path/'native.cpp').write_text(native)
        subprocess.run(['c++','-std=c++17','-O2',str(path/'oracle.cpp'),'-o',str(path/'oracle')],check=True)
        subprocess.run(['c++','-std=c++17','-O2','-I'+str(ROOT/'cpp/include'),'-I'+str(a.build/'generated'),
                        str(path/'native.cpp'),str(library),zlib,'-pthread','-o',str(path/'native')],check=True)
        query='\n'.join(rows)+'\n';actual=subprocess.check_output([str(path/'native')],input=query,text=True).splitlines()
        expected=subprocess.check_output([str(path/'oracle')],input=query,text=True).splitlines();assert len(actual)==len(expected)==len(rows)
        worst=0
        for i,(one,two) in enumerate(zip(actual,expected)):
            av,bv=list(map(float,one.split())),list(map(float,two.split()));assert len(av)==len(bv)
            for x,y in zip(av,bv):
                worst=max(worst,abs(x-y));assert math.isclose(x,y,rel_tol=3e-7,abs_tol=6e-8),(i,rows[i],one,two)
    result={'status':'passed','cases':len(rows),'source_commit':manifest['commit'],'source_sha256':checksum,
            'native_library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'max_absolute_difference':worst,
            'scope':'Actual production apply_training_weights linked without copying its implementation; direct source KL and complete redistribution block; zero/low/all/excess/reanalyze/reduced gates; row stochastic rounding is separately checked, never multiplied as loss weight',
            'precision':'Source float intermediates versus double internal frequency, both serialized float32; 3e-7 relative and 6e-8 absolute.'}
    if a.output:assert not a.output.exists();a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
