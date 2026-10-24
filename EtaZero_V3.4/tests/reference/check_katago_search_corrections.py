"""Execute fixed KataGo uncertainty/noise/logit-mix bodies against native helpers."""
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

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=Path('/home/sky/RL/SkyZero/KataGo'))
    parser.add_argument('--binary',type=Path,default=ROOT/'build/search_corrections_test')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    manifest=json.loads((ROOT/'reference_sources.json').read_text())['KataGo']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.source,text=True).strip()==manifest['commit']
    files=('cpp/search/searchupdatehelpers.cpp','cpp/neuralnet/cudaandrocmbackend.inc',
           'cpp/neuralnet/nneval.cpp','cpp/search/searchhelpers.cpp')
    hashes={f:hashlib.sha256((args.source/f).read_bytes()).hexdigest() for f in files}
    assert all(hashes[f]==manifest['sha256'][f] for f in files)
    update=(args.source/files[0]).read_text();backend=(args.source/files[1]).read_text()
    uncertainty=body(update,'double Search::computeWeightFromNNOutput(')
    noise=body(update,'double Search::pruneNoiseWeight(')
    mix='policyProbsTmp[i] = p + (pOpt-p) * policyOptimism;'
    assert backend.count(mix)==2
    nneval=(args.source/files[2]).read_text();helpers=(args.source/files[3]).read_text()
    start=nneval.index('    for(int i = 0; i<policySize; i++) {\n      float policyValue;')
    policy_max=nneval[start:nneval.index('    testAssert(legalCount > 0);',start)]
    start=nneval.index('    float policySum = 0.0f;',start)
    end=nneval.index('    if(!isfinite(policySum))',start)
    policy_exp=nneval[start:end]
    start=nneval.index('    if(policySum <= 0.0)',end)
    end=nneval.index('    // Fill everything out-of-bounds',start)
    policy_div=nneval[start:end]
    start=helpers.index('    double maxValue = 0.0;',helpers.index('    double rootPolicyTemperature ='))
    end=helpers.index('\n  if(searchParams.rootNoiseEnabled)',start)
    root_temperature=helpers[start:end].rsplit('  }',1)[0]
    source=r'''
#include <algorithm>
#include <cmath>
#include <cassert>
#include <iostream>
#include <vector>
using std::vector;using std::sqrt;using std::pow;using std::exp;using std::log;
#define testAssert assert
struct NNPos {static constexpr int MAX_NN_POLICY_SIZE=1000;static int locToPos(int,int,int,int){return 0;}};
struct Board {static constexpr int PASS_LOC=0;};
struct Logger {void write(const char*){}};
struct NNOutput {double whiteScoreMean=0,shorttermWinlossError=0,shorttermScoreError=0;};
struct NNEval {bool supported=true;bool supportsShorttermError()const{return supported;}};
struct MoreNodeStats {double selfUtility,weightAdjusted;};
struct Search {
 struct Params {bool useUncertainty=true;double winLossUtilityFactor=1,uncertaintyCoeff=.25,uncertaintyExponent=1,uncertaintyMaxWeight=8,noisePruneUtilityScale=.15,noisePruningCap=1e50;} searchParams;
 NNEval eval;const NNEval* nnEvaluator=&eval;
 double getApproxScoreUtilityDerivative(double)const{return 0;}
 double computeWeightFromNNOutput(const NNOutput*)const;
 double pruneNoiseWeight(vector<MoreNodeStats>&,int,double,const double*)const;
};
UNCERTAINTY
NOISE
int main(){std::cout.precision(17);char kind;
 while(std::cin>>kind){Search s;
  if(kind=='u'){NNOutput n;std::cin>>n.shorttermWinlossError>>s.eval.supported>>s.searchParams.uncertaintyCoeff>>s.searchParams.uncertaintyExponent>>s.searchParams.uncertaintyMaxWeight;std::cout<<s.computeWeightFromNNOutput(&n)<<'\n';}
  else if(kind=='n'){int count;std::cin>>count>>s.searchParams.noisePruneUtilityScale>>s.searchParams.noisePruningCap;
   vector<MoreNodeStats> stats(count);vector<double> policy(count);double total=0;
   for(int i=0;i<count;++i){std::cin>>policy[i]>>stats[i].weightAdjusted>>stats[i].selfUtility;policy[i]=std::max(1e-30,policy[i]);total+=stats[i].weightAdjusted;}
   s.pruneNoiseWeight(stats,count,total,policy.data());for(auto& n:stats)std::cout<<n.weightAdjusted<<' ';std::cout<<'\n';
  }else if(kind=='m'){float p,pOpt,policyOptimism;std::cin>>p>>pOpt>>policyOptimism;float policyProbsTmp[1];int i=0;MIX std::cout<<policyProbsTmp[0]<<'\n';}
  else if(kind=='p'){
    int policySize,allowed;double temperature,outputScaleMultiplier;float policyOptimism;
    std::cin>>policySize>>temperature>>outputScaleMultiplier>>policyOptimism>>allowed;
    if(!allowed)policyOptimism=0;
    vector<float> storage(policySize);float* policy=storage.data();vector<bool> isLegal(policySize);
    float* policyProbsTmp=policy;
    for(int i=0;i<policySize;++i){float p,pOpt;int legal;std::cin>>p>>pOpt>>legal;isLegal[i]=legal;MIX}
    float policyOutputScaling=outputScaleMultiplier/temperature,maxPolicy=-1e25f;int legalCount=0;
    POLICY_MAX
    struct {bool enablePassingHacks=false;} nnInputParams;
    int xSize=1,nnXLen=1,nnYLen=1;
    POLICY_EXP
    struct {bool errorLogLockout=false;} buf;Logger* logger=nullptr;const char* modelFileName="fixture";
    POLICY_DIV
    for(float p:storage)std::cout<<std::max(0.0f,p)<<' ';std::cout<<'\n';
  }else if(kind=='t'){
    int policySize;double rootPolicyTemperature;std::cin>>policySize>>rootPolicyTemperature;
    vector<float> storage(policySize);for(float& p:storage)std::cin>>p;float* noisedPolicyProbs=storage.data();
    ROOT_TEMPERATURE
    for(float p:storage)std::cout<<p<<' ';std::cout<<'\n';
  }else return 1;
 }}
'''.replace('UNCERTAINTY',uncertainty).replace('NOISE',noise).replace('MIX',mix).replace('POLICY_MAX',policy_max).replace('POLICY_EXP',policy_exp).replace('POLICY_DIV',policy_div).replace('ROOT_TEMPERATURE',root_temperature)
    source=source.replace('void write(const char*)','void write(const std::string&)').replace('using std::vector;','using std::string;using std::vector;')
    rows=[];groups=[]
    for u,support,c,p,maximum in itertools.product((0,.03125,.25,.5,1,10,1e4),(0,1),(.0001,.25,1),(0,.5,1,2),(1,8,100)):
        rows.append(f'u {u} {support} {c} {p} {maximum}');groups.append('uncertainty')
    rng=random.Random(606)
    for count,scale,cap in itertools.product((0,1,2,8),(.001,.15,10),(0,.1,1,1e50)):
        for repeat in range(5):
            entries=[(rng.choice((0,1e-10,.1,.9)),rng.choice((1e-7,.25,1,40,1000)),rng.choice((-1,-.5,0,.5,1))) for _ in range(count)]
            rows.append(f'n {count} {scale} {cap}\n'+'\n'.join(f'{p} {w} {q}' for p,w,q in entries));groups.append('noise_pruning')
    for p,o,optimism in itertools.product((-3.125,0,2.5,10),(-10,0,3.125), (0,.2,.5,1)):
        rows.append(f'm {p} {o} {optimism}');groups.append('float_logit_mix')
    for count,temp,scale,optimism,allowed in itertools.product((2,26,362,625),(.2,1,1.7,5),(.125,1,8),(0,.2,1),(0,1)):
        entries=[(rng.uniform(-200,200),rng.uniform(-200,200),int(i==0 or rng.random()>.2)) for i in range(count)]
        rows.append(f'p {count} {temp} {scale} {optimism} {allowed}\n'+'\n'.join(f'{p} {o} {legal}' for p,o,legal in entries));groups.append('float_policy_softmax')
    for logits in ((0,1e-9),(-1e30,-1e30),(0,-200)):
        rows.append('p 2 1 1 0 1\n'+'\n'.join(f'{p} {p} 1' for p in logits));groups.append('float_policy_softmax')
    for count,temp in itertools.product((2,26,362,625),(.2,1,1.7,5)):
        weights=[rng.choice((0,1e-38,.0002,.25,.5)) for _ in range(count)];weights[0]=.5
        rows.append(f't {count} {temp}\n'+' '.join(map(str,weights)));groups.append('float_root_temperature')
    query='\n'.join(rows)+'\n'
    with tempfile.TemporaryDirectory(prefix='etazero_corrections_reference_') as directory:
        path=Path(directory);(path/'oracle.cpp').write_text(source)
        subprocess.run(['c++','-std=c++17','-O2',str(path/'oracle.cpp'),'-o',str(path/'oracle')],check=True)
        expected=subprocess.check_output([str(path/'oracle')],input=query,text=True).splitlines()
        actual=subprocess.check_output([str(args.binary),'--oracle'],input=query,text=True).splitlines()
    assert len(actual)==len(expected)==len(rows)
    for index,(a,b) in enumerate(zip(actual,expected)):
        av,bv=list(map(float,a.split())),list(map(float,b.split()));assert len(av)==len(bv)
        if groups[index] in ('float_policy_softmax','float_root_temperature'):assert av==bv,(index,rows[index],a,b)
        else:assert all(math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-14) for x,y in zip(av,bv)),(index,rows[index],a,b)
    result={'status':'passed','cases':{name:groups.count(name) for name in sorted(set(groups))},'source_commit':manifest['commit'],'source_sha256':hashes,
            'mapping':'pure W-L, Go score utility derivative=0; policy mix, softmax and root temperature use extracted source statements; excluded passing hacks disabled; illegal policy sentinel mapped from -1 to 0; no optimization or search scheduling equivalence claim'}
    if args.output:
        assert not args.output.exists();args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
