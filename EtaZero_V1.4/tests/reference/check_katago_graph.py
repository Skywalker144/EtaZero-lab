"""Compare traced EtaZero diamonds with executable fixed KataGo scalar bodies.

Compile the original childWeight/childWeightSq, maybeCatchUpEdgeVisits and
recomputeNodeStats child aggregation loop. Shims supply atomics and unit NN
weight for NOVC WDL; Go score/no-result, bias, noise and reweighting are off.
No reference repo is modified, imported into production, or copied to a fixture.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def body(text, marker):
    start = text.index(marker)
    opening = text.index('{', start)
    # These selected functions contain no braces in string literals.
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('/home/sky/RL/SkyZero/KataGo'))
    parser.add_argument('--binary', type=Path, default=ROOT/'build/graph_test')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT/'reference_sources.json').read_text())['KataGo']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.source,text=True).strip() == manifest['commit']
    files = ('cpp/search/searchnode.h','cpp/search/search.cpp','cpp/search/searchupdatehelpers.cpp')
    hashes = {f:hashlib.sha256((args.source/f).read_bytes()).hexdigest() for f in files}
    assert all(hashes[f] == manifest['sha256'][f] for f in files)
    node, search, update = [(args.source/f).read_text() for f in files]
    functions = '\n'.join(body(node, 'inline static double '+name+'(') for name in ('childWeight','childWeightSq'))
    catch = body(search, 'bool Search::maybeCatchUpEdgeVisits(')
    loop = body(update, '  for(int i = 0; i<numGoodChildren; i++) {\n    const NodeStats& stats = statsBuf[i].stats;')
    source = r'''
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <vector>
struct NodeStats {
    double weightSum,weightSqSum,winLossValueAvg,noResultValueAvg,scoreMeanAvg,scoreMeanSqAvg,leadAvg,utilityAvg,utilitySqAvg;
    FUNCTIONS
};
struct MoreNodeStats {NodeStats stats;double weightAdjusted;};
struct SearchChildPointer {
    std::atomic<int64_t> edgeVisits{0};
    int64_t getEdgeVisits(){return edgeVisits.load();}
    bool compexweakEdgeVisits(int64_t& old,int64_t value){return edgeVisits.compare_exchange_weak(old,value);}
};
struct SearchNodeState {};
struct SearchNode {
    struct Stats {std::atomic<int64_t> visits{0};} stats;
    SearchChildPointer child;
    struct Children {SearchChildPointer& p;SearchChildPointer& operator[](int){return p;}};
    Children getChildren(const SearchNodeState&){return {child};}
};
using SearchNodeChildrenReference=SearchNode::Children;
struct SearchThread {struct Rand {bool coin=false;bool nextBool(double){return coin;}} rand;};
struct Search {
    struct Params {double graphSearchCatchUpLeakProb=0;} searchParams;
    bool maybeCatchUpEdgeVisits(SearchThread&,SearchNode&,const SearchNode*,const SearchNodeState&,int);
};
CATCH
int main() {
    std::cout.precision(17);char mode;
    while(std::cin>>mode) {
        if(mode=='c') {
            int64_t childVisits,edgeVisits;double leak;bool coin;std::cin>>childVisits>>edgeVisits>>leak>>coin;
            Search s;SearchNode n,c;SearchThread t;SearchNodeState state;
            s.searchParams.graphSearchCatchUpLeakProb=leak;c.stats.visits=childVisits;n.child.edgeVisits=edgeVisits;t.rand.coin=coin;
            bool caught=s.maybeCatchUpEdgeVisits(t,n,&c,state,0);std::cout<<caught<<' '<<n.child.edgeVisits.load()<<'\n';
        } else if(mode=='a') {
            int numGoodChildren;std::cin>>numGoodChildren;std::vector<MoreNodeStats> statsBuf(numGoodChildren);
            for(auto& entry:statsBuf) {
                int64_t childVisits,edgeVisits;auto& t=entry.stats;
                std::cin>>childVisits>>edgeVisits>>t.weightSum>>t.weightSqSum>>t.utilityAvg>>t.utilitySqAvg>>t.noResultValueAvg;
                t.winLossValueAvg=t.utilityAvg;t.scoreMeanAvg=t.scoreMeanSqAvg=t.leadAvg=0;
                entry.weightAdjusted=NodeStats::childWeight(edgeVisits,childVisits,t.weightSum);
            }
            double winLossValueSum=.4,noResultValueSum=.2,scoreMeanSum=0,scoreMeanSqSum=0,leadSum=0;
            double utilitySum=.4,utilitySqSum=.16,weightSqSum=1,weightSum=1;
            LOOP
            for(auto& entry:statsBuf)weightSum+=entry.weightAdjusted;
            std::cout<<weightSum<<' '<<weightSqSum<<' '<<utilitySum/weightSum<<' '<<utilitySqSum/weightSum<<' '<<noResultValueSum/weightSum<<'\n';
        } else if(mode=='w') {
            int64_t e,c;double w,s;std::cin>>e>>c>>w>>s;
            std::cout<<NodeStats::childWeight(e,c,w)<<' '<<NodeStats::childWeightSq(e,c,s)<<'\n';
        } else return 1;
    }
}
'''.replace('FUNCTIONS',functions).replace('CATCH',catch).replace('LOOP',loop)
    with tempfile.TemporaryDirectory(prefix='etazero_graph_reference_') as directory:
        path=Path(directory);(path/'oracle.cpp').write_text(source)
        subprocess.run(['c++','-std=c++17','-O2',str(path/'oracle.cpp'),'-o',str(path/'oracle')],check=True)
        cases=0;catchups=0
        for leak in (0,1):
            trace=[json.loads(s) for s in subprocess.check_output([str(args.binary),'--trace',str(leak)],text=True).splitlines()]
            previous={};queries=[];expected=[]
            for step in trace:
                nodes={n['id']:n for n in step['nodes']};caught=0
                for n in nodes.values():
                    old=previous.get(n['id'],{});old_edges={e[0]:e for e in old.get('edges',[])}
                    for action,child,visits,perspective in n['edges']:
                        before=old_edges.get(action,[action,child,0,perspective])[2]
                        assert visits-before in (0,1)
                        if visits>before:
                            child_visits=previous.get(child,{}).get('visits',0)
                            queries.append(f'c {child_visits} {before} {leak} {int(leak==1)}')
                            will_catch=before<child_visits and leak==0
                            expected.append([int(will_catch),before+int(will_catch)])
                            caught+=will_catch
                    if n['visits'] and (n['visits']!=old.get('visits',0) or n['root']):
                        if n['key']=='terminal':
                            assert n['value']==n['value_sq']==1 and n['draw']==0
                            assert n['weight']==n['weight_sq']==n['visits']
                        else:
                            good=[e for e in n['edges'] if e[2]>0 and nodes[e[1]]['visits']>0]
                            query=[f'a {len(good)}']
                            for _,child,visits,perspective in good:
                                c=nodes[child]
                                query.append(f"{c['visits']} {visits} {c['weight']} {c['weight_sq']} {perspective*c['value']} {c['value_sq']} {c['draw']}")
                            queries.append('\n'.join(query));expected.append([n[k] for k in ('weight','weight_sq','value','value_sq','draw')])
                assert caught==step['catch_ups'],(leak,step['step'],caught,step['catch_ups'])
                catchups+=caught;previous=nodes
            results=subprocess.check_output([str(path/'oracle')],input='\n'.join(queries)+'\n',text=True).splitlines()
            assert len(results)==len(expected)
            for actual,wanted in zip(results,expected):
                values=list(map(float,actual.split()));assert len(values)==len(wanted)
                assert all(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-14) for a,b in zip(values,wanted)),(actual,wanted)
                cases+=1
        # Distinguish LCB's linear weight-square mapping from parent aggregation's squared fraction.
        assert subprocess.check_output([str(path/'oracle')],input='w 2 8 12 20\na 1\n8 2 12 20 0.4 0.16 0.2\n',text=True).splitlines()==['3 5','4 2.25 0.40000000000000002 0.16 0.20000000000000001']
    result={'status':'passed','diamond_steps':200,'source_scalar_cases':cases,'catch_up_playouts':catchups,
            'source_commit':manifest['commit'],'source_sha256':hashes,
            'scope':'one-thread diamond; leak=0/1; terminal and shared node counts; raw moments and distinct LCB/aggregation weightSq',
            'limits':'Not a full Go search equivalence or concurrent scheduling/performance proof.'}
    if args.output:
        assert not args.output.exists(),'Refuse to overwrite evidence';args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
