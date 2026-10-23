"""Execute the pinned KataGo bias update, leaf and removal blocks verbatim."""
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

ROOT = Path(__file__).resolve().parents[2]


def block(text, start):
    opening = text.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('/home/sky/RL/SkyZero/KataGo'))
    parser.add_argument('--binary', type=Path, default=ROOT/'build/subtree_value_bias_test')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'reference_sources.json').read_text())['KataGo']
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.source, text=True).strip() == manifest['commit']
    paths = ('cpp/search/searchupdatehelpers.cpp', 'cpp/search/search.cpp')
    for path in paths:
        assert hashlib.sha256((args.source/path).read_bytes()).hexdigest() == manifest['sha256'][path]
    update = (args.source/paths[0]).read_text()
    marker = 'if(searchParams.subtreeValueBiasFactor != 0'
    leaf = block(update, update.index(marker))
    backup = block(update, update.index(marker, update.index(marker)+1))
    removal = body((args.source/paths[1]).read_text(), 'void Search::removeSubtreeValueBias(')
    oracle = r'''
#include <atomic>
#include <cmath>
#include <iostream>
#include <memory>
using std::pow;
struct SubtreeValueBiasEntry {
 double deltaUtilitySum=0,weightSum=0;
 std::atomic_flag entryLock=ATOMIC_FLAG_INIT;
};
struct SearchNode {
 std::shared_ptr<SubtreeValueBiasEntry> subtreeValueBiasTableEntry;
 double lastSubtreeValueBiasDeltaSum=0,lastSubtreeValueBiasWeight=0;
};
struct Search {
 struct Params {double subtreeValueBiasFactor,subtreeValueBiasWeightExponent,subtreeValueBiasFreeProp;} searchParams;
 double leaf(SearchNode& node,double utility) {
  bool isTerminal=false;
  LEAF
  return utility;
 }
 double update(SearchNode& node,double utility,double child,double origTotalChildWeight,double currentTotalChildWeight) {
  double utilitySum=child*currentTotalChildWeight;
  BACKUP
  return utility;
 }
 void removeSubtreeValueBias(SearchNode*);
};
REMOVAL
int main() {
 std::cout.precision(17);Search search;SearchNode nodes[2];
 auto shared=std::make_shared<SubtreeValueBiasEntry>();
 nodes[0].subtreeValueBiasTableEntry=nodes[1].subtreeValueBiasTableEntry=shared;
 char op;int n;
 while(std::cin>>op>>n) {
  if(op=='r') {shared=std::make_shared<SubtreeValueBiasEntry>();nodes[0]={shared,0,0};nodes[1]={shared,0,0};continue;}
  double utility;
  if(op=='u') {double child,original,current;
   std::cin>>utility>>child>>original>>current>>search.searchParams.subtreeValueBiasFactor>>search.searchParams.subtreeValueBiasWeightExponent;
   std::cout<<search.update(nodes[n],utility,child,original,current);
  } else if(op=='l') {
   std::cin>>utility>>search.searchParams.subtreeValueBiasFactor;std::cout<<search.leaf(nodes[n],utility);
  } else if(op=='d') {std::cin>>search.searchParams.subtreeValueBiasFreeProp;search.removeSubtreeValueBias(&nodes[n]);std::cout<<0;}
  else return 1;
  // KataGo leaves the last contribution fields intact after detaching the entry.
  // They are irrelevant once detached; print zero to normalize that bookkeeping.
  std::cout<<' '<<shared->deltaUtilitySum<<' '<<shared->weightSum<<' '
   <<(nodes[n].subtreeValueBiasTableEntry?nodes[n].lastSubtreeValueBiasDeltaSum:0)<<' '
   <<(nodes[n].subtreeValueBiasTableEntry?nodes[n].lastSubtreeValueBiasWeight:0)<<'\n';
 }
}
'''.replace('LEAF', leaf).replace('BACKUP', backup).replace('REMOVAL', removal)
    rows = []
    rng = random.Random(619)
    for factor, exponent, free in itertools.product((0, .1, .45, 1), (0, .5, .85, 1), (0, .8, 1)):
        rows.append('r 0')
        for i in range(150):
            n = rng.randrange(2)
            original = rng.choice((0, 1e-12, 1e-9, .0005, .001, .25, 1, 4, 400, 10000))
            current = original*rng.choice((0, .1, .9, 1))
            utility, child = rng.uniform(-1, 1), rng.uniform(-1, 1)
            line = f'u {n} {utility} {child} {original} {current} {factor} {exponent}'
            rows.extend((line, line, f'l {1-n} {utility} {factor}'))
        rows.extend((f'd 0 {free}', f'l 1 .4 {factor}', f'd 1 {free}', f'd 1 {free}'))
    query = '\n'.join(rows)+'\n'
    with tempfile.TemporaryDirectory(prefix='etazero_subtree_bias_reference_') as directory:
        path = Path(directory)
        (path/'oracle.cpp').write_text(oracle)
        subprocess.run(['c++', '-std=c++17', '-O2', str(path/'oracle.cpp'), '-o', str(path/'oracle')], check=True)
        expected = subprocess.check_output([str(path/'oracle')], input=query, text=True).splitlines()
        actual = subprocess.check_output([str(args.binary), '--oracle'], input=query, text=True).splitlines()
    assert len(actual) == len(expected)
    for index, (a, b) in enumerate(zip(actual, expected)):
        av, bv = list(map(float, a.split())), list(map(float, b.split()))
        assert len(av) == len(bv) == 5
        assert all(math.isclose(x, y, rel_tol=1e-11, abs_tol=1e-11) for x, y in zip(av, bv)), (index, a, b)
    print(json.dumps({'status': 'passed', 'cases': len(actual), 'source_commit': manifest['commit'],
                      'scope': 'verbatim shared-entry replacement, weighted residual, leaf bias, partial/double removal; W-L utility adaptation'}, indent=2))


if __name__ == '__main__':
    main()
