"""Use pinned KataGo's unmodified integer Q quantizer against the production core."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from check_katago_graph import body
ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    reference=json.loads((ROOT/'reference_sources.json').read_text())['KataGo']
    source=Path(reference['root'])/'cpp/dataio/trainingwrite.cpp'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==reference['sha256']['cpp/dataio/trainingwrite.cpp']
    assert subprocess.check_output(['git','-C',reference['root'],'rev-parse','HEAD'],text=True).strip()==reference['commit']
    function=body(source.read_text(),'static int16_t clampToRadius32000(')
    code=r'''
#include "etazero/record.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
struct Rand {
  std::mt19937_64 engine;
  Rand(unsigned long seed):engine(seed){}
  bool nextBool(float fraction){return std::bernoulli_distribution(fraction)(engine);}
};
__SOURCE__
int main(){size_t count=0;double worst=0;
 for(unsigned long seed:{0ul,1ul,7ul,123ul,999ul}) {
  Rand oracle(seed);std::mt19937_64 native(seed);
  for(int repeat=0;repeat<4;++repeat)for(int i=-1025;i<=1025;++i) {
   float value=float(i)/1024.0f;
   int actual=etazero::quantize_q_value(value,native);
   int expected=clampToRadius32000(value*32000.0f,oracle);
   if(actual!=expected){std::cerr<<seed<<' '<<value<<' '<<actual<<' '<<expected;return 1;}
   worst=std::max(worst,std::abs(double(actual)-value*32000.0f));++count;
  }
  for(float value:{std::nextafter(-1.f,0.f),std::nextafter(1.f,0.f),-.2500125f,.2500125f})
   for(int repeat=0;repeat<256;++repeat) {
    if(etazero::quantize_q_value(value,native)!=clampToRadius32000(value*32000.0f,oracle))return 2;
    ++count;
   }
 }
 std::cout<<count<<' '<<worst<<'\n';
}
'''.replace('__SOURCE__',function)
    build=ROOT/'build';zlib=re.search(r'^ZLIB_LIBRARY_RELEASE:FILEPATH=(.+)$',(build/'CMakeCache.txt').read_text(),re.M).group(1)
    with tempfile.TemporaryDirectory(prefix='etazero_q_reference_') as temp:
        directory=Path(temp);cpp=directory/'q.cpp';cpp.write_text(code)
        subprocess.run(['c++','-std=c++17','-O2','-I'+str(ROOT/'cpp/include'),'-I'+str(build/'generated'),str(cpp),
                        str(build/'libetazero_core.a'),zlib,'-pthread','-o',str(directory/'q')],check=True)
        cases,worst=subprocess.check_output([str(directory/'q')],text=True).split()
    result={'status':'verified','source_commit':reference['commit'],'source_sha256':reference['sha256']['cpp/dataio/trainingwrite.cpp'],
            'native_library_sha256':hashlib.sha256((build/'libetazero_core.a').read_bytes()).hexdigest(),
            'quantization_cases':int(cases),'max_distance_to_unclamped_scaled_value':float(worst),
            'scope':'Unmodified source float32 clampToRadius32000, fixed identical Bernoulli RNG shim; positive/negative fractions, zero, caps, adjacent float32 endpoints, independent repeats. Production core linked, not copied.',
            'adaptation':'Seed streams differ from Go writer (score quantization omitted); no source RNG-sequence equivalence claim.'}
    assert not args.output.exists();args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
