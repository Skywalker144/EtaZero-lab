#include "muzerowrite.h"
#include "trainingwrite.h"
#include "../core/test.h"

namespace {
template<typename T>
void appendRow(std::vector<T>& destination, const NumpyBuffer<T>& source) {
  destination.insert(destination.end(),source.data,source.data + source.getActualDataLen(1));
}

template<typename T>
void writeArray(ZipFile& zip, std::vector<std::shared_ptr<void>>& owners,
                const char* name, const std::vector<T>& data, const std::vector<int64_t>& shape) {
  auto owner = std::make_shared<NumpyBuffer<T>>(shape);
  owners.push_back(owner);
  NumpyBuffer<T>& buffer = *owner;
  testAssert(buffer.dataLen == (int64_t)data.size());
  std::copy(data.begin(),data.end(),buffer.data);
  uint64_t bytes = buffer.prepareHeaderWithNumRows(shape[0]);
  zip.writeBuffer(name,buffer.dataIncludingHeader,bytes);
}

template<typename T>
void writeTargets(ZipFile& zip, std::vector<std::shared_ptr<void>>& owners, const char* name,
                  const std::vector<std::shared_ptr<const MuZeroTrajectory>>& games,
                  std::vector<T> MuZeroTrajectory::* field, const std::vector<int64_t>& shape) {
  auto owner = std::make_shared<NumpyBuffer<T>>(shape);
  owners.push_back(owner);
  NumpyBuffer<T>& buffer = *owner;
  T* destination = buffer.data;
  for(const auto& game : games) {
    const auto& data = game.get()->*field;
    destination = std::copy(data.begin(),data.end(),destination);
  }
  testAssert(destination == buffer.data + buffer.dataLen);
  uint64_t bytes = buffer.prepareHeaderWithNumRows(shape[0]);
  zip.writeBuffer(name,buffer.dataIncludingHeader,bytes);
}
}

void MuZeroTrajectory::append(const TrainingWriteBuffers& row, int action, float weight) {
  testAssert(row.curRows == 1);
  actions.push_back(action);
  weights.push_back(weight);
  appendRow(policy,row.policyTargetsNCMove);
  appendRow(global,row.globalTargetsNC);
  appendRow(future,row.valueTargetsNCHW);
  appendRow(q,row.qValueTargetsNCMove);
}

void MuZeroWriteBuffers::addRoot(const std::shared_ptr<const MuZeroTrajectory>& trajectory, int turn) {
  // writeGame emits each game's main and side roots consecutively.
  if(games.empty() || games.back() != trajectory)
    games.push_back(trajectory);
  roots.push_back((int64_t)games.size()-1);
  roots.push_back(turn);
}

void MuZeroWriteBuffers::clear() {
  games.clear();
  roots.clear();
  numpyBuffers.clear();
}

void MuZeroWriteBuffers::writeToZipFile(ZipFile& zip, int numRoots, int xLen, int yLen, int policySize) {
  testAssert(roots.size() == (size_t)numRoots*2);
  std::vector<int64_t> offsets{0};
  std::vector<uint64_t> hashes;
  std::vector<uint8_t> finished;
  for(const auto& game : games) {
    offsets.push_back(offsets.back() + (int64_t)game->actions.size());
    hashes.push_back(game->gameHash.hash0);
    hashes.push_back(game->gameHash.hash1);
    finished.push_back(game->finished ? 1 : 0);
  }
  int64_t g = (int64_t)games.size();
  int64_t m = offsets.back();
  numpyBuffers.clear();
  writeArray(zip,numpyBuffers,"muZeroRootNC",roots,{numRoots,2});
  writeArray(zip,numpyBuffers,"muZeroGameOffsets",offsets,{g+1});
  writeArray(zip,numpyBuffers,"muZeroGameHash",hashes,{g,2});
  writeArray(zip,numpyBuffers,"muZeroGameFinished",finished,{g});
  writeTargets(zip,numpyBuffers,"muZeroActions",games,&MuZeroTrajectory::actions,{m});
  writeTargets(zip,numpyBuffers,"muZeroWeights",games,&MuZeroTrajectory::weights,{m});
  writeTargets(zip,numpyBuffers,"muZeroPolicyTargetsNCMove",games,&MuZeroTrajectory::policy,{m,2,policySize});
  writeTargets(zip,numpyBuffers,"muZeroGlobalTargetsNC",games,&MuZeroTrajectory::global,{m,80});
  writeTargets(zip,numpyBuffers,"muZeroValueTargetsNCHW",games,&MuZeroTrajectory::future,{m,2,yLen,xLen});
  writeTargets(zip,numpyBuffers,"muZeroQValueTargetsNCMove",games,&MuZeroTrajectory::q,{m,2,policySize});
}
