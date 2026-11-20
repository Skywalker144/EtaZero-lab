#pragma once
#include "muzero.h"
#include "../game/xiangqi.h"

namespace MuZero {
// The public observation is compact NCHW. Each device retains its own state storage.
inline std::vector<std::shared_ptr<const std::vector<uint8_t>>> observationMasks(
  int batch, int canvas, const std::string& game, const ModelDesc& desc,
  const std::vector<float>& spatial, const std::vector<float>& global, const std::vector<float>& metadata
) {
  const int area = canvas*canvas;
  if(spatial.size() != (size_t)batch*desc.numInputChannels*area ||
     global.size() != (size_t)batch*desc.numInputGlobalChannels ||
     metadata.size() != (size_t)batch*desc.numInputMetaChannels)
    throw StringError("MuZero initial observation dimensions differ");
  std::vector<std::shared_ptr<const std::vector<uint8_t>>> masks;
  for(int row = 0; row < batch; row++) {
    auto mask = std::make_shared<std::vector<uint8_t>>(area);
    bool nonempty = false;
    for(int point = 0; point < area; point++) {
      const float v = spatial[(size_t)row*desc.numInputChannels*area+point];
      if(v != 0 && v != 1) throw StringError("MuZero observation mask must be binary");
      if(game == "xiangqi" && (v != 0) != (point%canvas < Xiangqi::WIDTH && point/canvas < Xiangqi::HEIGHT))
        throw StringError("Xiangqi MuZero observation mask must cover exactly the 9 x 10 board");
      (*mask)[point] = v != 0;
      nonempty |= v != 0;
    }
    if(!nonempty) throw StringError("MuZero observation mask must be nonempty");
    masks.push_back(std::move(mask));
  }
  return masks;
}

inline std::array<int,2> actionSquares(int action, int canvas, const std::string& game, const std::vector<uint8_t>& mask) {
  const int area = canvas*canvas;
  const bool xiangqi = game == "xiangqi";
  if(action < 0 || action >= (xiangqi ? Xiangqi::NUM_ACTIONS : area+(game == "go")) ||
     (!xiangqi && action < area && !mask[action]))
    throw StringError("MuZero recurrent action is outside the inline game domain");
  if(!xiangqi) return {action,0};
  const auto& move = Xiangqi::MOVES[action];
  return {move[1]*canvas+move[0],move[3]*canvas+move[2]};
}

inline void unpackPrediction(Output& result, const std::string& game, int canvas, const ModelDesc& desc,
                             const std::vector<uint8_t>& mask, int row, const float* policy, const float* pass,
                             const float* value, const float* aux, int paddedArea, bool nhwc) {
  const int area = canvas*canvas;
  const bool xiangqi = game == "xiangqi";
  const int actions = xiangqi ? Xiangqi::NUM_ACTIONS : area+(game == "go");
  const int channels = desc.numPolicyChannels;
  const int spatialChannels = desc.policyHead.p2Conv.outChannels;
  const float scale = desc.postProcessParams.outputScaleMultiplier;
  result.policy.resize(channels*actions);
  for(int c = 0; c < channels; c++) {
    for(int a = 0; a < (xiangqi ? actions : area); a++) {
      const int idx = xiangqi ? Xiangqi::policyIndex(a,canvas,canvas) : a;
      const int point = idx%area;
      const int channel = xiangqi ? c*Xiangqi::NUM_DELTAS+idx/area : c;
      const size_t offset = (size_t)row*spatialChannels*paddedArea +
        (nhwc ? point*spatialChannels+channel : channel*paddedArea+point);
      result.policy[c*actions+a] = xiangqi || mask[point] ? policy[offset]*scale : -5000.0f;
    }
    if(game == "go") result.policy[c*actions+area] = pass[row*channels+c]*scale;
  }
  for(int c = 0; c < 4; c++) result.value[c] = value[row*4+c]*scale;
  for(int c = 0; c < 2; c++) result.auxiliary[c] = aux[row*2+c]*scale;
}
}
