#ifndef NEURALNET_NNPOSTPROCESS_H_
#define NEURALNET_NNPOSTPROCESS_H_

#include "nninputs.h"
#include "desc.h"
#include "../core/logger.h"

// Source evaluator math, shared by real-board and latent-state inference.
// Inputs are logits; policy has already been unrotated and optimism mixed.
namespace NNPostprocess {
void policy(NNOutput& output, const bool* isLegal, int policySize,
            float outputScale, float temperature, double optimism,
            bool& errorLogLockout, Logger* logger, const std::string& modelName);
void value(NNOutput& output, Player nextPlayer, bool suppressNoResult,
           int modelVersion, const ModelPostProcessParams& params);
}
#endif
