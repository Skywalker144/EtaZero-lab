#ifndef SEARCH_POLICYBUFFER_H_
#define SEARCH_POLICYBUFFER_H_

#include "../neuralnet/nninputs.h"
#include <algorithm>
#include <memory>

// Preserve source stack storage for point policies. Joint-action policies get
// additional storage only in the calls that actually need the larger domain.
template<typename T, int InlineSize = NNPos::MAX_NN_POLICY_SIZE>
class SearchPolicyBuffer {
  T local[InlineSize];
  std::unique_ptr<T[]> extended;
 public:
  explicit SearchPolicyBuffer(int size, bool zero = false)
    :extended(size > InlineSize ? new T[size] : nullptr) {
    if(zero) std::fill(data(),data()+size,T{});
  }
  T* data() { return extended ? extended.get() : local; }
};

#endif
