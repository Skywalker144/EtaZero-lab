#ifndef NEURALNET_NNBENCHMARK_H_
#define NEURALNET_NNBENCHMARK_H_

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

// Result of NNEvaluator::benchmarkPureForward
struct NNEvalBenchmarkResult {
  int batchSize;
  int numThreads;
  int numIterations;
  // Per thread, the wall time of each timed getOutput call, in order.
  std::vector<std::vector<double>> perThreadIterationSeconds;
  std::vector<double> perThreadMedianSeconds;
  std::vector<double> perThreadNNEvalsPerSec;
  // Sum over threads of batchSize / medianSeconds.
  double sumMedianNNEvalsPerSec;
  // Wall time of the timed region: all threads started together, ending at the last thread's
  // final iteration (warmups and compute handle teardown excluded), and
  // numThreads * batchSize * numIterations / that wall time. This is the primary
  // throughput metric since it accounts for real concurrency between threads.
  double actualWallSeconds;
  double actualWallNNEvalsPerSec;
};

// Source pure-forward timing and aggregation. Each worker owns its prepared
// context, whose forward() is statically dispatched and whose destructor runs
// after its end timestamp. Feature generation and setup are outside timing.
// Callers validate positive counts before starting workers.
template<typename PrepareThread>
NNEvalBenchmarkResult runNNBenchmark(int numWarmups, int numIterations, int batchSize,
                                    int numThreads, const PrepareThread& prepareThread) {
  NNEvalBenchmarkResult result;
  result.batchSize = batchSize;
  result.numThreads = numThreads;
  result.numIterations = numIterations;
  result.perThreadIterationSeconds.resize(numThreads);
  result.perThreadMedianSeconds.assign(numThreads, 0.0);
  result.perThreadNNEvalsPerSec.assign(numThreads, 0.0);
  result.sumMedianNNEvalsPerSec = 0.0;
  result.actualWallSeconds = 0.0;
  result.actualWallNNEvalsPerSec = 0.0;

  std::atomic<int> readyCount(0);
  std::atomic<bool> startFlag(false);
  std::atomic<bool> anyError(false);
  std::mutex errorMutex;
  std::exception_ptr firstError;

  // Per-thread end-of-timed-loop timestamps, so that wall time excludes compute handle
  // teardown (freeing the model is slow and an early finisher's frees can also perturb
  // other threads' tail iterations, but that perturbation is at least visible in samples).
  std::vector<std::chrono::steady_clock::time_point> threadEndTimes(numThreads);

  std::vector<std::thread> threads;
  threads.reserve(numThreads);
  for(int threadIdx = 0; threadIdx < numThreads; threadIdx++) {
    threads.emplace_back([&,threadIdx]() {
      try {
        auto context = prepareThread(threadIdx);
        for(int i = 0; i < numWarmups; i++)
          context->forward();

        readyCount.fetch_add(1);
        while(!startFlag.load(std::memory_order_acquire))
          std::this_thread::yield();

        std::vector<double>& times = result.perThreadIterationSeconds[threadIdx];
        times.reserve(numIterations);
        for(int i = 0; i < numIterations; i++) {
          std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
          context->forward();
          std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
          times.push_back(std::chrono::duration<double>(t1-t0).count());
        }
        threadEndTimes[threadIdx] = std::chrono::steady_clock::now();
      }
      catch(...) {
        anyError.store(true);
        std::lock_guard<std::mutex> lock(errorMutex);
        if(firstError == nullptr)
          firstError = std::current_exception();
      }
    });
  }

  while(readyCount.load() < numThreads && !anyError.load())
    std::this_thread::yield();
  std::chrono::steady_clock::time_point wallStart = std::chrono::steady_clock::now();
  startFlag.store(true, std::memory_order_release);
  for(std::thread& t: threads)
    t.join();

  if(firstError != nullptr)
    std::rethrow_exception(firstError);

  std::chrono::steady_clock::time_point wallEnd = wallStart;
  for(int threadIdx = 0; threadIdx < numThreads; threadIdx++)
    wallEnd = std::max(wallEnd, threadEndTimes[threadIdx]);

  for(int threadIdx = 0; threadIdx < numThreads; threadIdx++) {
    std::vector<double> sorted = result.perThreadIterationSeconds[threadIdx];
    std::sort(sorted.begin(), sorted.end());
    double median =
      sorted.size() % 2 == 1 ? sorted[sorted.size()/2] :
      0.5 * (sorted[sorted.size()/2-1] + sorted[sorted.size()/2]);
    result.perThreadMedianSeconds[threadIdx] = median;
    result.perThreadNNEvalsPerSec[threadIdx] = median > 0.0 ? (double)batchSize / median : 0.0;
    result.sumMedianNNEvalsPerSec += result.perThreadNNEvalsPerSec[threadIdx];
  }
  result.actualWallSeconds = std::chrono::duration<double>(wallEnd - wallStart).count();
  if(result.actualWallSeconds > 0.0)
    result.actualWallNNEvalsPerSec =
      (double)numThreads * (double)batchSize * (double)numIterations / result.actualWallSeconds;
  return result;
}

#endif
