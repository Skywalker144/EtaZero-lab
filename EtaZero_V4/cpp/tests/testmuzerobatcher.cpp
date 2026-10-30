#include "../neuralnet/muzerobatcher.h"
#include "../core/multithread.h"
#include "../external/nlohmann_json/json.hpp"
#include "tests.h"
#include <fstream>

namespace {
using json = nlohmann::json;

// Start all clients together and propagate failures after joining every thread.
void clients(int count, const std::function<void(int)>& run) {
  std::mutex mutex;
  std::condition_variable ready;
  int waiting = 0;
  bool start = false;
  std::vector<std::exception_ptr> errors(count);
  std::vector<std::thread> threads;
  for(int i = 0; i < count; i++) threads.emplace_back([&,i]() {
    {
      std::unique_lock<std::mutex> lock(mutex);
      waiting++;
      ready.notify_all();
      ready.wait(lock,[&]() { return start; });
    }
    try { run(i); } catch(...) { errors[i] = std::current_exception(); }
  });
  {
    std::unique_lock<std::mutex> lock(mutex);
    ready.wait(lock,[&]() { return waiting == count; });
    start = true;
  }
  ready.notify_all();
  for(auto& thread: threads) thread.join();
  for(const auto& error: errors) if(error) std::rethrow_exception(error);
}

std::string rejected(const std::function<void()>& run) {
  try { run(); }
  catch(const std::exception& e) { return e.what(); }
  throw StringError("Expected MuZero request rejection");
}

json heads(const MuZero::Output& output) {
  return {{"policy",output.policy},{"value",output.value},{"auxiliary",output.auxiliary}};
}
}

void Tests::runMuZeroBatcher(const std::string& directory, const std::string& request, const std::string& response) {
  std::ifstream in(request);
  const json input = json::parse(in);
  ConfigParser config(input.value("cuda_config",std::map<std::string,std::string>{}));
  const bool inputsUseNHWC = input.value("inputs_nhwc",true);
  const auto precision = input.at("fp16").get<bool>() ? enabled_t::True : enabled_t::False;
  const auto model = std::make_shared<MuZero::ModelBundle>(directory);
  const int rootCount = input.at("batch").get<int>();
  const int count = 24;
  const int maxBatch = 8;
  const auto split = [&](const std::vector<float>& values) {
    testAssert(values.size() % rootCount == 0);
    const size_t size = values.size()/rootCount;
    std::vector<std::vector<float>> rows;
    for(int i = 0; i < rootCount; i++) rows.emplace_back(values.begin()+i*size,values.begin()+(i+1)*size);
    return rows;
  };
  const auto spatial = split(input.at("spatial").get<std::vector<float>>());
  const auto global = split(input.at("global").get<std::vector<float>>());
  const auto metadata = split(input.value("metadata",std::vector<float>{}));
  const auto action = [&](int client, int step) { return model->game() == "go" && client%2 == 0 ? model->posLen()*model->posLen() : (client+step)%5; };
  json result;
  rejected([&]() { MuZero::Batcher bad(nullptr,maxBatch,{0},precision,config,nullptr,inputsUseNHWC); });
  rejected([&]() { MuZero::Evaluator bad(std::shared_ptr<const MuZero::ModelBundle>{},maxBatch,0,precision,config,nullptr,inputsUseNHWC); });
  if(model->hasTransformer()) {
#ifdef USE_EIGEN_BACKEND
    const auto error = rejected([&]() { MuZero::Batcher bad(model,maxBatch,{0,0},precision,config,nullptr,false); });
    testAssert(error.find("inputsUseNHWC = false") != std::string::npos);
#elif defined(USE_OPENCL_BACKEND)
    const auto error = rejected([&]() { MuZero::Batcher bad(model,maxBatch,{0,0},precision,config,nullptr,true); });
    testAssert(error.find("inputsUseNHWC = false required") != std::string::npos);
#else
    ConfigParser badConfig(std::map<std::string,std::string>{{"cudaUseNHWC","false"}});
    const auto error = rejected([&]() { MuZero::Batcher bad(model,maxBatch,{0,0},precision,badConfig); });
    testAssert(error.find("require NHWC") != std::string::npos);
#endif
    result["startup_error"] = error;
  }
  {
    // Independent workers on the same device (or host for Eigen). A latent may
    // only return to its own worker, even though both have device index zero.
    MuZero::Batcher batcher(model,maxBatch,{0,0},precision,config,nullptr,inputsUseNHWC);
    testAssert(batcher.stats().rows == 0 && batcher.stats().batches == 0);
    testAssert(batcher.getMaxRowsToSendPerBatch() == maxBatch);
    rejected([&]() { batcher.setMaxRowsToSendPerBatch(0); });
    rejected([&]() { batcher.setMaxRowsToSendPerBatch(maxBatch+1); });
    rejected([&]() { batcher.initial({},global[0],metadata[0]); });
    rejected([&]() { batcher.recurrent(nullptr,0); });
    {
      MuZero::Evaluator foreign(model,1,0,precision,config,nullptr,inputsUseNHWC);
      auto output = foreign.initial(1,spatial[0],global[0],metadata[0]);
      rejected([&]() { batcher.recurrent(output[0].state,0); });
    }
    std::vector<json> rows(count,json::array());
    std::vector<MuZero::State> states(count);
    clients(count,[&](int i) {
      auto output = batcher.initial(spatial[i%rootCount],global[i%rootCount],metadata[i%rootCount]);
      rows[i].push_back(heads(output));
      for(int step = 0; step < 3; step++) {
        std::weak_ptr<const MuZero::StateData> previous = output.state;
        output = batcher.recurrent(std::move(output.state),action(i,step));
        testAssert(previous.expired());
        rows[i].push_back(heads(output));
      }
      states[i] = std::move(output.state);
    });
    batcher.drain();
    batcher.waitForNextNNEvalIfAny();
    const auto burst = batcher.stats();
    testAssert(burst.rows == count*4 && burst.batches <= burst.rows);
    testAssert(burst.initialRows == count);
    testAssert(burst.rowsByWorker[0]+burst.rowsByWorker[1] == burst.rows);
    batcher.setMaxRowsToSendPerBatch(1);
    testAssert(batcher.getMaxRowsToSendPerBatch() == 1);
    clients(count,[&](int i) {
      auto output = batcher.recurrent(states[i],action(i,3));
      rows[i].push_back(heads(output));
    });
    batcher.drain();
    const auto capped = batcher.stats();
    testAssert(capped.rows-burst.rows == count && capped.batches-burst.batches == count);
    testAssert(capped.initialRows == burst.initialRows);
    result["outputs"] = rows;
    result["burst"] = {{"rows",burst.rows},{"batches",burst.batches},{"workers",burst.rowsByWorker}};
    result["capped"] = {{"rows",capped.rows},{"batches",capped.batches}};
    batcher.finish();
    batcher.finish();
    batcher.drain();
    testAssert(rejected([&]() { batcher.initial(spatial[0],global[0],metadata[0]); }).find("closing") != std::string::npos);
    // Retained device states are released after the GPU workers have exited.
    states.clear();
  }
  {
    MuZero::Batcher batcher(model,maxBatch,{0,0},precision,config,nullptr,inputsUseNHWC);
    std::mutex mutex;
    std::condition_variable progress;
    int successful = 0;
    std::exception_ptr failure;
    std::thread running([&]() {
      try {
        clients(count,[&](int i) {
          for(int step = 0; step < 1000; step++) {
            try { batcher.initial(spatial[i%rootCount],global[i%rootCount],metadata[i%rootCount]); }
            catch(const std::exception& e) {
              testAssert(std::string(e.what()).find("closing") != std::string::npos);
              break;
            }
            { std::lock_guard<std::mutex> lock(mutex); successful++; }
            progress.notify_all();
          }
        });
      }
      catch(...) { std::lock_guard<std::mutex> lock(mutex); failure = std::current_exception(); progress.notify_all(); }
    });
    {
      std::unique_lock<std::mutex> lock(mutex);
      progress.wait(lock,[&]() { return successful > 0 || failure; });
    }
    batcher.waitForNextNNEvalIfAny();
    batcher.finish();
    running.join();
    if(failure) std::rethrow_exception(failure);
    testAssert(batcher.stats().rows == (uint64_t)successful);
    result["drained_rows"] = successful;
  }
  {
    MuZero::Batcher batcher(model,maxBatch,{0,0},precision,config,nullptr,inputsUseNHWC);
    auto badMask = spatial[0];
    std::fill_n(badMask.begin(),model->posLen()*model->posLen(),0.0f);
    std::vector<std::string> errors(count);
    clients(count,[&](int i) {
      try { batcher.initial(i == 0 ? badMask : spatial[i%rootCount],global[i%rootCount],metadata[i%rootCount]); }
      catch(const std::exception& e) { errors[i] = e.what(); }
    });
    testAssert(errors[0].find(model->game() == "xiangqi" ? "exactly the 9 x 10 board" : "nonempty") != std::string::npos);
    for(const auto& error: errors) testAssert(error.empty() || error == errors[0]);
    testAssert(rejected([&]() { batcher.initial(spatial[0],global[0],metadata[0]); }) == errors[0]);
    testAssert(rejected([&]() { batcher.waitForNextNNEvalIfAny(); }) == errors[0]);
    testAssert(rejected([&]() { batcher.drain(); }) == errors[0]);
    testAssert(rejected([&]() { batcher.finish(); }) == errors[0]);
    result["runtime_error"] = errors[0];
  }
  {
    MuZero::Batcher batcher(model,maxBatch,{0},precision,config,nullptr,inputsUseNHWC);
    auto output = batcher.initial(spatial[0],global[0],metadata[0]);
    std::weak_ptr<const MuZero::StateData> state = output.state;
    const auto error = rejected([&]() { batcher.recurrent(std::move(output.state),-1); });
    testAssert(state.expired());
    testAssert(rejected([&]() { batcher.finish(); }) == error);
  }
  std::ofstream out(response);
  out << result.dump() << '\n';
}
