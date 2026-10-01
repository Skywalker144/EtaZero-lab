#include "etazero/record.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <system_error>
#include <unistd.h>
#include <zlib.h>
namespace etazero {
// KataGo play.cpp value surprise and per-game redistribution; trainingwrite.cpp row multiplicity.
void apply_training_weights(FinishedGame& game,double policy_factor,double value_factor,std::mt19937_64& rng) {
    if(!std::isfinite(policy_factor)||!std::isfinite(value_factor)||policy_factor<0||value_factor<0||policy_factor+value_factor>1)
        throw std::runtime_error("Invalid surprise weighting");
    WDL future{double(game.winner==1),double(game.winner==0),double(game.winner==-1)};
    double now=1/(1+game.size*game.size*0.016);
    for(size_t i=game.steps.size();i-->0;) {
        auto& step=game.steps[i];if(!step.trainable){step.target_weight=0;step.row_repeats=0;continue;}
        double kl=0;
        for(int j=0;j<3;++j) {
            int index=step.player==1?j:2-j;
            future[j]+=now*(step.search_wdl[index]-future[j]);
            if(future[j]>1e-100)kl+=future[j]*std::log(future[j]/std::max(1e-100,step.network_wdl[index]));
        }
        step.value_surprise=std::clamp(kl,0.0,1.0);
    }
    double sum_weights=0,sum_policy=0,sum_value=0;
    for(const auto& s:game.steps)if(s.trainable) {
        if(!std::isfinite(s.target_weight)||s.target_weight<0||s.target_weight>1)
            throw std::runtime_error("Invalid initial target weight");
        sum_weights+=s.target_weight;sum_policy+=s.target_weight*s.policy_surprise;sum_value+=s.target_weight*s.value_surprise;
    }
    if(sum_weights>=1 && (policy_factor>0 || value_factor>0)) {
        double avg_policy=sum_policy/sum_weights,avg_value=sum_value/sum_weights;
        value_factor*=std::min(1.0,avg_value/0.010);
        std::vector<double> policy_props(game.steps.size()),value_props(game.steps.size());sum_policy=sum_value=0;
        for(size_t i=0;i<game.steps.size();++i) {
            const auto& s=game.steps[i];if(!s.trainable)continue;
            policy_props[i]=s.target_weight*s.policy_surprise+(1-s.target_weight)*std::max(0.0,s.policy_surprise-1.5*avg_policy);
            value_props[i]=s.target_weight*s.value_surprise;sum_policy+=policy_props[i];sum_value+=value_props[i];
        }
        for(size_t i=0;i<game.steps.size();++i) {
            auto& s=game.steps[i];if(!s.trainable)continue;
            s.target_weight=(1-policy_factor-value_factor)*s.target_weight+
                policy_factor*policy_props[i]*sum_weights/std::max(sum_policy,1e-10)+
                value_factor*value_props[i]*sum_weights/std::max(sum_value,1e-10);
        }
    }
    for(auto& s:game.steps)if(s.trainable) {
        if(!std::isfinite(s.target_weight)||s.target_weight<0||s.target_weight>std::numeric_limits<int32_t>::max()-1.0)
            throw std::runtime_error("Invalid final target weight");
        int whole=static_cast<int>(s.target_weight);
        s.row_repeats=whole+(std::bernoulli_distribution(s.target_weight-whole)(rng)?1:0);
    }
}

std::string quote(const std::string& text) {
    std::string out = "\"";
    const char* digits = "0123456789abcdef";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (c < 32) { out += "\\u00"; out += digits[c >> 4]; out += digits[c & 15]; }
        else out += c;
    }
    return out + '"';
}
namespace {
void sync_path(const std::filesystem::path& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::system_error(errno, std::generic_category(), "Open for fsync");
    int result = ::fsync(fd), error = errno; ::close(fd);
    if (result) throw std::system_error(error, std::generic_category(), "fsync");
}
void u16(std::ostream& out, uint16_t n) { out.put(n & 255); out.put(n >> 8); }
void u32(std::ostream& out, uint32_t n) { u16(out, n & 65535); u16(out, n >> 16); }
struct Zip {
    struct Entry { std::string name; uint32_t crc, compressed, size, offset; };
    std::ofstream out;
    std::vector<Entry> entries;
    std::array<unsigned char,65536> compression_buffer;
    explicit Zip(const std::filesystem::path& path) : out(path, std::ios::binary) {
        out.exceptions(std::ios::failbit | std::ios::badbit);
    }
    template<class T>
    void array(const std::string& name, const std::vector<T>& data, const char* dtype, std::initializer_list<size_t> shape) {
        size_t count = 1; std::ostringstream dimensions;
        for (size_t n : shape) { count *= n; dimensions << n << ", "; }
        if (count != data.size()) throw std::runtime_error("NPY array shape mismatch: " + name);
        std::string header = "{'descr': '" + std::string(dtype) + "', 'fortran_order': False, 'shape': (" + dimensions.str() + "), }";
        header.append((64 - (10 + header.size() + 1) % 64) % 64, ' '); header += '\n';
        std::ostringstream raw(std::ios::binary | std::ios::out);
        raw.write("\x93NUMPY\x01\x00", 8); u16(raw, header.size()); raw << header;
        auto prefix = raw.str();
        const size_t byte_count=data.size()*sizeof(T);
        if (prefix.size()+byte_count>UINT32_MAX || out.tellp()>UINT32_MAX)
            throw std::runtime_error("NPZ shard exceeds classic ZIP limits");
        Entry e{name+".npy",0,0,static_cast<uint32_t>(prefix.size()+byte_count),static_cast<uint32_t>(out.tellp())};
        // ZIP data descriptors permit streaming compressed arrays without assembling
        // another full NPY byte string or compressBound-sized temporary buffer.
        u32(out,0x04034b50);u16(out,20);u16(out,8);u16(out,8);u16(out,0);u16(out,0);
        u32(out,0);u32(out,0);u32(out,0);u16(out,e.name.size());u16(out,0);out<<e.name;
        z_stream zs{};
        if(deflateInit2(&zs,Z_BEST_SPEED,Z_DEFLATED,-15,8,Z_DEFAULT_STRATEGY)!=Z_OK)
            throw std::runtime_error("Cannot initialize NPZ compressor");
        try {
            auto feed=[&](const void* data,size_t bytes,int flush) {
                auto input=static_cast<const Bytef*>(data);
                e.crc=crc32(e.crc,input,static_cast<uInt>(bytes));
                zs.next_in=const_cast<Bytef*>(input);zs.avail_in=bytes;
                int status;
                do {
                    zs.next_out=compression_buffer.data();zs.avail_out=compression_buffer.size();
                    status=deflate(&zs,flush);
                    if(status!=Z_OK && status!=Z_STREAM_END) throw std::runtime_error("NPZ compression failed");
                    out.write(reinterpret_cast<const char*>(compression_buffer.data()),compression_buffer.size()-zs.avail_out);
                } while(zs.avail_in || (flush==Z_FINISH && status!=Z_STREAM_END));
            };
            feed(prefix.data(),prefix.size(),Z_NO_FLUSH);
            feed(data.data(),byte_count,Z_FINISH);e.compressed=zs.total_out;
        } catch (...) {deflateEnd(&zs);throw;}
        deflateEnd(&zs);
        u32(out,0x08074b50);u32(out,e.crc);u32(out,e.compressed);u32(out,e.size);entries.push_back(e);
    }
    void finish() {
        uint32_t start = out.tellp();
        for (const auto& e : entries) {
            u32(out, 0x02014b50); u16(out, 20); u16(out, 20); u16(out, 8); u16(out, 8); u16(out, 0); u16(out, 0);
            u32(out, e.crc); u32(out, e.compressed); u32(out, e.size); u16(out, e.name.size());
            u16(out, 0); u16(out, 0); u16(out, 0); u16(out, 0); u32(out, 0); u32(out, e.offset); out << e.name;
        }
        uint32_t end = out.tellp();
        u32(out, 0x06054b50); u16(out, 0); u16(out, 0); u16(out, entries.size()); u16(out, entries.size());
        u32(out, end-start); u32(out, start); u16(out, 0); out.flush(); out.close();
    }
};
}
RecordWriter::RecordWriter(std::filesystem::path dir, Source source, size_t rows, size_t capacity, double seconds)
    : directory_(std::move(dir)), source_(std::move(source)), max_rows_(rows), capacity_(capacity), flush_seconds_(seconds) {
    if (!rows || !capacity || seconds <= 0) throw std::runtime_error("Invalid writer limits");
    std::filesystem::create_directories(directory_);
    if (!std::filesystem::is_empty(directory_)) throw std::runtime_error("Worker output directory must be new and empty");
    worker_ = std::thread(&RecordWriter::loop, this);
}
RecordWriter::~RecordWriter() { try { finish(); } catch (...) {} }
void RecordWriter::finish() {
    { std::lock_guard<std::mutex> lock(mutex_); closing_ = true; }
    changed_.notify_all();
    if (worker_.joinable()) worker_.join();
    if (failure_) std::rethrow_exception(failure_);
}
void RecordWriter::enqueue(FinishedGame game) {
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait(lock, [&] { return queue_.size() < capacity_ || closing_ || failure_; });
    if (failure_) std::rethrow_exception(failure_);
    if (closing_) throw std::runtime_error("Writer is closing");
    queue_.push_back(std::move(game)); lock.unlock(); changed_.notify_all();
}
struct RecordWriter::Buffers {
    int canvas, actions_count, packed_area;
    size_t row_capacity;
    std::vector<uint8_t> observations;
    std::vector<int8_t> players,rules,winners,reasons;
    std::vector<int16_t> sizes,opening_moves,balanced_moves,policy_moves;
    std::vector<int32_t> opening_attempts;
    std::vector<int8_t> opening_status;
    std::vector<uint8_t> train_mask,cheap_search;
    std::vector<int32_t> row_repeats;
    std::vector<float> target_weights,policy_surprises,value_surprises,network_wdl,search_wdl;
    std::vector<float> start_values, globals;
    std::vector<std::string> opening_failures;
    std::vector<uint64_t> ids,seeds;
    std::vector<int64_t> game_offsets,obs_offsets,visits;
    std::vector<int32_t> actions,simulations;
    std::vector<float> policies,temperatures,rewards;
    Buffers(int c,size_t rows) : canvas(c),actions_count(c*c),packed_area((c*c+7)/8),row_capacity(rows+c*c) {
        // At most one final whole game beyond the row threshold; T+1 observations
        // are retained for every trajectory. Allocations are reused after flushes.
        observations.reserve(2*row_capacity*INPUT_PLANES*packed_area);players.reserve(2*row_capacity);
        globals.reserve(2*row_capacity*GLOBAL_FEATURES);
        visits.reserve(row_capacity*actions_count);policies.reserve(row_capacity*actions_count);
        actions.reserve(row_capacity);simulations.reserve(row_capacity);temperatures.reserve(row_capacity);rewards.reserve(row_capacity);
        ids.reserve(row_capacity);seeds.reserve(row_capacity);sizes.reserve(row_capacity);
        rules.reserve(row_capacity);winners.reserve(row_capacity);reasons.reserve(row_capacity);
        game_offsets.reserve(row_capacity+1);obs_offsets.reserve(row_capacity+1);reset();
    }
    void reset() {
        observations.clear();globals.clear();players.clear();rules.clear();winners.clear();reasons.clear();sizes.clear();ids.clear();seeds.clear();
        visits.clear();actions.clear();simulations.clear();policies.clear();temperatures.clear();rewards.clear();
        opening_moves.clear();balanced_moves.clear();policy_moves.clear();opening_attempts.clear();
        row_repeats.clear();target_weights.clear();policy_surprises.clear();value_surprises.clear();
        network_wdl.clear();search_wdl.clear();cheap_search.clear();
        opening_status.clear();train_mask.clear();start_values.clear();opening_failures.clear();
        game_offsets.clear();obs_offsets.clear();game_offsets.push_back(0);obs_offsets.push_back(0);
    }
    void observation(const std::vector<float>& obs,int player) {
        if(obs.size()!=static_cast<size_t>(INPUT_PLANES*actions_count + GLOBAL_FEATURES))throw std::runtime_error("Writer observation shape mismatch");
        size_t base=observations.size();observations.resize(base+INPUT_PLANES*packed_area,0);
        for(int c=0;c<INPUT_PLANES;++c) for(int i=0;i<actions_count;++i) {
            float value=obs[c*actions_count+i];
            if(value!=0 && value!=1)throw std::runtime_error("Nonbinary writer observation");
            if(value)observations[base+c*packed_area+i/8]|=1<<(7-i%8);
        }
        globals.insert(globals.end(), obs.end()-GLOBAL_FEATURES, obs.end());
        players.push_back(player);
    }
};
void RecordWriter::append(const FinishedGame& g) {
    if(g.canvas<5 || g.canvas>25 || g.steps.empty() || g.steps.size()>static_cast<size_t>(g.canvas*g.canvas))
        throw std::runtime_error("Invalid finished game");
    if(!buffers_)buffers_=std::make_unique<Buffers>(g.canvas,max_rows_);
    auto& b=*buffers_;
    if(g.canvas!=b.canvas || b.actions.size()+g.steps.size()>b.row_capacity)throw std::runtime_error("Writer capacity/canvas mismatch");
    b.ids.push_back(g.id);b.seeds.push_back(g.seed);b.sizes.push_back(g.size);
    b.rules.push_back(static_cast<int8_t>(g.rule));b.winners.push_back(g.winner);b.reasons.push_back(g.reason);
    b.opening_moves.push_back(g.opening.actions.size());b.balanced_moves.push_back(g.opening.balanced_moves);
    b.policy_moves.push_back(g.opening.policy_moves);b.opening_attempts.push_back(g.opening.attempts);
    b.opening_status.push_back(static_cast<int8_t>(g.opening.status));b.start_values.push_back(g.opening.start_value);
    b.opening_failures.push_back(g.opening.failure);
    for(auto& s:g.steps) {
        b.train_mask.push_back(s.trainable);b.cheap_search.push_back(s.cheap_search);
        b.row_repeats.push_back(s.trainable?s.row_repeats:0);b.target_weights.push_back(s.trainable?s.target_weight:0);
        b.policy_surprises.push_back(s.policy_surprise);b.value_surprises.push_back(s.value_surprise);
        b.network_wdl.insert(b.network_wdl.end(),s.network_wdl.begin(),s.network_wdl.end());
        b.search_wdl.insert(b.search_wdl.end(),s.search_wdl.begin(),s.search_wdl.end());
        if(s.policy.size()!=static_cast<size_t>(b.actions_count) || s.visits.size()!=s.policy.size())
            throw std::runtime_error("Writer policy/visit shape mismatch");
        b.observation(s.observation,s.player);b.actions.push_back(s.action);b.simulations.push_back(s.simulations);
        b.temperatures.push_back(s.temperature);b.rewards.push_back(s.reward);
        b.policies.insert(b.policies.end(),s.policy.begin(),s.policy.end());b.visits.insert(b.visits.end(),s.visits.begin(),s.visits.end());
    }
    b.observation(g.final_observation,g.final_player);b.game_offsets.push_back(b.actions.size());b.obs_offsets.push_back(b.players.size());
}
void RecordWriter::loop() {
    try {
        auto interval=std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(flush_seconds_));
        auto deadline=std::chrono::steady_clock::now()+interval;
        for(;;) {
            FinishedGame game;bool have=false,done;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                changed_.wait_until(lock,deadline,[&]{return closing_ || !queue_.empty();});
                if(!queue_.empty()) {game=std::move(queue_.front());queue_.pop_front();have=true;}
                done=closing_ && queue_.empty();
            }
            changed_.notify_all();if(have)append(game);
            if(done || (buffers_ && buffers_->actions.size()>=max_rows_) || std::chrono::steady_clock::now()>=deadline) {
                if(buffers_ && !buffers_->ids.empty()) {publish();buffers_->reset();}
                deadline=std::chrono::steady_clock::now()+interval;
            }
            if(done)return;
        }
    } catch (...) {
        {std::lock_guard<std::mutex> lock(mutex_);failure_=std::current_exception();}changed_.notify_all();
    }
}
void RecordWriter::publish() {
    auto& b=*buffers_;
    const int canvas=b.canvas,actions_count=b.actions_count;
    const size_t n=b.ids.size(),t=b.actions.size(),o=b.players.size();
    std::string shard = "shard_" + std::to_string(next_shard_++) + ".npz";
    auto created = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    size_t rows=0;for(auto repeats:b.row_repeats)rows+=repeats;
    std::ostringstream meta;
    meta << "{\"contract\":" << quote(CONTRACT_ID) << ",\"canvas\":" << canvas << ",\"rows\":" << rows << ",\"plies\":" << t << ",\"games\":" << n
         << ",\"run_id\":" << quote(source_.run) << ",\"attempt_id\":" << quote(source_.attempt)
         << ",\"iteration_id\":" << source_.iteration << ",\"worker_id\":" << source_.worker << ",\"model_id\":" << quote(source_.model)
         << ",\"config_id\":" << quote(source_.config) << ",\"source_id\":" << quote(source_.source)
         << ",\"shard_id\":" << quote(source_.attempt+":"+std::to_string(source_.worker)+":"+shard)
         << ",\"created_ns\":" << created << ",\"opening_failures\":[";
    for(size_t i=0;i<n;++i){if(i)meta<<',';meta<<quote(b.opening_failures[i]);}meta<<"]}";
    std::string metadata = meta.str(); std::vector<uint8_t> meta_bytes(metadata.begin(), metadata.end());
    auto destination = directory_ / shard, temporary = destination; temporary += ".tmp";
    if (std::filesystem::exists(destination)) throw std::runtime_error("Refusing to overwrite raw data");
    Zip zip(temporary);
    zip.array("observations", b.observations, "|u1", {o, INPUT_PLANES, static_cast<size_t>(b.packed_area)});
    zip.array("globals", b.globals, "<f4", {o, GLOBAL_FEATURES});
    zip.array("players", b.players, "|i1", {o}); zip.array("actions", b.actions, "<i4", {t});
    zip.array("policies", b.policies, "<f4", {t, static_cast<size_t>(actions_count)});
    zip.array("visits", b.visits, "<i8", {t, static_cast<size_t>(actions_count)}); zip.array("simulations", b.simulations, "<i4", {t});
    zip.array("temperatures", b.temperatures, "<f4", {t}); zip.array("rewards", b.rewards, "<f4", {t});
    zip.array("game_offsets", b.game_offsets, "<i8", {n+1}); zip.array("observation_offsets", b.obs_offsets, "<i8", {n+1});
    zip.array("game_ids", b.ids, "<u8", {n}); zip.array("seeds", b.seeds, "<u8", {n}); zip.array("sizes", b.sizes, "<i2", {n});
    zip.array("rules", b.rules, "|i1", {n}); zip.array("winners", b.winners, "|i1", {n}); zip.array("reasons", b.reasons, "|i1", {n});
    zip.array("train_mask", b.train_mask, "|u1", {t});
    zip.array("cheap_search",b.cheap_search,"|u1",{t});zip.array("row_repeats",b.row_repeats,"<i4",{t});
    zip.array("target_weights",b.target_weights,"<f4",{t});
    zip.array("policy_surprises",b.policy_surprises,"<f4",{t});zip.array("value_surprises",b.value_surprises,"<f4",{t});
    zip.array("network_wdl",b.network_wdl,"<f4",{t,3});zip.array("search_wdl",b.search_wdl,"<f4",{t,3});
    zip.array("opening_moves", b.opening_moves, "<i2", {n});zip.array("balanced_moves", b.balanced_moves, "<i2", {n});
    zip.array("policy_moves", b.policy_moves, "<i2", {n});zip.array("opening_attempts", b.opening_attempts, "<i4", {n});
    zip.array("opening_status", b.opening_status, "|i1", {n});zip.array("start_values", b.start_values, "<f4", {n});
    zip.array("metadata", meta_bytes, "|u1", {meta_bytes.size()}); zip.finish(); sync_path(temporary);
    std::filesystem::rename(temporary, destination); sync_path(directory_);
    std::cout << "{\"event\":\"shard\",\"file\":" << quote(destination.string()) << ",\"rows\":" << rows << ",\"plies\":" << t << ",\"games\":" << n << "}" << std::endl;
}
}
