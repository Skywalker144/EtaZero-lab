#include "dataio/record.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <sstream>
#include <system_error>
#include <unistd.h>
#include <zlib.h>
namespace etazero {
SearchMetadata search_metadata(const SearchResult& result,const Game& game) {
    auto entropy=[](const std::vector<double>& probabilities) {
        double sum=0;for(double p:probabilities)if(p>1e-100)sum-=p*std::log(p);return std::max(0.0,sum);
    };
    return {entropy(result.search_policy),entropy(result.statistics_policy.empty()?result.policy:result.statistics_policy),entropy(result.network_policy),
        result.network_wdl[0]-result.network_wdl[2],result.policy_surprise,result.root_visits,game.turn(),game.training_metadata()};
}
size_t first_file_row_limit(size_t maximum,double minimum,double uniform) {
    if(!maximum || !std::isfinite(minimum) || minimum<0 || minimum>1 ||
       !std::isfinite(uniform) || uniform<0 || uniform>=1)throw std::runtime_error("Invalid first-file row limit");
    return maximum-static_cast<size_t>(maximum*(1-minimum)*uniform);
}
int16_t quantize_q_value(float winloss,std::mt19937_64& rng) {
    if(!std::isfinite(winloss))throw std::runtime_error("Nonfinite Q value");
    float x=winloss*32000.0f;
    if(x<=-32000)return -32000;
    if(x>=32000)return 32000;
    int low=static_cast<int>(std::floor(x));float fraction=x-low;
    return static_cast<int16_t>(low+(fraction>0 && std::bernoulli_distribution(fraction)(rng)));
}
namespace {
void append_q_targets(const std::vector<float>& values,const std::vector<int64_t>& visits,int repeats,int actions,
                      std::vector<int16_t>& packed_values,std::vector<int16_t>& packed_visits,std::mt19937_64& rng) {
    if(values.size()!=static_cast<size_t>(actions)||visits.size()!=values.size())
        throw std::runtime_error("Missing Q value/node-visit targets");
    for(int a=0;a<actions;++a) {
        if(!std::isfinite(values[a])||std::abs(values[a])>1.000001f||visits[a]<0 || (visits[a]==0 && values[a]!=0))
            throw std::runtime_error("Invalid Q value/node-visit target");
        packed_visits.push_back(static_cast<int16_t>(std::min<int64_t>(visits[a],32000)));
    }
    // Source quantizes Q independently on EACH final writer row. Visits are
    // unchanged by rounding, so retain them once per compact sampled position.
    for(int repeat=0;repeat<repeats;++repeat)for(int a=0;a<actions;++a)
        packed_values.push_back(visits[a]>0?quantize_q_value(values[a],rng):0);
}
}
// KataGo play.cpp value surprise and per-game redistribution; trainingwrite.cpp row multiplicity.
void compute_value_surprises(FinishedGame& game,bool direct) {
    if(game.opening.initial_position_kind==1 && game.opening.hint_action>=0) {
        size_t first=game.opening.actions.size();
        if(first<game.steps.size()) {
            auto& step=game.steps[first];
            if(first+1<game.steps.size()) {
                const auto& next=game.steps[first+1];
                for(int j=0;j<4;++j)step.search_wdl[j]=next.search_wdl[step.player==next.player || j==3?j:2-j];
            } else step.search_wdl=game.outcome().wdl(step.player);
        }
    }
    WDL future=game.outcome().wdl(1);
    int area=game_spec(game.rule).area(game.size);
    if(game.rule==Rule::GO)
        area=(game.options.go_width?game.options.go_width:game.size)*(game.options.go_height?game.options.go_height:game.size);
    double now=1/(1+area*0.016);
    for(size_t i=game.steps.size();i-->0;) {
        auto& step=game.steps[i];if(!step.trainable){step.target_weight=0;step.row_repeats=0;continue;}
        double kl=0;
        for(int j=0;j<4;++j) {
            int index=step.player==1 || j==3?j:2-j;
            if(direct)future[j]=step.search_wdl[index];
            else future[j]+=now*(step.search_wdl[index]-future[j]);
            if(future[j]>1e-100)kl+=future[j]*std::log(future[j]/std::max(1e-100,step.network_wdl[index]));
        }
        step.value_surprise=std::clamp(kl,0.0,1.0);
    }
}
void apply_training_weights(FinishedGame& game,double policy_factor,double value_factor,std::mt19937_64& rng,bool direct,bool use_reanalyze,double scale_data_weight) {
    if(!std::isfinite(scale_data_weight)||scale_data_weight<.01||scale_data_weight>10)throw std::runtime_error("Invalid scale_data_weight");
    if(!std::isfinite(policy_factor)||!std::isfinite(value_factor)||policy_factor<0||value_factor<0||policy_factor+value_factor>1)
        throw std::runtime_error("Invalid surprise weighting");
    compute_value_surprises(game,direct);
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
            policy_props[i]=s.target_weight*s.policy_surprise+(1-s.target_weight)*(use_reanalyze && s.cheap_search && !s.reanalyzed?0:std::max(0.0,s.policy_surprise-1.5*avg_policy));
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
        s.target_weight=static_cast<float>(s.target_weight*scale_data_weight);
        if(!std::isfinite(s.target_weight)||s.target_weight<0||s.target_weight>std::numeric_limits<int32_t>::max()-1.0)
            throw std::runtime_error("Invalid final target weight");
        int whole=static_cast<int>(s.target_weight);
        s.row_repeats=whole+(std::bernoulli_distribution(s.target_weight-whole)(rng)?1:0);
    }
    for(auto& side:game.side_positions) {
        side.target_weight=static_cast<float>(side.target_weight*scale_data_weight);
        if(!std::isfinite(side.target_weight)||side.target_weight<0||side.target_weight>std::numeric_limits<int32_t>::max()-1.0)
            throw std::runtime_error("Invalid final side target weight");
        int whole=static_cast<int>(side.target_weight);
        side.row_repeats=whole+(std::bernoulli_distribution(side.target_weight-whole)(rng)?1:0);
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
                // zlib treats a null pointer as CRC initialization, even at zero length.
                if(bytes)e.crc=crc32(e.crc,input,static_cast<uInt>(bytes));
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
RecordWriter::RecordWriter(std::filesystem::path dir, Source source, size_t rows, size_t capacity, double first_min, uint64_t seed)
    : directory_(std::move(dir)), source_(std::move(source)), max_rows_(rows), capacity_(capacity) {
    if (!rows || !capacity || !std::isfinite(first_min) || first_min<0 || first_min>1) throw std::runtime_error("Invalid writer limits");
    // Source: maxRows - int(maxRows * (1-firstFileMinRandProp) * rand.nextDouble()).
    // This stream is independent of gameplay, row multiplicities, dropout and Q.
    std::mt19937_64 rng(seed);
    first_rows_=first_file_row_limit(rows,first_min,std::uniform_real_distribution<double>(0,1)(rng));
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
    int canvas, actions_count, packed_area, planes, global_count, spatial_area;
    Rule layout_rule;
    std::vector<RuleOptions> options;
    size_t row_capacity;
    size_t row_begin=0,rows=0;
    std::vector<uint8_t> observations,side_observations,side_forbidden_input;
    std::vector<float> side_globals,side_wdl,side_target_weights;
    std::vector<int8_t> side_players;
    std::vector<int32_t> side_game_indices,side_row_repeats;
    std::vector<int16_t> side_policies;
    std::vector<int16_t> q_values,q_visits,side_q_values,side_q_visits;
    std::vector<int64_t> side_visits;
    std::vector<int8_t> players,rules,winners,reasons;
    std::vector<int16_t> sizes,opening_moves,balanced_moves,policy_moves,initial_position_moves;
    std::vector<int8_t> initial_position_kind;std::vector<int32_t> hint_actions;
    std::vector<int32_t> opening_attempts;
    std::vector<int8_t> opening_status;
    std::vector<uint8_t> train_mask,cheap_search,forbidden_input,reanalyzed,reanalysis_used_outcome;
    std::vector<int64_t> reanalysis_original_visits;
    std::vector<float> reanalysis_policy_surprise,reanalysis_value_surprise;
    std::vector<int32_t> row_repeats;
    std::vector<float> target_weights,policy_surprises,value_surprises,network_wdl,search_wdl;
    std::vector<float> start_values, globals, training_weights;
    std::vector<std::string> opening_failures;
    std::vector<uint64_t> ids,seeds;
    std::vector<double> root_policy_invalid_mass_sum;
    std::vector<uint64_t> root_policy_invalid_mass_count;
    std::vector<int64_t> game_offsets,obs_offsets,visits,sample_indices;
    std::vector<int32_t> actions;std::vector<int64_t> simulations;
    std::vector<int16_t> policies,opponent_policies;
    std::vector<int16_t> trajectory_policy,trajectory_q_values,trajectory_q_visits;
    std::vector<float> opponent_policy_weights,temperatures,rewards;
    std::vector<uint8_t> binaryInputNCHWPacked;
    std::vector<float> globalInputNC,globalTargetsNC,drawTargetsNC;
    std::vector<int16_t> policyTargetsNCMove,qValueTargetsNCMove;
    std::vector<int8_t> valueTargetsNCHW;
    Buffers(int c,Rule rule,size_t rows) : canvas(c),actions_count(game_spec(rule).actions(c)),packed_area((c*c+7)/8),
        planes(game_spec(rule).planes),global_count(game_spec(rule).globals),spatial_area(c*c),layout_rule(rule),row_capacity(rows+c*c) {
        // Training rows cap the shard, not trajectory plies. Sparse sampling may
        // require more trajectory storage; vectors grow and are reused on flush.
        observations.reserve(2*row_capacity*planes*packed_area);players.reserve(2*row_capacity);
        globals.reserve(2*row_capacity*global_count);
        visits.reserve(row_capacity*actions_count);policies.reserve(row_capacity*actions_count);
        opponent_policies.reserve(row_capacity*actions_count);opponent_policy_weights.reserve(row_capacity);
        actions.reserve(row_capacity);simulations.reserve(row_capacity);temperatures.reserve(row_capacity);rewards.reserve(row_capacity);
        ids.reserve(row_capacity);seeds.reserve(row_capacity);sizes.reserve(row_capacity);
        rules.reserve(row_capacity);winners.reserve(row_capacity);reasons.reserve(row_capacity);
        game_offsets.reserve(row_capacity+1);obs_offsets.reserve(row_capacity+1);reset();
    }
    void reset() {
        binaryInputNCHWPacked.clear();globalInputNC.clear();globalTargetsNC.clear();drawTargetsNC.clear();
        policyTargetsNCMove.clear();qValueTargetsNCMove.clear();valueTargetsNCHW.clear();
        trajectory_policy.clear();trajectory_q_values.clear();trajectory_q_visits.clear();
        row_begin=rows=0;options.clear();
        side_observations.clear();side_globals.clear();side_wdl.clear();side_target_weights.clear();
        side_players.clear();side_game_indices.clear();side_row_repeats.clear();side_policies.clear();
        side_visits.clear();side_forbidden_input.clear();
        q_values.clear();q_visits.clear();side_q_values.clear();side_q_visits.clear();
        observations.clear();globals.clear();players.clear();rules.clear();winners.clear();reasons.clear();sizes.clear();ids.clear();seeds.clear();
        root_policy_invalid_mass_sum.clear();root_policy_invalid_mass_count.clear();
        visits.clear();actions.clear();simulations.clear();policies.clear();temperatures.clear();rewards.clear();
        opponent_policies.clear();opponent_policy_weights.clear();
        opening_moves.clear();balanced_moves.clear();policy_moves.clear();opening_attempts.clear();
        initial_position_moves.clear();initial_position_kind.clear();hint_actions.clear();
        reanalyzed.clear();reanalysis_used_outcome.clear();reanalysis_original_visits.clear();
        reanalysis_policy_surprise.clear();reanalysis_value_surprise.clear();
        row_repeats.clear();target_weights.clear();policy_surprises.clear();value_surprises.clear();
        network_wdl.clear();search_wdl.clear();cheap_search.clear();sample_indices.clear();forbidden_input.clear();
        opening_status.clear();train_mask.clear();start_values.clear();opening_failures.clear();training_weights.clear();
        game_offsets.clear();obs_offsets.clear();game_offsets.push_back(0);obs_offsets.push_back(0);
    }
    void observation(const std::vector<float>& obs,int player) {
        if(obs.size()!=static_cast<size_t>(planes*spatial_area + global_count))throw std::runtime_error("Writer observation shape mismatch");
        size_t base=observations.size();observations.resize(base+planes*packed_area,0);
        for(int c=0;c<planes;++c) for(int i=0;i<spatial_area;++i) {
            float value=obs[c*spatial_area+i];
            if(value)observations[base+c*packed_area+i/8]|=1<<(7-i%8);
        }
        globals.insert(globals.end(), obs.end()-global_count, obs.end());
        players.push_back(player);
    }
};
void RecordWriter::append(const FinishedGame& g,size_t row_begin,size_t rows) {
    if(g.canvas<5 || g.canvas>25 || g.steps.empty())
        throw std::runtime_error("Invalid finished game");
    if(!buffers_)buffers_=std::make_unique<Buffers>(g.canvas,g.rule,max_rows_);
    auto& b=*buffers_;
    if(g.canvas!=b.canvas || !same_layout(game_spec(g.rule),game_spec(b.layout_rule)))throw std::runtime_error("Writer canvas mismatch");
    if(!std::isfinite(g.root_policy_invalid_mass_sum) || g.root_policy_invalid_mass_sum<0 ||
       g.root_policy_invalid_mass_sum>g.root_policy_invalid_mass_count ||
       (g.root_policy_invalid_mass_count && g.root_policy_invalid_mass_count!=g.steps.size()-g.opening.actions.size()))
        throw std::runtime_error("Invalid root policy game diagnostics");
    if(b.ids.empty())b.row_begin=row_begin;
    else if(row_begin!=0)throw std::runtime_error("Only the first trajectory may start at a partial training row");
    b.rows+=rows;
    b.options.push_back(g.options);b.ids.push_back(g.id);b.seeds.push_back(g.seed);b.sizes.push_back(g.size);
    b.root_policy_invalid_mass_sum.push_back(g.root_policy_invalid_mass_sum);
    b.root_policy_invalid_mass_count.push_back(g.root_policy_invalid_mass_count);
    b.rules.push_back(static_cast<int8_t>(g.rule));b.winners.push_back(g.winner);b.reasons.push_back(static_cast<int8_t>(g.reason));
    b.initial_position_moves.push_back(g.opening.initial_position_moves);b.initial_position_kind.push_back(g.opening.initial_position_kind);b.hint_actions.push_back(g.opening.hint_action);
    b.opening_moves.push_back(g.opening.actions.size());b.balanced_moves.push_back(g.opening.balanced_moves);
    b.policy_moves.push_back(g.opening.policy_moves);b.opening_attempts.push_back(g.opening.attempts);
    b.opening_status.push_back(static_cast<int8_t>(g.opening.status));b.start_values.push_back(g.opening.start_value);
    if(!std::isfinite(g.training_weight)||g.training_weight<0)throw std::runtime_error("Invalid per-game training weight");
    b.training_weights.push_back(g.training_weight);
    b.opening_failures.push_back(g.opening.failure);
    if(!std::isfinite(g.forbidden_feature_dropout_prob) || g.forbidden_feature_dropout_prob<0 || g.forbidden_feature_dropout_prob>1)
        throw std::runtime_error("Invalid forbidden feature dropout probability");
    std::mt19937_64 feature_rng(g.seed ^ 0xD1B54A32D192ED03ULL);
    // Removed Go ownership/score quantization has no shared RNG stream here.
    // Isolate the Q rounding stream from gameplay and forbidden-plane dropout.
    std::mt19937_64 q_rng(g.seed ^ 0xDB4F0B9175AE2165ULL);
    std::bernoulli_distribution use_forbidden(1-g.forbidden_feature_dropout_prob);
    // Same finite-horizon TD mixture as trainingwrite.cpp, extended by a draw
    // channel. Four-way values are always transformed through absolute Black.
    std::vector<std::array<WDL,3>> td(g.steps.size());
    std::vector<double> variance(g.steps.size());
    std::array<WDL,3> future;future.fill(g.outcome().wdl(1));
    const double factors[3]={.176,.056,.016};
    double next_value=g.outcome().value(1),variance_sum=0,variance_time=0;
    for(size_t t=g.steps.size();t-->0;) {
        const auto& step=g.steps[t];
        double value=(step.search_wdl[0]-step.search_wdl[2])*step.player;
        variance_sum+=(next_value-value)*(next_value-value);variance_time+=variance_sum;
        variance[t]=variance_time;next_value=value;
        for(int k=0;k<3;++k)for(int j=0;j<4;++j) {
            int absolute=step.player==1 || j==3?j:2-j;
            double now=1/(1+std::accumulate(g.final_observation.begin(),g.final_observation.begin()+g.canvas*g.canvas,0.0)*factors[k]);
            future[k][absolute]=now*step.search_wdl[j]+(1-now)*future[k][absolute];
            td[t][k][j]=future[k][absolute];
        }
    }
    size_t output_row=0;
    auto source_row=[&](const std::vector<float>& obs,int player,const std::vector<int16_t>& policy,
                        const std::vector<int16_t>* opponent,const WDL& value,const std::array<WDL,3>& values,
                        const WDL& now,double outcome_weight,double var_time,size_t turn,bool use_features,
                        const int16_t* q,const int16_t* qvisits,const Step* step,const SearchMetadata& stats) {
        size_t row=output_row++;
        if(row<row_begin || row>=row_begin+rows)return;
        size_t start=b.binaryInputNCHWPacked.size();b.binaryInputNCHWPacked.resize(start+b.planes*b.packed_area,0);
        for(int c=0;c<b.planes;++c)for(int i=0;i<b.spatial_area;++i) {
            if(g.rule==Rule::RENJU && !use_features && c>=3)continue;
            if(obs[c*b.spatial_area+i])b.binaryInputNCHWPacked[start+c*b.packed_area+i/8]|=1<<(7-i%8);
        }
        b.globalInputNC.insert(b.globalInputNC.end(),obs.end()-b.global_count,obs.end());
        if(g.rule==Rule::RENJU && !use_features)b.globalInputNC[b.globalInputNC.size()-b.global_count+3]=0;
        b.policyTargetsNCMove.insert(b.policyTargetsNCMove.end(),policy.begin(),policy.end());
        if(opponent)b.policyTargetsNCMove.insert(b.policyTargetsNCMove.end(),opponent->begin(),opponent->end());
        else b.policyTargetsNCMove.insert(b.policyTargetsNCMove.end(),b.actions_count,1);
        std::array<float,80> global{};
        auto target=[&](int slot,const WDL& wdl) {
            global[4*slot]=wdl[0];global[4*slot+1]=wdl[2];global[4*slot+2]=wdl[3];
            b.drawTargetsNC.push_back(wdl[1]);
        };
        target(0,value);for(int k=0;k<3;++k)target(k+1,values[k]);target(4,now);
        global[22]=var_time;global[25]=g.training_weight;global[26]=1;global[28]=opponent?1:0;
        global[27]=outcome_weight*(value[3]==0);global[33]=outcome_weight;global[62]=step?1:0;global[63]=3;
        global[51]=stats.turn;global[53]=g.opening.actions.size();
        // Source initialization modes; exact multi-game start kind remains in the trajectory extension.
        int modes[]={0,5,2,2,6,4};
        global[55]=modes[g.opening.initial_position_kind];
        const int pda_index=game_spec(g.rule).kind==GameKind::GO?15:4;
        if(g.opening.initial_position_kind==0 && obs[b.planes*b.spatial_area+pda_index]!=0)global[55]=7;
        global[31]=stats.policy_entropy;global[32]=stats.search_entropy;global[59]=stats.raw_policy_entropy;
        global[30]=stats.policy_surprise;global[57]=stats.raw_winloss;global[60]=stats.visits;
        global[47]=stats.rules[0];global[48]=stats.rules[1];global[56]=stats.rules[2];
        global[68]=stats.rules[3];global[69]=stats.rules[4];
        // Legacy history flags are still written, although the current source reader samples fresh flags.
        std::mt19937_64 history_rng(g.seed ^ (0x9E3779B97F4A7C15ULL*(row+1)));
        bool history=true;for(int k=36;k<=40;++k){history=history&&std::bernoulli_distribution(.98)(history_rng);global[k]=history;}
        uint64_t hash0=14695981039346656037ULL;
        for(unsigned char c:source_.run+source_.attempt+std::to_string(source_.worker)+":"+std::to_string(g.id))
            hash0=(hash0^c)*1099511628211ULL;
        uint64_t hash1=g.seed;
        for(int k=0;k<2;++k){uint64_t hash=k?hash1:hash0;global[41+3*k]=hash&0x3FFFFF;global[42+3*k]=(hash>>22)&0x3FFFFF;global[43+3*k]=(hash>>44)&0xFFFFF;}
        if(step) {
            global[30]=step->policy_surprise;global[57]=(step->network_wdl[0]-step->network_wdl[2]);
            global[64]=step->reanalyzed;
            if(step->reanalyzed){global[65]=step->reanalysis_policy_surprise;global[66]=step->reanalysis_value_surprise;global[67]=step->reanalysis_original_visits;}
        }
        // Go score, ownership and seki targets are intentionally unsupervised.
        // Future occupancy retains the source slots 2 and 3 and int8 encoding.
        size_t spatial_start=b.valueTargetsNCHW.size();b.valueTargetsNCHW.resize(spatial_start+5*b.spatial_area,0);
        if(step)for(int k=0;k<2;++k) {
            size_t t=std::min(turn+(k==0?8:32),g.steps.size());
            const auto& position=t==g.steps.size()?g.final_observation:g.steps[t].observation;
            int sign=(t==g.steps.size()?g.final_player:g.steps[t].player)*player;
            for(int i=0;i<b.spatial_area;++i) {
                float own=0,opp=0;
                int pieces=game_spec(g.rule).kind==GameKind::XIANGQI?7:1;
                for(int c=0;c<pieces;++c){own+=position[(1+c)*b.spatial_area+i];opp+=position[(1+pieces+c)*b.spatial_area+i];}
                b.valueTargetsNCHW[spatial_start+(k+2)*b.spatial_area+i]=static_cast<int8_t>((own-opp)*sign);
            }
        }
        b.globalTargetsNC.insert(b.globalTargetsNC.end(),global.begin(),global.end());
        b.qValueTargetsNCMove.insert(b.qValueTargetsNCMove.end(),q,q+b.actions_count);
        b.qValueTargetsNCMove.insert(b.qValueTargetsNCMove.end(),b.actions_count,0);
        b.qValueTargetsNCMove.insert(b.qValueTargetsNCMove.end(),qvisits,qvisits+b.actions_count);
    };
    for(size_t turn=0;turn<g.steps.size();++turn) {
        const auto& s=g.steps[turn];
        b.reanalyzed.push_back(s.reanalyzed);b.reanalysis_used_outcome.push_back(s.reanalyzed && s.reanalysis_used_outcome);
        b.reanalysis_original_visits.push_back(s.reanalysis_original_visits);
        b.reanalysis_policy_surprise.push_back(s.reanalysis_policy_surprise);b.reanalysis_value_surprise.push_back(s.reanalysis_value_surprise);
        b.train_mask.push_back(s.trainable);b.cheap_search.push_back(s.cheap_search);
        b.row_repeats.push_back(s.trainable?s.row_repeats:0);b.target_weights.push_back(s.trainable?s.target_weight:0);
        b.policy_surprises.push_back(s.policy_surprise);b.value_surprises.push_back(s.value_surprise);
        b.network_wdl.insert(b.network_wdl.end(),s.network_wdl.begin(),s.network_wdl.end());
        b.search_wdl.insert(b.search_wdl.end(),s.search_wdl.begin(),s.search_wdl.end());
        if(s.policy.size()!=static_cast<size_t>(b.actions_count) || s.visits.size()!=s.policy.size())
            throw std::runtime_error("Writer policy/visit shape mismatch");
        // Retain the full trajectory even when only a span of its final training
        // rows belongs to this shard. TD/opponent targets and per-row randomness
        // must be identical on both sides of a shard boundary.
        if(source_.unroll_steps>0) {
            // Independent per-position rounding, stable across writer row splits.
            std::mt19937_64 trajectory_rng(g.seed ^ (0xD6E8FEB86659FD93ULL*(turn+1)));
            if(s.trainable) {
                if(s.policy_target.size()!=s.policy.size())throw std::runtime_error("Missing MuZero trajectory policy");
                b.trajectory_policy.insert(b.trajectory_policy.end(),s.policy_target.begin(),s.policy_target.end());
                append_q_targets(s.q_values,s.q_visits,1,b.actions_count,b.trajectory_q_values,b.trajectory_q_visits,trajectory_rng);
            } else {
                b.trajectory_policy.insert(b.trajectory_policy.end(),b.actions_count,0);
                b.trajectory_q_values.insert(b.trajectory_q_values.end(),b.actions_count,0);
                b.trajectory_q_visits.insert(b.trajectory_q_visits.end(),b.actions_count,0);
            }
        }
        if(s.trainable && s.row_repeats>0) {
            for(int repeat=0;repeat<s.row_repeats;++repeat)
                b.forbidden_input.push_back(g.rule==Rule::RENJU && use_forbidden(feature_rng));
            b.sample_indices.push_back(b.actions.size());
            if(s.policy_target.size()!=s.policy.size())throw std::runtime_error("Missing quantized policy target");
            b.policies.insert(b.policies.end(),s.policy_target.begin(),s.policy_target.end());
            b.visits.insert(b.visits.end(),s.visits.begin(),s.visits.end());
            append_q_targets(s.q_values,s.q_visits,s.row_repeats,b.actions_count,b.q_values,b.q_visits,q_rng);
            // KataGo trainingwrite.cpp uses the actual next turn's search target,
            // even if that turn has zero sampling weight (e.g. a cheap search).
            bool has_next=turn+1<g.steps.size() && !(s.reanalyzed && !s.reanalysis_used_outcome);
            b.opponent_policy_weights.push_back(has_next?1.0f:0.0f);
            if(has_next) {
                const auto& next=g.steps[turn+1];
                if(!next.trainable || next.policy_target.size()!=s.policy.size())
                    throw std::runtime_error("Missing next-turn opponent policy");
                b.opponent_policies.insert(b.opponent_policies.end(),next.policy_target.begin(),next.policy_target.end());
            } else {
                // Normalizable placeholder only; both opponent losses have zero weight.
                b.opponent_policies.insert(b.opponent_policies.end(),b.actions_count,1);
            }
            for(int repeat=0;repeat<s.row_repeats;++repeat) {
                size_t qrow=b.q_values.size()/b.actions_count-s.row_repeats+repeat;
                bool full=!s.reanalyzed || s.reanalysis_used_outcome;
                source_row(s.observation,s.player,s.policy_target,has_next?&g.steps[turn+1].policy_target:nullptr,
                    g.outcome().wdl(s.player),td[turn],s.search_wdl,full,variance[turn],turn,
                    b.forbidden_input[b.forbidden_input.size()-s.row_repeats+repeat],
                    b.q_values.data()+qrow*b.actions_count,b.q_visits.data()+b.q_visits.size()-b.actions_count,&s,s.metadata);
            }
        }
        b.observation(s.observation,s.player);b.actions.push_back(s.action);b.simulations.push_back(s.simulations);
        b.temperatures.push_back(s.temperature);b.rewards.push_back(s.reward);
    }
    std::mt19937_64 side_feature_rng(g.seed ^ 0x94D049BB133111EBULL);
    std::mt19937_64 side_q_rng(g.seed ^ 0xBBE0563303A4615FULL);
    const auto kind=game_spec(b.layout_rule).kind;
    const int pda_global=kind==GameKind::GO?15:kind==GameKind::XIANGQI?3:4;
    for(const auto& side:g.side_positions) {
        if(side.observation.size()!=static_cast<size_t>(b.planes*b.spatial_area+b.global_count) ||
           side.policy_target.size()!=static_cast<size_t>(b.actions_count) || side.visits.size()!=side.policy_target.size() ||
           (side.player!=1 && side.player!=-1) || side.row_repeats<0 || !std::isfinite(side.target_weight) || side.target_weight<0 ||
           std::abs(side.row_repeats-side.target_weight)>1.00001)
            throw std::runtime_error("Invalid side position");
        if(!std::all_of(side.observation.end()-b.global_count,side.observation.end(),[](float v){return std::isfinite(v);}))
            throw std::runtime_error("Nonfinite side globals");
        if(side.observation[b.planes*b.spatial_area+pda_global]!=0 || side.observation[b.planes*b.spatial_area+pda_global+1]!=0)
            throw std::runtime_error("Side rows must clear PDA conditioning");
        size_t base=b.side_observations.size();b.side_observations.resize(base+b.planes*b.packed_area,0);
        for(int c=0;c<b.planes;++c)for(int i=0;i<b.spatial_area;++i) {
            float value=side.observation[c*b.spatial_area+i];
            if(value)b.side_observations[base+c*b.packed_area+i/8]|=1<<(7-i%8);
        }
        b.side_globals.insert(b.side_globals.end(),side.observation.end()-b.global_count,side.observation.end());
        b.side_players.push_back(side.player);b.side_game_indices.push_back(b.ids.size()-1);
        b.side_policies.insert(b.side_policies.end(),side.policy_target.begin(),side.policy_target.end());
        b.side_visits.insert(b.side_visits.end(),side.visits.begin(),side.visits.end());
        append_q_targets(side.q_values,side.q_visits,side.row_repeats,b.actions_count,b.side_q_values,b.side_q_visits,side_q_rng);
        b.side_wdl.insert(b.side_wdl.end(),side.search_wdl.begin(),side.search_wdl.end());
        b.side_target_weights.push_back(side.target_weight);b.side_row_repeats.push_back(side.row_repeats);
        for(int repeat=0;repeat<side.row_repeats;++repeat) {
            b.side_forbidden_input.push_back(g.rule==Rule::RENJU && use_forbidden(side_feature_rng));
            std::array<WDL,3> values;values.fill(side.search_wdl);
            size_t qrow=b.side_q_values.size()/b.actions_count-side.row_repeats+repeat;
            source_row(side.observation,side.player,side.policy_target,nullptr,side.search_wdl,values,side.search_wdl,
                0,0,0,b.side_forbidden_input.back(),b.side_q_values.data()+qrow*b.actions_count,
                b.side_q_visits.data()+b.side_q_visits.size()-b.actions_count,nullptr,side.metadata);
        }
    }
    b.observation(g.final_observation,g.final_player);b.game_offsets.push_back(b.actions.size());b.obs_offsets.push_back(b.players.size());
}
void RecordWriter::loop() {
    try {
        for(;;) {
            FinishedGame game;bool have=false,done;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                changed_.wait(lock,[&]{return closing_ || !queue_.empty();});
                if(!queue_.empty()) {game=std::move(queue_.front());queue_.pop_front();have=true;}
                done=closing_ && queue_.empty();
            }
            changed_.notify_all();
            if(have) {
                size_t total=0;
                for(const auto& step:game.steps) {
                    if(step.row_repeats<0)throw std::runtime_error("Negative main row repeats");
                    if(step.trainable)total+=static_cast<size_t>(step.row_repeats);
                }
                for(const auto& side:game.side_positions) {
                    if(side.row_repeats<0)throw std::runtime_error("Negative side row repeats");
                    total+=static_cast<size_t>(side.row_repeats);
                }
                size_t begin=0;
                do {
                    size_t limit=first_file_?first_rows_:max_rows_;
                    size_t available=limit-(buffers_?buffers_->rows:0);
                    size_t take=std::min(total-begin,available);
                    append(game,begin,take);begin+=take;
                    if(buffers_->rows==limit) {publish();buffers_->reset();first_file_=false;}
                } while(begin<total);
            }
            if(done && buffers_ && !buffers_->ids.empty()) {publish();buffers_->reset();}
            if(done)return;
        }
    } catch (...) {
        {std::lock_guard<std::mutex> lock(mutex_);failure_=std::current_exception();}changed_.notify_all();
    }
}
void RecordWriter::publish() {
    auto& b=*buffers_;
    const int canvas=b.canvas,actions_count=b.actions_count;
    const size_t n=b.ids.size(),t=b.actions.size(),o=b.players.size(),s=b.sample_indices.size();
    // Independent file naming does not consume the game RNG. Source validation
    // assigns whole files by the MD5 of their path.
    std::random_device file_random;
    const uint64_t filename_bits=(uint64_t(file_random())<<32)^file_random();
    std::ostringstream filename;
    filename<<std::hex<<std::setfill('0')<<std::setw(16)<<filename_bits<<".npz";
    std::string shard=filename.str();
    auto created = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::ostringstream meta;
    meta << "{\"canvas\":" << canvas << ",\"rows\":" << b.rows << ",\"row_begin\":" << b.row_begin << ",\"plies\":" << t << ",\"games\":" << n
         << ",\"run_id\":" << quote(source_.run) << ",\"attempt_id\":" << quote(source_.attempt)
         << ",\"iteration_id\":" << source_.iteration << ",\"worker_id\":" << source_.worker << ",\"model_id\":" << quote(source_.model)
         << ",\"config_id\":" << quote(source_.config) << ",\"source_id\":" << quote(source_.source)
         << ",\"shard_id\":" << quote(source_.attempt+":"+std::to_string(source_.worker)+":"+shard)
         << ",\"created_ns\":" << created << ",\"opening_failures\":[";
    for(size_t i=0;i<n;++i){if(i)meta<<',';meta<<quote(b.opening_failures[i]);}meta<<"]";
    meta<<",\"game\":"<<quote(game_spec(b.layout_rule).name)
        <<",\"input_planes\":"<<b.planes<<",\"global_features\":"<<b.global_count<<",\"action_count\":"<<b.actions_count
        <<",\"rule_options\":[";
    for(size_t i=0;i<n;++i){if(i)meta<<',';const auto& o=b.options[i];
        meta<<"{\"go_rules\":"<<quote(o.go_rules)<<",\"komi\":"<<o.komi<<",\"fen\":"<<quote(o.fen)<<",\"go_position\":"<<quote(o.go_position)<<",\"go_width\":"<<o.go_width<<",\"go_height\":"<<o.go_height<<",\"go_pass_alive\":"<<(o.go_pass_alive?"true":"false")<<",\"go_exclude_atari\":"<<(o.go_exclude_atari?"true":"false")<<'}';}meta<<']';
    if(source_.gumbel)meta<<",\"root_search_algo\":\"gumbel\",\"nonroot_search_algo\":"<<quote(source_.full_gumbel?"gumbel":"puct");
    if(source_.unroll_steps>0) {
        meta<<",\"algorithm\":\"muzero\",\"unroll_steps\":"<<source_.unroll_steps
            <<",\"root_policy_invalid_mass_sum\":["<<std::setprecision(17);
        for(size_t i=0;i<n;++i){if(i)meta<<',';meta<<b.root_policy_invalid_mass_sum[i];}
        meta<<"],\"root_policy_invalid_mass_count\":[";
        for(size_t i=0;i<n;++i){if(i)meta<<',';meta<<b.root_policy_invalid_mass_count[i];}
        meta<<"]";
    }
    meta<<",\"game_row_totals\":[";
    for(size_t g=0;g<n;++g) {
        int64_t rows=0;for(int64_t i=b.game_offsets[g];i<b.game_offsets[g+1];++i)rows+=b.row_repeats[i];
        for(size_t i=0;i<b.side_game_indices.size();++i)if(b.side_game_indices[i]==int(g))rows+=b.side_row_repeats[i];
        if(g)meta<<',';meta<<rows;
    }meta<<"]";
    meta<<"}";
    std::string metadata = meta.str(); std::vector<uint8_t> meta_bytes(metadata.begin(), metadata.end());
    auto destination = directory_ / shard, temporary = destination; temporary += ".tmp";
    if (std::filesystem::exists(destination)) throw std::runtime_error("Refusing to overwrite raw data");
    Zip zip(temporary);
    zip.array("binaryInputNCHWPacked",b.binaryInputNCHWPacked,"|u1",{b.rows,static_cast<size_t>(b.planes),static_cast<size_t>(b.packed_area)});
    zip.array("globalInputNC",b.globalInputNC,"<f4",{b.rows,static_cast<size_t>(b.global_count)});
    zip.array("policyTargetsNCMove",b.policyTargetsNCMove,"<i2",{b.rows,2,static_cast<size_t>(actions_count)});
    zip.array("globalTargetsNC",b.globalTargetsNC,"<f4",{b.rows,80});
    zip.array("drawTargetsNC",b.drawTargetsNC,"<f4",{b.rows,5});
    zip.array("valueTargetsNCHW",b.valueTargetsNCHW,"|i1",{b.rows,5,static_cast<size_t>(canvas),static_cast<size_t>(canvas)});
    zip.array("qValueTargetsNCMove",b.qValueTargetsNCMove,"<i2",{b.rows,3,static_cast<size_t>(actions_count)});
    if(source_.unroll_steps>0) {
        zip.array("trajectory_policy",b.trajectory_policy,"<i2",{t,static_cast<size_t>(actions_count)});
        zip.array("trajectory_q_values",b.trajectory_q_values,"<i2",{t,static_cast<size_t>(actions_count)});
        zip.array("trajectory_q_visits",b.trajectory_q_visits,"<i2",{t,static_cast<size_t>(actions_count)});
    }
    zip.array("policy_moves", b.policy_moves, "<i2", {n});
    zip.array("winners", b.winners, "|i1", {n});
    zip.array("opening_moves", b.opening_moves, "<i2", {n});
    zip.array("opening_status", b.opening_status, "|i1", {n});
    zip.array("game_offsets", b.game_offsets, "<i8", {n+1});
    zip.array("game_ids", b.ids, "<u8", {n});
    zip.array("opening_attempts", b.opening_attempts, "<i4", {n});
    zip.array("reasons", b.reasons, "|i1", {n});
    if(source_.unroll_steps>0||source_.research_records) {
    zip.array("observations", b.observations, "|u1", {o, static_cast<size_t>(b.planes), static_cast<size_t>(b.packed_area)});
    zip.array("globals", b.globals, "<f4", {o, static_cast<size_t>(b.global_count)});
    zip.array("players", b.players, "|i1", {o}); zip.array("actions", b.actions, "<i4", {t});
    zip.array("sample_indices",b.sample_indices,"<i8",{s});
    zip.array("policies", b.policies, "<i2", {s, static_cast<size_t>(actions_count)});
    zip.array("opponent_policies", b.opponent_policies, "<i2", {s, static_cast<size_t>(actions_count)});
    zip.array("opponent_policy_weights", b.opponent_policy_weights, "<f4", {s});
    zip.array("visits", b.visits, "<i8", {s, static_cast<size_t>(actions_count)}); zip.array("simulations", b.simulations, "<i8", {t});
    zip.array("q_values",b.q_values,"<i2",{b.forbidden_input.size(),static_cast<size_t>(actions_count)});
    zip.array("q_visits",b.q_visits,"<i2",{s,static_cast<size_t>(actions_count)});
    zip.array("temperatures", b.temperatures, "<f4", {t}); zip.array("rewards", b.rewards, "<f4", {t});
     zip.array("observation_offsets", b.obs_offsets, "<i8", {n+1});
     zip.array("seeds", b.seeds, "<u8", {n}); zip.array("sizes", b.sizes, "<i2", {n});
    zip.array("rules", b.rules, "|i1", {n});
    zip.array("train_mask", b.train_mask, "|u1", {t});
    zip.array("cheap_search",b.cheap_search,"|u1",{t});zip.array("row_repeats",b.row_repeats,"<i4",{t});
    zip.array("reanalyzed",b.reanalyzed,"|u1",{t});zip.array("reanalysis_used_outcome",b.reanalysis_used_outcome,"|u1",{t});
    zip.array("reanalysis_original_visits",b.reanalysis_original_visits,"<i8",{t});
    zip.array("reanalysis_policy_surprise",b.reanalysis_policy_surprise,"<f4",{t});
    zip.array("reanalysis_value_surprise",b.reanalysis_value_surprise,"<f4",{t});
    zip.array("target_weights",b.target_weights,"<f4",{t});
    zip.array("policy_surprises",b.policy_surprises,"<f4",{t});zip.array("value_surprises",b.value_surprises,"<f4",{t});
    zip.array("network_wdl",b.network_wdl,"<f4",{t,4});zip.array("search_wdl",b.search_wdl,"<f4",{t,4});
    zip.array("initial_position_moves",b.initial_position_moves,"<i2",{n});zip.array("initial_position_kind",b.initial_position_kind,"|i1",{n});zip.array("hint_actions",b.hint_actions,"<i4",{n});
    zip.array("balanced_moves", b.balanced_moves, "<i2", {n});

    zip.array("forbidden_input", b.forbidden_input, "|u1", {b.forbidden_input.size()});
    zip.array("start_values", b.start_values, "<f4", {n});
    zip.array("training_weights", b.training_weights, "<f4", {n});
    size_t d=b.side_players.size();
    zip.array("side_observations",b.side_observations,"|u1",{d,static_cast<size_t>(b.planes),static_cast<size_t>(b.packed_area)});
    zip.array("side_globals",b.side_globals,"<f4",{d,static_cast<size_t>(b.global_count)});
    zip.array("side_players",b.side_players,"|i1",{d});zip.array("side_game_indices",b.side_game_indices,"<i4",{d});
    zip.array("side_policies",b.side_policies,"<i2",{d,static_cast<size_t>(actions_count)});
    zip.array("side_visits",b.side_visits,"<i8",{d,static_cast<size_t>(actions_count)});
    zip.array("side_q_values",b.side_q_values,"<i2",{b.side_forbidden_input.size(),static_cast<size_t>(actions_count)});
    zip.array("side_q_visits",b.side_q_visits,"<i2",{d,static_cast<size_t>(actions_count)});
    zip.array("side_wdl",b.side_wdl,"<f4",{d,4});zip.array("side_target_weights",b.side_target_weights,"<f4",{d});
    zip.array("side_row_repeats",b.side_row_repeats,"<i4",{d});
    zip.array("side_forbidden_input",b.side_forbidden_input,"|u1",{b.side_forbidden_input.size()});
    }
    zip.array("metadata", meta_bytes, "|u1", {meta_bytes.size()}); zip.finish();
    std::filesystem::rename(temporary, destination);
    std::cout << "{\"event\":\"shard\",\"file\":" << quote(destination.string()) << ",\"rows\":" << b.rows
              << ",\"plies\":" << t << ",\"games\":" << n-(b.row_begin>0?1:0) << "}" << std::endl;
}
}
