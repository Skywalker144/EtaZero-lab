#include "etazero/record.h"
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <system_error>
#include <unistd.h>
#include <zlib.h>
namespace etazero {
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
        raw.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
        auto bytes = raw.str();
        if (bytes.size() > UINT32_MAX || out.tellp() > UINT32_MAX) throw std::runtime_error("NPZ shard exceeds classic ZIP limits");
        std::vector<unsigned char> compressed(compressBound(bytes.size()));
        z_stream zs{};
        if (deflateInit2(&zs, Z_BEST_SPEED, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY) != Z_OK)
            throw std::runtime_error("Cannot initialize NPZ compressor");
        zs.next_in = reinterpret_cast<Bytef*>(bytes.data()); zs.avail_in = bytes.size();
        zs.next_out = compressed.data(); zs.avail_out = compressed.size();
        int status = deflate(&zs, Z_FINISH); uint32_t length = zs.total_out; deflateEnd(&zs);
        if (status != Z_STREAM_END) throw std::runtime_error("NPZ compression failed");
        Entry e{name + ".npy", static_cast<uint32_t>(crc32(0, reinterpret_cast<const Bytef*>(bytes.data()), bytes.size())),
                length, static_cast<uint32_t>(bytes.size()), static_cast<uint32_t>(out.tellp())};
        u32(out, 0x04034b50); u16(out, 20); u16(out, 0); u16(out, 8); u16(out, 0); u16(out, 0);
        u32(out, e.crc); u32(out, e.compressed); u32(out, e.size); u16(out, e.name.size()); u16(out, 0);
        out << e.name; out.write(reinterpret_cast<const char*>(compressed.data()), length); entries.push_back(e);
    }
    void finish() {
        uint32_t start = out.tellp();
        for (const auto& e : entries) {
            u32(out, 0x02014b50); u16(out, 20); u16(out, 20); u16(out, 0); u16(out, 8); u16(out, 0); u16(out, 0);
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
void RecordWriter::loop() {
    try {
        std::vector<FinishedGame> games; size_t rows = 0;
        auto interval = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(flush_seconds_));
        auto deadline = std::chrono::steady_clock::now() + interval;
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex_);
            changed_.wait_until(lock, deadline, [&] { return closing_ || !queue_.empty(); });
            if (!queue_.empty()) {
                rows += queue_.front().steps.size(); games.push_back(std::move(queue_.front())); queue_.pop_front();
                changed_.notify_all();
            }
            bool done = closing_ && queue_.empty(); lock.unlock();
            if (rows >= max_rows_ || done || std::chrono::steady_clock::now() >= deadline) {
                if (!games.empty()) publish(games);
                games.clear(); rows = 0; deadline = std::chrono::steady_clock::now() + interval;
            }
            if (done) return;
        }
    } catch (...) {
        { std::lock_guard<std::mutex> lock(mutex_); failure_ = std::current_exception(); }
        changed_.notify_all();
    }
}
void RecordWriter::publish(const std::vector<FinishedGame>& games) {
    const int canvas = games.front().canvas, actions_count = canvas * canvas;
    std::vector<uint8_t> observations;
    std::vector<int8_t> players, rules, winners, reasons;
    std::vector<int16_t> sizes;
    std::vector<uint64_t> ids, seeds;
    std::vector<int64_t> game_offsets{0}, obs_offsets{0}, visits;
    std::vector<int32_t> actions, simulations;
    std::vector<float> policies, temperatures, rewards;
    for (auto& g : games) {
        if (g.canvas != canvas || g.steps.empty()) throw std::runtime_error("Invalid finished game");
        ids.push_back(g.id); seeds.push_back(g.seed); sizes.push_back(g.size);
        rules.push_back(static_cast<int8_t>(g.rule)); winners.push_back(g.winner); reasons.push_back(g.reason);
        for (auto& s : g.steps) {
            observations.insert(observations.end(), s.observation.begin(), s.observation.end()); players.push_back(s.player);
            actions.push_back(s.action); simulations.push_back(s.simulations); temperatures.push_back(s.temperature); rewards.push_back(s.reward);
            policies.insert(policies.end(), s.policy.begin(), s.policy.end()); visits.insert(visits.end(), s.visits.begin(), s.visits.end());
        }
        observations.insert(observations.end(), g.final_observation.begin(), g.final_observation.end()); players.push_back(g.final_player);
        game_offsets.push_back(actions.size()); obs_offsets.push_back(players.size());
    }
    const size_t n = games.size(), t = actions.size(), o = players.size();
    std::string shard = "shard_" + std::to_string(next_shard_++) + ".npz";
    auto created = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::ostringstream meta;
    meta << "{\"contract\":" << quote(CONTRACT_ID) << ",\"canvas\":" << canvas << ",\"rows\":" << t << ",\"games\":" << n
         << ",\"run_id\":" << quote(source_.run) << ",\"attempt_id\":" << quote(source_.attempt)
         << ",\"cycle_id\":" << source_.cycle << ",\"worker_id\":" << source_.worker << ",\"model_id\":" << quote(source_.model)
         << ",\"config_id\":" << quote(source_.config) << ",\"source_id\":" << quote(source_.source)
         << ",\"shard_id\":" << quote(source_.attempt+":"+std::to_string(source_.worker)+":"+shard)
         << ",\"created_ns\":" << created << '}';
    std::string metadata = meta.str(); std::vector<uint8_t> meta_bytes(metadata.begin(), metadata.end());
    auto destination = directory_ / shard, temporary = destination; temporary += ".tmp";
    if (std::filesystem::exists(destination)) throw std::runtime_error("Refusing to overwrite raw data");
    Zip zip(temporary);
    zip.array("observations", observations, "|u1", {o, INPUT_PLANES, static_cast<size_t>(canvas), static_cast<size_t>(canvas)});
    zip.array("players", players, "|i1", {o}); zip.array("actions", actions, "<i4", {t});
    zip.array("policies", policies, "<f4", {t, static_cast<size_t>(actions_count)});
    zip.array("visits", visits, "<i8", {t, static_cast<size_t>(actions_count)}); zip.array("simulations", simulations, "<i4", {t});
    zip.array("temperatures", temperatures, "<f4", {t}); zip.array("rewards", rewards, "<f4", {t});
    zip.array("game_offsets", game_offsets, "<i8", {n+1}); zip.array("observation_offsets", obs_offsets, "<i8", {n+1});
    zip.array("game_ids", ids, "<u8", {n}); zip.array("seeds", seeds, "<u8", {n}); zip.array("sizes", sizes, "<i2", {n});
    zip.array("rules", rules, "|i1", {n}); zip.array("winners", winners, "|i1", {n}); zip.array("reasons", reasons, "|i1", {n});
    zip.array("metadata", meta_bytes, "|u1", {meta_bytes.size()}); zip.finish(); sync_path(temporary);
    std::filesystem::rename(temporary, destination); sync_path(directory_);
    std::cout << "{\"event\":\"shard\",\"file\":" << quote(destination.string()) << ",\"rows\":" << t << ",\"games\":" << n << "}" << std::endl;
}
}
