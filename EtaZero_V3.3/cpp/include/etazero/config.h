#pragma once
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cctype>
namespace etazero {
inline std::string trim(std::string s) {
    auto start = s.find_first_not_of(" \t\r\n"), end = s.find_last_not_of(" \t\r\n");
    return start == std::string::npos ? "" : s.substr(start, end - start + 1);
}
inline int parse_integer(const std::string& value, const std::string& field) {
    size_t consumed=0;int result=std::stoi(value,&consumed);
    if(consumed!=value.size())throw std::runtime_error("Invalid integer: "+field+"="+value);
    return result;
}
class Config {
    std::map<std::string, std::string> fields_;
public:
    explicit Config(const std::string& path) {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("Cannot open resolved config: " + path);
        std::string line, section;
        while (std::getline(in, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;
            if (line.front() == '[' && line.back() == ']') { section = line.substr(1, line.size()-2); continue; }
            auto pos = line.find('=');
            if (pos == std::string::npos || section.empty()) throw std::runtime_error("Malformed resolved config");
            auto key = section + "." + trim(line.substr(0, pos));
            if (!fields_.emplace(key, trim(line.substr(pos+1))).second) throw std::runtime_error("Duplicate native key");
        }
    }
    void override_field(const std::string& key,const std::string& value) {if(!contains(key))throw std::runtime_error("Unsupported overrideSettings key: "+key);fields_[key]=value;}
    bool contains(const std::string& k) const { return fields_.count(k)!=0; }
    std::string text(const std::string& k) const {
        auto it=fields_.find(k);
        if(it==fields_.end())throw std::runtime_error("Missing native configuration field: "+k);
        return it->second;
    }
    int integer(const std::string& k) const { return parse_integer(text(k),k); }
    int64_t integer64(const std::string& k) const {
        auto value=text(k);size_t consumed=0;int64_t result=std::stoll(value,&consumed);
        if(consumed!=value.size())throw std::runtime_error("Invalid int64: "+k);
        return result;
    }
    double number(const std::string& k) const {auto value=text(k);size_t used=0;double result=std::stod(value,&used);if(used!=value.size()||!std::isfinite(result))throw std::runtime_error("Invalid number: "+k);return result;}
    bool boolean(const std::string& k) const {auto value=text(k);for(auto& c:value)c=std::tolower(c);if(value!="true"&&value!="false")throw std::runtime_error("Invalid boolean: "+k);return value=="true";}
    std::vector<std::string> list(const std::string& k) const {
        std::istringstream s(text(k)); std::string item; std::vector<std::string> result;
        while (std::getline(s, item, ',')) { item=trim(item); if (!item.empty()) result.push_back(item); }
        return result;
    }
};
}
