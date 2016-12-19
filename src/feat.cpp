

#include <sstream>
#include "feat.h"
#include "utils.h"

namespace myccg {
namespace cat {
    
#ifdef JAPANESE
// parse "mod=nm,form=base,fin=f"
Feature::Feature(const std::string& string): contains_wildcard_(false) {
    std::istringstream s(string);
    std::string pair;
    while (std::getline(s, pair, ',')) {
        int eq = pair.find("=");
        std::string key = pair.substr(0, eq);
        std::string value = pair.substr(eq + 1);
        contains_wildcard_ |= value[0] == 'X';
        values_.emplace_back(key, value);
    }
}

std::string Feature::ToStr() const {
    if (IsEmpty()) return "";
    std::stringstream res;
    res << "[";
    for (unsigned i = 0; i < values_.size(); i++) {
        if (i > 0) res << ",";
        auto& pair = values_[i];
        res << pair.first << "=" << pair.second;
    }
    res << "]";
    return res.str();
}

bool Feature::Matches(const Feature* other) const {
    if (GetId() == other->GetId()) return true;
    if ((!this->ContainsWildcard() && 
            !other->ContainsWildcard()) ||
            this->values_.size() != other->values_.size())
        return false;

    for (unsigned i = 0; i < values_.size(); i++) {
        auto& t_v = this->values_[i];
        auto& o_v = other->values_[i];
        if (t_v.first != o_v.first ||
                (t_v.second != o_v.second &&
                 t_v.second[0] != 'X' &&
                 o_v.second[0] != 'X'))
            return false;
    }
    return true;
}

std::string Feature::SubstituteWildcard(const std::string& string) const {
    std::string res(string);
    for (unsigned i = 0; i < values_.size(); i++)
        utils::ReplaceAll(&res, "X" + std::to_string(i+1), values_[i].second);
    return res;
}

#else
// S[X] --> S[feat] ; for English grammar
std::string Feature::SubstituteWildcard(const std::string& string) const {
    std::string res(string);
    utils::ReplaceAll(&res, "[X]", this->ToStr());
    return res;
}
#endif

Feat Feature::Parse(const std::string& string) {
    if (Cacheable::Count(string) > 0) {
        return Cacheable::Get<Feat>(string);
    } else {
        Feat res = new Feature(string);
        res->RegisterCache(string);
        return res;
    }
}

} // namespace cat
} // namespace myccg

