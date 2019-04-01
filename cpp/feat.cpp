

#include <iostream>
#include <sstream>
#include "feat.h"
#include "utils.h"

namespace myccg {
    
Feat Feature::Parse(const std::string& string) {
    if (Cacheable::Count(string) > 0) {
        return Cacheable::Get(string);
    } else {
        Feat res;
        if (string.find(",") != std::string::npos)
            res = new MultiValueFeature(string);
        else
            res = new SingleValueFeature(string);
        res->RegisterCache(string);
        return res;
    }
}

// parse "mod=nm,form=base,fin=f"
MultiValueFeature::MultiValueFeature(const std::string& string)
    : contains_wildcard_(false) {
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

std::string MultiValueFeature::ToStr() const {
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

bool MultiValueFeature::Matches(Feat other) const {
    const MultiValueFeature* o;
    if ((o = dynamic_cast<const MultiValueFeature*>(other)) == nullptr)
        return false;
    if (GetId() == o->GetId()) return true;
    if ((!this->ContainsWildcard() && 
            !o->ContainsWildcard()) ||
            this->values_.size() != o->values_.size())
        return false;

    for (unsigned i = 0; i < values_.size(); i++) {
        auto& t_v = this->values_[i];
        auto& o_v = o->values_[i];
        if (t_v.first != o_v.first ||
                (t_v.second != o_v.second &&
                 t_v.second[0] != 'X' &&
                 o_v.second[0] != 'X'))
            return false;
    }
    return true;
}

std::string MultiValueFeature::SubstituteWildcard(const std::string& string) const {
    std::string res(string);
    for (unsigned i = 0; i < values_.size(); i++)
        utils::ReplaceAll(&res, "X" + std::to_string(i+1), values_[i].second);
    return res;
}

bool MultiValueFeature::ContainsKeyValue(
        const std::string& key, const std::string& value) const {
    for (auto&& pair : values_) {
        if (pair.first == key && pair.second == value)
            return true;
    }
    return false;
}

// S[X] --> S[feat] ; for English grammar
std::string SingleValueFeature::SubstituteWildcard(const std::string& string) const {
    std::string res(string);
    utils::ReplaceAll(&res, "[X]", this->ToStr());
    return res;
}

bool SingleValueFeature::Matches(Feat other) const {
    const SingleValueFeature* o;
    if ((o = dynamic_cast<const SingleValueFeature*>(other)) == nullptr)
        return false;
    return (GetId() == o->GetId() ||
            this->ContainsWildcard() || o->ContainsWildcard());
}

Feat SingleValueFeature::ToMultiValue() const {
    return IsEmpty() ? this  : Feature::Parse(value_ + "=true");
}

} // namespace myccg

