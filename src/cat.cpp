
#include <iostream>
#include <sstream>
#include "cat.h"
#include "utils.h"

namespace myccg {
namespace cat {

const char* slashes = "/\\|";
std::unordered_map<std::string, const Cacheable*> cache;
int Cacheable::ids = 0;


template<> Cat Compose<0>(Cat head, const Slash& op, Cat tail) {
    return Make(head, op, tail->GetRight());
}

template<> bool Category::HasFunctorAtLeft<0>() const {
    return this->IsFunctor();
}

template<> bool Category::HasFunctorAtRight<0>() const {
    return this->IsFunctor();
}

template<> Cat Category::GetLeft<0>() const { return this; }
template<> Cat Category::GetRight<0>() const { return this; }

Cacheable::Cacheable() {
    #pragma omp atomic capture
    id_ = ids++;
}

void Cacheable::RegisterCache(const std::string& key) const {
    #pragma omp critical(RegisterCache)
    cache.emplace(key, this);
}

#ifdef JAPANESE
// "mod=nm,form=base,fin=f"
Feature::Feature(const std::string& string) : contains_wildcard_(false) {
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
    std::stringstream res;
    res << "[";
    for (unsigned i = 0; i < values_.size(); i++) {
        if (i > 0) res << ","
        auto& pair = values_[i];
        res << pair.first << "=" << pair.second;
    }
    res << "]";
    return res.to_str();
}

bool Feature::Matches(const Feature* other) const {
    if (GetId() == other->GetId()) return true;
    if (!this->ContainsWildcard() || 
            !other->ContainsWildcard() ||
            this->values_.size() != other->values_.size())
        return false;

    for (unsigned i = 0; i < values_.size(); i++) {
        auto& t_v = this->values_[i];
        auto& o_v = other->values_[i];
        if (t_v.first != o_v.first ||
                t_v.second != o_v.second ||
                t_v.second[0] != 'X' ||
                o_v.second[0] != 'X')
            return false;
    }
    return true;
}

std::string SubstituteWildcard(const std::string& string) const {
    return NULL;
}

#else
// S[X] --> S[feat] ; for English grammar
std::string Feature::SubstituteWildcard(const std::string& string) const {
    return utils::ReplaceAll(string, "X", this->ToStr());
}
#endif

std::string AtomicCategory::ToStrWithoutFeat() const {
#ifdef JAPANESE
    return type_;
#else
    return utils::ReplaceAll(utils::ReplaceAll(ToStr(), "[X]", ""), "[nb]", "");
#endif
}

Feat Feature::Parse(const std::string& string) {
    if (cache.count(string) > 0) {
        return static_cast<Feat>(cache[string]);
    } else {
        Feat res = new Feature(string);
        res->RegisterCache(string);
        return res;
    }
}

Cat Category::Parse(const std::string& cat) {
    Cat res;
    if (cache.count(cat) > 0) {
        return static_cast<Cat>(cache[cat]);
    } else {
        const std::string name = utils::DropBrackets(cat);
        if (cache.count(name) > 0) {
            res = static_cast<Cat>(cache[name]);
        } else {
            res = ParseUncached(name);
            if (name != cat) {
                res->RegisterCache(name);
            }
        }
        res->RegisterCache(cat);
        return res;
    }
}


Cat Category::ParseUncached(const std::string& cat) {
    std::string new_cat = cat;
    std::string semantics;
    if (new_cat.back() == '}') {
        int open_idx = new_cat.rfind("{");
        semantics = new_cat.substr(open_idx + 1, new_cat.size() - open_idx - 1);
        new_cat = new_cat.substr(0, open_idx);
    } else {
        semantics = "";
    }
    new_cat = utils::DropBrackets(new_cat);
    int op_idx = utils::FindNonNestedChar(new_cat, slashes);

    if (op_idx == -1) {
        int feat_idx = new_cat.find("[");
        std::string feat_str;
        std::string type = feat_idx == -1 ? new_cat : new_cat.substr(0, feat_idx);
        if (feat_idx > -1)
            feat_str = new_cat.substr(
                    feat_idx + 1, new_cat.find("]", feat_idx) - feat_idx - 1);
        else
            feat_str = "";

        Feat feat = Feature::Parse(feat_str);
        return new AtomicCategory(type, feat, semantics);
    } else {
        Cat left = Category::Parse(new_cat.substr(0, op_idx));
        Slash slash = Slash(new_cat[op_idx]);
        Cat right = Category::Parse(new_cat.substr(op_idx + 1));
        return new Functor(left, slash, right, semantics);
    }
}

Cat Category::Substitute(Feat feat) const {
    if (feat->IsEmpty()) return this;
    return Category::Parse(feat->SubstituteWildcard(str_));
}

Cat Make(Cat left, const Slash& op, Cat right) {
    return Category::Parse(left->WithBrackets() + op.ToStr() + right->WithBrackets());
}

Cat CorrectWildcardFeatures(Cat to_correct, Cat match1, Cat match2) {
    return to_correct->Substitute(match1->GetSubstitution(match2));
}

Cat COMMA       = Category::Parse(",");
Cat SEMICOLON   = Category::Parse(";");
Cat CONJ        = Category::Parse("conj");
Cat N           = Category::Parse("N");
Cat LQU         = Category::Parse("LQU");
Cat LRB         = Category::Parse("LRB");
Cat NP          = Category::Parse("NP");
Cat NPbNP       = Category::Parse("NP\\NP");
Cat PP          = Category::Parse("PP");
Cat PREPOSITION = Category::Parse("PP/NP");
Cat PR          = Category::Parse("PR");
Feat kWILDCARD  = Feature::Parse("X");
Feat kNONE      = Feature::Parse("");
Feat kNB        = Feature::Parse("nb");



} // namespace cat
} // namespace myccg
