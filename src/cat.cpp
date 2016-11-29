
#include <iostream>
#include "utils.h"
#include "cat.h"

namespace myccg {
namespace cat {

const char* slashes = "/\\|";

template<> Cat Compose<0>(Cat head, Slash op, Cat tail) {
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

std::unordered_map<std::string, const Cacheable*> cache;

int Cacheable::ids = 0;

Cacheable::Cacheable() {
    #pragma omp atomic capture
    id_ = ids++;
}

void Cacheable::RegisterCache(const std::string& key) const {
    #pragma omp critical(RegisterCache)
    cache.emplace(key, this);
}

Slash Slashes::fwd_ptr = new Slashes(FwdApp);
Slash Slashes::bwd_ptr = new Slashes(BwdApp);
Slash Slashes::either_ptr = new Slashes(EitherApp);

std::string AtomicCategory::ToStrWithoutFeat() const {
    return utils::ReplaceAll(utils::ReplaceAll(ToStr(), "[X]", ""), "[nb]", "");
}

inline Feat FeatFromStr(const std::string& string) {
    Feat res = new FeatureValue(string);
    res->RegisterCache(string);
    return res;
}

Cat Parse_uncached(const std::string& cat);

Cat Parse(const std::string& cat) {
    Cat res;
    if (cache.count(cat) > 0) {
        return static_cast<Cat>(cache[cat]);
    } else {
        const std::string name = utils::DropBrackets(cat);
        if (cache.count(name) > 0) {
            res = static_cast<Cat>(cache[name]);
        } else {
            res = Parse_uncached(name);
            if (name != cat) {
                res->RegisterCache(name);
            }
        }
        res->RegisterCache(cat);
        return res;
    }
}


Cat Parse_uncached(const std::string& cat) {
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

        Feat feat;
        if (cache.count(feat_str) > 0) {
            feat = static_cast<Feat>(cache[feat_str]);
        } else {
            feat = FeatFromStr(feat_str);
        }
        return new AtomicCategory(type, feat, semantics);
    } else {
        Cat left = Parse(new_cat.substr(0, op_idx));
        Slash slash = Slashes::FromStr(new_cat.substr(op_idx, 1));
        Cat right = Parse(new_cat.substr(op_idx + 1));
        return new Functor(left, slash, right, semantics);
    }
}

Cat Category::Substitute(Feat feat) const {
    if (feat->IsEmpty())
        return this;
    return cat::Parse(utils::ReplaceAll(str_, kWILDCARD->ToStr(), feat->ToStr()));
}

Cat Make(Cat left, Slash op, Cat right) {
    return Parse(left->WithBrackets() + op->ToStr() + right->WithBrackets());
}


Cat CorrectWildcardFeatures(Cat to_correct, Cat match1, Cat match2) {
    return to_correct->Substitute(match1->GetSubstitution(match2));
}

Cat COMMA       = Parse(",");
Cat SEMICOLON   = Parse(";");
Cat CONJ        = Parse("conj");
Cat N           = Parse("N");
Cat LQU         = Parse("LQU");
Cat LRB         = Parse("LRB");
Cat NP          = Parse("NP");
Cat NPbNP       = Parse("NP\\NP");
Cat PP          = Parse("PP");
Cat PREPOSITION = Parse("PP/NP");
Cat PR          = Parse("PR");
Feat kWILDCARD  = FeatFromStr("X");
Feat kNONE      = FeatFromStr("");
Feat kNB        = FeatFromStr("nb");


} // namespace cat
} // namespace myccg
