
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

std::unordered_map<std::string, Cat> cache;

int Category::num_cats = 0;

Slash Slashes::fwd_ptr = new Slashes(FwdApp);
Slash Slashes::bwd_ptr = new Slashes(BwdApp);
Slash Slashes::either_ptr = new Slashes(EitherApp);

std::string AtomicCategory::ToStrWithoutFeat() const {
    return utils::ReplaceAll(utils::ReplaceAll(ToStr(), "[X]", ""), "[nb]", "");
}

Cat Parse(const std::string& cat) {
    Cat res;
    if (cache.count(cat) != 0) {
        return cache[cat];
    } else {
        const std::string name = utils::DropBrackets(cat);
        if (cache.count(name) != 0) {
            res = cache[name];
        } else {
            res = Parse_uncached(name);
            if (name != cat) {
                #pragma omp critical(parse_name)
                cache.emplace(name, res);
            }
        }
        #pragma omp critical(parse_cat)
        cache.emplace(cat, res);
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
        std::string feat;
        std::string type = feat_idx == -1 ? new_cat : new_cat.substr(0, feat_idx);
        if (feat_idx > -1)
            feat = new_cat.substr(feat_idx + 1, new_cat.find("]", feat_idx) - feat_idx - 1);
        else
            feat = "";

        return new AtomicCategory(type, new FeatureValue(feat), semantics);
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


} // namespace cat
} // namespace myccg
