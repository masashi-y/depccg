
#include <iostream>
#include "utils.h"
#include "cat.h"

namespace myccg {
namespace cat {

const char* slashes = "/\\|";

std::unordered_map<std::string, Cat> cache;

int Category::num_cats = 0;

const Slash* Slash::fwd_ptr = new Slash(FwdApp);
const Slash* Slash::bwd_ptr = new Slash(BwdApp);
const Slash* Slash::either_ptr = new Slash(EitherApp);

Cat parse(const std::string& cat) {
    Cat res;
    if (cache.count(cat) != 0) {
        return cache[cat];
    } else {
        const std::string name = utils::DropBrackets(cat);
        if (cache.count(name) != 0) {
            res = cache[name];
        } else {
            res = parse_uncached(name);
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


Cat parse_uncached(const std::string& cat) {
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

        return new AtomicCategory(type, feat, semantics);
    } else {
        Cat left = parse(new_cat.substr(0, op_idx));
        const Slash* slash = Slash::FromStr(new_cat.substr(op_idx, 1));
        Cat right = parse(new_cat.substr(op_idx + 1));
        return new Functor(left, slash, right, semantics);
    }
}

Cat Category::Substitute(const std::string& sub) const {
    return cat::parse(utils::ReplaceAll(str_, kWILDCARD, sub));
}

Cat make(Cat left, const Slash* op, Cat right) {
    return parse(left->WithBrackets() + op->ToStr() + right->WithBrackets());
}

Cat CorrectWildcardFeatures(Cat to_correct, Cat match1, Cat match2) {
    return to_correct->Substitute(match1->GetSubstitution(match2));
}

Cat COMMA       = parse(",");
Cat SEMICOLON   = parse(";");
Cat CONJ        = parse("conj");
Cat N           = parse("N");
Cat LQU         = parse("LQU");
Cat LRB         = parse("LRB");
Cat NP          = parse("NP");
Cat NPbNP       = parse("NP\\NP");
Cat PP          = parse("PP");
Cat PREPOSITION = parse("PP/NP");
Cat PR          = parse("PR");


} // namespace cat
} // namespace myccg

