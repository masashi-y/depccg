
#include "tree.h"
#include "cat.h"

#define REPEAT(out, size, string) for (int __sp__ = 0; __sp__ < (size); __sp__++) \
                                                    (out) << (string)
#define SPACE(out, size) REPEAT(out, size, " ")

namespace myccg {
namespace tree {

std::vector<const Leaf*> GetLeaves(const Tree* tree) {
    std::vector<const Leaf*> res;
    tree->GetLeaves(&res);
    return res;
}

int Leaf::ShowDerivation(int lwidth, std::ostream& out) const {
    int rwidth = lwidth;
    return std::max(std::max(
                rwidth, 2 + lwidth + (int)cat_->ToStr().size()),
                2 + lwidth + (int)word_.size());
}

int Tree::ShowDerivation(int lwidth, std::ostream& out) const {
    int rwidth = lwidth;
    rwidth = std::max(rwidth, (lchild_->ShowDerivation(rwidth, out)));
    if (NULL != rchild_)
        rwidth = std::max(rwidth, (rchild_->ShowDerivation(rwidth, out)));

    std::string str_res = cat_->ToStr();
    int respadlen = (rwidth - lwidth - str_res.size()) / 2 + lwidth;

    SPACE(out, lwidth);
    REPEAT(out, (rwidth - lwidth), "-");
    out << rule_->ToStr() << std::endl;;
    SPACE(out, respadlen);
    out << str_res << std::endl;;
    return rwidth;
}

void ShowDerivation(const Tree* tree, std::ostream& out) {
    std::stringstream cats;
    std::stringstream words;
    std::vector<const Leaf*> leaves = GetLeaves(tree);
    for (unsigned i = 0; i < leaves.size(); i++) {
        std::string str_cat = leaves[i]->GetCategory()->ToStr();
        std::string str_word = leaves[i]->GetWord();
        int nextlen = 2 + std::max(str_cat.size(), str_word.size());
        int lcatlen = (nextlen - str_cat.size()) / 2;
        int rcatlen = lcatlen + (nextlen - str_cat.size()) % 2;
        int lwordlen = (nextlen - str_word.size()) / 2;
        int rwordlen = lwordlen + (nextlen - str_word.size()) % 2;
        SPACE(cats, lcatlen);
        cats << str_cat;
        SPACE(cats, rcatlen);
        SPACE(words, lwordlen);
        words << str_word;
        SPACE(words, rwordlen);
    }

    out << cats.str() << std::endl;
    out << words.str() << std::endl;
    tree->ShowDerivation(0, out); 
}

void ShowDerivation(std::shared_ptr<const Node> tree, std::ostream& out) {
    ShowDerivation(static_cast<const Tree*>(tree.get()), out);
}

} // namespace tree
} // namespace myccg

