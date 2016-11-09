
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

std::size_t Leaf::ShowDerivation(std::size_t lwidth, std::ostream& out) const {
    std::size_t rwidth = lwidth;
    return std::max(std::max(
                rwidth, 2 + lwidth + cat_->ToStr().size()),
                2 + lwidth + word_.size());
}

std::size_t Tree::ShowDerivation(std::size_t lwidth, std::ostream& out) const {
    std::size_t rwidth = lwidth;
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
    for (int i = 0; i < leaves.size(); i++) {
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

#define APPLY_BINARY(comb, left, right) new myccg::tree::Tree( \
        (comb)->Apply((left)->GetCategory(), (right)->GetCategory()), \
        (comb)->HeadIsLeft((left)->GetCategory(), (right)->GetCategory()), \
        (left), (right), (comb));

#define APPLY_UNARY(cat, child) new myccg::tree::Tree((cat), (child))

#define APPLICABLE(comb, left, right) std::cout << #comb": " << \
    (left)->GetCategory()->ToStr() << ", " << (right)->GetCategory()->ToStr() \
    << " --> " << \
    ((comb)->CanApply((left)->GetCategory(), (right)->GetCategory()) ? "OK" : "NO" )\
    << std::endl;

#define TEST(cond)    std::cout << #cond" --> " << ( (cond) ? "yes":"no") << std::endl;

void test()
{
    std::cout << "----" << __FILE__ << "----" << std::endl;

    auto fwd  = new combinator::ForwardApplication();
    auto bwd  = new combinator::BackwardApplication();
    auto Bx   = new combinator::BackwardComposition(cat::Slash::Fwd(), cat::Slash::Bwd(), cat::Slash::Fwd());
    auto conj = new combinator::Conjunction();
    auto un   = new combinator::UnaryRule();
    auto rp   = new combinator::RemovePunctuation(false);

    const Node* leaves[] = {
        new Leaf("this",     cat::parse("NP"),              0),
        new Leaf("is",       cat::parse("(S[dcl]\\NP)/NP"), 1),
        new Leaf("a",        cat::parse("NP[nb]/N"),        2),
        new Leaf("new",      cat::parse("N/N"),             3),
        new Leaf("sentence", cat::parse("N"),               4),
        new Leaf(".",        cat::parse("."),               5),
    };

    const Tree* tree1 = APPLY_BINARY(fwd, leaves[3], leaves[4]);
    const Tree* tree2 = APPLY_BINARY(fwd, leaves[2], tree1);
    const Tree* tree3 = APPLY_BINARY(fwd, leaves[1], tree2);
    const Tree* tree4 = APPLY_BINARY(bwd, leaves[0], tree3);
    const Tree* tree5 = APPLY_BINARY(rp, tree4, leaves[5]);

    APPLICABLE(fwd, leaves[3], leaves[4]);
    APPLICABLE(fwd, leaves[2], tree1);
    APPLICABLE(fwd, leaves[1], tree2);
    APPLICABLE(bwd, leaves[0], tree3);
    APPLICABLE(rp, tree4, leaves[5]);

    print(tree5->ToStr());
    ShowDerivation(tree5);

    const Node* leaves2[] = {
        new Leaf("Ed",      cat::parse("N"),                0),
        new Leaf("saw",     cat::parse("(S[dcl]\\NP)/NP"),  1),
        new Leaf("briefly", cat::parse("(S\\NP)\\(S\\NP)"), 2),
        new Leaf("Tom",     cat::parse("N"),                3),
        new Leaf("and",     cat::parse("conj"),             4),
        new Leaf("Taro",    cat::parse("N"),                5),
        new Leaf(".",       cat::parse("."),                6),
    };
    const Tree* tree2_1 = APPLY_UNARY(cat::parse("NP"), leaves2[0]); // Ed NP
    const Tree* tree2_2 = APPLY_UNARY(cat::parse("NP"), leaves2[3]); // Tom NP
    const Tree* tree2_3 = APPLY_UNARY(cat::parse("NP"), leaves2[5]); // Taro NP
    const Tree* tree2_4 = APPLY_BINARY(Bx, leaves2[1], leaves2[2]); // saw briefly (S[dcl]\NP)/NP
    const Tree* tree2_5 = APPLY_BINARY(conj, leaves2[4], tree2_3); // and Taro NP\NP
    const Tree* tree2_6 = APPLY_BINARY(bwd, tree2_2, tree2_5); // Tom and Taro NP
    const Tree* tree2_7 = APPLY_BINARY(fwd, tree2_4, tree2_6);
    const Tree* tree2_8 = APPLY_BINARY(bwd, tree2_1, tree2_7);
    const Tree* tree2_9 = APPLY_BINARY(rp, tree2_8, leaves2[6]);

    APPLICABLE(Bx, leaves2[1], leaves2[2]);
    APPLICABLE(conj, leaves2[4], tree2_3);
    APPLICABLE(bwd, tree2_2, tree2_5);
    APPLICABLE(fwd, tree2_4, tree2_6);
    APPLICABLE(bwd, tree2_1, tree2_7);
    APPLICABLE(rp, tree2_8, leaves2[6]);

    print(tree2_9->ToStr());
    ShowDerivation(tree2_9);


}

} // namespace tree
} // namespace myccg

