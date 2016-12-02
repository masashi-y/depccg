
#include "tree.h"
#include "cat.h"
#include "utils.h"

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
#ifdef JAPANESE
        int nextlen = 2 + std::max(
                (unsigned)str_cat.size(), utils::utf8_strlen(str_word));
#else
        int nextlen = 2 + std::max(str_cat.size(), str_word.size());
#endif
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

std::string EscapeGTLT(const std::string& input) {
    std::string s(input);
    utils::ReplaceAll(&s, "<", "&lt;");
    utils::ReplaceAll(&s, ">", "&gt;");
    return s;
}

void Leaf::ToXML(std::ostream& out) const {
    out << "<lf start=\"" << position_
        << "\" span=\"" << 1
        << "\" word=\"" << word_
        << "\" lemma=\"" << word_
        << "\" pos=\"DT\" chunk=\"I-NP\" entity=\"O\" cat=\""
        << cat_->ToStrWithoutFeat() << "\" />"
        << std::endl;
}
void Tree::ToXML(std::ostream& out) const {
   out << "<rule type=\""
       << EscapeGTLT(rule_->ToStr())
       << "\" cat=\""
       << cat_->ToStrWithoutFeat() << "\">"
       << std::endl;
   lchild_->ToXML(out);
   if (NULL != rchild_)
       rchild_->ToXML(out);
   out << "</rule>" << std::endl;
}

void ToXML(std::vector<std::shared_ptr<const Node>>&
        trees, std::ostream& out) {
    std::vector<const Tree*> v(trees.size());
    for (unsigned i = 0; i < trees.size(); i++) {
        v[i] = static_cast<const Tree*>(trees[i].get());
    }
    ToXML(v, out);
}

void ToXML(std::vector<const Tree*>& trees, std::ostream& out) {
    out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl;
    out << "<?xml-stylesheet type=\"text/xsl\" href=\"candc.xml\"?>" << std::endl;
    out << "<candc>" << std::endl;
    for (auto&& tree: trees) {
        out << "<ccg>" << std::endl;
        tree->ToXML(out);
        out << "</ccg>" << std::endl;
    }
    out << "</candc>" << std::endl;
}
} // namespace tree
} // namespace myccg

