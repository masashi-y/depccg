
#include "tree.h"
#include "cat.h"
#include "utils.h"

#define REPEAT(out, size, string) for (int __sp__ = 0; __sp__ < (size); __sp__++) \
                                                    (out) << (string)
#define SPACE(out, size) REPEAT(out, size, " ")

namespace myccg {

const std::string Node::ToStr() const {
    auto res = AUTO(this);
    return res.Get();
}

int Derivation::Visit(const Leaf* leaf) {
    return std::max(std::max(
                lwidth_, 2 + lwidth_ + (int)leaf->GetCategory()->ToStr().size()),
                2 + lwidth_ + (int)leaf->GetWord().size());
}

int Derivation::Visit(const Tree* tree) {
    int lwidth = lwidth_;
    int rwidth = lwidth;
    rwidth = std::max(rwidth, (tree->GetLeftChild()->Accept(*this)));
    if (NULL != tree->GetRightChild()) {
        lwidth_ = rwidth;
        rwidth = std::max(rwidth, (tree->GetRightChild()->Accept(*this)));
    }

    std::string str_res = tree->GetCategory()->ToStr();
    int respadlen = (rwidth - lwidth - str_res.size()) / 2 + lwidth;

    SPACE(out_, lwidth);
    REPEAT(out_, (rwidth - lwidth), "-");
    out_ << tree->GetRule() << std::endl;;
    SPACE(out_, respadlen);
    out_ << str_res << std::endl;;
    return rwidth;
}

void Derivation::Process() {
    std::stringstream cats;
    std::stringstream words;
    std::vector<const Leaf*> leaves = GetLeaves()(tree_);
    for (unsigned i = 0; i < leaves.size(); i++) {
        std::string str_cat = leaves[i]->GetCategory()->ToStr();
        std::string str_word = leaves[i]->GetWord();
        int nextlen = 2 + std::max(
                (unsigned)str_cat.size(), utils::utf8_strlen(str_word));
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

    out_ << cats.str() << std::endl;
    out_ << words.str() << std::endl;
    tree_->Accept(*this);
}

std::string EscapeGTLT(const std::string& input) {
    std::string s(input);
    utils::ReplaceAll(&s, "<", "&lt;");
    utils::ReplaceAll(&s, ">", "&gt;");
    return s;
}

std::string EscapeAMP(const std::string& input) {
    std::string s(input);
    utils::ReplaceAll(&s, "&", "&amp;");
    return s;
}

int XML::Visit(const Leaf* leaf) {
    out_ << "<lf start=\"" << leaf->GetPosition()
         << "\" span=\"" << 1
         << "\" word=\"" << EscapeAMP(leaf->GetWord())
         << "\" lemma=\"" << EscapeAMP(leaf->GetWord())
         << "\" pos=\"DT\" chunk=\"I-NP\" entity=\"O\" cat=\""
         << leaf->GetCategory()->ToStrWithoutFeat() << "\" />"
         << std::endl;
   return 0;
}

int XML::Visit(const Tree* tree) {
   out_ << "<rule type=\""
        << EscapeGTLT(tree->GetRule()->ToStr())
        << "\" cat=\""
        << tree->GetCategory()->ToStrWithoutFeat() << "\">"
        << std::endl;
   tree->GetLeftChild()->Accept(*this);
   if (! tree->IsUnary())
       tree->GetRightChild()->Accept(*this);
   out_ << "</rule>" << std::endl;
   return 0;
}

void ToXML(std::vector<std::shared_ptr<const Node>>& trees, std::ostream& out) {
    std::vector<const Node*> v(trees.size());
    for (unsigned i = 0; i < trees.size(); i++) {
        v[i] = trees[i].get();
    }
    ToXML(v, out);
}

void ToXML(std::vector<const Node*>& trees, std::ostream& out) {
    out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl;
    out << "<?xml-stylesheet type=\"text/xsl\" href=\"candc.xml\"?>" << std::endl;
    out << "<candc>" << std::endl;
    for (auto&& tree: trees) {
        out << "<ccg>" << std::endl;
        out << XML(tree);
        out << "</ccg>" << std::endl;
    }
    out << "</candc>" << std::endl;
}
} // namespace myccg

