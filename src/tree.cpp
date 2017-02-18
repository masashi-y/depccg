
#include "tree.h"
#include "cat.h"
#include "utils.h"
#include "grammar.h"

#define REPEAT(out, size, string) for (int __sp__ = 0; __sp__ < (size); __sp__++) \
                                                    (out) << (string)
#define SPACE(out, size) REPEAT(out, size, " ")

namespace myccg {

RuleType GetUnaryRuleType(Cat cat) {
    return cat->IsForwardTypeRaised() ?  FWD_TYPERAISE :
        (cat->IsBackwardTypeRaised() ? BWD_TYPERAISE : UNARY);
}

const std::string Node::ToStr() const {
    AUTO res(this);
    return res.Get();
}

int Derivation::Visit(const Leaf* leaf) {
    std::string cat_str = feat_ ? leaf->GetCategory()->ToStr()
                                : leaf->GetCategory()->ToStrWithoutFeat();
    return std::max(std::max(
                lwidth_, 2 + lwidth_ + (int)cat_str.size()),
                2 + lwidth_ + (int)utils::utf8_strlen(leaf->GetWord()));
}

int Derivation::Visit(const Tree* tree) {
    int lwidth = lwidth_;
    int rwidth = lwidth;
    rwidth = std::max(rwidth, (tree->GetLeftChild()->Accept(*this)));
    if (NULL != tree->GetRightChild()) {
        lwidth_ = rwidth;
        rwidth = std::max(rwidth, (tree->GetRightChild()->Accept(*this)));
    }

    std::string str_res = feat_ ? tree->GetCategory()->ToStr()
                                    : tree->GetCategory()->ToStrWithoutFeat();
    if ( !tree->IsUnary() )
        str_res += tree->HeadIsLeft() ? " ->" : " <-";
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
        std::string str_cat = feat_ ? leaves[i]->GetCategory()->ToStr()
                                    : leaves[i]->GetCategory()->ToStrWithoutFeat();
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

int AUTO::Visit(const Tree* tree) {
    out_ << "(<T "
         << tree->GetCategory() << " "
         << (tree->HeadIsLeft() ? "0 " : "1 ")
         << (tree->IsUnary() ? "1" : "2")
         << "> ";
    tree->GetLeftChild()->Accept(*this);
    if (! tree->IsUnary()) {
        out_ << " ";
        tree->GetRightChild()->Accept(*this);
        // if (IsNormalFormExtended(tree->GetRuleTypeOld(),
        //             tree->GetCategory(), tree->GetLeftChild(),
        //             tree->GetRightChild())) {
        //     std::cerr << "NORMAL" << std::endl;
        // } else {
        //     std::cerr << "NOT" << std::endl;
        // }
    }
    out_ << " )";
    return 0;
}

int AUTO::Visit(const Leaf* leaf) {
    std::string pos = "POS";
    out_ << "(<L "
         << leaf->GetCategory() << " "
         << pos << " "
         << pos << " "
         << leaf->GetWord() << " "
         << leaf->GetCategory() << ">)";
    return 0;
}

std::string JaResolveCombinatorName(const Combinator* comb) {
}

int JaCCG::Visit(const Tree* tree) {
    out_ << "{";
    if (tree->IsUnary()) {
        Cat child = tree->GetLeftChild()->GetCategory();
        Feat ch_feat = child->Arg(0)->GetFeat();
        if (ch_feat->ContainsKeyValue("mod", "adn")) {
            if (child->StripFeat()->ToStr() == "S")
                out_ << "ADNext ";
            else
                out_ << "ADNint ";
        } else if (ch_feat->ContainsKeyValue("mod", "adv")) {
            if (tree->GetCategory()->StripFeat()->ToStr() == "(S\\NP)/(S\\NP)")
                out_ << "ADV1 ";
            else
                out_ << "ADV0 ";
        } else {
            throw std::runtime_error("JaCCG::Visit: " + child->ToStr());
        }

    } else {
         out_ << tree->GetRule() << " ";
    }
    out_ << tree->GetCategory() << " ";
    tree->GetLeftChild()->Accept(*this);
    if (! tree->IsUnary()) {
        out_ << " ";
        tree->GetRightChild()->Accept(*this);
    }
    out_ << "}";
    return 0;
}

int JaCCG::Visit(const Leaf* leaf) {
    out_ << "{"
         << leaf->GetCategory() << " "
         << leaf->GetWord() << "/"
         << leaf->GetWord() << "/"
         << "**"
         << "}";
    return 0;
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
    Cat c = leaf->GetCategory();
    out_ << "<lf start=\"" << leaf->GetPosition()
         << "\" span=\"" << 1
         << "\" word=\"" << EscapeAMP(leaf->GetWord())
         << "\" lemma=\"" << EscapeAMP(leaf->GetWord())
         << "\" pos=\"DT\" chunk=\"I-NP\" entity=\"O\" cat=\""
         << ( feat_ ? c->ToStr() : c->ToStrWithoutFeat() ) << "\" />"
         << std::endl;
   return 0;
}

std::string ToCAndCStr(const Tree* tree) {
    if (tree->IsUnary()) {
        Cat init = tree->GetLeftChild()->GetCategory();
        if ((init->Matches(Category::Parse("NP")) ||
                init->Matches(Category::Parse("PP")))
                && tree->GetCategory()->IsTypeRaised())
            return "tr";
        else
            return "lex";
    }
    switch (tree->GetRule()->GetRuleType()) {
        case FA: return "fa";
        case BA: return "ba";
        case FC: return "fc";
        case BC: return "bx";
        case GFC: return "gfc";
        case GBC: return "gbx";
        case FX: return "fx";
        case BX: return "bx";
        case CONJ: return "conj";
        case CONJ2: return "conj";
        case COORD: return "ba";
        case RP: return "rp";
        case LP: return "lp";
        case NOISE: return "lp";
        default:
            return "other";
    }
}

int XML::Visit(const Tree* tree) {
    Cat c = tree->GetCategory();
    out_ << "<rule type=\""
         << ToCAndCStr(tree)
         << "\" cat=\""
         << ( feat_ ? c->ToStr() : c->ToStrWithoutFeat() ) << "\">"
         << std::endl;
    tree->GetLeftChild()->Accept(*this);
    if (! tree->IsUnary())
        tree->GetRightChild()->Accept(*this);
    out_ << "</rule>" << std::endl;
    return 0;
}

void ToXML(std::vector<std::shared_ptr<const Node>>& trees, bool feat, std::ostream& out) {
    std::vector<const Node*> v(trees.size());
    for (unsigned i = 0; i < trees.size(); i++) {
        v[i] = trees[i].get();
    }
    ToXML(v, feat, out);
}

void ToXML(std::vector<const Node*>& trees, bool feat, std::ostream& out) {
    out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl;
    out << "<?xml-stylesheet type=\"text/xsl\" href=\"candc.xml\"?>" << std::endl;
    out << "<candc>" << std::endl;
    for (auto&& tree: trees) {
        out << "<ccg>" << std::endl;
        out << XML(tree, feat);
        out << "</ccg>" << std::endl;
    }
    out << "</candc>" << std::endl;
}

CoNLL::CoNLL(const Node* tree)
    : id_(0),
      tree_(tree),
      length_(tree->GetLength()),
      heads_(new int[length_]),
      leaves_(new const Leaf*[length_]) { Process(); }

CoNLL::CoNLL(NodeType tree)
    : id_(0),
      tree_(tree.get()),
      length_(tree->GetLength()),
      heads_(new int[length_]),
      leaves_(new const Leaf*[length_]) { Process(); }

CoNLL::~CoNLL() {
    delete[] heads_;
    delete[] leaves_;
}

void CoNLL::Process() {
    for (int i = 0; i < length_; i++) heads_[i] = 0;
    tree_->Accept(*this);
    const Leaf* leaf;
    for (int i = 0; i < length_; i++) {
        leaf = leaves_[i];
        out_ << i+1 << "\t"
             << leaf->GetWord() << "\t"
             << leaf->GetCategory() << "\t"
             << heads_[i] << std::endl;
    } 
}

int CoNLL::Visit(const Tree* tree) {
   int lhead = tree->GetLeftChild()->Accept(*this);
   if (! tree->IsUnary()) {
       int rhead = tree->GetRightChild()->Accept(*this);
       int head =  tree->HeadIsLeft() ? lhead : rhead;
       int child = tree->HeadIsLeft() ? rhead : lhead;
       heads_[child] = head + 1;
       return head;
   }
    return lhead;
}

int CoNLL::Visit(const Leaf* leaf) {
    leaves_[id_++] = leaf;
    return leaf->GetHeadId();
}

} // namespace myccg


