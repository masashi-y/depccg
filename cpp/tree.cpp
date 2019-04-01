
#include "tree.h"
#include "cat.h"
#include "utils.h"
#include <stack>
#include <algorithm>
#include <string>

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

int Derivation::Visit(const CTree* tree) {
    int lwidth = lwidth_;
    int rwidth = lwidth;
    rwidth = std::max(rwidth, (tree->GetLeftChild()->Accept(*this)));
    if (NULL != tree->GetRightChild()) {
        lwidth_ = rwidth;
        rwidth = std::max(rwidth, (tree->GetRightChild()->Accept(*this)));
    }

    std::string str_res = feat_ ? tree->GetCategory()->ToStr()
                                    : tree->GetCategory()->ToStrWithoutFeat();
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

    out_ << words.str() << std::endl;
    out_ << cats.str() << std::endl;
    tree_->Accept(*this);
}

int AUTO::Visit(const CTree* tree) {
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

// std::string JaResolveCombinatorName(const Combinator* comb) {
// }

int JaCCG::Visit(const CTree* tree) {
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

int PyXML::Visit(const Leaf* leaf) {
    Cat c = leaf->GetCategory();
    int position = leaf->GetPosition();
    out_ << "<lf start=\"" << position
         << "\" span=\"" << 1
         << "\" word=\"" << EscapeAMP(leaf->GetWord())
         << "\" lemma=\"{" << position << ".lemma}\" "
         << "pos=\"{" << position << ".pos}\" "
         << "chunk=\"{" << position << ".chunk}\" "
         << "entity=\"{" << position << ".entity}\" cat=\""
         << ( feat_ ? c->ToStr() : c->ToStrWithoutFeat() ) << "\" />"
         << std::endl;
   return 0;
}

std::string ToCAndCStr(const CTree* tree) {
    if (tree->IsUnary()) {
        Cat init = tree->GetLeftChild()->GetCategory();
        if ((init->Matches(CCategory::Parse("NP")) ||
                init->Matches(CCategory::Parse("PP")))
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

int XMLVisitBase(FormatVisitor* visitor,
        const CTree* tree, std::ostream& out, bool feat) {
    Cat c = tree->GetCategory();
    out << "<rule type=\""
         << ToCAndCStr(tree)
         << "\" cat=\""
         << ( feat ? c->ToStr() : c->ToStrWithoutFeat() ) << "\">"
         << std::endl;
    tree->GetLeftChild()->Accept(*visitor);
    if (! tree->IsUnary())
        tree->GetRightChild()->Accept(*visitor);
    out << "</rule>" << std::endl;
    return 0;
}

int XML::Visit(const CTree* tree) {
    XMLVisitBase(this, tree, out_, feat_);
    return 0;
}

int PyXML::Visit(const CTree* tree) {
    XMLVisitBase(this, tree, out_, feat_);
    return 0;
}

class PrologCatStr: public CatVisitor {
public:
    PrologCatStr(Cat cat) { cat->Accept(*this); }
    friend std::ostream& operator<<(std::ostream& ost, const PrologCatStr& p) {
        ost << p.stack.top();
        return ost;
    }

    std::string Get() { return stack.top(); }

    int Visit(const AtomicCategory* cat) {
        std::string cstr = cat->ToStrWithoutFeat();
        std::transform(cstr.begin(), cstr.end(), cstr.begin(), ::tolower);
        if (false) {
        } else if (*cat == *CCategory::Parse(".")) {
            stack.push("period");
        } else if (*cat == *CCategory::Parse(",")) {
            stack.push("comma");
        } else if (*cat == *CCategory::Parse(":")) {
            stack.push("colon");
        } else if (*cat == *CCategory::Parse(";")) {
            stack.push("semicolon");
        } else if (cat->GetFeat()->IsEmpty()) {
            stack.push(cstr);
        } else {
            std::string feat = cat->GetFeat()->ToStr();
            stack.push(
                cstr + ":" + feat.substr(1, feat.size() - 2));
        }
        return 0;
    }

    int Visit(const Functor* cat) {
        std::string left, right;
        cat->GetLeft()->Accept(*this);
        cat->GetRight()->Accept(*this);
        right = stack.top();
        stack.pop();
        left = stack.top();
        stack.pop();
        stack.push(
            "(" + left + cat->GetSlash().ToStr() + right + ")");
        return 0;
    }

    std::stack<std::string> stack;
};

std::string escapeProlog(std::string in) {
    if (in.find("'") != std::string::npos)
        utils::ReplaceAll(&in, "'", "\\'");
    return in;
}

int Prolog::Visit(const Leaf* leaf) {
    Cat c = leaf->GetCategory();
    int position = 1 + leaf->GetPosition();
    Indent();
    out_ << "t(" << PrologCatStr(c)
         << ", \'" << escapeProlog(leaf->GetWord())
         << "\', \'{" << position << ".lemma}"
         << "\', \'{" << position << ".pos}"
         << "\', \'{" << position << ".chunk}"
         << "\', \'{" << position << ".entity}\')";
   return 0;
}

// TODO: Fix this
int Prolog::Visit(const CTree* tree) {
    bool child = false, arg = false, noise = false, conj2 = false;
    Indent();
    if (tree->IsUnary()) {
        out_ << "lx(";
        child = true;
    } else {
        switch (tree->GetRule()->GetRuleType()) {
            case FA: out_ << "fa("; break;
            case COORD:
            case BA: out_ << "ba("; break;
            case FX:
            case FC: out_ << "fc("; break;
            case BX:
            case BC: out_ << "bxc("; break;
            case GFC: out_ << "gfc("; break;
            case GBC: out_ << "gbx("; break;
            case RP: out_ << "rp("; break;
            case LP:
            case NOISE: out_ << "lx(";
                        noise = true;
                        break;
            case CONJ: out_ << "conj(";
                        arg = true;
                        break;
            case CONJ2: out_ << "lx(";
                        conj2 = true;
                        break;
            default:
                out_ << "other(";
        }
    }
    out_ << PrologCatStr(tree->GetCategory()) << ", ";
    if (conj2) {
        std::string c = PrologCatStr(tree->GetRightChild()->GetCategory()).Get();
        out_ << c << "\\" << c << ", ";
        out_ << std::endl;
        depth++;
        Indent();
        out_ << "conj(" << c << "\\" << c << ", " << c << ", ";
    }
    if (noise) {
        out_ << PrologCatStr(tree->GetRightChild()->GetCategory()) << ", ";
        out_ << std::endl;
        depth++;
        Indent();
        out_ << "lp("
             << PrologCatStr(tree->GetRightChild()->GetCategory()) << ", ";
    }
    if (child)
        out_ << PrologCatStr(tree->GetLeftChild()->GetCategory()) << ", ";
    if (arg)
        out_ << PrologCatStr(tree->GetCategory()->GetLeft()) << ", ";
    out_ << std::endl;
    depth++;
    tree->GetLeftChild()->Accept(*this);
    if (! tree->IsUnary()) {
        out_ << "," << std::endl;
        tree->GetRightChild()->Accept(*this);
    }
    out_ << ")";
    depth--;
    if (conj2 || noise) {
        out_ << ")";
        depth--;
    }
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

int CoNLL::Visit(const CTree* tree) {
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

std::string EnResolveCombinatorName(const Node* parse) {
    const CTree* tree;
    if ( (tree = dynamic_cast<const CTree*>(parse)) == nullptr )
        throw std::runtime_error("This node is leaf and does not have combinator!");
    if (tree->IsUnary()) {
        Cat init = tree->GetLeftChild()->GetCategory();
        if ((init->Matches(CCategory::Parse("NP")) ||
                init->Matches(CCategory::Parse("PP")))
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


std::string JaResolveCombinatorName(const Node* parse) {
   const CTree* tree;
   if ( (tree = dynamic_cast<const CTree*>(parse)) == nullptr )
       throw std::runtime_error("This node is leaf and does not have combinator!");
    Cat child;
    Feat ch_feat;
    if ( tree->IsUnary() ) {
        child = tree->GetLeftChild()->GetCategory();
        ch_feat = child->Arg(0)->GetFeat();
        if ( ch_feat->ContainsKeyValue("mod", "adn") ) {
            if ( child->StripFeat()->ToStr() == "S" ) {
                return "ADNext";
            } else {
                return "ADNint";
            }
        } else if ( ch_feat->ContainsKeyValue("mod", "adv") ) {
            if ( tree->GetCategory()->StripFeat()->ToStr() == "(S\\NP)/(S\\NP)" ) {
                return "ADV1";
            } else {
                return "ADV0";
            }
        }
    }
    return tree->GetRule()->ToStr();
}

} // namespace myccg


