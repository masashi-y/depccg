
#include "tree.h"
#include "cat.h"
#include "utils.h"
#include <stack>
#include <algorithm>
#include <string>

namespace myccg {

RuleType GetUnaryRuleType(Cat cat) {
    return cat->IsForwardTypeRaised() ?  FWD_TYPERAISE :
        (cat->IsBackwardTypeRaised() ? BWD_TYPERAISE : UNARY);
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
    Indent();
    out_ << "t(" << PrologCatStr(c)
         << ", \'" << escapeProlog(leaf->GetWord())
         << "\', \'{" << position << ".lemma}"
         << "\', \'{" << position << ".pos}"
         << "\', \'{" << position << ".chunk}"
         << "\', \'{" << position << ".entity}\')";
    position++;
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

} // namespace myccg


