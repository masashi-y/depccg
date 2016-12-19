
#include "parser_tools.h"

namespace myccg {
namespace parser {

bool IsModifier(Cat cat) {
    return (cat->IsFunctor() &&
            !cat->GetLeft()->IsFunctor() &&
            !cat->GetRight()->IsFunctor() &&
            cat->GetLeft()->GetType() == cat->GetRight()->GetType());
}

bool IsVerb(NodeType tree) {
    Cat cat = tree->GetCategory();
    if (cat->IsFunctor()) {
        return (cat->GetSlash().IsBackward() &&
                cat->Arg(0)->GetType() == "S" &&
                cat->Arg(1)->GetType() == "NP");
    } else {
        return (tree->GetLeftMostChild()->GetCategory()->ToStrWithoutFeat() == "S" &&
                tree->GetLeftMostChild()->GetRuleType() == combinator::LEXICON);
    }
}

bool IsAdjective(Cat cat) {
    return (IsModifier(cat) &&
            cat->GetSlash().IsForward() &&
            cat->Arg(0)->GetType() == "NP" &&
            cat->Arg(1)->GetType() == "NP");
}

bool IsAdverb(Cat cat) {
    return (IsModifier(cat) &&
            cat->GetSlash().IsForward() &&
            cat->Arg(0)->GetType() == "S" &&
            cat->Arg(1)->GetType() == "S");
}

bool IsAUX(Cat cat) {
    return (IsModifier(cat) &&
            cat->GetSlash().IsBackward() &&
            cat->Arg(0)->GetType() == "S" &&
            cat->Arg(1)->GetType() == "S");
}

bool IsPeriod(Cat cat) {
#ifdef JAPANESE
    if (IsModifier(cat) && cat->GetSlash().IsBackward()) {
        Cat first = cat->Arg(0);
        Cat second = cat->Arg(1);
        return (first->GetType() == "S" &&
                second->GetType() == "S" &&
                first->GetFeat()->ContainsKeyValue("fin", "t") &&
                second->GetFeat()->ContainsKeyValue("fin", "f"));
    }
#endif
    return false;
}

}
}
