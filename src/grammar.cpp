
#include <unordered_set>
#include "grammar.h"

#define F Slash::Fwd()
#define B Slash::Bwd()

namespace myccg {

bool Ja::IsAcceptable(RuleType rule_type, NodeType left, NodeType right) {
    if (right->IsUnary())
        return false;
    if (left->IsUnary())
        return rule_type == FA;
    if (IsPeriod(right->GetCategory()))
        return rule_type == BA;
    // std::cout << left->GetCategory()->ToStrWithoutFeat() << std::endl;
    if (left->GetCategory()->ToStrWithoutFeat() == "(NP/NP)") {
        std::string rcat = right->GetCategory()->ToStrWithoutFeat();
        return ((rcat == "NP" && rule_type == FA) || 
                (rcat == "(NP/NP)" && rule_type == FC));
    }
    if (IsAUX(left->GetCategory()) && IsAUX(right->GetCategory()))
        return false;
    if (IsVerb(left) && IsAUX(right->GetCategory()))
        return rule_type == BC || rule_type == BA;
    if (rule_type == FC) {
        // only allow S/S S/S or NP/NP NP/NP pairs
        Cat lcat = left->GetCategory();
        Cat rcat = right->GetCategory();
        return (IsModifier(lcat) && lcat->GetSlash().IsForward() &&
                IsModifier(rcat) && rcat->GetSlash().IsForward());
    }
    if (rule_type == FX) {
        return (IsVerb(right) &&
                IsAdverb(left->GetCategory()));
    }
    return true;
}

bool IsNormalForm(RuleType rule_type, NodeType left, NodeType right) {
    if ( (left->GetRuleType() == FC ||
                left->GetRuleType() == GFC) &&
            (rule_type == FA ||
             rule_type == FC ||
             rule_type == GFC) )
        return false;
    if ( (right->GetRuleType() == BC ||
                right->GetRuleType() == GBC) &&
            (rule_type == BA ||
             rule_type == BC ||
             rule_type == GBC) )
        return false;
    if ( left->GetRuleType() == UNARY &&
            rule_type == FA &&
            left->GetCategory()->IsForwardTypeRaised() )
        return false;
    if ( right->GetRuleType() == UNARY &&
            rule_type == BA &&
            right->GetCategory()->IsBackwardTypeRaised() )
        return false;
    return true;
}


bool En::IsAcceptable(RuleType rule_type, NodeType left, NodeType right) {
    return IsNormalForm(rule_type, left, right);
}

bool Ja::IsModifier(Cat cat) {
    return (cat->IsFunctor() &&
            !cat->GetLeft()->IsFunctor() &&
            !cat->GetRight()->IsFunctor() &&
            cat->GetLeft()->GetType() == cat->GetRight()->GetType());
}

bool Ja::IsVerb(NodeType tree) {
    Cat cat = tree->GetCategory();
    if (cat->IsFunctor()) {
        return (cat->GetSlash().IsBackward() &&
                cat->Arg(0)->GetType() == "S" &&
                cat->Arg(1)->GetType() == "NP");
    } else {
        return (tree->GetLeftMostChild()->GetCategory()->ToStrWithoutFeat() == "S" &&
                tree->GetLeftMostChild()->GetRuleType() == LEXICON);
    }
}

bool Ja::IsAdjective(Cat cat) {
    return (IsModifier(cat) &&
            cat->GetSlash().IsForward() &&
            cat->Arg(0)->GetType() == "NP" &&
            cat->Arg(1)->GetType() == "NP");
}

bool Ja::IsAdverb(Cat cat) {
    return (IsModifier(cat) &&
            cat->GetSlash().IsForward() &&
            cat->Arg(0)->GetType() == "S" &&
            cat->Arg(1)->GetType() == "S");
}

bool Ja::IsAUX(Cat cat) {
    return (IsModifier(cat) &&
            cat->GetSlash().IsBackward() &&
            cat->Arg(0)->GetType() == "S" &&
            cat->Arg(1)->GetType() == "S");
}

bool Ja::IsPeriod(Cat cat) {
    if (IsModifier(cat) && cat->GetSlash().IsBackward()) {
        Cat first = cat->Arg(0);
        Cat second = cat->Arg(1);
        return (first->GetType() == "S" &&
                second->GetType() == "S" &&
                first->GetFeat()->ContainsKeyValue("fin", "t") &&
                second->GetFeat()->ContainsKeyValue("fin", "f"));
    }
    return false;
}

const std::unordered_set<Cat> En::possible_root_cats = {
    Category::Parse("S[dcl]"),
    Category::Parse("S[wq]"),
    Category::Parse("S[q]"),
    Category::Parse("S[qem]"),
    Category::Parse("NP")
};

const std::vector<Combinator*> En::binary_rules = {
    new ForwardApplication(),
    new BackwardApplication(),
    new GeneralizedForwardComposition<0, FC>(F, F, F),
    new GeneralizedBackwardComposition<0, BC>(F, B, F),
    new GeneralizedForwardComposition<1, GFC>(F, F, F),
    new GeneralizedBackwardComposition<1, GBC>(F, B, F),
    new Conjunction(),
    new RemovePunctuation(false),
    new RemovePunctuationLeft()
};

const std::unordered_set<Cat> Ja::possible_root_cats = {
    Category::Parse("NP[case=nc,mod=nm,fin=f]"),    // 170
    Category::Parse("NP[case=nc,mod=nm,fin=t]"),    // 2972
    Category::Parse("S[mod=nm,form=attr,fin=t]"),   // 2
    Category::Parse("S[mod=nm,form=base,fin=f]"),   // 68
    Category::Parse("S[mod=nm,form=base,fin=t]"),   // 19312
    Category::Parse("S[mod=nm,form=cont,fin=f]"),   // 3
    Category::Parse("S[mod=nm,form=cont,fin=t]"),   // 36
    Category::Parse("S[mod=nm,form=da,fin=f]"),     // 1
    Category::Parse("S[mod=nm,form=da,fin=t]"),     // 68
    Category::Parse("S[mod=nm,form=hyp,fin=t]"),    // 1
    Category::Parse("S[mod=nm,form=imp,fin=f]"),    // 3
    Category::Parse("S[mod=nm,form=imp,fin=t]"),    // 15
    Category::Parse("S[mod=nm,form=r,fin=t]"),      // 2
    Category::Parse("S[mod=nm,form=s,fin=t]"),      // 1
    Category::Parse("S[mod=nm,form=stem,fin=f]"),   // 11
    Category::Parse("S[mod=nm,form=stem,fin=t]")    // 710
};

const std::vector<Combinator*> Ja::binary_rules = {
    new Conjoin(),
    new ForwardApplication(),
    new BackwardApplication(),
    new GeneralizedForwardComposition<0, FC>(F, F, F), // >B
    new GeneralizedBackwardComposition<0, BC>(B, B, B), // <B1
    new GeneralizedBackwardComposition<1, BC>(B, B, B), // <B2
    new GeneralizedBackwardComposition<2, BC>(B, B, B), // <B3
    new GeneralizedBackwardComposition<3, BC>(B, B, B), // <B4
    new GeneralizedForwardComposition<0, FX>(F, B, B), // >Bx1
    new GeneralizedForwardComposition<1, FX>(F, B, B), // >Bx2
    new GeneralizedForwardComposition<2, FX>(F, B, B), // >Bx3
};

}
