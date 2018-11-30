
#include <memory>
#include "grammar.h"

namespace myccg {

std::string Ja::ResolveCombinatorName(const Node* parse) {
   const Tree* tree;
   if ( (tree = dynamic_cast<const Tree*>(parse)) == nullptr )
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

bool Ja::IsModifier(Cat cat) {
    return (cat->IsFunctor() &&
            !cat->GetLeft()->IsFunctor() &&
            !cat->GetRight()->IsFunctor() &&
            cat->GetLeft()->GetType() == cat->GetRight()->GetType());
}

bool Ja::IsVerb(const Node* tree) {
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

bool Ja::IsVerb(NodeType tree) {
    return IsVerb(tree.get());
}

// std::cerr << cat << " " << (res ? "yes" : "no") << std::endl;
bool Ja::IsAdjective(Cat cat) {
    return cat->StripFeat() == CCategory::Parse("NP/NP");
}

bool Ja::IsAdverb(Cat cat) {
    return cat->StripFeat() == CCategory::Parse("S/S");
}

bool Ja::IsAuxiliary(Cat cat) {
    return cat->StripFeat() == CCategory::Parse("S\\S");
}

bool Ja::IsPunct(NodeType tree) {
    if ( tree->GetLength() != 1 )
        return false;
    std::string w = tree->GetWord();
    return (w == "。" || w == "." || w == "．" ||
            w == "," || w == "，" || w == "、");
}

bool Ja::IsComma(NodeType tree) {
    if ( tree->GetLength() != 1 )
        return false;
    std::string w = tree->GetWord();
    return (w == "," || w == "，" || w == "、");
}

bool Ja::IsPeriod(NodeType tree) {
    if ( tree->GetLength() != 1 )
        return false;
    std::string w = tree->GetWord();
    return (w == "。" || w == "." || w == "．");
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

bool IsWord(NodeType tree, const std::string& str) {
    if ( ! tree->IsLeaf() )
       return false;
    return tree->GetWord() == str;
}

bool Ja::IsAcceptableUnary(Cat result, NodeType parse) {
    return true;
}

bool Ja::IsAcceptableBinary(
        RuleType rule_type, Cat result, NodeType left, NodeType right) {
    return IsAcceptableBinary(rule_type, left, right);
}

bool Ja::IsAcceptableBinary(RuleType rule_type, NodeType left, NodeType right) {

    // unary subtree is only allowed as left child
    if (right->IsUnary())
        return false;

    // unary always results in S/S or NP/NP and should FA S or NP
    if (left->IsUnary() && ! (rule_type == FA || rule_type == FX))
        return false;

    // left is never punct
    if (IsPunct(left))
        return false;

    return true;
    // { { X } { L comma } } never be right argument
    // if (! right->IsLeaf() && ! right->IsUnary() && IsComma(right->GetRightChild()))
    //     return false;

    // disallow {<B S\S S\S }
    // if (IsAuxiliary(left->GetCategory()) && IsAuxiliary(right->GetCategory()))
    //     return false;

    // try to combine period as late as possible
    // if (IsPeriod(right))
    //     return is_start_of_sentence && rule_type == BA;

    if (IsWord(left, "」") || IsWord(left, "』"))
        return false;

    if (IsWord(right, "「") || IsWord(right, "『"))
        return false;

    // paranthesis
    if (IsWord(right, "」"))
        return ! left->IsLeaf() && IsWord(left->GetLeftChild(), "「");

    if (IsWord(right, "』"))
        return ! left->IsLeaf() && IsWord(left->GetLeftChild(), "『");

    // disallow for verb to combine with any other before composing with auxiliaries
    // e.g. {< {< NP S\NP } S\S }
    if (IsAuxiliary(right->GetCategory()) &&
            ! left->IsLeaf() && ! left->IsUnary() &&
            IsVerb(left->GetRightChild()->GetHeadLeaf())) {
        return false;
    }

    return true;
    // allow only {> NP/NP NP } or {>B NP/NP NP/NP }
    if (left->GetCategory()->ToStrWithoutFeat() == "(NP/NP)") {
        std::string rcat = right->GetCategory()->ToStrWithoutFeat();
        return ((rcat == "NP" && rule_type == FA) || 
                (rcat == "(NP/NP)" && rule_type == FC));
    }

    if (IsVerb(left) && IsAuxiliary(right->GetCategory()))
        return rule_type == BC || rule_type == BA;

    // only allow S/S S/S or NP/NP NP/NP pairs in FC
    if (rule_type == FC) {
        Cat lcat = left->GetCategory();
        Cat rcat = right->GetCategory();
        return (IsModifier(lcat) && lcat->GetSlash().IsForward() &&
                IsModifier(rcat) && rcat->GetSlash().IsForward());
    }

    // only allow {>Bx S/S S\NP } in >Bx
    // if (IsVerb(right) && IsAdverb(left->GetCategory()))
        // return rule_type == FX;

    return true;
}

const std::unordered_set<Cat> Ja::possible_root_cats = {
    CCategory::Parse("NP[case=nc,mod=nm,fin=f]"),    // 170
    CCategory::Parse("NP[case=nc,mod=nm,fin=t]"),    // 2972
    CCategory::Parse("S[mod=nm,form=attr,fin=t]"),   // 2
    CCategory::Parse("S[mod=nm,form=base,fin=f]"),   // 68
    CCategory::Parse("S[mod=nm,form=base,fin=t]"),   // 19312
    CCategory::Parse("S[mod=nm,form=cont,fin=f]"),   // 3
    CCategory::Parse("S[mod=nm,form=cont,fin=t]"),   // 36
    CCategory::Parse("S[mod=nm,form=da,fin=f]"),     // 1
    CCategory::Parse("S[mod=nm,form=da,fin=t]"),     // 68
    CCategory::Parse("S[mod=nm,form=hyp,fin=t]"),    // 1
    CCategory::Parse("S[mod=nm,form=imp,fin=f]"),    // 3
    CCategory::Parse("S[mod=nm,form=imp,fin=t]"),    // 15
    CCategory::Parse("S[mod=nm,form=r,fin=t]"),      // 2
    CCategory::Parse("S[mod=nm,form=s,fin=t]"),      // 1
    CCategory::Parse("S[mod=nm,form=stem,fin=f]"),   // 11
    CCategory::Parse("S[mod=nm,form=stem,fin=t]")    // 710
};

const std::vector<Op> Ja::binary_rules = {
    new Conjoin(),
    new JAForwardApplication(),
    new JABackwardApplication(),
    new JAGeneralizedForwardComposition<0, FC>(F, F, F, ">B"),
    new JAGeneralizedBackwardComposition<0, BC>(B, B, B, "<B1"),
    new JAGeneralizedBackwardComposition<1, BC>(B, B, B, "<B2"),
    new JAGeneralizedBackwardComposition<2, BC>(B, B, B, "<B3"),
    new JAGeneralizedBackwardComposition<3, BC>(B, B, B, "<B4"),
    new JAGeneralizedForwardComposition<0, FX>(F, B, B, ">Bx1"),
    new JAGeneralizedForwardComposition<1, FX>(F, B, B, ">Bx2"),
    new JAGeneralizedForwardComposition<2, FX>(F, B, B, ">Bx3"),
};

const std::vector<Op> Ja::headfinal_binary_rules = {
    HeadFinal(Conjoin()),
    HeadFinal(JAForwardApplication()),
    HeadFinal(JABackwardApplication()),
    HeadFinal(JAGeneralizedForwardComposition<0, FC>(F, F, F, ">B")),
    HeadFinal(JAGeneralizedBackwardComposition<0, BC>(B, B, B, "<B1")),
    HeadFinal(JAGeneralizedBackwardComposition<1, BC>(B, B, B, "<B2")),
    HeadFinal(JAGeneralizedBackwardComposition<2, BC>(B, B, B, "<B3")),
    HeadFinal(JAGeneralizedBackwardComposition<3, BC>(B, B, B, "<B4")),
    HeadFinal(JAGeneralizedForwardComposition<0, FX>(F, B, B, ">Bx1")),
    HeadFinal(JAGeneralizedForwardComposition<1, FX>(F, B, B, ">Bx2")),
    HeadFinal(JAGeneralizedForwardComposition<2, FX>(F, B, B, ">Bx3")),
};

const std::vector<Op> Ja::cg_binary_rules = {
    HeadFinal(JAForwardApplication()),
    HeadFinal(JABackwardApplication()),
};

}
