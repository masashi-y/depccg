
#include "grammar.h"

namespace myccg {

std::string En::ResolveCombinatorName(const Node* parse) {
    const Tree* tree;
    if ( (tree = dynamic_cast<const Tree*>(parse)) == nullptr )
        throw std::runtime_error("This node is leaf and does not have combinator!");
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

bool En::IsAcceptableUnary(Cat result, NodeType parse) {
    bool is_not_punct = parse->GetRuleType() != LP && parse->GetRuleType() != RP;
    return is_not_punct || result->IsTypeRaised();
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


bool En::IsAcceptableBinary(RuleType rule_type, NodeType left, NodeType right) {
    return IsNormalForm(rule_type, left, right);
}

const std::unordered_set<Cat> En::possible_root_cats = {
    Category::Parse("S[dcl]"),
    Category::Parse("S[wq]"),
    Category::Parse("S[q]"),
    Category::Parse("S[qem]"),
    Category::Parse("NP")
};

const std::vector<Op> En::binary_rules = {
    new ENForwardApplication(),
    new ENBackwardApplication(),
    new GeneralizedForwardComposition<0, FC>(F, F, F),
    new GeneralizedBackwardComposition<0, BC>(F, B, F),
    new GeneralizedForwardComposition<1, GFC>(F, F, F),
    new GeneralizedBackwardComposition<1, GBC>(F, B, F),
    new Conjunction(),
    new RemovePunctuation(false),
    new RemovePunctuation(true),
    new CommaAndVerbPhraseToAdverb(),
    new ParentheticalDirectSpeech()
};

const std::vector<Op> En::dep_binary_rules = {
    new ENForwardApplication(),
    new ENBackwardApplication(),
    new GeneralizedForwardComposition<0, FC>(F, F, F),
    new GeneralizedBackwardComposition<0, BC>(F, B, F),
    new GeneralizedForwardComposition<1, GFC>(F, F, F),
    new GeneralizedBackwardComposition<1, GBC>(F, B, F),
    new Conjunction(),
    new Conjunction2(),
    new Coordinate(),
    new RemovePunctuation(false),
    new RemovePunctuation(true),
    // new RemovePunctuationLeft(),
    new CommaAndVerbPhraseToAdverb(),
    new ParentheticalDirectSpeech()
};

const std::vector<Op> En::headfirst_binary_rules = {
    HeadFirst(ENForwardApplication()),
    HeadFirst(ENBackwardApplication()),
    HeadFirst(GeneralizedForwardComposition<0, FC>(F, F, F)),
    HeadFirst(GeneralizedBackwardComposition<0, BC>(F, B, F)),
    HeadFirst(GeneralizedForwardComposition<1, GFC>(F, F, F)),
    HeadFirst(GeneralizedBackwardComposition<1, GBC>(F, F, F)),
    HeadFirst(Conjunction()),
    HeadFirst(Conjunction2()),
    HeadFirst(Coordinate()),
    HeadFirst(RemovePunctuation(false)),
    HeadFirst(RemovePunctuation(true)),
    HeadFirst(CommaAndVerbPhraseToAdverb()),
    HeadFirst(ParentheticalDirectSpeech())
};

bool IsNormalFormExtended(
        RuleType rule_type, Cat result, NodeType left, NodeType right) {
    bool is_start_of_sentence = left->GetLeftMostChild()->GetHeadId() == 0;
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

    if ( rule_type == FA &&
            left->GetRuleType() == FWD_TYPERAISE )
        return false;
    if ( rule_type == BA &&
            right->GetRuleType() == BWD_TYPERAISE )
        return false;

    if ( right->GetRuleType() == CONJ &&
            (left->GetRuleType() == FWD_TYPERAISE || left->GetRuleType() == BWD_TYPERAISE) )
        return false;

    if ( (left->GetRuleType() == FC ||
                left->GetRuleType() == GFC) &&
            (rule_type == FA ||
             rule_type == FC) )
        return false;
    if ( (right->GetRuleType() == BC ||
                right->GetRuleType() == GBC) &&
            (rule_type == BA ||
             rule_type == BC) )
        return false;

    if ( left->GetRuleType() == FC &&
            (rule_type == FC || rule_type == GFC) )
        return false;
    if ( right->GetRuleType() == BC &&
            (rule_type == BC || rule_type == GBC) )
        return false;

    if ( left->GetRuleType() == FWD_TYPERAISE &&
            right->GetRuleType() == GBC && rule_type == FC )
        return false;

    if ( right->GetRuleType() == BWD_TYPERAISE &&
            left->GetRuleType() == GFC && rule_type == BC )
        return false;

    // if ( rule_type == LP && ! is_start_of_sentence )
    //     return false;

    // if ( (left->GetRuleType() == LP && rule_type != RP && is_start_of_sentence)
    //         || right->GetRuleType() == LP )
    //     return false;

    //
    // if ( right->GetRuleType() == RP && left->GetRuleType() != LP )
    //     return false;

    if ( rule_type == LP && (right->GetRuleType() == FWD_TYPERAISE || right->GetRuleType() == BWD_TYPERAISE) )
        return false;

    if ( rule_type == RP && (left->GetRuleType() == FWD_TYPERAISE || left->GetRuleType() == BWD_TYPERAISE) )
        return false;

    if ( (rule_type == RP && left->GetRuleType() == FC)
            || (rule_type == LP && right->GetRuleType() == BC) )
        return false;

    if ( left->GetRuleType() == CONJ )
        return false;

    // if ( right->GetRuleType() == CONJ && rule_type != BA )
    //     return false;

    if (rule_type == FA && *right->GetCategory() == *result &&
            right->GetRuleType() == B_MOD )
        return false;

    if ( right->GetRuleType() == FC &&
            rule_type == FA && *right->GetCategory() == *result )
        return false;

    if ( rule_type == COORD
            && ! (right->GetRuleType() == CONJ || 
                right->GetRuleType() == CONJ2 ))
        return false;

    return true;
}

bool En::IsAcceptableBinary(
        RuleType rule_type, Cat result, NodeType left, NodeType right) {
    return true;
    return IsNormalFormExtended(rule_type, result, left, right);
}

}
