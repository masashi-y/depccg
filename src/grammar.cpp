
#include <unordered_set>
#include "grammar.h"

#define F cat::Slash::Fwd()
#define B cat::Slash::Bwd()

namespace myccg {
namespace grammar {

const std::unordered_set<Cat> en::possible_root_cats = {
    cat::Category::Parse("S[dcl]"),
    cat::Category::Parse("S[wq]"),
    cat::Category::Parse("S[q]"),
    cat::Category::Parse("S[qem]"),
    cat::Category::Parse("NP")
};

const std::vector<Combinator*> en::binary_rules = {
    new ForwardApplication(),
    new BackwardApplication(),
    new GeneralizedForwardComposition<0>(F, F, F),
    new GeneralizedBackwardComposition<0>(F, B, F),
    new GeneralizedForwardComposition<1>(F, F, F),
    new GeneralizedBackwardComposition<1>(F, B, F),
    new Conjunction(),
    new RemovePunctuation(false),
    new RemovePunctuationLeft()
};

const std::unordered_set<Cat> ja::possible_root_cats = {
    cat::Category::Parse("NP[case=nc,mod=nm,fin=f]{I1}"),    // 170
    cat::Category::Parse("NP[case=nc,mod=nm,fin=t]{I1}"),    // 2972
    cat::Category::Parse("S[mod=nm,form=attr,fin=t]{I1}"),   // 2
    cat::Category::Parse("S[mod=nm,form=base,fin=f]{I1}"),   // 68
    cat::Category::Parse("S[mod=nm,form=base,fin=t]{I1}"),   // 19312
    cat::Category::Parse("S[mod=nm,form=cont,fin=f]{I1}"),   // 3
    cat::Category::Parse("S[mod=nm,form=cont,fin=t]{I1}"),   // 36
    cat::Category::Parse("S[mod=nm,form=da,fin=f]{I1}"),     // 1
    cat::Category::Parse("S[mod=nm,form=da,fin=t]{I1}"),     // 68
    cat::Category::Parse("S[mod=nm,form=hyp,fin=t]{I1}"),    // 1
    cat::Category::Parse("S[mod=nm,form=imp,fin=f]{I1}"),    // 3
    cat::Category::Parse("S[mod=nm,form=imp,fin=t]{I1}"),    // 15
    cat::Category::Parse("S[mod=nm,form=r,fin=t]{I1}"),      // 2
    cat::Category::Parse("S[mod=nm,form=s,fin=t]{I1}"),      // 1
    cat::Category::Parse("S[mod=nm,form=stem,fin=f]{I1}"),   // 11
    cat::Category::Parse("S[mod=nm,form=stem,fin=t]{I1}")    // 710
};

const std::vector<Combinator*> ja::binary_rules = {
    new ForwardApplication(),
    new BackwardApplication(),
    new GeneralizedForwardComposition<0>(F, F, F), // >B
    new GeneralizedBackwardComposition<0>(B, B, B), // <B1
    new GeneralizedBackwardComposition<1>(B, B, B), // <B2
    new GeneralizedBackwardComposition<2>(B, B, B), // <B3
    new GeneralizedBackwardComposition<3>(B, B, B), // <B4
    new GeneralizedForwardComposition<0>(F, B, B), // >Bx1
    new GeneralizedForwardComposition<1>(F, B, B), // >Bx2
    new GeneralizedForwardComposition<2>(F, B, B), // >Bx3
};
}
}
