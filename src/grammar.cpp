
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
    new GeneralizedForwardComposition<0, combinator::FC>(F, F, F),
    new GeneralizedBackwardComposition<0, combinator::BC>(F, B, F),
    new GeneralizedForwardComposition<1, combinator::GFC>(F, F, F),
    new GeneralizedBackwardComposition<1, combinator::GBC>(F, B, F),
    new Conjunction(),
    new RemovePunctuation(false),
    new RemovePunctuationLeft()
};

const std::unordered_set<Cat> ja::possible_root_cats = {
    cat::Category::Parse("NP[case=nc,mod=nm,fin=f]"),    // 170
    cat::Category::Parse("NP[case=nc,mod=nm,fin=t]"),    // 2972
    cat::Category::Parse("S[mod=nm,form=attr,fin=t]"),   // 2
    cat::Category::Parse("S[mod=nm,form=base,fin=f]"),   // 68
    cat::Category::Parse("S[mod=nm,form=base,fin=t]"),   // 19312
    cat::Category::Parse("S[mod=nm,form=cont,fin=f]"),   // 3
    cat::Category::Parse("S[mod=nm,form=cont,fin=t]"),   // 36
    cat::Category::Parse("S[mod=nm,form=da,fin=f]"),     // 1
    cat::Category::Parse("S[mod=nm,form=da,fin=t]"),     // 68
    cat::Category::Parse("S[mod=nm,form=hyp,fin=t]"),    // 1
    cat::Category::Parse("S[mod=nm,form=imp,fin=f]"),    // 3
    cat::Category::Parse("S[mod=nm,form=imp,fin=t]"),    // 15
    cat::Category::Parse("S[mod=nm,form=r,fin=t]"),      // 2
    cat::Category::Parse("S[mod=nm,form=s,fin=t]"),      // 1
    cat::Category::Parse("S[mod=nm,form=stem,fin=f]"),   // 11
    cat::Category::Parse("S[mod=nm,form=stem,fin=t]")    // 710
};

const std::vector<Combinator*> ja::binary_rules = {
    new Conjoin(),
    new ForwardApplication(),
    new BackwardApplication(),
    new GeneralizedForwardComposition<0, combinator::FC>(F, F, F), // >B
    new GeneralizedBackwardComposition<0, combinator::BC>(B, B, B), // <B1
    new GeneralizedBackwardComposition<1, combinator::BC>(B, B, B), // <B2
    new GeneralizedBackwardComposition<2, combinator::BC>(B, B, B), // <B3
    new GeneralizedBackwardComposition<3, combinator::BC>(B, B, B), // <B4
    new GeneralizedForwardComposition<0, combinator::FX>(F, B, B), // >Bx1
    new GeneralizedForwardComposition<1, combinator::FX>(F, B, B), // >Bx2
    new GeneralizedForwardComposition<2, combinator::FX>(F, B, B), // >Bx3
};

}
}
