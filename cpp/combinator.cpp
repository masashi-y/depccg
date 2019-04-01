
#include "combinator.h"

namespace myccg {

#define F Slash::Fwd()
#define B Slash::Bwd()

CCombinator* unary_rule = new UnaryRule();

const std::unordered_set<Cat> Conjoin::ja_possible_root_cats = {
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

const std::vector<Op> en_binary_rules = {
    new HeadFirstCombinator(new ENForwardApplication()),
    new HeadFirstCombinator(new ENBackwardApplication()),
    new HeadFirstCombinator(new GeneralizedForwardComposition<0, FC>(F, F, F)),
    new HeadFirstCombinator(new GeneralizedBackwardComposition<0, BC>(F, B, F)),
    new HeadFirstCombinator(new GeneralizedForwardComposition<1, GFC>(F, F, F)),
    new HeadFirstCombinator(new GeneralizedBackwardComposition<1, GBC>(F, F, F)),
    new HeadFirstCombinator(new Conjunction()),
    new HeadFirstCombinator(new Conjunction2()),
    new HeadFirstCombinator(new RemovePunctuation(false)),
    new HeadFirstCombinator(new RemovePunctuation(true)),
    new HeadFirstCombinator(new CommaAndVerbPhraseToAdverb()),
    new HeadFirstCombinator(new ParentheticalDirectSpeech())
};

const std::vector<Op> ja_binary_rules = {
    new HeadFinalCombinator(new Conjoin()),
    new HeadFinalCombinator(new JAForwardApplication()),
    new HeadFinalCombinator(new JABackwardApplication()),
    new HeadFinalCombinator(new JAGeneralizedForwardComposition<0, FC>(F, F, F, ">B")),
    new HeadFinalCombinator(new JAGeneralizedBackwardComposition<0, BC>(B, B, B, "<B1")),
    new HeadFinalCombinator(new JAGeneralizedBackwardComposition<1, BC>(B, B, B, "<B2")),
    new HeadFinalCombinator(new JAGeneralizedBackwardComposition<2, BC>(B, B, B, "<B3")),
    new HeadFinalCombinator(new JAGeneralizedBackwardComposition<3, BC>(B, B, B, "<B4")),
    new HeadFinalCombinator(new JAGeneralizedForwardComposition<0, FX>(F, B, B, ">Bx1")),
    new HeadFinalCombinator(new JAGeneralizedForwardComposition<1, FX>(F, B, B, ">Bx2")),
    new HeadFinalCombinator(new JAGeneralizedForwardComposition<2, FX>(F, B, B, ">Bx3")),
};

} // namespace myccg

