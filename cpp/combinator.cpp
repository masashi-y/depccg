
#include "combinator.h"

namespace myccg {

#define F Slash::Fwd()
#define B Slash::Bwd()

Combinator* unary_rule = new UnaryRule();

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
    HeadFirst(ENForwardApplication()),
    HeadFirst(ENBackwardApplication()),
    HeadFirst(GeneralizedForwardComposition<0, FC>(F, F, F)),
    HeadFirst(GeneralizedBackwardComposition<0, BC>(F, B, F)),
    HeadFirst(GeneralizedForwardComposition<1, GFC>(F, F, F)),
    HeadFirst(GeneralizedBackwardComposition<1, GBC>(F, F, F)),
    HeadFirst(Conjunction()),
    HeadFirst(Conjunction2()),
    HeadFirst(RemovePunctuation(false)),
    HeadFirst(RemovePunctuation(true)),
    HeadFirst(CommaAndVerbPhraseToAdverb()),
    HeadFirst(ParentheticalDirectSpeech())
};

const std::vector<Op> ja_binary_rules = {
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

} // namespace myccg

