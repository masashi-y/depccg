
#include "combinator.h"

namespace myccg {

#define F Slash::Fwd()
#define B Slash::Bwd()

CCombinator* unary_rule = new UnaryRule();

// const std::unordered_set<Cat> Conjoin::ja_possible_root_cats = {
//     CCategory::Parse("NP[case=nc,mod=nm,fin=f]"),    // 170
//     CCategory::Parse("NP[case=nc,mod=nm,fin=t]"),    // 2972
//     CCategory::Parse("S[mod=nm,form=attr,fin=t]"),   // 2
//     CCategory::Parse("S[mod=nm,form=base,fin=f]"),   // 68
//     CCategory::Parse("S[mod=nm,form=base,fin=t]"),   // 19312
//     CCategory::Parse("S[mod=nm,form=cont,fin=f]"),   // 3
//     CCategory::Parse("S[mod=nm,form=cont,fin=t]"),   // 36
//     CCategory::Parse("S[mod=nm,form=da,fin=f]"),     // 1
//     CCategory::Parse("S[mod=nm,form=da,fin=t]"),     // 68
//     CCategory::Parse("S[mod=nm,form=hyp,fin=t]"),    // 1
//     CCategory::Parse("S[mod=nm,form=imp,fin=f]"),    // 3
//     CCategory::Parse("S[mod=nm,form=imp,fin=t]"),    // 15
//     CCategory::Parse("S[mod=nm,form=r,fin=t]"),      // 2
//     CCategory::Parse("S[mod=nm,form=s,fin=t]"),      // 1
//     CCategory::Parse("S[mod=nm,form=stem,fin=f]"),   // 11
//     CCategory::Parse("S[mod=nm,form=stem,fin=t]")    // 710
// };

} // namespace myccg

