
#include "combinator.h"

namespace myccg {
namespace combinator {

#define F Slashes::Fwd()
#define B Slashes::Bwd()

std::vector<Combinator*> binary_rules = {
    new ForwardApplication(),
    new BackwardApplication(),
    new GeneralizedForwardComposition<0>(Slashes::Fwd(), Slashes::Fwd(), Slashes::Fwd()),
    new GeneralizedBackwardComposition<0>(Slashes::Fwd(), Slashes::Bwd(), Slashes::Fwd()),
    new GeneralizedForwardComposition<1>(Slashes::Fwd(), Slashes::Fwd(), Slashes::Fwd()),
    new GeneralizedBackwardComposition<1>(Slashes::Fwd(), Slashes::Bwd(), Slashes::Fwd()),
    new Conjunction(),
    new RemovePunctuation(false),
    new RemovePunctuationLeft()
};


std::vector<Combinator*> ja_combinators = {
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

Combinator* unary_rule = new UnaryRule();

} // namespace combinator
} // namespace myccg

