
#include "combinator.h"

namespace myccg {
namespace combinator {

std::vector<Combinator*> binary_rules = {
    new Conjunction(),
    new RemovePunctuation(false),
    new RemovePunctuationLeft(),
    new ForwardApplication(),
    new BackwardApplication(),
    new ForwardComposition(Slash::Fwd(), Slash::Fwd(), Slash::Fwd()),
    new BackwardComposition(Slash::Fwd(), Slash::Bwd(), Slash::Fwd()),
    new GeneralizedForwardComposition(Slash::Fwd(), Slash::Fwd(), Slash::Fwd()),
    new GeneralizedBackwardComposition(Slash::Fwd(), Slash::Bwd(), Slash::Fwd())
};

Combinator* unary_rule = new UnaryRule();

} // namespace combinator
} // namespace myccg
