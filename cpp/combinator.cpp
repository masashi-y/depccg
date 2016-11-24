
#include "combinator.h"

namespace myccg {
namespace combinator {

// GeneralizedForwardComposition<0> corresponds combinator with order 2 e.g. >Bx2
// GeneralizedForwardComposition<0>(Fwd(), Fwd(), Fwd())
// A/B B/C/D --> A/C/D
//
// GeneralizedForwardComposition<1>(Fwd(), Bwd(), Bwd()) >Bx3
// S/S S\NP\NP\NP --> S\NP\NP\NP
//
// GeneralizedBackwardComposition<0> corresponds <B3
// GeneralizedBackwardComposition<0>(Fwd(), Fwd(), Fwd())
// A/B/C D/A --> D/B/C
//
std::vector<Combinator*> binary_rules = {
    new ForwardApplication(),
    new BackwardApplication(),
    new ForwardComposition(Slash::Fwd(), Slash::Fwd(), Slash::Fwd()),
    new BackwardComposition(Slash::Fwd(), Slash::Bwd(), Slash::Fwd()),
    new GeneralizedForwardComposition<0>(Slash::Fwd(), Slash::Fwd(), Slash::Fwd()),
    new GeneralizedBackwardComposition<0>(Slash::Fwd(), Slash::Bwd(), Slash::Fwd()),
    new Conjunction(),
    new RemovePunctuation(false),
    new RemovePunctuationLeft()
};

Combinator* unary_rule = new UnaryRule();

} // namespace combinator
} // namespace myccg

using namespace myccg;

#define test(v) std::cout << #v": " << (v ? "true" : "false") << std::endl;

int main() {
    cat::Cat ex = myccg::cat::parse("(((A/B)\\C)/D)");
    cat::Cat ex1 = myccg::cat::parse("((A/B)\\C)");
    std::cout << ex->ToStr() << std::endl;
    std::cout << ex->GetLeft<0>()->ToStr() << std::endl;
    std::cout << ex->GetLeft<1>()->ToStr() << std::endl;
    std::cout << ex->GetLeft<2>()->ToStr() << std::endl;

    cat::Cat ex2 = myccg::cat::parse("(A/(B\\(C/D)))");
    std::cout << ex2->ToStr() << std::endl;
    std::cout << ex2->GetRight<0>()->ToStr() << std::endl;
    std::cout << ex2->GetRight<1>()->ToStr() << std::endl;
    std::cout << ex2->GetRight<2>()->ToStr() << std::endl;

    test(ex->HasFunctorAtLeft<0>());
    test(ex->HasFunctorAtLeft<1>());
    test(ex->HasFunctorAtLeft<2>());
    test(ex->HasFunctorAtLeft<3>());
    test(ex->HasFunctorAtLeft<4>());
    test(ex->HasFunctorAtLeft<5>());

    test(ex->HasFunctorAtRight<0>());
    test(ex->HasFunctorAtRight<1>());
    test(ex->HasFunctorAtRight<2>());
    test(ex->HasFunctorAtRight<3>());
    test(ex->HasFunctorAtRight<4>());
    test(ex->HasFunctorAtRight<5>());
    auto bx = new combinator::GeneralizedForwardComposition<0>(
            cat::Slash::Fwd(), cat::Slash::Fwd(), cat::Slash::Fwd());
    auto bx1 = new combinator::GeneralizedForwardComposition<1>(
            cat::Slash::Fwd(), cat::Slash::Fwd(), cat::Slash::Fwd());

    std::cout << ex->GetLeft<2>()->ToStr() << std::endl;
    cat::Cat ex3 = myccg::cat::parse("(S/A)");
    test(bx->CanApply(ex3, ex1));
    test(bx1->CanApply(ex3, ex));
    std::cout << bx->Apply(ex3, ex1)->ToStr() << std::endl;
    std::cout << bx1->Apply(ex3, ex)->ToStr() << std::endl;

    return 0;
}
