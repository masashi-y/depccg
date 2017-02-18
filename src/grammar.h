
#ifndef INCLUDE_GRAMMAR_H_
#define INCLUDE_GRAMMAR_H_

#include "cat.h"
#include "combinator.h"
#include "tree.h"

#define F Slash::Fwd()
#define B Slash::Bwd()

namespace myccg {

bool IsNormalFormExtended(
        RuleType rule_type, Cat result, NodeType left, NodeType right);


template <typename T>
class SimpleHeadCombinator: public Combinator
{
public:
    SimpleHeadCombinator(T comb, bool head_is_left)
    : Combinator(comb.GetRuleType()),
      comb_(comb), head_is_left_(head_is_left) {};

    bool CanApply(Cat left, Cat right) const {
        return comb_.CanApply(left, right); }

    Cat Apply(Cat left, Cat right) const {
        return comb_.Apply(left, right); }

    bool HeadIsLeft(Cat left, Cat right) const {
        return head_is_left_; }

    const std::string ToStr() const {
        return comb_.ToStr(); }

private:
    T comb_;
    bool head_is_left_;
};

template <typename T>
class HeadFirstCombinator: public SimpleHeadCombinator<T>
{
public:
    HeadFirstCombinator(T comb): SimpleHeadCombinator<T>(comb, true) {};
};

template <typename T>
HeadFirstCombinator<T>* HeadFirst(T comb) { return new HeadFirstCombinator<T>(comb); }

template <typename T>
class HeadFinalCombinator: public SimpleHeadCombinator<T>
{
public:
    HeadFinalCombinator(T comb): SimpleHeadCombinator<T>(comb, false) {};
};

template <typename T>
HeadFinalCombinator<T>* HeadFinal(T comb) { return new HeadFinalCombinator<T>(comb); }

struct En {

public:
class ENBackwardApplication: public BackwardApplication
{
public:
    ENBackwardApplication(): BackwardApplication() {}

    bool CanApply(Cat left, Cat right) const {
        if (*right == *Category::Parse("S[em]\\S[em]") &&
                *left == *Category::Parse("S[dcl]"))
            return true;
        return (right->IsFunctor() &&
                right->GetSlash().IsBackward() &&
                right->GetRight()->Matches(left));
    }
    bool HeadIsLeft(Cat left, Cat right) const {
        return (right->IsModifier() || right->IsTypeRaised()) &&
            !(*right == *Category::Parse("S[dcl]\\S[dcl]"));
    }
};


class ENForwardApplication: public ForwardApplication
{
public:
    ENForwardApplication(): ForwardApplication() {}

    bool HeadIsLeft(Cat left, Cat right) const {
        return !(left->IsModifier() ||
                left->IsTypeRaised() ||
                *left == *Category::Parse("NP[nb]/N") ||
                *left == *Category::Parse("NP/N"));}

};

static bool IsAcceptableUnary(Cat result, NodeType parse);

static bool IsAcceptableBinary(RuleType rule_type, NodeType left, NodeType right);

static bool IsAcceptableBinary(RuleType rule_type, Cat result, NodeType left, NodeType right);

static bool IsModifier(Cat cat);

static bool IsVerb(NodeType tree);

static bool IsAdjective(Cat cat);

static bool IsAdverb(Cat cat);

static bool IsAuxiliary(Cat cat);

static bool IsPeriod(Cat cat);

static const std::unordered_set<Cat> possible_root_cats;
static const std::vector<Op> binary_rules;
static const std::vector<Op> dep_binary_rules;
static const std::vector<Op> headfirst_binary_rules;

};

struct Ja {
            
class Conjoin: public Combinator
{
public:
    Conjoin(): Combinator(SSEQ) {}
    bool CanApply(Cat left, Cat right) const {
        return (possible_root_cats.count(left) > 0 &&
                *left == *right &&
                !left->IsFunctor());
    }
    Cat Apply(Cat left, Cat right) const { return right; }
    bool HeadIsLeft(Cat left, Cat right) const { return false; }
    const std::string ToStr() const { return "SSEQ"; };
};

class JAForwardApplication: public ForwardApplication
{
public:
    bool HeadIsLeft(Cat left, Cat right) const {
        return !(left->IsModifier() || left->IsTypeRaised());
    }
};

class JABackwardApplication: public BackwardApplication
{
public:
    bool HeadIsLeft(Cat left, Cat right) const {
        return right->IsModifier() || right->IsTypeRaised();
    }
};

template<int Order, RuleType Rule=FC>
class JAGeneralizedForwardComposition
    : public GeneralizedForwardComposition<Order, Rule>
{
public:
    JAGeneralizedForwardComposition(
            const Slash& left, const Slash& right,
            const Slash& result, const std::string& string)
        : GeneralizedForwardComposition<Order, Rule>(left, right, result),
          string_(string) {}
    bool HeadIsLeft(Cat left, Cat right) const {
        return ! (left->IsModifier() || left->IsTypeRaised());
    }

    const std::string ToStr() const { return string_; }
    std::string string_;
};

template<int Order, RuleType Rule=BC>
class JAGeneralizedBackwardComposition
    : public GeneralizedBackwardComposition<Order, Rule>
{
public:
    JAGeneralizedBackwardComposition(
            const Slash& left, const Slash& right,
            const Slash& result, const std::string& string)
        : GeneralizedBackwardComposition<Order, Rule>(left, right, result),
          string_(string) {}
    bool HeadIsLeft(Cat left, Cat right) const {
        return right->IsModifier() || right->IsTypeRaised();
    }

    const std::string ToStr() const { return string_; }
    std::string string_;
};
static bool IsAcceptableUnary(Cat result, NodeType parse);

static bool IsAcceptableBinary(RuleType rule_type, NodeType left, NodeType right);

static bool IsAcceptableBinary(RuleType rule_type, Cat result, NodeType left, NodeType right);

static bool IsModifier(Cat cat);

static bool IsVerb(const Node* tree);

static bool IsVerb(NodeType tree);

static bool IsAdjective(Cat cat);

static bool IsAdverb(Cat cat);

static bool IsAuxiliary(Cat cat);

static bool IsPunct(NodeType tree);

static bool IsComma(NodeType tree);

static bool IsPeriod(NodeType tree);

static bool IsPeriod(Cat cat);

static const std::unordered_set<Cat> possible_root_cats;
static const std::vector<Op> binary_rules;
static const std::vector<Op> headfinal_binary_rules;

};

} // namespace myccg

#endif // include
