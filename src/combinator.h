
#ifndef INCLUDE_COMBINATOR_H_
#define INCLUDE_COMBINATOR_H_

#include "cat.h"
#include <vector>
#include <stdexcept>
#include <unordered_set>

namespace myccg {
namespace combinator {

using cat::Slash;
using cat::Cat;

enum RuleType {
    FA      = 0,
    BA      = 1,
    FC      = 2,
    BX      = 3,
    GFC     = 4,
    GBX     = 5,
    CONJ    = 6,
    RP      = 7,
    LP      = 8,
    NOISE   = 9,
    UNARY   = 10,
    LEXICON = 11,
    NONE    = 12,
    SSEQ    = 13
};

class Combinator
{
public:
    Combinator(RuleType ruletype): ruletype_(ruletype) {}
    virtual bool CanApply(Cat left, Cat right) const = 0;
    virtual Cat Apply(Cat left, Cat right) const = 0;
    virtual bool HeadIsLeft(Cat left, Cat right) const = 0;
    virtual const std::string ToStr() const = 0;

    RuleType GetRuleType() const { return ruletype_; }

private:
    RuleType ruletype_;
};

class UnaryRule: public Combinator
{
public:
    UnaryRule(): Combinator(UNARY) {}
    bool CanApply(Cat left, Cat right) const { return false; }
    Cat Apply(Cat left, Cat right) const { 
        throw std::runtime_error("UnaryRule::Apply is not implemented.");
    }

    bool HeadIsLeft(Cat left, Cat right) const {
        throw std::runtime_error("UnaryRule::HeadIsLeft is not implemented.");
    }

    const std::string ToStr() const { return "<un>"; };
};

class Conjoin: public Combinator
{
public:
    Conjoin(): Combinator(SSEQ) {}
    bool CanApply(Cat left, Cat right) const {
        return (*left == *right &&
                !left->IsFunctor() &&
                left->GetType() == "S");
    }
    Cat Apply(Cat left, Cat right) const { return left; }
    bool HeadIsLeft(Cat left, Cat right) const { return false; }
    const std::string ToStr() const { return "SSEQ"; };
};

class Conjunction: public Combinator
{
public:
    Conjunction(): Combinator(CONJ),
      puncts_({cat::Category::Parse(","),
               cat::Category::Parse(";"),
               cat::Category::Parse("conj")}) {}

    bool CanApply(Cat left, Cat right) const {
        if (cat::Category::Parse("NP\\NP")->Matches(right))
            return false;
        return (puncts_.count(left) > 0 &&
                !right->IsPunct() &&
                !right->IsTypeRaised() &&
                ! (!right->IsFunctor() &&
                        right->GetType() == "N"));
    }

    Cat Apply(Cat left, Cat right) const {
        return cat::Make(right, Slash::Bwd(), right);
    }

    bool HeadIsLeft(Cat left, Cat right) const { return false; }
    const std::string ToStr() const { return "<Φ>"; }

private:
    std::unordered_set<Cat> puncts_;
};

class RemovePunctuation: public Combinator
{
public:
    RemovePunctuation(bool punct_is_left)
        : Combinator(RP), punct_is_left_(punct_is_left) {}

    bool CanApply(Cat left, Cat right) const {
        return punct_is_left_ ? left->IsPunct() :
            (right->IsPunct() &&
            !left->IsFunctor() &&
            left->GetType() != "N");
    }
    Cat Apply(Cat left, Cat right) const {
        return punct_is_left_ ? right : left;
    }
    bool HeadIsLeft(Cat left, Cat right) const {
        return !punct_is_left_;
    }
    const std::string ToStr() const { return "<rp>"; };

private:
    bool punct_is_left_;
};

class RemovePunctuationLeft: public Combinator
{
public:
    RemovePunctuationLeft(): Combinator(LP),
      puncts_({cat::Category::Parse("LQU"),
               cat::Category::Parse("LRB"),}) {}

    bool CanApply(Cat left, Cat right) const {
        return puncts_.count(left) > 0;
    }

    Cat Apply(Cat left, Cat right) const { return right; }
    bool HeadIsLeft(Cat left, Cat right) const { return false; }
    const std::string ToStr() const { return "<rp>"; };

private:
    std::unordered_set<Cat> puncts_;
};

class SpecialCombinator: public Combinator
{
    public:
    SpecialCombinator(Cat left, Cat right, Cat result, bool head_is_left)
    : Combinator(NOISE), left_(left), right_(right), result_(result), head_is_left_(head_is_left) {}

    bool CanApply(Cat left, Cat right) const {
        return left_->Matches(left) && right_->Matches(right);
    }
    Cat Apply(Cat left, Cat right) const { return result_; }
    bool HeadIsLeft(Cat left, Cat right) const {return head_is_left_; }
    const std::string ToStr() const {return "<sp>"; };

private:
    Cat left_;
    Cat right_;
    Cat result_;
    bool head_is_left_;
};

class ForwardApplication: public Combinator
{
    public:
    ForwardApplication(): Combinator(FA) {}
    bool CanApply(Cat left, Cat right) const {
        return (left->IsFunctor() &&
                left->GetSlash().IsForward() &&
                left->GetRight()->Matches(right));
    }
    Cat Apply(Cat left, Cat right) const {
        if (left->IsModifier()) return right;
        Cat result = left->GetLeft();
        return cat::CorrectWildcardFeatures(result, left->GetRight(), right);
    }

    bool HeadIsLeft(Cat left, Cat right) const {
        return !(left->IsModifier() || left->IsTypeRaised());}

    const std::string ToStr() const { return ">"; };
};

class BackwardApplication: public Combinator
{
    public:
    BackwardApplication(): Combinator(BA) {}
    bool CanApply(Cat left, Cat right) const {
        return (right->IsFunctor() &&
                right->GetSlash().IsBackward() &&
                right->GetRight()->Matches(left));
    }

    Cat Apply(Cat left, Cat right) const {
        Cat res = right->IsModifier() ? left : right->GetLeft();
        return cat::CorrectWildcardFeatures(res, right->GetRight(), left);
    }

    bool HeadIsLeft(Cat left, Cat right) const {
        return right->IsModifier() || right->IsTypeRaised();
    }

    const std::string ToStr() const { return "<"; }
};

template<int Order>
class GeneralizedForwardComposition: public Combinator
{
    public:
    GeneralizedForwardComposition(const Slash& left, const Slash& right, const Slash& result)
        : Combinator(Order == 0 ? FC : GFC), left_(left), right_(right), result_(result) {}
    bool CanApply(Cat left, Cat right) const {
        return (left->IsFunctor() &&
                right->HasFunctorAtLeft<Order>() &&
                left->GetRight()->Matches(right->GetLeft<Order+1>()) &&
                left->GetSlash() == left_ &&
                right->GetLeft<Order>()->GetSlash() == right_);
    }

    Cat Apply(Cat left, Cat right) const {
        Cat res = left->IsModifier() ? right :
            cat::Compose<Order>(left->GetLeft(), result_, right);
        return cat::CorrectWildcardFeatures(res,
                right->GetLeft<Order+1>(), left->GetRight());
    }

    bool HeadIsLeft(Cat left, Cat right) const {
        return ! (left->IsModifier() || left->IsTypeRaised());
    }

    const std::string ToStr() const { return ">B" + std::to_string(Order); }

private:
    Slash left_;
    Slash right_;
    Slash result_;
};

template<int Order>
class GeneralizedBackwardComposition: public Combinator
{
    public:
    GeneralizedBackwardComposition(const Slash& left, const Slash& right, const Slash& result)
        : Combinator(Order == 0 ? BX : GBX), left_(left), right_(right), result_(result) {}
    bool CanApply(Cat left, Cat right) const {
        return (right->IsFunctor() &&
                left->HasFunctorAtLeft<Order>() &&
                right->GetRight()->Matches(left->GetLeft<Order+1>()) &&
                left->GetLeft<Order>()->GetSlash() == left_ &&
                right->GetSlash() == right_ && //);
                ! left->GetLeft<Order+1>()->IsNorNP());
    }

    Cat Apply(Cat left, Cat right) const {
        Cat res = right->IsModifier() ? left :
            cat::Compose<Order>(right->GetLeft(), result_, left);
        return cat::CorrectWildcardFeatures(
                res, left->GetLeft<Order+1>(), right->GetRight());
    }
    bool HeadIsLeft(Cat left, Cat right) const {
        return right->IsModifier() || right->IsTypeRaised();
    }
    const std::string ToStr() const { return "<B" + std::to_string(Order); }

private:
    Slash left_;
    Slash right_;
    Slash result_;
};

extern std::vector<Combinator*> binary_rules;
extern Combinator* unary_rule;

} // namespace combinator
} // namespace myccg

#endif // include
