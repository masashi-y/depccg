
#ifndef INCLUDE_COMBINATOR_H_
#define INCLUDE_COMBINATOR_H_

#include <vector>
#include <unordered_set>
#include "cat.h"
#include "debug.h"

namespace myccg {

enum RuleType {
    FA      = 0,
    BA      = 1,
    FC      = 2,
    BC      = 3,
    GFC     = 4,
    GBC     = 5,
    FX      = 6,
    BX      = 7,
    CONJ    = 8,
    RP      = 9,
    LP      = 10,
    NOISE   = 11,
    UNARY   = 12,
    LEXICON = 13,
    NONE    = 14,
    SSEQ    = 15
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

    friend std::ostream& operator<<(std::ostream& ost, const Combinator* comb) {
        ost << comb->ToStr();
        return ost;
    }

private:
    RuleType ruletype_;
};


using Op = const Combinator*;

class UnaryRule: public Combinator
{
public:
    UnaryRule(): Combinator(UNARY) {}
    bool CanApply(Cat left, Cat right) const { return false; }
    Cat Apply(Cat left, Cat right) const NO_IMPLEMENTATION
    bool HeadIsLeft(Cat left, Cat right) const NO_IMPLEMENTATION

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
      puncts_({Category::Parse(","),
               Category::Parse(";"),
               Category::Parse("conj")}) {}

    bool CanApply(Cat left, Cat right) const {
        if (Category::Parse("NP\\NP")->Matches(right))
            return false;
        return (puncts_.count(left) > 0 &&
                !right->IsPunct() &&
                !right->IsTypeRaised() &&
                ! (!right->IsFunctor() &&
                        right->GetType() == "N"));
    }

    Cat Apply(Cat left, Cat right) const {
        return Category::Make(right, Slash::Bwd(), right);
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
             !(!left->IsFunctor() &&
              left->GetType() == "N"));
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
      puncts_({Category::Parse("LQU"),
               Category::Parse("LRB"),}) {}

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
        return Category::CorrectWildcardFeatures(result, left->GetRight(), right);
    }

    bool HeadIsLeft(Cat left, Cat right) const {
        return !(left->IsModifierWithoutFeat() || left->IsTypeRaisedWithoutFeat());}

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
        return Category::CorrectWildcardFeatures(res, right->GetRight(), left);
    }

    bool HeadIsLeft(Cat left, Cat right) const {
        return right->IsModifierWithoutFeat() || right->IsTypeRaisedWithoutFeat();
    }

    const std::string ToStr() const { return "<"; }
};

template<int Order, RuleType Rule=FC>
class GeneralizedForwardComposition: public Combinator
{
    public:
    GeneralizedForwardComposition(const Slash& left, const Slash& right, const Slash& result)
        : Combinator(Rule), left_(left), right_(right), result_(result) {}
    bool CanApply(Cat left, Cat right) const {
        return (left->IsFunctor() &&
                right->HasFunctorAtLeft<Order>() &&
                left->GetRight()->Matches(right->GetLeft<Order+1>()) &&
                left->GetSlash() == left_ &&
                right->GetLeft<Order>()->GetSlash() == right_);
    }

    Cat Apply(Cat left, Cat right) const {
        Cat res = left->IsModifier() ? right :
            Category::Compose<Order>(left->GetLeft(), result_, right);
        return Category::CorrectWildcardFeatures(res,
                right->GetLeft<Order+1>(), left->GetRight());
    }

    bool HeadIsLeft(Cat left, Cat right) const {
        return ! (left->IsModifierWithoutFeat() || left->IsTypeRaisedWithoutFeat());
    }

    const std::string ToStr() const { return ">B" + std::to_string(Order + 1); }

private:
    Slash left_;
    Slash right_;
    Slash result_;
};

template<int Order, RuleType Rule=BC>
class GeneralizedBackwardComposition: public Combinator
{
    public:
    GeneralizedBackwardComposition(const Slash& left, const Slash& right, const Slash& result)
        : Combinator(Rule), left_(left), right_(right), result_(result) {}
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
            Category::Compose<Order>(right->GetLeft(), result_, left);
        return Category::CorrectWildcardFeatures(
                res, left->GetLeft<Order+1>(), right->GetRight());
    }
    bool HeadIsLeft(Cat left, Cat right) const {
        return right->IsModifierWithoutFeat() || right->IsTypeRaisedWithoutFeat();
    }
    const std::string ToStr() const { return "<B" + std::to_string(Order + 1); }

private:
    Slash left_;
    Slash right_;
    Slash result_;
};

extern std::vector<Combinator*> binary_rules;
extern Combinator* unary_rule;

} // namespace myccg

#endif // include
