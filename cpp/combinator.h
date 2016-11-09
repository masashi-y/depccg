
#ifndef INCLUDE_COMBINATOR_H_
#define INCLUDE_COMBINATOR_H_

#include "cat.h"
#include <vector>
#include <stdexcept>

namespace myccg {
namespace combinator {

using namespace cat;

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
    NONE    = 12
};

class Combinator
{
public:
    Combinator(RuleType ruletype): ruletype_(ruletype) {}
    virtual bool CanApply(const Category* left, const Category* right) const = 0;
    virtual const Category* Apply(const Category* left, const Category* right) const = 0;
    virtual bool HeadIsLeft(const Category* left, const Category* right) const = 0;
    virtual const std::string ToStr() const = 0;

    RuleType GetRuleType() const { return ruletype_; }

private:
    RuleType ruletype_;
};

class UnaryRule: public Combinator
{
public:
    UnaryRule(): Combinator(UNARY) {}
    bool CanApply(const Category* left, const Category* right) const { return false; }
    const Category* Apply(const Category* left, const Category* right) const { 
        throw std::runtime_error("UnaryRule::Apply is not implemented.");
    }

    bool HeadIsLeft(const Category* left, const Category* right) const {
        throw std::runtime_error("UnaryRule::HeadIsLeft is not implemented.");
    }

    const std::string ToStr() const { return "<un>"; };
};

class Conjunction: public Combinator
{
public:
    Conjunction(): Combinator(CONJ) {}
    bool CanApply(const Category* left, const Category* right) const {
        if (cat::parse("NP\\NP")->Matches(right))
            return false;
        return (*left == *cat::CONJ ||
                *left == *cat::COMMA ||
                *left == *cat::SEMICOLON) &&
                !right->IsPunct() &&
                !right->IsTypeRaised() &&
                ! (!right->IsFunctor() &&
                        right->GetType() == "N");
    }

    const Category* Apply(const Category* left, const Category* right) const {
        return make(right, Slash::Bwd(), right);
    }

    bool HeadIsLeft(const Category* left, const Category* right) const { return false; }
    const std::string ToStr() const { return "<Φ>"; }
};

class RemovePunctuation: public Combinator
{
public:
    RemovePunctuation(bool punct_is_left)
        : Combinator(RP), punct_is_left_(punct_is_left) {}

    bool CanApply(const Category* left, const Category* right) const {
        return punct_is_left_ ? left->IsPunct() :
            (right->IsPunct() && !cat::N->Matches(left));
    }
    const Category* Apply(const Category* left, const Category* right) const {
        return punct_is_left_ ? right : left;
    }
    bool HeadIsLeft(const Category* left, const Category* right) const {
        return !punct_is_left_;
    }
    const std::string ToStr() const { return "<rp>"; };

private:
    bool punct_is_left_;
};

class RemovePunctuationLeft: public Combinator
{
public:
    RemovePunctuationLeft(): Combinator(LP) {}

    bool CanApply(const Category* left, const Category* right) const {
        return *left == *cat::LQU || *left == *cat::LRB;
    }

    const Category* Apply(const Category* left, const Category* right) const { return right; }
    bool HeadIsLeft(const Category* left, const Category* right) const { return false; }
    const std::string ToStr() const { return "<rp>"; };
};

class SpecialCombinator: public Combinator
{
    public:
    SpecialCombinator(const Category* left, const Category* right, const Category* result, bool head_is_left)
    : Combinator(NOISE), left_(left), right_(right), result_(result), head_is_left_(head_is_left) {}
    bool CanApply(const Category* left, const Category* right) const {
        return left_->Matches(left) && right_->Matches(right);
    }
    const Category* Apply(const Category* left, const Category* right) const { return result_; }
    bool HeadIsLeft(const Category* left, const Category* right) const {return head_is_left_; }
    const std::string ToStr() const {return "<sp>"; };

private:
    const Category* left_;
    const Category* right_;
    const Category* result_;
    bool head_is_left_;
};

class ForwardApplication: public Combinator
{
    public:
    ForwardApplication(): Combinator(FA) {}
    bool CanApply(const Category* left, const Category* right) const {
        if (left->IsFunctor())
            return (left->GetSlash() == Slash::Fwd() &&
                    left->GetRight()->Matches(right));
        return false;
    }
    const Category* Apply(const Category* left, const Category* right) const {
        if (left->IsModifier()) return right;
        const Category* result = left->GetLeft();
        return cat::CorrectWildcardFeatures(result, left->GetRight(), right);
    }

    bool HeadIsLeft(const Category* left, const Category* right) const {
        return !(left->IsModifier() || left->IsTypeRaised());}

    const std::string ToStr() const { return ">"; };
};

class BackwardApplication: public Combinator
{
    public:
    BackwardApplication(): Combinator(BA) {}
    bool CanApply(const Category* left, const Category* right) const {
        if (right->IsFunctor())
            return (right->GetSlash() == Slash::Bwd() &&
                    right->GetRight()->Matches(left));
        return false;
    }

    const Category* Apply(const Category* left, const Category* right) const {
        const Category* res;
        if (right->IsModifier())
            res = left;
        else
            res = right->GetLeft();
        return cat::CorrectWildcardFeatures(res, right->GetRight(), left);
    }

    bool HeadIsLeft(const Category* left, const Category* right) const {
        return right->IsModifier() || right->IsTypeRaised();
    }

    const std::string ToStr() const { return "<"; }
};

class ForwardComposition: public Combinator
{
    public:
    ForwardComposition(const Slash* left, const Slash* right, const Slash* result)
        : Combinator(FC), left_(left), right_(right), result_(result) {}

    bool CanApply(const Category* left, const Category* right) const {
        if (left->IsFunctor() && right->IsFunctor())
            return (left->GetRight()->Matches(right->GetLeft()) &&
                    left->GetSlash() == left_ &&
                    right->GetSlash() == right_);
        return false;
    }

    const Category* Apply(const Category* left, const Category* right) const {
        const Category* res;
        if (left->IsModifier())
            res = right;
        else
            res = cat::make(left->GetLeft(), result_, right->GetRight());
        return cat::CorrectWildcardFeatures(res, right->GetLeft(), left->GetRight());
    }

    bool HeadIsLeft(const Category* left, const Category* right) const {
        return ! (left->IsModifier() || left->IsTypeRaised());
    }

    const std::string ToStr() const { return ">B"; }

private:
    const Slash* left_;
    const Slash* right_;
    const Slash* result_;
};

class BackwardComposition: public Combinator
{
    public:
    BackwardComposition(const Slash* left, const Slash* right, const Slash* result)
        : Combinator(BX), left_(left), right_(right), result_(result) {}

    bool CanApply(const Category* left, const Category* right) const {
        if (left->IsFunctor() && right->IsFunctor())
            return (right->GetRight()->Matches(left->GetLeft()) &&
                    left->GetSlash() == left_ && right->GetSlash() == right_ &&
                    ! left->GetLeft()->IsNorNP());
        return false;
    }

    const Category* Apply(const Category* left, const Category* right) const {
        const Category* res;
        if (right->IsModifier())
            res = left;
        else
            res = cat::make(right->GetLeft(), result_, left->GetRight());
        return cat::CorrectWildcardFeatures(res, left->GetLeft(), right->GetRight());
    }

    bool HeadIsLeft(const Category* left, const Category* right) const {
        return right->IsModifier() || right->IsTypeRaised();
    }

    const std::string ToStr() const { return "<B"; }

private:
    const Slash* left_;
    const Slash* right_;
    const Slash* result_;
};

class GeneralizedForwardComposition: public Combinator
{
    public:
    GeneralizedForwardComposition(const Slash* left, const Slash* right, const Slash* result)
        : Combinator(GFC), left_(left), right_(right), result_(result) {}
    bool CanApply(const Category* left, const Category* right) const {
        if (left->IsFunctor() && right->IsFunctor() && right->GetLeft()->IsFunctor())
            return (left->GetRight()->Matches(right->GetLeft()->GetLeft()) &&
                    left->GetSlash() == left_ &&
                    right->GetLeft()->GetSlash() == right_);
    }

    const Category* Apply(const Category* left, const Category* right) const {
        if (left->IsModifier()) return right;
        const Category* res = cat::make(
                cat::make(left->GetLeft(), result_, right->GetLeft()->GetRight()),
                    right->GetSlash(), right->GetRight());
        return cat::CorrectWildcardFeatures(res,
                right->GetLeft()->GetLeft(), left->GetRight());
    }

    bool HeadIsLeft(const Category* left, const Category* right) const {
        return ! (left->IsModifier() || left->IsTypeRaised());
    }

    const std::string ToStr() const { return ">Bx"; }

private:
    const Slash* left_;
    const Slash* right_;
    const Slash* result_;
};

class GeneralizedBackwardComposition: public Combinator
{
    public:
    GeneralizedBackwardComposition(const Slash* left, const Slash* right, const Slash* result)
        : Combinator(GBX), left_(left), right_(right), result_(result) {}
    bool CanApply(const Category* left, const Category* right) const {
        if (left->IsFunctor() && right->IsFunctor() && left->GetLeft()->IsFunctor())
            return (right->GetRight()->Matches(left->GetLeft()->GetLeft()) &&
                    left->GetLeft()->GetSlash() == left_ &&
                    right->GetSlash() == right_ &&
                    ! left->GetLeft()->IsNorNP());
        return false;
    }

    const Category* Apply(const Category* left, const Category* right) const {
        if (right->IsModifier()) return left;
        const Category* res = cat::make(cat::make(
                    right->GetLeft(), result_, left->GetLeft()->GetRight()),
                left->GetSlash(), left->GetRight());
        return cat::CorrectWildcardFeatures(
                res, left->GetLeft()->GetLeft(), right->GetRight());
    }
    bool HeadIsLeft(const Category* left, const Category* right) const {
        return right->IsModifier() || right->IsTypeRaised();
    }
    const std::string ToStr() const { return "<Bx"; }

private:
    const Slash* left_;
    const Slash* right_;
    const Slash* result_;
};

extern std::vector<Combinator*> binary_rules;
void test();

} // namespace combinator
} // namespace myccg

#endif // include
