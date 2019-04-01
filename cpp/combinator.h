
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
    CONJ2   = 9,
    RP      = 10,
    LP      = 11,
    NOISE   = 12,
    UNARY   = 13,
    LEXICON = 14,
    NONE    = 15,
    SSEQ    = 16,
    F_MOD   = 17,
    B_MOD   = 18,
    FWD_TYPERAISE = 19,
    BWD_TYPERAISE = 20,
    COORD = 21
};

class CCombinator: public Cacheable<CCategory>
{
public:
    CCombinator(RuleType ruletype): ruletype_(ruletype) {}
    virtual bool CanApply(Cat left, Cat right) const = 0;
    virtual Cat Apply(Cat left, Cat right) const = 0;
    virtual bool HeadIsLeft(Cat left, Cat right) const = 0;
    virtual std::string ToStr() const = 0;

    RuleType GetRuleType() const { return ruletype_; }

    friend std::ostream& operator<<(std::ostream& ost, const CCombinator* comb) {
        ost << comb->ToStr();
        return ost;
    }

private:
    RuleType ruletype_;
};


using Op = const CCombinator*;

class UnaryRule: public CCombinator
{
public:
    UnaryRule(): CCombinator(UNARY) {}
    bool CanApply(Cat left, Cat right) const { return false; }
    Cat Apply(Cat left, Cat right) const NO_IMPLEMENTATION
    bool HeadIsLeft(Cat left, Cat right) const NO_IMPLEMENTATION

    std::string ToStr() const { return "<un>"; };
};

class Conjunction2: public CCombinator
{
public:
    Conjunction2(): CCombinator(CONJ2),
      puncts_({CCategory::Parse(",")}) {}
               // CCategory::Parse(";")}) {}

    bool CanApply(Cat left, Cat right) const {
        if (*left == *CCategory::Parse("conj")
                && *right == *CCategory::Parse("NP\\NP"))
            return true;
        return false;
    }

    Cat Apply(Cat left, Cat right) const { return right; }

    bool HeadIsLeft(Cat left, Cat right) const { return false; }
    std::string ToStr() const { return "<Φ>"; }

private:
    std::unordered_set<Cat> puncts_;
};

class Coordinate: public CCombinator
{
public:
    Coordinate(): CCombinator(COORD) {}

    bool CanApply(Cat left, Cat right) const {
        if (*left == *CCategory::Parse("NP")
                && *right == *CCategory::Parse("NP\\NP"))
            return true;
        return false;
    }

    Cat Apply(Cat left, Cat right) const { return left; }

    bool HeadIsLeft(Cat left, Cat right) const { return false; }
    std::string ToStr() const { return "<Φ>"; }

};

class Conjunction: public CCombinator
{
public:
    Conjunction(): CCombinator(CONJ),
      puncts_({CCategory::Parse(","),
               CCategory::Parse(";"),
               CCategory::Parse("conj")}) {}

    bool CanApply(Cat left, Cat right) const {
        if (CCategory::Parse("NP\\NP")->Matches(right))
            return false;
        return (puncts_.count(left) > 0 &&
                !right->IsPunct() &&
                !right->IsTypeRaised()); // &&
                // ! (!right->IsFunctor() &&
                        // right->GetType() == "N"));
    }

    Cat Apply(Cat left, Cat right) const {
        return CCategory::Make(right, Slash::Bwd(), right);
    }

    bool HeadIsLeft(Cat left, Cat right) const { return false; }
    std::string ToStr() const { return "<Φ>"; }

private:
    std::unordered_set<Cat> puncts_;
};

//  ,  S[ng|pss]\NP
// -----------------
//   (S\NP)\(S\NP)
class CommaAndVerbPhraseToAdverb: public CCombinator
{
public:
    CommaAndVerbPhraseToAdverb()
        : CCombinator(NOISE),
          ngVP_(CCategory::Parse("S[ng]\\NP")),
          pssVP_(CCategory::Parse("S[pss]\\NP")),
          result_(CCategory::Parse("(S\\NP)\\(S\\NP)")) {}

    bool CanApply(Cat left, Cat right) const {
        return (!left->IsFunctor() &&
                left->GetType() == "," &&
                (*right == *ngVP_ || *right == *pssVP_));
    }

    Cat Apply(Cat left, Cat right) const { return result_; }

    bool HeadIsLeft(Cat left, Cat right) const { return false; }

    std::string ToStr() const { return "<*>"; };

private:
    Cat ngVP_;
    Cat pssVP_;
    Cat result_;
};

//  ,  S[dcl]/S[dcl]
// -----------------
//   (S\NP)\(S\NP)
class ParentheticalDirectSpeech: public CCombinator
{
public:
    ParentheticalDirectSpeech()
        : CCombinator(NOISE),
          SdclSdcl_(CCategory::Parse("S[dcl]/S[dcl]")),
          result_(CCategory::Parse("(S\\NP)/(S\\NP)")) {}

    bool CanApply(Cat left, Cat right) const {
        return (!left->IsFunctor() &&
                left->GetType() == "," &&
                *right == *SdclSdcl_);
    }

    Cat Apply(Cat left, Cat right) const { return result_; }

    bool HeadIsLeft(Cat left, Cat right) const { return false; }

    std::string ToStr() const { return "<*>"; };

private:
    Cat SdclSdcl_;
    Cat result_;
};

class RemovePunctuation: public CCombinator
{
public:
    RemovePunctuation(bool punct_is_left)
        : CCombinator(punct_is_left ? LP : RP), punct_is_left_(punct_is_left) {}

    bool CanApply(Cat left, Cat right) const {
        return punct_is_left_ ? left->IsPunct() : (right->IsPunct()); //&&
             // !(!left->IsFunctor() &&
              // left->GetType() == "N"));
    }
    Cat Apply(Cat left, Cat right) const {
        return punct_is_left_ ? right : left;
    }
    bool HeadIsLeft(Cat left, Cat right) const {
        return !punct_is_left_;
    }
    std::string ToStr() const { return "<rp>"; };

private:
    bool punct_is_left_;
};

class RemovePunctuationLeft: public CCombinator
{
public:
    RemovePunctuationLeft(): CCombinator(LP),
      puncts_({CCategory::Parse("LQU"),
               CCategory::Parse("LRB"),}) {}

    bool CanApply(Cat left, Cat right) const {
        return puncts_.count(left) > 0;
    }

    Cat Apply(Cat left, Cat right) const {
        return CCategory::Make(right, Slash::Bwd(), right);
    }
    bool HeadIsLeft(Cat left, Cat right) const { return false; }
    std::string ToStr() const { return "<rp>"; };

private:
    std::unordered_set<Cat> puncts_;
};

class SpecialCombinator: public CCombinator
{
    public:
    SpecialCombinator(Cat left, Cat right, Cat result, bool head_is_left)
    : CCombinator(NOISE), left_(left), right_(right), result_(result), head_is_left_(head_is_left) {}

    bool CanApply(Cat left, Cat right) const {
        return left_->Matches(left) && right_->Matches(right);
    }
    Cat Apply(Cat left, Cat right) const { return result_; }
    bool HeadIsLeft(Cat left, Cat right) const {return head_is_left_; }
    std::string ToStr() const {return "<sp>"; };

private:
    Cat left_;
    Cat right_;
    Cat result_;
    bool head_is_left_;
};

class ForwardApplication: public CCombinator
{
    public:
    ForwardApplication(): CCombinator(FA) {}
    bool CanApply(Cat left, Cat right) const {
        return (left->IsFunctor() &&
                left->GetSlash().IsForward() &&
                left->GetRight()->Matches(right));
    }
    Cat Apply(Cat left, Cat right) const {
        if (left->IsModifier()) return right;
        Cat result = left->GetLeft();
        return CCategory::CorrectWildcardFeatures(result, left->GetRight(), right);
    }

    bool HeadIsLeft(Cat left, Cat right) const {
        return !(left->IsModifier() || left->IsTypeRaised());
    }

    std::string ToStr() const { return ">"; };
};

class BackwardApplication: public CCombinator
{
    public:
    BackwardApplication(): CCombinator(BA) {}
    bool CanApply(Cat left, Cat right) const {
        return (right->IsFunctor() &&
                right->GetSlash().IsBackward() &&
                right->GetRight()->Matches(left));
    }

    Cat Apply(Cat left, Cat right) const {
        Cat res = right->IsModifier() ? left : right->GetLeft();
        return CCategory::CorrectWildcardFeatures(res, right->GetRight(), left);
    }

    bool HeadIsLeft(Cat left, Cat right) const {
        return (right->IsModifier() || right->IsTypeRaised());
    }

    std::string ToStr() const { return "<"; }
};

template<int Order, RuleType Rule=FC>
class GeneralizedForwardComposition: public CCombinator
{
    public:
    GeneralizedForwardComposition(const Slash& left, const Slash& right, const Slash& result)
        : CCombinator(Rule), left_(left), right_(right), result_(result) {}
    bool CanApply(Cat left, Cat right) const {
        return (left->IsFunctor() &&
                right->HasFunctorAtLeft<Order>() &&
                left->GetRight()->Matches(right->GetLeft<Order+1>()) &&
                left->GetSlash() == left_ &&
                right->GetLeft<Order>()->GetSlash() == right_);
    }

    Cat Apply(Cat left, Cat right) const {
        Cat res = left->IsModifier() ? right :
            CCategory::Compose<Order>(left->GetLeft(), result_, right);
        return CCategory::CorrectWildcardFeatures(res,
                right->GetLeft<Order+1>(), left->GetRight());
    }

    bool HeadIsLeft(Cat left, Cat right) const {
        return ! (left->IsModifier() || left->IsTypeRaised());
    }

    std::string ToStr() const { return ">B" + std::to_string(Order + 1); }

private:
    Slash left_;
    Slash right_;
    Slash result_;
};

template<int Order, RuleType Rule=BC>
class GeneralizedBackwardComposition: public CCombinator
{
    public:
    GeneralizedBackwardComposition(const Slash& left, const Slash& right, const Slash& result)
        : CCombinator(Rule), left_(left), right_(right), result_(result) {}
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
            CCategory::Compose<Order>(right->GetLeft(), result_, left);
        return CCategory::CorrectWildcardFeatures(
                res, left->GetLeft<Order+1>(), right->GetRight());
    }
    bool HeadIsLeft(Cat left, Cat right) const {
        return right->IsModifier() || right->IsTypeRaised();
    }
    std::string ToStr() const { return "<B" + std::to_string(Order + 1); }

private:
    Slash left_;
    Slash right_;
    Slash result_;
};

class SimpleHeadCombinator: public CCombinator
{
public:
    SimpleHeadCombinator(Op comb, bool head_is_left)
    : CCombinator(comb->GetRuleType()),
      comb_(comb), head_is_left_(head_is_left) {};

    bool CanApply(Cat left, Cat right) const {
        return comb_->CanApply(left, right); }

    Cat Apply(Cat left, Cat right) const {
        return comb_->Apply(left, right); }

    bool HeadIsLeft(Cat left, Cat right) const {
        return head_is_left_; }

    std::string ToStr() const {
        return comb_->ToStr(); }

private:
    Op comb_;
    bool head_is_left_;
};

class HeadFirstCombinator: public SimpleHeadCombinator
{
public:
    HeadFirstCombinator(Op comb): SimpleHeadCombinator(comb, true) {};
};

class HeadFinalCombinator: public SimpleHeadCombinator
{
public:
    HeadFinalCombinator(Op comb): SimpleHeadCombinator(comb, false) {};
};

class ENBackwardApplication: public BackwardApplication
{
public:
    ENBackwardApplication(): BackwardApplication() {}

    bool CanApply(Cat left, Cat right) const {
        if (*right == *CCategory::Parse("S[em]\\S[em]") &&
                *left == *CCategory::Parse("S[dcl]"))
            return true;
        return (right->IsFunctor() &&
                right->GetSlash().IsBackward() &&
                right->GetRight()->Matches(left));
    }
    bool HeadIsLeft(Cat left, Cat right) const {
        return (right->IsModifier() || right->IsTypeRaised()) &&
            !(*right == *CCategory::Parse("S[dcl]\\S[dcl]"));
    }
};

class ENForwardApplication: public ForwardApplication
{
public:
    ENForwardApplication(): ForwardApplication() {}

    bool HeadIsLeft(Cat left, Cat right) const {
        return !(left->IsModifier() ||
                left->IsTypeRaised() ||
                *left == *CCategory::Parse("NP[nb]/N") ||
                *left == *CCategory::Parse("NP/N"));}

};


class Conjoin: public CCombinator
{
public:
    Conjoin(): CCombinator(SSEQ) {}
    bool CanApply(Cat left, Cat right) const {
        return (ja_possible_root_cats.count(left) > 0 &&
                *left == *right &&
                !left->IsFunctor());
    }
    Cat Apply(Cat left, Cat right) const { return right; }
    bool HeadIsLeft(Cat left, Cat right) const { return false; }
    std::string ToStr() const { return "SSEQ"; };

private:
    static const std::unordered_set<Cat> ja_possible_root_cats;
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

    std::string ToStr() const { return string_; }
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

    std::string ToStr() const { return string_; }
    std::string string_;
};

class RemoveDisfluency: public CCombinator
{
    public:
    RemoveDisfluency(): CCombinator(NOISE) {}
    bool CanApply(Cat left, Cat right) const {
        return left->Matches(CCategory::Parse("X"))
                || right->Matches(CCategory::Parse("X"));
    }

    Cat Apply(Cat left, Cat right) const {
        if (left->Matches(CCategory::Parse("X")))
            return right;
        return left;
    }

    bool HeadIsLeft(Cat left, Cat right) const {
        return ! left->Matches(CCategory::Parse("X"));
    }

    std::string ToStr() const { return "<X>"; }
};

class UnknownCombinator: public CCombinator
{
    public:
    UnknownCombinator(): CCombinator(NONE) {}
    bool CanApply(Cat left, Cat right) const {
        throw std::runtime_error("don't use this combinator");
    }

    Cat Apply(Cat left, Cat right) const {
        throw std::runtime_error("don't use this combinator");
    }

    bool HeadIsLeft(Cat left, Cat right) const {
        throw std::runtime_error("don't use this combinator");
    }

    std::string ToStr() const { return "<*>"; }
};

extern CCombinator* unary_rule;

extern const std::vector<Op> en_binary_rules;
extern const std::vector<Op> ja_binary_rules;

} // namespace myccg

#endif // include
