
#ifndef INCLUDE_CAT_H_
#define INCLUDE_CAT_H_

#include <iostream>
#include <string>
#include <stdexcept>
#include "cacheable.h"
#include "feat.h"
#include "debug.h"

namespace myccg {

class CCategory;

typedef const CCategory* Cat;
typedef std::pair<Cat, Cat> CatPair;

extern Feat kNONE;
extern Feat kNB;

class Slash
{
public:
    static Slash Fwd() { return Slash('/'); }
    static Slash Bwd() { return Slash('\\'); }
    static Slash Either() { return Slash('|'); }
    bool IsForward() const { return slash_ == '/'; } 
    bool IsBackward() const { return slash_ == '\\'; } 

    Slash(char slash): slash_(slash) {}
    const std::string ToStr() const { return std::string(1, slash_); }

    bool Matches(const Slash& other) const {
        return  (other.slash_ == this->slash_ ||
                this->slash_ == '|' || other.slash_ == '|');
    }

    bool operator==(const Slash& other) const {
        return this->slash_ == other.slash_;
    }

private:
    char slash_;
};

class AtomicCategory;
class Functor;

class CatVisitor {
public:
    virtual int Visit(const AtomicCategory* leaf) = 0;
    virtual int Visit(const Functor* leaf) = 0;
};

class CCategory: public Cacheable<CCategory>
{
public:
    static Cat Parse(const std::string& cat);
    static Cat ParseUncached(const std::string& cat);
    static Cat Make(Cat left, const Slash& op, Cat right);
    static Cat CorrectWildcardFeatures(Cat to_correct, Cat match1, Cat match2);
    template<int Order>
    static Cat Compose(Cat head, const Slash& op, Cat tail);


    template<typename T>
    const T& As() const { return static_cast<const T&>(*this); }
    
    template<typename T>
    T& As() { return static_cast<T&>(*this); }

    std::string ToStr() { return str_; }
    std::string ToStr() const { return str_; }

    friend std::ostream& operator<<(std::ostream& ost, Cat cat) {
        ost << cat->ToStr();
        return ost;
    }

    virtual std::string ToStrWithoutFeat() const = 0;

    typedef const std::string& Str;
    Cat StripFeat() const;
    Cat StripFeat(Str f1) const;
    Cat StripFeat(Str f1, Str f2) const;
    Cat StripFeat(Str f1, Str f2, Str f3) const;
    Cat StripFeat(Str f1, Str f2, Str f3, Str f4) const;

    virtual const std::string& GetType() const = 0;
    virtual Feat GetFeat() const = 0;
    virtual Cat GetLeft() const = 0;
    virtual Cat GetRight() const = 0;

    // get i'th left (or right) child category of functor
    // when i== 0, return `this`.
    // performing GetLeft (GetRight) with exceeding `i` will
    // result in undefined behavior. e.g.GetLeft<10>(cat::Parse("(((A/B)/C)/D)"))
    template<int i> Cat GetLeft() const;
    template<int i> Cat GetRight() const;

    // test if i'th left (or right) child category is functor.
    // when i == 0, check if the category itself is functor.
    template<int i> bool HasFunctorAtLeft() const;
    template<int i> bool HasFunctorAtRight() const;
    virtual Slash GetSlash() const = 0;

    virtual const std::string WithBrackets() const = 0;
    virtual bool IsModifier() const = 0;
    virtual bool IsModifierWithoutFeat() const = 0;
    virtual bool IsTypeRaised() const = 0;
    virtual bool IsTypeRaisedWithoutFeat() const = 0;
    virtual bool IsForwardTypeRaised() const = 0;
    virtual bool IsBackwardTypeRaised() const = 0;
    virtual bool IsFunctor() const = 0;
    virtual bool IsPunct() const = 0;
    virtual bool IsNorNP() const = 0;
    virtual int NArgs() const = 0;
    virtual Feat GetSubstitution(Cat other) const = 0;
    virtual bool Matches(Cat other) const = 0;

    virtual Cat Arg(int argn) const = 0;
    virtual Cat LeftMostArg() const = 0;
    virtual bool IsFunctionInto(Cat cat) const = 0;
    virtual Cat ToMultiValue() const = 0;

    Cat Substitute(Feat feat) const;
    virtual int Accept(CatVisitor& visitor) const = 0;

protected:
    CCategory(const std::string& str, const std::string& semantics)
        : str_(semantics.empty() ? str : str + "{" + semantics + "}") {}

private:
    int id_;
    std::string str_;
};


// perform composition where `Order` is size of `tail` - 1.
//   e.g.  A/B B/C/D/E -> A/C/D/E
//
// Cat ex = cat::Parse("(((B/C)/D)/E)");
// compose<3>(cat::Parse("A"), Slash::Fwd(), ex);
//   --> (((A/C)/D)/E)

template<int Order>
Cat CCategory::Compose(Cat head, const Slash& op, Cat tail) {
    Cat target = tail->GetLeft<Order>();
    target = target->GetRight();
    return CCategory::Compose<Order-1>(Make(head, op, target),
            tail->GetLeft<Order-1>()->GetSlash(), tail);
}

template<> Cat CCategory::Compose<0>(Cat head, const Slash& op, Cat tail);

template<int i> bool CCategory::HasFunctorAtLeft() const {
    return this->IsFunctor() ? GetLeft()->HasFunctorAtLeft<i-1>() : false; }

template<int i> bool CCategory::HasFunctorAtRight() const {
    return this->IsFunctor() ? GetRight()->HasFunctorAtRight<i-1>() : false; }

template<> bool CCategory::HasFunctorAtLeft<0>() const;
template<> bool CCategory::HasFunctorAtRight<0>() const;

template<int i> Cat CCategory::GetLeft() const {
    return GetLeft()->GetLeft<i-1>(); }

template<int i> Cat CCategory::GetRight() const {
    return GetRight()->GetRight<i-1>(); }

template<> Cat CCategory::GetLeft<0>() const;
template<> Cat CCategory::GetRight<0>() const;


class Functor: public CCategory
{
public:
    std::string ToStrWithoutFeat() const {
        return "(" + left_->ToStrWithoutFeat() +
                slash_.ToStr() + right_->ToStrWithoutFeat() + ")";
    }

    const std::string& GetType() const NO_IMPLEMENTATION
    Feat GetFeat() const NO_IMPLEMENTATION
    Cat GetLeft() const { return left_;  }
    Cat GetRight() const { return right_;  }
    Slash GetSlash() const { return slash_;  }

    const std::string WithBrackets() const { return "(" + ToStr() + ")"; }
    bool IsModifier() const { return *this->left_ == *this->right_; }
    bool IsModifierWithoutFeat() const { return *left_->StripFeat() == *right_->StripFeat(); }

    bool IsTypeRaised() const {
        return (right_->IsFunctor() && *right_->GetLeft() == *left_);
    }

    bool IsTypeRaisedWithoutFeat() const {
        return (right_->IsFunctor() &&
                *right_->GetLeft()->StripFeat() == *left_->StripFeat());
    }

    bool IsForwardTypeRaised() const {
        return (this->IsTypeRaised() && this->slash_.IsForward());
    }

    bool IsBackwardTypeRaised() const {
        return (this->IsTypeRaised() && this->slash_.IsForward());
    }

    bool IsFunctor() const { return true; }
    bool IsPunct() const { return false; }
    bool IsNorNP() const { return false; }
    int NArgs() const { return 1 + this->left_->NArgs(); }

    Feat GetSubstitution(Cat other) const {
        Feat res = this->right_->GetSubstitution(other->GetRight());
        if (res->IsEmpty())
            return this->left_->GetSubstitution(other->GetLeft());
        return res;
    }

    bool Matches(Cat other) const {
        return (other->IsFunctor() &&
                left_->Matches(other->GetLeft()) &&
                right_->Matches(other->GetRight()) &&
                slash_.Matches(other->GetSlash()));
    }

    Cat LeftMostArg() const {
        if (left_->IsFunctor()) 
            return left_->LeftMostArg();
        else
            return left_;
    }

    Cat Arg(int argn) const {
        if (argn == this->NArgs()) {
            return this->right_;
        } else {
            return this->left_->Arg(argn);
        }
    }

    bool IsFunctionInto(Cat cat) const {
        return (cat->Matches(this) || left_->IsFunctionInto(cat));
    }

    Cat ToMultiValue() const {
        return CCategory::Parse("(" + left_->ToMultiValue()->ToStr() + ")" +
                        slash_.ToStr() + "(" + right_->ToMultiValue()->ToStr() + ")");
    }

    int Accept(CatVisitor& visitor) const { return visitor.Visit(this); }

    Functor(Cat left, const Slash& slash, Cat right, std::string& semantics)
    : CCategory(left->WithBrackets() + slash.ToStr() + right->WithBrackets(),
            semantics), left_(left), right_(right), slash_(slash) {}

private:
    Cat left_;
    Cat right_;
    Slash slash_;
};

class AtomicCategory: public CCategory
{
public:
    std::string ToStrWithoutFeat() const { return type_; }

    const std::string& GetType() const { return type_; }

    Feat GetFeat() const { return feat_; }

    Cat GetLeft() const NO_IMPLEMENTATION
    Cat GetRight() const NO_IMPLEMENTATION
    Slash GetSlash() const NO_IMPLEMENTATION

    const std::string WithBrackets() const { return ToStr(); }

    bool IsModifier() const { return false; }
    bool IsModifierWithoutFeat() const { return false; }
    bool IsTypeRaisedWithoutFeat() const { return false; }
    bool IsTypeRaised() const { return false; }
    bool IsForwardTypeRaised() const { return false; }
    bool IsBackwardTypeRaised() const { return false; }
    bool IsFunctor() const { return false; }

    bool IsPunct() const {
        return (!( ('A' <= type_[0] && type_[0] <= 'Z') ||
                    ('a' <= type_[0] && type_[0] <= 'z')) ||
                type_ == "LRB" || type_ == "RRB" ||
                type_ == "LQU" || type_ == "RQU");
    }

    bool IsNorNP() const { return type_ == "N" || type_ == "NP"; }

    int NArgs() const { return 0; }

    Feat GetSubstitution(Cat other) const {
        if (this->feat_->ContainsWildcard()) {
            return other->GetFeat();
        } else if (other->GetFeat()->ContainsWildcard()) {
            return this->GetFeat();
        }
        return kNONE;
    }

    bool Matches(Cat other) const {
        return (!other->IsFunctor() &&
                this->type_ == other->GetType() &&
                (this->feat_->IsEmpty() ||
                 other->GetFeat()->IsEmpty() ||
                 this->feat_->Matches(other->GetFeat()) ||
                 this->feat_->Matches(kNB) ||
                 other->GetFeat()->Matches(kNB)));
    }

    Cat LeftMostArg() const NO_IMPLEMENTATION

    Cat Arg(int argn) const {
        if (argn == 0)
            return this;
        throw std::runtime_error("Error getting argument of category");
    }

    bool IsFunctionInto(Cat cat) const {
        return false;
    }

    Cat ToMultiValue() const {
        return CCategory::Parse(
                type_ + feat_->ToMultiValue()->ToStr());
    }

    int Accept(CatVisitor& visitor) const { return visitor.Visit(this); }

    AtomicCategory(const std::string& type, Feat feat, const std::string& semantics)
        : CCategory(type + feat->ToStr(), semantics), type_(type), feat_(feat) {}

private:
    std::string type_;
    Feat feat_;
};


} // namespace myccg


namespace std {

template<>
struct equal_to<myccg::Cat>
{
    inline bool operator () (myccg::Cat c1, myccg::Cat c2) const {
        return c1->GetId() == c2->GetId();
    }
};

template<>
struct hash<myccg::Cat>
{
    inline size_t operator () (myccg::Cat c) const {
        return c->GetId();
    }
};

template<>
struct hash<myccg::CatPair>
{
    inline size_t operator () (const myccg::CatPair& p) const {
        return ((p.first->GetId() << 31) | (p.second->GetId()));
    }
};

} // namespace std
#endif
