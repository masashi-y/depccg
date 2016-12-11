
#ifndef INCLUDE_CAT_H_
#define INCLUDE_CAT_H_

#include <iostream>
#include <string>
#include <stdexcept>
#include "cacheable.h"
#include "feat.h"

#define print(value) std::cout << (value) << std::endl;

namespace myccg {
namespace cat {

class Category;
class Slash;

typedef const Category* Cat;
typedef std::pair<Cat, Cat> CatPair;

Cat Make(Cat left, const Slash& op, Cat right);
Cat CorrectWildcardFeatures(Cat to_correct, Cat match1, Cat match2);

extern Feat kNONE;
extern Feat kNB;

class Slash
{
public:
    static Slash Fwd() { return Slash('/'); }
    static Slash Bwd() { return Slash('\\'); }
    static Slash Either() { return Slash('|'); }

    Slash(char slash): slash_(slash) {}
    const std::string ToStr() const { return std::string(1, slash_); }

    bool Matches(const Slash& other) const {
        return  (other.slash_ == this->slash_ ||
                this->slash_ == '|' ||
                other.slash_ == '|');
    }

    bool operator==(const Slash& other) const {
        return this->slash_ == other.slash_;
    }

    bool IsForward() const { return slash_ == '/'; } 
    bool IsBackward() const { return slash_ == '\\'; } 

private:
    char slash_;
};

class Category: public Cacheable
{
public:
    static Cat Parse(const std::string& cat);
    static Cat ParseUncached(const std::string& cat);

    template<typename T>
    const T& As() const { return dynamic_cast<const T&>(*this); }
    
    template<typename T>
    T& As() { return dynamic_cast<T&>(*this); }

    std::string ToStr() { return str_; }
    std::string ToStr() const { return str_; }

    virtual std::string ToStrWithoutFeat() const = 0;

    Cat StripFeat() const { return Category::Parse(this->ToStrWithoutFeat()); }

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
    virtual bool IsTypeRaised() const = 0;
    virtual bool IsForwardTypeRaised() const = 0;
    virtual bool IsBackwardTypeRaised() const = 0;
    virtual bool IsFunctor() const = 0;
    virtual bool IsPunct() const = 0;
    virtual bool IsNorNP() const = 0;
    virtual int NArgs() const = 0;
    virtual Feat GetSubstitution(Cat other) const = 0;
    virtual bool Matches(Cat other) const = 0;

    virtual Cat Arg(int argn) const = 0;
    virtual Cat HeadCat() const = 0;
    virtual bool IsFunctionInto(Cat cat) const = 0;
    virtual bool IsFunctionIntoModifier() const = 0;

    Cat Substitute(Feat feat) const;

protected:
    Category(const std::string& str, const std::string& semantics)
        : str_(semantics.empty() ? str : str + "{" + semantics + "}") {
    }

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
Cat Compose(Cat head, const Slash& op, Cat tail) {
    Cat target = tail->GetLeft<Order>();
    target = target->GetRight();
    return Compose<Order-1>(Make(head, op, target),
            tail->GetLeft<Order-1>()->GetSlash(), tail);
}

template<> Cat Compose<0>(Cat head, const Slash& op, Cat tail);


template<int i> bool Category::HasFunctorAtLeft() const {
    return this->IsFunctor() ? GetLeft()->HasFunctorAtLeft<i-1>() : false; }

template<int i> bool Category::HasFunctorAtRight() const {
    return this->IsFunctor() ? GetRight()->HasFunctorAtRight<i-1>() : false; }

template<> bool Category::HasFunctorAtLeft<0>() const;
template<> bool Category::HasFunctorAtRight<0>() const;

template<int i> Cat Category::GetLeft() const {
    return GetLeft()->GetLeft<i-1>(); }

template<int i> Cat Category::GetRight() const {
    return GetRight()->GetRight<i-1>(); }

template<> Cat Category::GetLeft<0>() const;
template<> Cat Category::GetRight<0>() const;


class Functor: public Category
{
public:
    std::string ToStrWithoutFeat() const {
        return "(" + left_->ToStrWithoutFeat() +
                slash_.ToStr() + right_->ToStrWithoutFeat() + ")";
    }

    virtual const std::string& GetType() const { throw std::logic_error("not implemented"); }

    virtual Feat GetFeat() const { throw std::logic_error("not implemented"); }

    Cat GetLeft() const { return left_;  }

    Cat GetRight() const { return right_;  }

    Slash GetSlash() const { return slash_;  }

    const std::string WithBrackets() const { return "(" + ToStr() + ")"; }

    bool IsModifier() const { return *this->left_ == *this->right_; }
    bool IsTypeRaised() const {
        return (right_->IsFunctor() && *right_->GetLeft() == *left_);
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

    Cat Arg(int argn) const {
        if (argn == this->NArgs()) {
            return this->right_;
        } else {
            return this->left_->Arg(argn);
        }
    }

    Cat HeadCat() const { return this->left_->HeadCat(); }

    bool IsFunctionInto(Cat cat) const {
        return (cat->Matches(this) || this->left_->IsFunctionInto(cat));
    }

    bool IsFunctionIntoModifier() const {
        return (this->IsModifier() || this->left_->IsModifier());
    }

    Functor(Cat left, const Slash& slash, Cat right, std::string& semantics)
    : Category(left->WithBrackets() + slash.ToStr() + right->WithBrackets(),
            semantics), left_(left), right_(right), slash_(slash) {}

private:
    Cat left_;
    Cat right_;
    Slash slash_;
};

class AtomicCategory: public Category
{
public:
    std::string ToStrWithoutFeat() const; // { return type_; }

    const std::string& GetType() const { return type_; }

    Feat GetFeat() const { return feat_; }

    Cat GetLeft() const { throw std::logic_error("not implemented"); }

    Cat GetRight() const { throw std::logic_error("not implemented"); }

    Slash GetSlash() const { throw std::logic_error("not implemented"); }

    const std::string WithBrackets() const { return ToStr(); }

    bool IsModifier() const { return false; }
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
                 this->feat_->Matches(other->GetFeat()) ||
                 this->feat_->Matches(kNB)));
    }

    const AtomicCategory* Arg(int argn) const {
        if (argn == 0)
            return this;
        throw std::runtime_error("Error getting argument of category");
    }

    const AtomicCategory* HeadCat() const { return this; }

    bool IsFunctionInto(Cat cat) const { return cat->Matches(this); }

    bool IsFunctionIntoModifier() const { return false; }

    AtomicCategory(const std::string& type, Feat feat, const std::string& semantics)
        : Category(type + feat->ToStr(), semantics), type_(type), feat_(feat) {}

private:
    std::string type_;
    Feat feat_;
};

} // namespace cat
} // namespace myccg


namespace std {

template<>
struct equal_to<myccg::cat::Cat>
{
    inline bool operator () (myccg::cat::Cat c1, myccg::cat::Cat c2) const {
        return c1->GetId() == c2->GetId();
    }
};

template<>
struct hash<myccg::cat::Cat>
{
    inline size_t operator () (myccg::cat::Cat c) const {
        return c->GetId();
    }
};

template<>
struct hash<myccg::cat::CatPair>
{
    inline size_t operator () (const myccg::cat::CatPair& p) const {
        return ((p.first->GetId() << 31) | (p.second->GetId()));
    }
};

} // namespace std
#endif
