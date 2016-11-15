
#ifndef INCLUDE_CAT_H_
#define INCLUDE_CAT_H_

#include <iostream>
#include <string>
#include <stdexcept>
#include <regex>
#include <omp.h>
#include <unordered_map>
#include <unordered_set>
#include "feat.h"

#define print(value) std::cout << (value) << std::endl;

namespace myccg {
namespace cat {

class Category;
class Slash;
class FeatureValue;

typedef const Category* Cat;
typedef std::pair<Cat, Cat> CatPair;

Cat parse_uncached(const std::string& cat);

Cat parse(const std::string& cat);

Cat make(Cat left, const Slash* op, Cat right);

Cat CorrectWildcardFeatures(Cat to_correct, Cat match1, Cat match2);

// const std::regex reg_no_punct("[a-zA-Zz]+");

class Slash
{
public:
    enum SlashE {
        FwdApp = 0,
        BwdApp = 1,
        EitherApp = 2,
    };

public:
    static const Slash* Fwd() { return fwd_ptr; }
    static const Slash* Bwd() { return bwd_ptr; }
    static const Slash* Either() { return either_ptr; }

    const std::string ToStr() const {
        switch (slash_) {
            case 0:
                return "/";
            case 1:
                return "\\";
            case 2:
                return "|";
            default:
                throw;
        }
    }

    static const Slash* FromStr(const std::string& string) {
        if (string ==  "/")
                return fwd_ptr;
        else if (string == "\\")
                return bwd_ptr;
        else if (string == "|")
                return either_ptr;
        throw std::runtime_error("Slash must be initialized with slash string.");
    }

    bool Matches(const Slash* other) const {
        return (this->slash_ == EitherApp ||
                other->slash_ == EitherApp ||
                other->slash_ == this->slash_); }

    bool operator==(const Slash* other) const { return this->slash_ == other->slash_; }

    bool IsForward() const { return slash_ == FwdApp; } 
    bool IsBackward() const { return slash_ == BwdApp; } 

private:
    Slash(SlashE slash): slash_(slash) {}

    static const Slash* fwd_ptr;
    static const Slash* bwd_ptr;
    static const Slash* either_ptr;

    SlashE slash_;
};

class Category
{
private:
    static int num_cats;

public:
    bool operator==(const Category& other) { return this->id_ == other.id_; }
    bool operator==(const Category& other) const { return this->id_ == other.id_; }

    template<typename T>
    const T& As() const { return dynamic_cast<const T&>(*this); }
    
    template<typename T>
    T& As() { return dynamic_cast<T&>(*this); }

    std::string ToStr() { return str_; }
    std::string ToStr() const { return str_; }

    virtual std::string ToStrWithoutFeat() const = 0;

    Cat StripFeat() const { return parse(this->ToStrWithoutFeat()); }

    inline int Hashcode() { return id_; }

    const int GetId() const { return id_; }

    virtual const std::string& GetType() const = 0;
    virtual Feat GetFeat() const = 0;
    virtual Cat GetLeft() const = 0;
    virtual Cat GetRight() const = 0;
    virtual const Slash* GetSlash() const = 0;

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
        #pragma omp atomic capture
        id_ = num_cats++;
    }

private:
    int id_;
    std::string str_;
};


class Functor: public Category
{
public:
    std::string ToStrWithoutFeat() const {
        return "(" + left_->ToStrWithoutFeat() +
                slash_->ToStr() + right_->ToStrWithoutFeat() + ")";
    }

    virtual const std::string& GetType() const { throw std::logic_error("not implemented"); }

    virtual Feat GetFeat() const { throw std::logic_error("not implemented"); }

    Cat GetLeft() const { return left_;  }

    Cat GetRight() const { return right_;  }

    const Slash* GetSlash() const { return slash_;  }

    const std::string WithBrackets() const { return "(" + ToStr() + ")"; }

    bool IsModifier() const { return this->left_ == this->right_; }
    bool IsTypeRaised() const {
        return (right_->IsFunctor() && right_->GetLeft() == left_);
    }

    bool IsForwardTypeRaised() const {
        return (this->IsTypeRaised() && this->slash_->IsForward());
    }

    bool IsBackwardTypeRaised() const {
        return (this->IsTypeRaised() && this->slash_->IsForward());
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
        if (other->IsFunctor()) {
            return (this->left_->Matches(other->GetLeft()) &&
                    this->right_->Matches(other->GetRight()) &&
                    this->slash_->Matches(other->GetSlash()));
        }
        return false;
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

private:
    Functor(Cat left, const Slash* slash, Cat right, std::string& semantics)
    : Category(left->WithBrackets() + slash->ToStr() + right->WithBrackets(),
            semantics), left_(left), right_(right), slash_(slash) {
    }

    friend Cat parse_uncached(const std::string& cat);

    Cat left_;
    Cat right_;
    const Slash* slash_;
};

class AtomicCategory: public Category
{
public:
    std::string ToStrWithoutFeat() const { return type_; }

    const std::string& GetType() const { return type_; }

    Feat GetFeat() const { return feat_; }

    Cat GetLeft() const { throw std::logic_error("not implemented"); }

    Cat GetRight() const { throw std::logic_error("not implemented"); }

    const Slash* GetSlash() const { throw std::logic_error("not implemented"); }

    const std::string WithBrackets() const { return ToStr(); }

    bool IsModifier() const { return false; }
    bool IsTypeRaised() const { return false; }
    bool IsForwardTypeRaised() const { return false; }
    bool IsBackwardTypeRaised() const { return false; }
    bool IsFunctor() const { return false; }

    bool IsPunct() const {
        // return (!std::regex_match(type_, reg_no_punct) ||
        return (!( 'A' <= type_[0] && type_[0] <= 'Z' ) ||
                type_ == "LRB" || type_ == "RRB" ||
                type_ == "LQU" || type_ == "RQU");
    }

    bool IsNorNP() const { return type_ == "N" || type_ == "NP"; }

    int NArgs() const { return 0; }

    Feat GetSubstitution(Cat other) const {
        if (this->feat_->Matches(kWILDCARD)) {
            return other->GetFeat();
        } else if (other->GetFeat()->Matches(kWILDCARD)) {
            return this->GetFeat();
        }
        return kNONE;
    }

    bool Matches(Cat other) const {
        if (!other->IsFunctor()) {
            return (this->type_ == other->GetType() &&
                    (this->feat_->IsEmpty() ||
                     this->feat_->Matches(other->GetFeat()) ||
                     kWILDCARD->Matches(this->feat_) ||
                     kWILDCARD->Matches(other->GetFeat()) ||
                     this->feat_->Matches(kNB)));
        }
        return false;
    }

    const AtomicCategory* Arg(int argn) const {
        if (argn == 0)
            return this;
        throw std::runtime_error("Error getting argument of category");
    }

    const AtomicCategory* HeadCat() const { return this; }

    bool IsFunctionInto(Cat cat) const { return cat->Matches(this); }

    bool IsFunctionIntoModifier() const { return false; }

private:
    AtomicCategory(const std::string& type, Feat feat, const std::string& semantics)
        : Category(type + feat->ToStr(), semantics),
          type_(type), feat_(feat) {}

    friend Cat parse_uncached(const std::string& cat);

    std::string type_;
    Feat feat_;
};

extern Cat COMMA;
extern Cat SEMICOLON;
extern Cat CONJ;
extern Cat N;
extern Cat LQU;
extern Cat LRB;
extern Cat NP;
extern Cat NPbNP;
extern Cat PP;
extern Cat PREPOSITION;
extern Cat PR;

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
