
#ifndef INCLUDE_CAT_H_
#define INCLUDE_CAT_H_

#include <iostream>
#include <string>
#include <stdexcept>
#include <regex>
#include <omp.h>
#include <unordered_map>

#define print(value) std::cout << (value) << std::endl;

namespace myccg {
namespace cat {

class Category;
class Slash;
typedef const Category* Cat;
typedef std::pair<Cat, Cat> CatPair;

Cat parse_uncached(const std::string& cat);

Cat parse(const std::string& cat);

Cat make(Cat left, const Slash* op, Cat right);

Cat CorrectWildcardFeatures(Cat to_correct, Cat match1, Cat match2);

const std::regex reg_no_punct("([A-Za-z]+)");

const std::string kWILDCARD = "X";

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

    inline int Hashcode() { return id_; }

    const int GetId() const { return id_; }

    virtual const std::string& GetType() const = 0;
    virtual const std::string& GetFeat() const = 0;
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
    virtual const std::string GetSubstitution(Cat other) const = 0;
    virtual bool Matches(Cat other) const = 0;

    // def replace_arg(self, argn, new_cat):
    //     if argn == self.n_args:
    //         return Cat.make(self.left, self.slash, new_cat)
    //     else:
    //         return Cat.make(
    //                 self.left.replace_arg(argn, new_cat), self.slash, self.right)

    virtual Cat Arg(int argn) const = 0;
    virtual Cat HeadCat() const = 0;
    virtual bool IsFunctionInto(const Category& cat) const = 0;
    virtual bool IsFunctionIntoModifier() const = 0;

    // def drop_PP_and_PR_feat(self):
    //     return Cat.make(self.left.drop_PP_and_PR_feat(),
    //                      self.slash,
    //                      self.right.drop_PP_and_PR_feat())

    Cat Substitute(const std::string& sub) const;

protected:
    Category(const std::string& str, const std::string& semantics)
        : id_(-1), str_(semantics.empty() ? str : str + "{" + semantics + "}") {
        #pragma omp critical
        id_ = num_cats++;
    }

private:
    int id_;
    std::string str_;
};


class Functor: public Category
{
public:
    virtual const std::string& GetType() const { throw std::logic_error("not implemented"); }

    virtual const std::string& GetFeat() const { throw std::logic_error("not implemented"); }

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

    const std::string GetSubstitution(Cat other) const {
        const Functor& o = other->As<Functor>();
        const std::string res = this->right_->GetSubstitution(o.right_);
        if (res.empty())
            return this->left_->GetSubstitution(o.left_);
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

    // def replace_arg(self, argn, new_cat):
    //     if argn == self.n_args:
    //         return Cat.make(self.left, self.slash, new_cat)
    //     else:
    //         return Cat.make(
    //                 self.left.replace_arg(argn, new_cat), self.slash, self.right)

    Cat Arg(int argn) const {
        if (argn == this->NArgs()) {
            return this->right_;
        } else {
            return this->left_->Arg(argn);
        }
    }

    Cat HeadCat() const { return this->left_->HeadCat(); }

    bool IsFunctionInto(const Category& cat) const {
        return (cat.Matches(this) || this->left_->IsFunctionInto(cat));
    }

    bool IsFunctionIntoModifier() const {
        return (this->IsModifier() || this->left_->IsModifier());
    }

    // def drop_PP_and_PR_feat(self):
    //     return Cat.make(self.left.drop_PP_and_PR_feat(),
    //                      self.slash,
    //                      self.right.drop_PP_and_PR_feat())

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
    const std::string& GetType() const { return type_; }

    const std::string& GetFeat() const { return feat_; }

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
        return (!std::regex_match(type_, reg_no_punct) ||
                type_ == "LRB" || type_ == "RRB" ||
                type_ == "LQU" || type_ == "RQU");
    }

    bool IsNorNP() const { return type_ == "N" || type_ == "NP"; }

    int NArgs() const { return 0; }

    const std::string GetSubstitution(Cat other) const {
        const AtomicCategory& o = other->As<AtomicCategory>();
        if (this->feat_ == kWILDCARD) {
            return o.feat_;
        } else if (o.feat_ == kWILDCARD) {
            return this->feat_;
        } return "";
    }

    bool Matches(Cat other) const {
        if (!other->IsFunctor()) {
            return (this->type_ == other->GetType() &&
                    (this->feat_.empty() ||
                     this->feat_ == other->GetFeat() ||
                     kWILDCARD == this->feat_ ||
                     kWILDCARD == other->GetFeat() ||
                     this->feat_ == "nb"));
        }
        return false;
    }

    const Category& replace_arg(int argn, const Category& new_cat) const {
        if (argn == 0)
            return new_cat;
        throw std::runtime_error("Error replacing argument of category");
    }

    const AtomicCategory* Arg(int argn) const {
        if (argn == 0)
            return this;
        throw std::runtime_error("Error getting argument of category");
    }

    const AtomicCategory* HeadCat() const { return this; }

    bool IsFunctionInto(const Category& cat) const { return cat.Matches(this); }

    bool IsFunctionIntoModifier() const { return false; }

private:
    AtomicCategory(const std::string& type, const std::string& feat,
            const std::string& semantics)
        : Category(type + (feat.empty() ? "" : "[" + feat + "]"), semantics),
          type_(type), feat_(feat) {}

    friend Cat parse_uncached(const std::string& cat);

    std::string type_;
    std::string feat_;
};

extern Cat COMMA;
extern Cat SEMICOLON;
extern Cat CONJ;
extern Cat N;
extern Cat LQU;
extern Cat LRB;
extern Cat NP;
extern Cat PP;
extern Cat PREPOSITION;
extern Cat PR;

struct hash_cat_pair
{
    inline size_t operator () (const CatPair& p) const {
        return ((p.first->GetId() << 31) | (p.second->GetId()));
    }
};

struct cat_hash {
    inline size_t operator () (Cat c) const { return c->GetId(); }
};

struct cat_eq {
    inline bool operator () (Cat c1, Cat c2) const { return c1->GetId() == c2->GetId(); }
};

template<typename Ty>
using CatMap = std::unordered_map<Cat, Ty, cat_hash, cat_eq>;

} // namespace cat
} // namespace myccg
#endif
