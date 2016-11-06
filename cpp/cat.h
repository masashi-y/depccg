
#ifndef INCLUDE_CAT_H_
#define INCLUDE_CAT_H_

#include <iostream>
#include <string>
#include <stdexcept>
#include <regex>
#define print(value) std::cout << (value) << std::endl;

namespace myccg {
namespace cat {

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
    static const Slash* fwd_ptr;
    static const Slash* bwd_ptr;
    static const Slash* either_ptr;

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

    SlashE slash_;
};

class Category
{
private:
    static int num_cats;

public:
    Category(const std::string& str, const std::string& semantics)
        : str_(semantics.empty() ? str : str + "{" + semantics + "}"), id_(num_cats++) {}

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
    virtual const Category* GetLeft() const = 0;
    virtual const Category* GetRight() const = 0;
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
    virtual const std::string GetSubstitution(const Category* other) const = 0;
    virtual bool Matches(const Category* other) const = 0;

    // def replace_arg(self, argn, new_cat):
    //     if argn == self.n_args:
    //         return Cat.make(self.left, self.slash, new_cat)
    //     else:
    //         return Cat.make(
    //                 self.left.replace_arg(argn, new_cat), self.slash, self.right)

    virtual const Category* Arg(int argn) const = 0;
    virtual const Category* HeadCat() const = 0;
    virtual bool IsFunctionInto(const Category& cat) const = 0;
    virtual bool IsFunctionIntoModifier() const = 0;

    // def drop_PP_and_PR_feat(self):
    //     return Cat.make(self.left.drop_PP_and_PR_feat(),
    //                      self.slash,
    //                      self.right.drop_PP_and_PR_feat())

    const Category* Substitute(const std::string& sub) const;

private:
    const int id_;
    const std::string str_;
};


class Functor: public Category
{
public:
    Functor(const Category* left, const Slash* slash,
            const Category* right, std::string& semantics)
    : Category(left->WithBrackets() + slash->ToStr() + right->WithBrackets(),
            semantics), left_(left), right_(right), slash_(slash) {
    }

    virtual const std::string& GetType() const { throw std::logic_error("not implemented"); }

    virtual const std::string& GetFeat() const { throw std::logic_error("not implemented"); }

    // const Category& GetLeft() { return left_;  }
    const Category* GetLeft() const { return left_;  }

    // const Category& GetRight() { return right_;  }
    const Category* GetRight() const { return right_;  }

    // const Slash& GetSlash() { return slash_;  }
    const Slash* GetSlash() const { return slash_;  }

    const std::string WithBrackets() const { return "(" + ToStr() + ")"; }

    bool IsModifier() const { return this->left_ == this->right_; }
    bool IsTypeRaised() const {
        return (right_->IsFunctor() &&
                right_->GetLeft() == left_);
    }

    bool IsForwardTypeRaised() const {
        return (this->IsTypeRaised() &&
                this->slash_->IsForward());
    }

    bool IsBackwardTypeRaised() const {
        return (this->IsTypeRaised() &&
                this->slash_->IsForward());
    }

    bool IsFunctor() const { return true; }
    bool IsPunct() const { return false; }
    bool IsNorNP() const { return false; }
    int NArgs() const { return 1 + this->left_->NArgs(); }

    const std::string GetSubstitution(const Category* other) const {
        const Functor& o = other->As<Functor>();
        const std::string res = this->right_->GetSubstitution(o.right_);
        if (res.empty())
            return this->left_->GetSubstitution(o.left_);
        return res;
    }

    bool Matches(const Category* other) const {
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

    const Category* Arg(int argn) const {
        if (argn == this->NArgs()) {
            return this->right_;
        } else {
            return this->left_->Arg(argn);
        }
    }

    const Category* HeadCat() const { return this->left_->HeadCat(); }

    bool IsFunctionInto(const Category& cat) const {
        return (cat.Matches(this) ||
                this->left_->IsFunctionInto(cat));
    }

    bool IsFunctionIntoModifier() const {
        return (this->IsModifier() ||
                this->left_->IsModifier());
    }

    // def drop_PP_and_PR_feat(self):
    //     return Cat.make(self.left.drop_PP_and_PR_feat(),
    //                      self.slash,
    //                      self.right.drop_PP_and_PR_feat())

private:
    const Category* left_;
    const Category* right_;
    const Slash* slash_;
};

class AtomicCategory: public Category
{
public:
    AtomicCategory(const std::string& type, const std::string& feat,
            const std::string& semantics)
        : Category(type + (feat.empty() ? "" : "[" + feat + "]"), semantics),
          type_(type), feat_(feat) {}

    // const std::string& GetType() { return type_; }
    const std::string& GetType() const { return type_; }

    // const std::string& GetFeat() { return feat_; }
    const std::string& GetFeat() const { return feat_; }

    const Category* GetLeft() const { throw std::logic_error("not implemented"); }

    const Category* GetRight() const { throw std::logic_error("not implemented"); }

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

    const std::string GetSubstitution(const Category* other) const {
        const AtomicCategory& o = other->As<AtomicCategory>();
        if (this->feat_ == kWILDCARD) {
            return o.feat_;
        } else if (o.feat_ == kWILDCARD) {
            return this->feat_;
        } return "";
    }

    bool Matches(const Category* other) const {
        if (!other->IsFunctor()) {
            return (this->type_ == other->GetType() &&
                    (this->feat_.empty() ||
                     this->feat_ == other->GetFeat() ||
                     kWILDCARD == this->feat_ ||
                     kWILDCARD == other->GetFeat() ||
                     this->feat_ == "nb"));
        }
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
    std::string type_;
    std::string feat_;
};

const Category* parse_uncached(const std::string& cat);

const Category* parse(const std::string& cat);

const Category* make(const Category* left, const Slash* op, const Category* right);

const Category* CorrectWildcardFeatures(const Category* to_correct,
        const Category* match1, const Category* match2);

extern const Category* COMMA;
extern const Category* SEMICOLON;
extern const Category* CONJ;
extern const Category* N;
extern const Category* LQU;
extern const Category* LRB;
extern const Category* NP;
extern const Category* PP;
extern const Category* PREPOSITION;
extern const Category* PR;

} // namespace cat
} // namespace myccg
#endif
