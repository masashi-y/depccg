
#ifndef INCLUDE_TREE_H_
#define INCLUDE_TREE_H_

#include "cat.h"
#include "combinator.h"
#include <sstream>
#include <vector>
#include <memory>


namespace myccg {
namespace tree {

using cat::Cat;

class Leaf;
class Tree;

class Node
{
public:
    Node(Cat cat, const combinator::RuleType rule_type)
    : cat_(cat), rule_type_(rule_type) {}

    virtual ~Node() {}

    Cat GetCategory() { return cat_; }
    Cat GetCategory() const { return cat_; }

    const combinator::RuleType GetRuleType() { return rule_type_; }
    const combinator::RuleType GetRuleType() const { return rule_type_; }

    virtual const std::string ToStr() const = 0;
    virtual int GetHeadId() const = 0;
    virtual int GetDependencyLength() const = 0;

    // to call ShowDerivation
    friend Tree;

private:
    virtual int ShowDerivation(int lwidth, std::ostream& out) const = 0;
    virtual void GetLeaves(std::vector<const Leaf*>* out) const = 0;

protected:
    Cat cat_;
    const combinator::RuleType rule_type_;
};
        
class Leaf: public Node
{
public:
    Leaf(const std::string& word, Cat cat, int position)
    : Node(cat, combinator::LEXICON), word_(word), position_(position) {}

    ~Leaf() {}

    const std::string ToStr() const {
        std::stringstream out;
        std::string pos = "POS";
        out << "(<L ";
        out << cat_->ToStr() << " ";
        out << pos << " ";
        out << pos << " ";
        out << word_ << " ";
        out << cat_->ToStr() << ">)";
        return out.str();
    }

    std::string GetWord() const { return word_; }

    int GetPosition() const { return position_; }

    int GetHeadId() const { return position_; }

    int GetDependencyLength() const { return 0; }

private:
    int ShowDerivation(int lwidth, std::ostream& out) const;

    void GetLeaves(std::vector<const Leaf*>* out) const {
        out->push_back(this);
    }

private:
    const std::string word_;
    const int position_;
};

class Tree: public Node
{
public:
    Tree(Cat cat, bool left_is_head, const Node* lchild,
            const Node* rchild, const combinator::Combinator* rule)
    : Node(cat, rule->GetRuleType()), left_is_head_(left_is_head),
      lchild_(lchild), rchild_(rchild), rule_(rule) {}

    Tree(Cat cat, bool left_is_head, std::shared_ptr<const Node> lchild,
            std::shared_ptr<const Node> rchild, const combinator::Combinator* rule)
    : Node(cat, rule->GetRuleType()), left_is_head_(left_is_head),
      lchild_(lchild), rchild_(rchild), rule_(rule) {}

    Tree(Cat cat, const Node* lchild)
    : Node(cat, combinator::UNARY), left_is_head_(true),
      lchild_(lchild), rchild_(NULL), rule_(combinator::unary_rule) {}

    Tree(Cat cat, std::shared_ptr<const Node> lchild)
    : Node(cat, combinator::UNARY), left_is_head_(true),
      lchild_(lchild), rchild_(NULL), rule_(combinator::unary_rule) {}

    ~Tree() {}

    const std::string ToStr() const {
        std::stringstream out;
        out << "(<T ";
        out << this->cat_->ToStr() << " ";
        out << (left_is_head_ ? "0 " : "1 ");
        out << (NULL == rchild_ ? "1 " : "2 ");
        out << lchild_->ToStr();
        if (NULL !=  rchild_)
            out << " " << rchild_->ToStr();
        out << " )";
        return out.str();
    }

    const Node* GetLChild() const { return lchild_.get(); }

    const Node* GetRChild() const { return rchild_.get(); }

    int GetHeadId() const {
        if (NULL == rchild_)
            return lchild_->GetHeadId();
        else
            return left_is_head_ ? lchild_->GetHeadId() : rchild_->GetHeadId();
    }

    int GetDependencyLength() const {
        if (NULL == rchild_)
            return lchild_->GetDependencyLength();
        else
            return (rchild_->GetHeadId() - lchild_->GetHeadId() +
                    rchild_->GetDependencyLength() + lchild_->GetDependencyLength());
    }

private:
    int ShowDerivation(int lwidth, std::ostream& out) const;

    void GetLeaves(std::vector<const Leaf*>* out) const {
        lchild_->GetLeaves(out);
        if (NULL != rchild_)
            rchild_->GetLeaves(out);
    }

    friend std::vector<const Leaf*> GetLeaves(const Tree* tree);
    friend void ShowDerivation(const Tree* tree, std::ostream& out);

private:
    bool left_is_head_;
    std::shared_ptr<const Node> lchild_;
    std::shared_ptr<const Node> rchild_;
    const combinator::Combinator* rule_;
};

std::vector<const Leaf*> GetLeaves(const Tree* tree);

void ShowDerivation(const Tree* tree, std::ostream& out=std::cout);

void ShowDerivation(std::shared_ptr<const Node> tree, std::ostream& out=std::cout);

void test();

} // namespace tree
} // namespace myccg

#endif
