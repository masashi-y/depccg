
#ifndef INCLUDE_TREE_H_
#define INCLUDE_TREE_H_

#include "cat.h"
#include "combinator.h"
#include <sstream>
#include <vector>
#include <memory>


namespace myccg {


class Node;
class Leaf;
class Tree;
typedef std::shared_ptr<const Node> NodeType;
typedef std::shared_ptr<const Tree> TreeType;
typedef std::shared_ptr<const Leaf> LeafType;

class FormatVisitor {
public:
    virtual int Visit(const Leaf* leaf) = 0;
    virtual int Visit(const Tree* leaf) = 0;
};

class Node
{
public:
    Node(Cat cat, const RuleType rule_type)
    : cat_(cat), rule_type_(rule_type) {}

    virtual ~Node() {}

    Cat GetCategory() { return cat_; }
    Cat GetCategory() const { return cat_; }
    const RuleType GetRuleType() { return rule_type_; }
    const RuleType GetRuleType() const { return rule_type_; }
    const std::string ToStr() const;
    virtual int GetHeadId() const = 0;
    virtual int GetDependencyLength() const = 0;
    virtual bool HeadIsLeft() const = 0;
    virtual bool IsUnary() const = 0;
    virtual int NumDescendants() const = 0;
    virtual int RightNumDescendants() const = 0;
    virtual int LeftNumDescendants() const = 0;
    virtual int Accept(FormatVisitor& visitor) const = 0;
    virtual const Node* GetLeftMostChild() const = 0;


    friend std::ostream& operator<<(std::ostream& ost, const Node* node) {
        ost << node->ToStr();
        return ost;
    }

protected:
    Cat cat_;
    const RuleType rule_type_;
};
        
class Leaf: public Node
{
public:
    Leaf(const std::string& word, Cat cat, int position)
    : Node(cat, LEXICON), word_(word), position_(position) {}

    ~Leaf() {}
    std::string GetWord() const { return word_; }
    int GetPosition() const { return position_; }
    int GetHeadId() const { return position_; }
    int GetDependencyLength() const { return 0; }
    bool HeadIsLeft() const { return false; }
    bool IsUnary() const { return false; }
    int NumDescendants() const { return 0; }
    int RightNumDescendants() const { return 0; }
    int LeftNumDescendants() const { return 0; }
    int Accept(FormatVisitor& visitor) const { return visitor.Visit(this); }
    const Node* GetLeftMostChild() const { return this; }

private:
    const std::string word_;
    const int position_;
};

class Tree: public Node
{
public:
    typedef std::shared_ptr<const Node> ChildType;

    Tree(Cat cat, bool left_is_head, const Node* lchild,
            const Node* rchild, Op rule)
    : Node(cat, rule->GetRuleType()), left_is_head_(left_is_head),
      lchild_(lchild), rchild_(rchild), rule_(rule),
      dependency_length_(rchild_->GetHeadId() - lchild_->GetHeadId() +
            rchild_->GetDependencyLength() + lchild_->GetDependencyLength()) {}

    Tree(Cat cat, bool left_is_head, ChildType lchild,
            ChildType rchild, Op rule)
    : Node(cat, rule->GetRuleType()), left_is_head_(left_is_head),
      lchild_(lchild), rchild_(rchild), rule_(rule),
      dependency_length_(rchild_->GetHeadId() - lchild_->GetHeadId() +
            rchild_->GetDependencyLength() + lchild_->GetDependencyLength()) {}

    Tree(Cat cat, const Node* lchild)
    : Node(cat, UNARY), left_is_head_(true),
      lchild_(lchild), rchild_(NULL), rule_(unary_rule),
      dependency_length_(lchild_->GetDependencyLength()) {}

    Tree(Cat cat, ChildType lchild)
    : Node(cat, UNARY), left_is_head_(true),
      lchild_(lchild), rchild_(NULL), rule_(unary_rule),
      dependency_length_(lchild_->GetDependencyLength()) {}

    ~Tree() {}

    int GetHeadId() const {
        if (NULL == rchild_)
            return lchild_->GetHeadId();
        else
            return left_is_head_ ? lchild_->GetHeadId() : rchild_->GetHeadId();
    }

    int GetDependencyLength() const { return dependency_length_; }
    bool HeadIsLeft() const { return left_is_head_; }
    bool IsUnary() const { return NULL == rchild_; }
    int NumDescendants() const {
        return ( rchild_ == NULL ? 0 : RightNumDescendants() ) + LeftNumDescendants();
    }

    int LeftNumDescendants() const { return lchild_->NumDescendants() + 1; }
    int RightNumDescendants() const { return rchild_->NumDescendants() + 1; }
    ChildType GetLeftChild() const { return lchild_; }
    ChildType GetRightChild() const { return rchild_; }
    Op GetRule() const { return rule_; }
    int Accept(FormatVisitor& visitor) const { return visitor.Visit(this); }
    const Node* GetLeftMostChild() const { return lchild_->GetLeftMostChild(); }

private:
    bool left_is_head_;
    ChildType lchild_;
    ChildType rchild_;
    Op rule_;
    int dependency_length_;
};

void ToXML(std::vector<std::shared_ptr<const Node>>&
        trees, std::ostream& out=std::cout);

void ToXML(std::vector<const Node*>& trees, std::ostream& out=std::cout);


class GetLeaves: public FormatVisitor {
    typedef std::vector<const Leaf*> result_type;

public:
    GetLeaves() {}
    result_type operator()(const Node* node) {
        node->Accept(*this);
        return leaves_;
    }

    int Visit(const Tree* tree) {
        tree->GetLeftChild()->Accept(*this);
        if (! tree->IsUnary())
            tree->GetRightChild()->Accept(*this);
        return 0;
    }

    int Visit(const Leaf* leaf) {
        leaves_.push_back(leaf);
        return 0;
    }

result_type leaves_;
};

class Derivation: public FormatVisitor {

public:
    Derivation(const Node* tree): tree_(tree), lwidth_(0) { Process(); }
    Derivation(NodeType tree): tree_(tree.get()), lwidth_(0) { Process(); }

    void Process();
    std::string Get() const { return out_.str(); }
    friend std::ostream& operator<<(std::ostream& ost, const Derivation& deriv) {
        ost << deriv.out_.str();
        return ost;
    }
    int Visit(const Tree* tree);
    int Visit(const Leaf* leaf);

private:
    const Node* tree_;
    std::stringstream out_;
    int lwidth_;
};

class AUTO: public FormatVisitor {
public:
    AUTO(const Node* tree): tree_(tree) { Process(); }
    AUTO(NodeType tree): tree_(tree.get()) { Process(); }

    void Process() { tree_->Accept(*this); }
    std::string Get() const { return out_.str(); }

    int Visit(const Tree* tree) {
        out_ << "(<T "
             << tree->GetCategory() << " "
             << (tree->HeadIsLeft() ? "0 " : "1 ")
             << (tree->IsUnary() ? "1" : "2")
             << "> ";
        tree->GetLeftChild()->Accept(*this);
        if (! tree->IsUnary())
            tree->GetRightChild()->Accept(*this);
        out_ << " )";
        return 0;
    }

    int Visit(const Leaf* leaf) {
        std::string pos = "POS";
        out_ << "(<L "
             << leaf->GetCategory() << " "
             << pos << " "
             << pos << " "
             << leaf->GetWord() << " "
             << leaf->GetCategory() << ">)";
        return 0;
    }

private:
    const Node* tree_;
    std::stringstream out_;
};

class XML: public FormatVisitor {

public:
    XML(const Node* tree): tree_(tree) { Process(); }
    XML(NodeType tree): tree_(tree.get()) { Process(); }

    void Process() { tree_->Accept(*this); }
    std::string Get() const { return out_.str(); }
    friend std::ostream& operator<<(std::ostream& ost, const XML& xml) {
        return ost << xml.out_.str();
    }

    int Visit(const Tree* tree);
    int Visit(const Leaf* leaf);

private:
    const Node* tree_;
    std::stringstream out_;
};

} // namespace myccg

#endif
