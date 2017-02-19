
# from libcpp.memory cimport shared_ptr
# from libcpp.vector cimport vector
#
# cdef extern from "tree.h" namespace "myccg" nogil:
#     cdef cppclass Node:
#         pass
#
#     cdef cppclass Leaf:
#         pass
#
#     cdef cppclass Tree:
#         pass
#
#     ctypedef shared_ptr[const Node] NodeType
#     ctypedef shared_ptr[const Tree] TreeType
#     ctypedef shared_ptr[const Leaf] LeafType

    # void ToXML(vector<shared_ptr<const Node>>&
    #         trees, bool feat, std::ostream& out=std::cout);
    #
    # void ToXML(std::vector<const Node*>& trees, bool feat, std::ostream& out=std::cout);
    #
    #
    # class GetLeaves: public FormatVisitor {
    #     typedef std::vector<const Leaf*> result_type;
    #
    # public:
    #     GetLeaves() {}
    #     result_type operator()(const Node* node) {
    #         node->Accept(*this);
    #         return leaves_;
    #     }
    #
    #     int Visit(const Tree* tree) {
    #         tree->GetLeftChild()->Accept(*this);
    #         if (! tree->IsUnary())
    #             tree->GetRightChild()->Accept(*this);
    #         return 0;
    #     }
    #
    #     int Visit(const Leaf* leaf) {
    #         leaves_.push_back(leaf);
    #         return 0;
    #     }
    #
    # result_type leaves_;
    # };
    #
    # class Derivation: public FormatVisitor {
    #
    # public:
    #     Derivation(const Node* tree, bool feat=true)
    #         : tree_(tree), lwidth_(0), feat_(feat) { Process(); }
    #     Derivation(NodeType tree, bool feat=true)
    #         : tree_(tree.get()), lwidth_(0), feat_(feat) { Process(); }
    #
    #     void Process();
    #     std::string Get() const { return out_.str(); }
    #     friend std::ostream& operator<<(std::ostream& ost, const Derivation& deriv) {
    #         ost << deriv.out_.str();
    #         return ost;
    #     }
    #     int Visit(const Tree* tree);
    #     int Visit(const Leaf* leaf);
    #
    # private:
    #     const Node* tree_;
    #     std::stringstream out_;
    #     int lwidth_;
    #     bool feat_;
    # };
    #
    # class AUTO: public FormatVisitor {
    # public:
    #     AUTO(const Node* tree): tree_(tree) { Process(); }
    #     AUTO(NodeType tree): tree_(tree.get()) { Process(); }
    #
    #     void Process() { tree_->Accept(*this); }
    #     std::string Get() const { return out_.str(); }
    #
    #     int Visit(const Tree* tree);
    #     int Visit(const Leaf* leaf);
    #
    # private:
    #     const Node* tree_;
    #     std::stringstream out_;
    # };
    #
    # class JaCCG: public FormatVisitor {
    # public:
    #     JaCCG(const Node* tree): tree_(tree) { Process(); }
    #     JaCCG(NodeType tree): tree_(tree.get()) { Process(); }
    #
    #     void Process() { tree_->Accept(*this); }
    #     friend std::ostream& operator<<(std::ostream& ost, const JaCCG& deriv) {
    #         ost << deriv.out_.str();
    #         return ost;
    #     }
    #     std::string Get() const { return out_.str(); }
    #
    #     int Visit(const Tree* tree);
    #     int Visit(const Leaf* leaf);
    #
    # private:
    #     const Node* tree_;
    #     std::stringstream out_;
    # };
    #
    # class XML: public FormatVisitor {
    #
    # public:
    #     XML(const Node* tree, bool feat=true): tree_(tree), feat_(feat) { Process(); }
    #     XML(NodeType tree, bool feat=true): tree_(tree.get()), feat_(feat) { Process(); }
    #
    #     void Process() { tree_->Accept(*this); }
    #     std::string Get() const { return out_.str(); }
    #     friend std::ostream& operator<<(std::ostream& ost, const XML& xml) {
    #         return ost << xml.out_.str();
    #     }
    #
    #     int Visit(const Tree* tree);
    #     int Visit(const Leaf* leaf);
    #
    # private:
    #     const Node* tree_;
    #     bool feat_;
    #     std::stringstream out_;
    # };
    #
    # class CoNLL: public FormatVisitor {
    #
    # public:
    #     CoNLL(const Node* tree);
    #     CoNLL(NodeType tree);
    #     ~CoNLL();
    #
    #     void Process();
    #     std::string Get() const { return out_.str(); }
    #     friend std::ostream& operator<<(std::ostream& ost, const CoNLL& xml) {
    #         return ost << xml.out_.str();
    #     }
    #
    #     int Visit(const Tree* tree);
    #     int Visit(const Leaf* leaf);
    #
    # private:
    #     int id_;
    #     const Node* tree_;
    #     int length_;
    #     int* heads_;
    #     const Leaf** leaves_;
    #     std::stringstream out_;
    # };
    #
