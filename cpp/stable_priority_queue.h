#include <queue>

template<typename T, typename Compare> 
struct stable_compare {
  bool operator()(const std::pair<T,unsigned>& x, const std::pair<T,unsigned>& y) const {
    if ( comp_(x.first, y.first) ) return true;
    if ( comp_(y.first, x.first) ) return false;
    return x.second > y.second;
  }
  Compare comp_;
};

template<typename T, typename Compare> 
struct helper {
  typedef std::pair<T,unsigned> value_type;
  typedef std::vector<value_type> container_type;
  typedef stable_compare<T,Compare> compare_type;
  typedef std::priority_queue<value_type, container_type, compare_type> stable_priority_queue;
};

