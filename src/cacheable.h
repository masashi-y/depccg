
#ifndef INCLUDE_CACHEABLE_H_
#define INCLUDE_CACHEABLE_H_

#include <unordered_map>

namespace myccg {

class Cacheable
{
public:
    Cacheable();
    static std::unordered_map<std::string, const Cacheable*>& cache_() {
        static std::unordered_map<std::string, const Cacheable*> cache;
        return cache;
    }
    bool operator==(const Cacheable& other) { return this->id_ == other.id_; }
    bool operator==(const Cacheable& other) const { return this->id_ == other.id_; }
    inline int GetId() const { return id_; }
    void RegisterCache(const std::string& key) const;
    static unsigned Count(const std::string& string) { return cache_().count(string); }


    template <typename T>
    static T Get(const std::string& key) {
        return static_cast<T>(cache_()[key]);
    }

private:
    static int ids;
    int id_;
};

} // namespace myccg

#endif
