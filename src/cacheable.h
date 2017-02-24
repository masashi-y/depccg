
#ifndef INCLUDE_CACHEABLE_H_
#define INCLUDE_CACHEABLE_H_

#include <omp.h>
#include <unordered_map>

namespace myccg {

template<typename T>
class Cacheable
{
public:
    typedef const T* Cached;
    Cacheable() {
    #pragma omp atomic capture
        id_ = ids++;
    }
    static std::unordered_map<std::string, Cached>& cache_() {
        static std::unordered_map<std::string, Cached> cache;
        return cache;
    }
    bool operator==(const Cacheable<T>& other) { return this->id_ == other.id_; }
    bool operator==(const Cacheable<T>& other) const { return this->id_ == other.id_; }
    inline int GetId() const { return id_; }
    void RegisterCache(const std::string& key) const {
    #pragma omp critical(RegisterCache)
        cache_().emplace(key, static_cast<Cached>(this));
    }
    static unsigned Count(const std::string& string) { return cache_().count(string); }


    static Cached Get(const std::string& key) {
        return cache_()[key];
    }

private:
    static int ids;
    int id_;
};

template<typename T>
int Cacheable<T>::ids = 0;

} // namespace myccg

#endif
