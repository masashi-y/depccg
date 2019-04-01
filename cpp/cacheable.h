
#ifndef INCLUDE_CACHEABLE_H_
#define INCLUDE_CACHEABLE_H_

#ifdef _OPENMP
#include <omp.h>
#endif
#include <unordered_map>
#include <iostream>

namespace myccg {

template<typename T>
class Cacheable
{
public:
    typedef const T* Cached;
    Cacheable() {
    #pragma omp atomic capture
        id_ = ids_()++;
    }
    static int& ids_() {
        static int ids = 0;
        return ids;
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
    int id_;
};

} // namespace myccg

#endif
