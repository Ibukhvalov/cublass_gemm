 #include <vector>
 #include <random>

template <typename T>
std::vector<T> rand_vector(size_t size) {
    std::random_device rnd_device;
    std::mt19937 eng(rnd_device());
    std::uniform_real_distribution<T> dist(-10.f, 10.f);

    std::vector<T> vec(size);
    std::generate(vec.begin(), vec.end(), [&]() {
        return dist(eng);
    });

    return vec;
}
