#include <iostream>
#include <stdexcept>

#include "halo_exchange.hpp"

bool check_owner(
    int index,
    int N,
    int ranks,
    int expected
) {
    const int actual = find_owner_rank(index, N, ranks);

    if (actual != expected) {
        std::cerr << "FAIL: index=" << index
                  << ", expected=" << expected
                  << ", actual=" << actual << "\n";
        return false;
    }

    std::cout << "PASS: index=" << index
              << ", owner=" << actual << "\n";
    return true;
}

int main() {
    bool ok = true;

    // Even partition: N=9, ranks=3.
    ok = check_owner(0, 9, 3, 0) && ok;
    ok = check_owner(4, 9, 3, 1) && ok;
    ok = check_owner(8, 9, 3, 2) && ok;

    // Uneven partition: N=9, ranks=2.
    ok = check_owner(4, 9, 2, 0) && ok;
    ok = check_owner(5, 9, 2, 1) && ok;

    // Zero-row ranks: N=4, ranks=5.
    ok = check_owner(3, 4, 5, 3) && ok;

    // Invalid index.
    bool caught = false;
    try {
        find_owner_rank(9, 9, 2);
    } catch (const std::invalid_argument&) {
        caught = true;
    }

    if (!caught) {
        std::cerr << "FAIL: Invalid index accepted\n";
        ok = false;
    } else {
        std::cout << "PASS: Invalid index rejected\n";
    }

    if (ok) {
        std::cout << "PASS: All Halo owner tests\n";
    }

    return ok ? 0 : 1;
}
