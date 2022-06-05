import math
def calculate_first_n_primes(n):
    primes = []
    for curr_candidate in range(2,n+1):
        if curr_candidate % 500 == 0:
            print(f"curr_candiate == {curr_candidate}")
        is_prime = True
        for witness in range(2, math.ceil(math.sqrt(curr_candidate))+1):
            if curr_candidate % witness == 0:
                is_prime = False
        if is_prime:
            primes.append(curr_candidate)
    return primes

if __name__ == "__main__":
    primes = calculate_first_n_primes(1_000_000)
    print(primes)