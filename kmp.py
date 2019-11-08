def force_match(text, pattern):
    if not text or not pattern:
        return -1
    i = 0
    j = 0
    while (i < len(text) and j < len(pattern)):
        if text[i] == pattern[j]:
            i += 1
            j += 1
        else:
            i = i - j + 1
            j = 0
    if j == len(pattern):
        return (i - j)
    return -1


def next_positions(pattern):
    if not pattern:
        return []
    next = [-1, ]
    k, j = -1, 0
    while j < len(pattern) - 1:
        if k == -1 or pattern[j] == pattern[k]:
            k, j = k + 1, j + 1
            if pattern[j] != pattern[k]:
                next.append(k)
            else:
                next.append(next[k])
        else:
            k = next[k]
    return next


def kmp_match(text, pattern):
    next = next_positions(pattern)
    i, j = 0, 0
    while i < len(text) and j < len(pattern):
        if j == -1 or text[i] == pattern[j]:
            i, j = i + 1, j + 1
        else:
            j = next[j]
    if j == len(pattern):
        return i - j
    return -1


if __name__ == '__main__':
    text = "abcd"
    pattern = "bc"
    pattern2 = "dabcdabde"
    print(force_match(text, pattern))
    print(kmp_match(text, pattern))
    # print(next_positions(pattern))
    # print(next_positions(pattern2))
    print("new case", kmp_match("abacababc", "abab"))
