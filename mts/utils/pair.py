def generate_exhaustive(no):
    return [(i, j) for i in range(no) for j in range(i + 1, no)]
