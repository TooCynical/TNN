# Parse a file containing training patterns in the following format:
# For each pattern, one line should be provided, containing first the entries
# of the input vector, separated by a single space, then at least 2 spaces,
# followed by the entries of the desired output vector, separated by a single space.
# The behaviour of this function is undefined for wrongly formatted input files!
def parse_training_file(filepath):
    file = open(filepath)
    lines = file.readlines()

    patterns = []
    for line in lines:
        spl = re.split("  +", line)
        if len(spl) < 2:
            spl = re.split("\t", line)
        X = np.fromstring(spl[0], sep=" ")
        Y = np.fromstring(spl[1], sep=" ")
        patterns.append(pattern(X, Y))    
    return patterns