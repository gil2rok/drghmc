def compute_accept_prob(diagnostic):
    """From individual proposal acceptance probabilities, compute overall acceptance
    probability"""
    # extract acceptance probabilities from diagnostic dict
    acceptance_probs = []
    for k, v in diagnostic.items():
        if "acceptance" in k:
            acceptance_probs.append(v)

    # compute overall acceptance probability
    acceptance = 0
    for i, a in enumerate(acceptance_probs):
        term = a
        for j in range(i):
            term *= 1 - acceptance_probs[j]
        acceptance += term
    return acceptance