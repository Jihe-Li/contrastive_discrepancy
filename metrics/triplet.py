def comp_triplet(flow, l1=True):
    if l1:
        diff = flow.abs().sum(-1).mean()
    else:
        diff = (flow ** 2).sum(-1).mean()

    return diff
