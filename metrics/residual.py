def field_norm(flow, order=1):
    """Mean norm of a displacement field ``[N, 3]``.

    This is the discrepancy function behind CD3: the residual field returned by
    registering the two warped moving images against each other.
    """
    if order == 1:
        return flow.abs().sum(-1).mean()
    return flow.norm(dim=-1).mean()
