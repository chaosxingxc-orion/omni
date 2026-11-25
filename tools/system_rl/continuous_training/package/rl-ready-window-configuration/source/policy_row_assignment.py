"""Length-balanced policy-only ownership for provisional microbatch production."""
from dependencies import require

def assign_policy_rows(lengths, world, micro, accumulation):
    """Same deterministic ownership for the CPU producer and the gradient consumer."""
    require(type(world) is int and world>0 and all(type(n) is int and n>0 for n in lengths),
            'invalid row geometry')
    loads = [0]*world; owners = [-1]*len(lengths)
    if micro is not None:
        require(type(micro) is int and micro > 0 and type(accumulation) is int and accumulation > 0
                and len(lengths) <= world*micro*accumulation, 'cohort exceeds configured policy accumulation capacity')
    counts = [0]*world
    for row in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        rank = min((r for r in range(world) if micro is None or counts[r]<micro*accumulation), key=lambda r: (loads[r], r))
        counts[rank] += 1
        owners[row] = rank; loads[rank] += int(lengths[row])
    return owners,loads
