import torch
import itertools

def mk_level_sampler(max_level=4):
    R = 8
    P = .5
    d = torch.distributions.TransformedDistribution(
        torch.distributions.NegativeBinomial(R, P),
        [torch.distributions.AffineTransform(loc=1, scale=1.0)],
    )
    return lambda x: int(d.sample().item()) if x < max_level else 99999999

def level_generator(sampler):
    rls = []
    rls.append(sampler(0))
    
    def get_level(level):
        if level == len(rls):
            rls.append(sampler(level))
        return rls[level]
            
    while True:
        yield from (0 for _ in range(rls[0]))
        rls[0] = sampler(0)
        
        for i in itertools.count(1):
            if i == len(rls):
                rls.append(sampler(i))
            
            yield i
            rls[i] -= 1 
            if rls[i] == 0:
                rls[i] = sampler(i)
            else:
                break


def mk_level_mask(levels):
    N = len(levels)
    A = []
    for level in range(levels.max().item()+1):
        A.append((levels > level).cumsum(dim=0))
    A = torch.stack(A)
    
    dl = levels[:, None] - levels[None, :]
    dA = A[:, :, None] - A[:, None, :]

    M = dl <= 1
    M &= (dl < 0) | (torch.stack([dA[levels[i], :, i] for i in range(N)], dim=1) == dl)
    M &= (dl > 0) | (torch.stack([dA[levels[i], i, :] for i in range(N)]) == 0)
    M.tril_()
    return M
