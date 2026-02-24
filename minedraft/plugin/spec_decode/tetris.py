import torch
from torch_scatter import scatter_max
from vllm.spec_decode.interfaces import SpeculativeProposals


def select_proposals_no_priority(capacity: int, proposals: SpeculativeProposals) -> SpeculativeProposals:
    mask = proposals.proposal_lens > 0
    drafts_values = torch.gather(proposals.proposal_logprobs[mask], dim=-1, index=proposals.proposal_token_ids[mask].unsqueeze(-1)).squeeze(-1)
    drafts_values = drafts_values.cumsum(dim=-1)
    drafts_values_flatten = drafts_values.flatten()
    _, best_indices = torch.topk(drafts_values_flatten, min(capacity, len(drafts_values_flatten)), largest=True)
    indices = best_indices // drafts_values.size(1)
    vals = (best_indices % drafts_values.size(1)) + 1
    
    best_frontier, _  = scatter_max(vals, indices, dim_size=proposals.proposal_lens[mask].size(0))
    proposals.proposal_lens[mask] = best_frontier
    
    
    return proposals