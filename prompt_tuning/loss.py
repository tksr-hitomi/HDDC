"""Build primary-secondary and present-absent ranking pairs."""

import torch


def build_ranking_pairs(
    logits: torch.Tensor, labels: torch.Tensor, primary_indices: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    higher: list[torch.Tensor] = []
    lower: list[torch.Tensor] = []
    for sample_index in range(logits.shape[0]):
        sample_logits = logits[sample_index]
        positive = torch.nonzero(labels[sample_index] == 1, as_tuple=True)[0].tolist()
        negative = torch.nonzero(labels[sample_index] == 0, as_tuple=True)[0].tolist()
        primary = int(primary_indices[sample_index].item())
        if len(positive) > 1:
            for secondary in positive:
                if secondary != primary:
                    higher.append(sample_logits[primary])
                    lower.append(sample_logits[secondary])
        for positive_index in positive:
            for negative_index in negative:
                higher.append(sample_logits[positive_index])
                lower.append(sample_logits[negative_index])
    if not higher:
        return (None, None)
    return (torch.stack(higher), torch.stack(lower))
