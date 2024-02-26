import torch
import torch.nn as nn
import torch.nn.functional as f


class Wasserstein(nn.Module):
    def __init__(self, reg: float = 0.1, nit: int = 500, device=torch.device("cpu")):
        """The wasserstein distance model.

        Args:
            reg: The regularization factor used with "emd" (Default: 0.1).
            nit: The maximum number of iterations used with "emd" (Default: 500).
        """
        super(Wasserstein, self).__init__()
        self.reg = reg
        self.nit = nit
        self.device = device

    def forward(
        self,
        C: torch.Tensor = None,
        dist_1: torch.Tensor = None,
        dist_2: torch.Tensor = None,
        as_prob: bool = False,
    ):
        """Calculate the wasserstein distance.

        Args:
            C: The cost matrix.
            dist_1: The distribution of the first input.
            dist_2: The distribution of the second input.
        """
        # solve the optimal transport problem
        T = self.sinkhorn(dist_1, dist_2, C, self.reg, self.nit)
        # calculate the distances
        dists = (C * T).view(C.shape[0], -1).sum(dim=1)

        if as_prob:
            dists = torch.exp(-(dists**2))

        # return the loss, transport and cost matrices
        return dists, C, T

    def get_cost_matrix(self, embeds_1: torch.Tensor, embeds_2: torch.Tensor):
        """Calculates the cost matrix of the embeddings

        Args:
            embeds_1: The first embeddings tensor.
            embeds_2: The seconds embeddings tensor.
        Returns:
            torch.Tensor: The cost matrix between the first and second embeddings.

        """
        # normalize the embeddings
        embeds_1 = f.normalize(embeds_1, p=2, dim=2)
        embeds_2 = f.normalize(embeds_2, p=2, dim=2)
        # calculate and return the cost matrix
        cost_matrix = embeds_1.matmul(embeds_2.transpose(1, 2))
        cost_matrix = torch.ones_like(cost_matrix) - cost_matrix
        return cost_matrix

    def get_distributions(self, attention: torch.FloatTensor) -> torch.Tensor:
        """Generates the distribution tensor

        Args:
            attention: The attention tensor.

        Returns:
            torch.Tensor: The distribution tensor.

        """
        dist = torch.ones_like(attention) * attention
        dist = dist / dist.sum(dim=1).view(-1, 1).repeat(1, attention.shape[1])
        dist.requires_grad = False
        return dist

    def sinkhorn(
        self,
        dist_1: torch.Tensor,
        dist_2: torch.Tensor,
        cost_matrix: torch.Tensor,
        reg: float = 0.1,
        nit: int = 500,
    ):
        """Documentation
        The sinkhorn algorithm adapted for PyTorch from the
            PythonOT library <https://pythonot.github.io/>.

        Args:
            dist_1: The distribution of the first input.
            dist_2: The distribution of the second input.
            cost_matrix: The cost matrix.
            reg: The regularization factor. Default 0.1.
            nit: Number of maximum iterations. Default 500.

        Returns:
            torch.Tensor: The transportation matrix.

        """
        # asset the dimensions
        assert dist_1.shape[0] == cost_matrix.shape[0]
        assert dist_2.shape[0] == cost_matrix.shape[0]
        # prepare the initial variables
        dist_1 = dist_1.to(self.device)
        dist_2 = dist_2.to(self.device)
        K = torch.exp(-cost_matrix / reg).to(self.device)
        Kp = ((1 / dist_1).reshape(dist_1.shape[0], -1, 1) * K).to(self.device)
        # initialize the u and v tensor
        u = torch.ones_like(dist_1).to(self.device)
        v = torch.ones_like(dist_2).to(self.device)
        istep = 0
        while istep < nit:
            # calculate K.T * u for each example in batch
            KTransposeU = K.transpose(1, 2).bmm(u.unsqueeze(2)).squeeze(2)
            # calculate the v_{i} tensor
            v = dist_2 / KTransposeU
            # calculate the u_{i} tensor
            u = 1.0 / Kp.bmm(v.unsqueeze(2)).squeeze(2)
            # go to next step
            istep = istep + 1
        # calculate the transport matrix
        U = torch.diag_embed(u)
        V = torch.diag_embed(v)
        return U.bmm(K).bmm(V).cpu()
