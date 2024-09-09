from torch import nn
import torch


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.D = D
        self.embedding_a = nn.Embedding(K, D)
        self.embedding_a.weight.data.uniform_(-1./K, 1./K)
        self.embedding_v = nn.Embedding(K, D)
        self.embedding_v.weight.data.uniform_(-1./K, 1./K)

    def _straight_through(self, z_e_x, embedding):
        """
        :param z_e_x: [B, D]
        :return:
        """
        indices = z_e_x.mm(embedding.weight.detach().T)  # [B, K]
        indices_flatten = indices.argmax(-1)  # [B]
        z_q_x_st = torch.index_select(embedding.weight.detach(), dim=0, index=indices_flatten)  # [B, D]
        z_q_x = torch.index_select(embedding.weight, dim=0, index=indices_flatten)  # [B, D]
        return z_q_x_st, z_q_x, indices

    def straight_through(self, z_e_x):
        z_q_x_st_a, z_q_x_a, indices_a = self._straight_through(z_e_x[:, :self.D], self.embedding_a)
        z_q_x_st_v, z_q_x_v, indices_v = self._straight_through(z_e_x[:, self.D:], self.embedding_v)
        z_q_x_st = torch.cat([z_q_x_st_a, z_q_x_st_v], dim=-1)  # [B, 2D]
        z_q_x = torch.cat([z_q_x_a, z_q_x_v], dim=-1)  # [B, 2D]
        indices = torch.cat([indices_a, indices_v], dim=-1)  # [B, 2K]
        return z_q_x_st, z_q_x, indices


class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, dim, K=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4 * dim),
            nn.BatchNorm1d(4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, 2 * dim),
            nn.BatchNorm1d(2 * dim),
            nn.ReLU()
        )

        self.codebook = VQEmbedding(K, dim)

        self.decoder = nn.Sequential(
            nn.Linear(2 * dim, 4 * dim),
            nn.BatchNorm1d(4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU()
        )

        self.apply(weights_init)

    def forward(self, x):
        """
        :param x: [B, 161]
        :return:
        """
        z_e_x = self.encoder(x)  # [B, dim]
        z_q_x_st, z_q_x, indices = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x, indices


class BasicAE(VectorQuantizedVAE):
    def forward(self, x):
        z_e_x = self.encoder(x)
        x_tilde = self.decoder(z_e_x)
        return x_tilde
