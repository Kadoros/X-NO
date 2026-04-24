
class HeinnXConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, degree):
        super().__init__()
        self.degree = degree
        self.W_base = nn.Parameter(
            torch.randn(in_ch, out_ch, degree + 1) / math.sqrt(in_ch * (degree + 1))
        )
        self.W_int = nn.Parameter(
            torch.randn(in_ch, out_ch, degree + 2) / math.sqrt(in_ch * (degree + 2))
        )
        self.register_buffer("S", build_S_matrix(degree))
        self._cache = {}

    def forward(self, x):
        B, C, N = x.shape
        d_eff = min(self.degree, N // 2)
        if (N, d_eff) not in self._cache:
            self._cache[(N, d_eff)] = (
                chebyshev_matrices(N, d_eff, x.device)[0],
                chebyshev_matrices(N, d_eff, x.device)[1],
                chebyshev_sum_matrix(N, d_eff, x.device),
            )
        T_pinv, T, T_sum = self._cache[(N, d_eff)]

        c = torch.einsum("bcn,kn->bck", x, T_pinv)

        m_base = torch.einsum("bck,cok->bok", c, self.W_base[:, :, : d_eff + 1])
        out_base = torch.einsum("bok,nk->bon", m_base, T)

        c_int = torch.einsum("bck,lk->bcl", c, self.S[: d_eff + 2, : d_eff + 1])
        m_int = torch.einsum("bcl,col->bol", c_int, self.W_int[:, :, : d_eff + 2])
        out_int = torch.einsum("bol,nl->bon", m_int, T_sum)

        return out_base + out_int
