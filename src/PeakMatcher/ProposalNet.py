import torch
from torch import nn


import torch
import torch.nn as nn
import torch.nn.init as init
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional
from tqdm import tqdm
class ProposalNet(nn.Module):
    pass
def init_zero_logits(model: ProposalNet):
    """Zero-initialise the final d→1 projection so raw_logits = 0."""
    final_lin = model.scoring_mlp[-1]  # nn.Linear(d, 1)
    assert isinstance(final_lin, nn.Linear), "expected Linear at end"
    init.zeros_(final_lin.weight)  # weight ← 0
    init.zeros_(final_lin.bias)

class ProposalNet(nn.Module):
    """Cross‑attention proposal network for SMC data association.

    Parameters
    ----------
    d : int, default 128
        Model (embedding) dimension.
    num_heads : int, default 4
        Number of attention heads.
    num_layers : int, default 2
        Number of stacked cross‑attention blocks.
    dropout : float, default 0.1
        Dropout used in FFNs and attention.
    use_pos_emb : bool, default False
        If ``True`` add learnable positional embeddings for rows and columns.
    """
    def __init__(self, d: int = 128, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.1,
                 use_pos_emb: bool = False):
        super().__init__()
        self.d = d
        self.use_pos_emb = use_pos_emb

        # --- embedding MLPs -------------------------------------------------
        # * Row features have last‑axis dimension (n + 2) at run‑time, so we
        #   defer building the first linear layer until we know *n* (lazy).
        # * Same for column features (input dim m + 1).
        self._row_mlp = None
        self._col_mlp = None

        # --- cross‑attention stacks ----------------------------------------
        self.layers = nn.ModuleList([
            _CrossAttnBlock(d, num_heads, dropout,torch.float32) for _ in range(num_layers)
        ])

        # --- scoring head ---------------------------------------------------
        self.scoring_mlp = nn.Sequential(
            nn.Linear(2 * d + 1, d,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(d, 1,dtype=torch.float32)
        )

    # --------------------------------------------------------------------- #
    #  Public interface
    # --------------------------------------------------------------------- #
    def forward(self, logL: torch.Tensor,
                avail_rows: torch.Tensor,
                avail_cols: torch.Tensor) -> torch.Tensor:
        """Compute row‑column joint logits.

        Parameters
        ----------
        logL : Tensor, shape (B, m, n+1)
            Per‑row raw log‑likelihoods (last column = 'unmatched' slot).
        avail_rows : BoolTensor, shape (B, m)
            Mask of rows still available (``True`` = available).
        avail_cols : BoolTensor, shape (B, n)
            Mask of real columns still available (``True`` = available).

        Returns
        -------
        logits : Tensor, shape (B, m, n+1)
            Normalised log‑probabilities over pairs (i, j).
        """
        logL = logL.type(dtype=torch.float32)
        logL = logL.expand((avail_rows.shape[0],-1,-1))
        avail_cols = avail_cols[...,:-1]
        B, m, n1 = logL.shape
        n = n1 - 1  # number of real columns

        # ------------------------------------------------------------------ #
        # Build per‑row feature tensor: (B, m, n + 2)
        feat_rows = torch.cat(
            [logL,                          # (B, m, n+1)
             avail_rows.unsqueeze(-1).float()],  # (B, m, 1)
            dim=-1
        )

        # ------------------------------------------------------------------ #
        # Build per‑column feature tensor
        #   * real columns: (B, n,   m+1)
        #   * unmatched   : (B, 1,   m+1)  (always available)
        logL_real = logL[:, :, :n]               # (B, m, n)
        feat_cols_real = torch.cat(
            [logL_real.permute(0, 2, 1),         # (B, n, m)
             avail_cols.unsqueeze(-1).float()],   # (B, n, 1)
            dim=-1
        )

        # unmatched column uses per‑row likelihoods to all rows
        um = logL[:, :, -1].unsqueeze(1)         # (B, 1, m)
        um = torch.cat([um,
                        torch.ones_like(um[:, :, :1])],  # availability flag = 1
                       dim=-1)                           # (B, 1, m+1)

        feat_cols = torch.cat([feat_cols_real, um], dim=-2)  # (B, n+1, m+1)

        # ------------------------------------------------------------------ #
        # Lazy initialisation of embedding MLPs (sizes depend on m & n)
        if self._row_mlp is None:
            self._row_mlp = nn.Sequential(
                nn.Linear(n + 2, self.d,dtype=feat_cols.dtype),
                nn.ReLU(),
                nn.Linear(self.d, self.d,dtype=feat_cols.dtype)
            ).to(logL.device)

        if self._col_mlp is None:
            self._col_mlp = nn.Sequential(
                nn.Linear(m + 1, self.d,dtype=feat_cols.dtype),
                nn.ReLU(),
                nn.Linear(self.d, self.d,dtype=feat_cols.dtype)
            ).to(logL.device)

        # ------------------------------------------------------------------ #
        # Embed rows / columns
        row_emb = self._row_mlp(feat_rows)   # (B, m,   d)
        col_emb = self._col_mlp(feat_cols)   # (B, n+1, d)

        # optional learnable positional embeddings
        if self.use_pos_emb:
            row_pe = self._get_pos_emb('row', m, logL.device)
            col_pe = self._get_pos_emb('col', n1, logL.device)
            row_emb = row_emb + row_pe
            col_emb = col_emb + col_pe

        # Extend availability to include unmatched column (always True)
        avail_cols_full = torch.cat(
            [avail_cols,
             torch.ones(B, 1, dtype=avail_cols.dtype, device=avail_cols.device)],
            dim=1
        )  # (B, n+1)

        # ------------------------------------------------------------------ #
        # Cross‑attention stack
        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb,
                                     avail_rows, avail_cols_full)

        # ------------------------------------------------------------------ #
        # Pair‑wise scoring
        r = row_emb.unsqueeze(2).expand(-1, -1, n1, -1)   # (B,m,n+1,d)
        c = col_emb.unsqueeze(1).expand(-1, m, -1, -1)    # (B,m,n+1,d)
        l = logL.unsqueeze(-1)                            # (B,m,n+1,1)
        pair_feat = torch.cat([r, c, l], dim=-1)          # (B,m,n+1,2d+1)
        logits = self.scoring_mlp(pair_feat).squeeze(-1)  # (B,m,n+1)

        # invalidate consumed rows / columns
        mask_rows = (~avail_rows).unsqueeze(-1)           # (B,m,1)
        mask_cols = (~avail_cols_full).unsqueeze(1)       # (B,1,n+1)
        invalid = mask_rows | mask_cols                   # (B,m,n+1)
        logits = logits.masked_fill(invalid, -1e9)
        logits_normalized = logits - logits.logsumexp(dim=(-1,-2),keepdim=True)
        # ------------------------------------------------------------------ #
        return logits_normalized

    # --------------------------------------------------------------------- #
    #  Private helpers
    # --------------------------------------------------------------------- #
    def _get_pos_emb(self, kind: str, length: int, device: torch.device):
        name = f'_{kind}_pos_emb'
        if not hasattr(self, name):
            setattr(self, name, nn.Parameter(torch.randn(1, length, self.d)))
        return getattr(self, name).to(device)


class _CrossAttnBlock(nn.Module):
    """Row↔Column bi‑directional cross‑attention block."""
    def __init__(self, d: int, num_heads: int, dropout: float, dtype):
        super().__init__()
        self.row2col = nn.MultiheadAttention(d, num_heads,
                                             dropout=dropout,
                                             batch_first=True,
                                             dtype=dtype)
        self.col2row = nn.MultiheadAttention(d, num_heads,
                                             dropout=dropout,
                                             batch_first=True,
                                             dtype=dtype)

        self.row_norm1 = nn.LayerNorm(d,dtype=dtype)
        self.row_norm2 = nn.LayerNorm(d,dtype=dtype)
        self.col_norm1 = nn.LayerNorm(d,dtype=dtype)
        self.col_norm2 = nn.LayerNorm(d,dtype=dtype)

        self.row_ffn = _FFN(d, dropout)
        self.col_ffn = _FFN(d, dropout)

    def forward(self, row_emb, col_emb,
                avail_rows, avail_cols):
        # key‑padding masks (True = pad)
        kpm_rows = ~avail_rows.bool()   # (B, m)
        kpm_cols = ~avail_cols.bool()   # (B, n+1)

        # 1) rows attend to columns
        r2c, _ = self.row2col(row_emb, col_emb, col_emb,
                              key_padding_mask=kpm_cols)
        row_emb = self.row_norm1(row_emb + r2c)
        row_emb = self.row_norm2(row_emb + self.row_ffn(row_emb))

        # 2) columns attend to rows
        c2r, _ = self.col2row(col_emb, row_emb, row_emb,
                              key_padding_mask=kpm_rows)
        col_emb = self.col_norm1(col_emb + c2r)
        col_emb = self.col_norm2(col_emb + self.col_ffn(col_emb))
        return row_emb, col_emb


class _FFN(nn.Module):
    """Position‑wise feed‑forward network."""
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ProposalTrainer:
    """Encapsulated training/eval harness for ProposalNet."""

    def __init__(
        self,
        d: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        #self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device="cpu"
        self.net = ProposalNet(d=d, num_heads=num_heads, num_layers=num_layers).to(self.device)
        self.opt = AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.grad_clip = grad_clip

    # ------------------------------------------------------------------ #
    def warm_start(self, batch):
        """Run a dummy forward so lazy layers exist, then zero Δ‑logits."""
        logL, r, c, _ = [t.to(self.device) for t in batch]
        _ = self.net(logL, r, c, normalise=False)
        init_zero_logits(self.net)

    # ------------------------------------------------------------------ #
    def train_step(self,
                    logL: torch.tensor,
                    rmask: torch.tensor,
                    cmask: torch.tensor,
                    n_samples: int,
                    n_steps: int):
        self.net.train()

        previous_loss = torch.zeros((1,))
        for step in range(n_steps):
            self.opt.zero_grad(set_to_none=True)
            logits = self.net(logL, rmask, cmask)  # CE needs normalised

            log_probability_matrix = torch.where(rmask.unsqueeze(-1)&cmask.unsqueeze(1),logL + logits,-torch.inf)
            probability_matrix = log_probability_matrix  - log_probability_matrix.logsumexp(dim=(-2, -1), keepdim=True)
            sampled_decisions = torch.multinomial(probability_matrix.reshape(probability_matrix.shape[0], -1).exp(),
                              num_samples=1,
                              replacement=True).type(torch.int32)
            weight_matrix = (logL.unsqueeze(0) - probability_matrix).reshape(logits.shape[0],-1)
            weights = weight_matrix[torch.arange(0,logits.shape[0]).unsqueeze(-1),sampled_decisions]
            assert weights.isfinite().all()
            log_sum_w = weights.logsumexp(dim=-1)
            log_sum_w2 = (2.0*weights).logsumexp(dim=-1)
            log_ess = 2.0*log_sum_w - log_sum_w2
            loss = (-log_ess.exp()/sampled_decisions.shape[-1]).sum()/n_samples
            diff = torch.abs(loss - previous_loss)
            if  diff < 1E-5 or 1.0+loss < 1E-3:
                break
            previous_loss = loss
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
            self.opt.step()

    def simple_step(self,weights: torch.tensor):
        self.opt.zero_grad(set_to_none=True)
        log_sum_w = weights.logsumexp(dim=-1)
        log_sum_w2 = (2.0 * weights).logsumexp(dim=-1)
        log_ess = 2.0 * log_sum_w - log_sum_w2
        loss = (-log_ess.exp() / weights.shape[-1]).sum()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.opt.step()
        return loss


        return logits.detach()


