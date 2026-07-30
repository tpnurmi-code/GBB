import torch
import torch.nn as nn
import torch.nn.functional as F
import config 

# ==========================================
# 1. FastKAN Building Blocks
# ==========================================

class FastKANLinear(nn.Module):
    """
    Linear layer augmented with fixed Gaussian RBF basis expansion.
    Works with inputs of shape (..., in_features).
    """
    def __init__(self, in_features, out_features, num_grids=5, grid_range=(-4.0, 4.0)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_grids = num_grids

        min_grid, max_grid = grid_range

        # Fixed Gaussian basis parameters
        mu = torch.linspace(min_grid, max_grid, num_grids)
        sigma = torch.ones(num_grids) * (max_grid - min_grid) / (num_grids - 1)

        self.register_buffer("mu", mu)
        self.register_buffer("sigma", sigma)

        # KAN/RBF weights
        self.spline_weights = nn.Parameter(
            torch.randn(out_features, in_features, num_grids) * config.SPLINECOEFF
        )

        # Residual linear term
        self.base_linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        original_shape = x.shape[:-1]   # preserve leading dims
        x_flat = x.reshape(-1, self.in_features)   # (B*, In)

        # Expand input over Gaussian basis
        x_expanded = x_flat.unsqueeze(-1)  # (B*, In, 1)
        basis = torch.exp(-((x_expanded - self.mu) ** 2) / (2 * (self.sigma ** 2)))  # (B*, In, G)

        # Directly contract over input and grid -> (B*, Out)
        spline_out = torch.einsum('big,oig->bo', basis, self.spline_weights)

        # Base linear residual
        base_out = self.base_linear(x_flat)  # (B*, Out)

        out = base_out + spline_out
        out = out.reshape(*original_shape, self.out_features)

        return out

# ==========================================
# 2. Feature Extractors (H1 - H6)
#    Standardized to take 'input_channels=1' (Siamese Mode)
# ==========================================

class H1_CNN_Extractor(nn.Module):
    def __init__(self, input_length, hidden_dim, layers=2, dropout=0.5, stride=1):
        super().__init__()
        # Input: (Batch, 1, Time)
        # We use a robust 2-layer block. 
        # Making CNN depth dynamic is risky due to dimension shrinking, 
        # so we focus on stride and dropout.
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), # Takes 1 channel (Per-Node)
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=stride),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Flatten(),
            # Calculate linear input size based on stride
            nn.Linear(64 * (input_length // stride), hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x shape: (Batch, 1, Time) or (Batch, Time, 1)
        # CNN expects (Batch, Channels, Time)
        if x.shape[-1] == 1:
             x = x.permute(0, 2, 1)
        return self.net(x)

class H2_MLP_Extractor(nn.Module):
    def __init__(self, input_length, hidden_dim, layers=3, dropout=0.5):
        super().__init__()
        # Input: (Batch, Time, 1) -> Flatten to (Batch, Time)
        
        # Dynamic Layer Construction based on config.MLP_LAYERS
        modules = [nn.Flatten()]
        
        # Input Layer
        modules.append(nn.Linear(input_length, 256))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(dropout))
        
        # Hidden Layers (Dynamic Depth)
        # We add (layers - 2) intermediate layers
        for _ in range(max(0, layers - 2)):
            modules.append(nn.Linear(256, 256))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
            
        # Output Layer
        modules.append(nn.Linear(256, hidden_dim))
        modules.append(nn.ReLU()) # Non-linearity on embedding
        
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

class H3_CNNLSTM_Extractor(nn.Module):
    def __init__(self, input_length, hidden_dim, dropout=0.5, cnn_layers=2, rnn_layers=1, stride=1):
        super().__init__()
        # CNN Part
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2) # Halves dimension
        )
        # LSTM Part (Dynamic Depth)
        self.lstm = nn.LSTM(32, hidden_dim, num_layers=rnn_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, 1, Time)
        if x.shape[-1] == 1: x = x.permute(0, 2, 1)
        
        c_out = self.cnn(x) # (Batch, 32, Time/2)
        c_out = c_out.permute(0, 2, 1) # (Batch, Time/2, 32)
        
        _, (h_n, _) = self.lstm(c_out)
        # h_n shape: (num_layers, Batch, Hidden)
        return self.dropout(h_n[-1])

class H4_LSTM_Extractor(nn.Module):
    def __init__(self, input_length, hidden_dim, dropout=0.5, rnn_layers=2):
        super().__init__()
        # Input size is 1 (Scalar time series)
        self.lstm = nn.LSTM(1, hidden_dim, num_layers=rnn_layers, batch_first=True, dropout=dropout)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: (Batch, Time, 1)
        if x.shape[1] == 1: x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        return self.proj(h_n[-1])

class H5_Transformer_Extractor(nn.Module):
    def __init__(self, input_length, hidden_dim, layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(1, 64)
        # Learnable Position Encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, input_length, 64))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=dropout, batch_first=True)
        # Dynamic Depth
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.proj = nn.Linear(64, hidden_dim)

    def forward(self, x):
        # x: (Batch, Time, 1)
        if x.shape[1] == 1: x = x.permute(0, 2, 1)
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer(x)
        return self.proj(x.mean(dim=1)) # Average pooling

class H6_LocalTransformer_Extractor(nn.Module):
    # Same as H5 but potentially smaller/lighter
    def __init__(self, input_length, hidden_dim, dropout=0.1, layers=1):
        super().__init__()
        self.embedding = nn.Linear(1, 32)
        self.pos_encoder = nn.Parameter(torch.randn(1, input_length, 32))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=2, dim_feedforward=64, dropout=dropout, batch_first=True)
        # Dynamic Depth
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.proj = nn.Linear(32, hidden_dim)

    def forward(self, x):
        if x.shape[1] == 1: x = x.permute(0, 2, 1)
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer(x)
        return self.proj(x[:, -1, :]) # Last token

# ==========================================
# 3. Graph Layer (MultiHead FastKAN)
# ==========================================

class MultiHeadFastKANLayer(nn.Module):
    """
    Tier II: Multi-Head FastKAN for Graph Connectivity.
    """
    def __init__(self, num_nodes, in_dim, out_dim, num_heads=3, num_basis=5):
        super().__init__()
        self.num_heads = num_heads
        
        # Learnable Adjacency Weights: (N, N, Heads, Basis)
        self.spline_coeffs = nn.Parameter(torch.randn(num_nodes, num_nodes, num_heads, num_basis) * config.SPLINECOEFF)
        
        # This is an edge attenuation gate, not a true temporal conduction delay.
        # Kept as conduction_delays for checkpoint compatibility.
        self.conduction_delays = nn.Parameter(torch.rand(num_nodes, num_nodes, num_heads))

        # Shared Basis Functions
        self.register_buffer("mu", torch.linspace(-4, 4, num_basis))
        step = 8.0 / num_basis
        self.register_buffer("sigma", torch.ones(num_basis) * step)
        
        # Project combined features (Source + Target) -> 1 Scalar for Attention
        self.pair_proj = nn.Linear(in_dim * 2, 1) 
        
        head_signs = torch.ones(num_heads)
        if getattr(config, "ENABLE_SIGNED_HEAD_MESSAGES", False):
            configured = getattr(config, "GRAPH_HEAD_SIGNS", [1.0, -1.0, 0.5])
            for i in range(min(num_heads, len(configured))):
                head_signs[i] = float(configured[i])

        self.register_buffer("head_message_signs", head_signs)

    def forward(self, h, adj):
        #B, N, C = h.shape
    
        # Create pairwise features
        #h_i = h.unsqueeze(2).expand(-1, -1, N, -1)   # (B, N, N, C)
        #h_j = h.unsqueeze(1).expand(-1, N, -1, -1)   # (B, N, N, C)
        #pair_feat = torch.cat([h_i, h_j], dim=-1)    # (B, N, N, 2C)
    
        ## Pairwise scalar interaction
        #x = self.pair_proj(pair_feat)                 # (B, N, N, 1)
        B, N, C = h.shape
        
        # pair_proj: Linear(2C -> 1)
        w = self.pair_proj.weight.squeeze(0)         # (2C,)
        w_i = w[:C].unsqueeze(0)                     # (1, C)
        w_j = w[C:].unsqueeze(0)                     # (1, C)
        b = self.pair_proj.bias.view(1, 1, 1, 1)     # scalar bias
        
        # Compute source and target contributions separately
        score_i = F.linear(h, w_i)                   # (B, N, 1)
        score_j = F.linear(h, w_j)                   # (B, N, 1)
        
        # Broadcast to pairwise score without constructing pair_feat
        x = score_i.unsqueeze(2) + score_j.unsqueeze(1) + b   # (B, N, N, 1)    
    
    
    
        # FastKAN RBF basis expansion
        x_exp = x.unsqueeze(-1)                       # (B, N, N, 1, G)
        rbf = torch.exp(
            -((x_exp - self.mu.view(1, 1, 1, 1, -1)) ** 2) /
            (2 * self.sigma.view(1, 1, 1, 1, -1) ** 2)
        )                                             # (B, N, N, 1, G)
    
        # Per-head edge scores
        head_scores = torch.einsum('bnmzg,nmhg->bnmh', rbf, self.spline_coeffs)  # (B, N, N, H)
    
        # Conduction delay gate (shared with current implementation)
        delay_gate = torch.exp(-torch.abs(self.conduction_delays))                # (N, N, H)
        head_scores = head_scores * delay_gate.unsqueeze(0)                       # (B, N, N, H)
    
        if adj is not None:
            if adj.ndim == 2:
                # Weighted adjacency prior: encourages nearby/allowed edges
                adj_safe = adj.clamp_min(1e-6)
                head_scores = head_scores + config.ADJ_PRIOR_STRENGTH * torch.log(adj_safe).unsqueeze(0).unsqueeze(-1)
    
                # Binary support mask
                support = (adj > 0)                                               # (N, N)
                head_scores = head_scores.masked_fill(~support.unsqueeze(0).unsqueeze(-1), -1e9)
    
                # Guard against rows with no valid neighbors
                no_neighbors = (support.sum(dim=-1) == 0)                         # (N,)
                if no_neighbors.any():
                    forced = torch.full_like(head_scores, -1e9)
                    eye = torch.eye(N, device=h.device, dtype=torch.bool)
                    forced[:, eye, :] = 0.0                                       # self-loop only
                    head_scores[:, no_neighbors, :, :] = forced[:, no_neighbors, :, :]
    
            elif adj.ndim == 3:
                # If batched adjacency is ever used
                adj_safe = adj.clamp_min(1e-6)
                head_scores = head_scores + config.ADJ_PRIOR_STRENGTH * torch.log(adj_safe).unsqueeze(-1)
    
                support = (adj > 0)                                               # (B, N, N)
                head_scores = head_scores.masked_fill(~support.unsqueeze(-1), -1e9)
    
                no_neighbors = (support.sum(dim=-1) == 0)                         # (B, N)
                if no_neighbors.any():
                    eye = torch.eye(N, device=h.device, dtype=torch.bool).unsqueeze(0).unsqueeze(-1)  # (1,N,N,1)
                    forced = torch.full_like(head_scores, -1e9)
                    forced = forced.masked_fill(eye, 0.0)
                    for b in range(B):
                        if no_neighbors[b].any():
                            head_scores[b, no_neighbors[b], :, :] = forced[b, no_neighbors[b], :, :]
    
            else:
                raise ValueError(f"Unexpected adj shape: {adj.shape}")
    
        # Softmax over SOURCE dimension
        attn_heads = torch.softmax(head_scores, dim=2)                            # (B, N, N, H)
    
        # Message passing per head
        out_heads = torch.einsum('bnmh,bmc->bnhc', attn_heads, h)  # (B,N,H,C)
        
        if getattr(config, "ENABLE_SIGNED_HEAD_MESSAGES", False):
            signs = self.head_message_signs.to(device=h.device, dtype=h.dtype)
            signs = signs.view(1, 1, self.num_heads, 1)
        
            # Signed messages: head 1 can subtract/inhibit.
            out_heads = out_heads * signs
        
            # Signed effective map for interpretation/export.
            signed_attn_heads = attn_heads * signs.squeeze(-1).unsqueeze(2)  # (1,1,1,H)
            attn_map = signed_attn_heads.mean(dim=-1)
        else:
            attn_map = attn_heads.mean(dim=-1)
        
        out = out_heads.mean(dim=2)
        
        return out, attn_map, attn_heads

# ==========================================
# 4. Neural ODE (CfC)
# ==========================================

class CfCLayer(nn.Module):
    """
    Tier III: Continuous-Time Dynamics (Liquid Neural Network).
    Processes each node independently but with node-specific bias.
    """
    def __init__(self, input_size, hidden_size, num_nodes):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_nodes = num_nodes
        
        # Liquid Time-Constant Network components
        self.f_neural = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.g_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid() # Gate 0-1
        )
        
        # Dedicated heads for disentangled mechanistic quantities.
        # tau_system: positive time constant via softplus in forward/get_tau_secs
        # drive_head: signed intrinsic drive for causal-drive maps/regularization
        self.tau_system = nn.Linear(hidden_size, 1)
        self.drive_head = nn.Linear(hidden_size, 1)
        
        # Node-Specific Bias (The "Identity" of each region)
        # Allows S1 to behave differently from M1 even with shared weights
        self.node_bias = nn.Parameter(torch.randn(num_nodes, hidden_size) * 0.1)

    def forward(self, x, h_prev, timespan):
        """
        x: Input (Batch, Nodes, Features) - from Graph Layer
        h_prev: Previous state (Batch, Nodes, Features)
        timespan: Delta t (Batch, 1)
        """
        batch_size, num_nodes, _ = x.shape
        
        # Add Node Bias to input
        # Broadcast bias: (1, N, C)
        bias_broadcast = self.node_bias.unsqueeze(0).expand(batch_size, -1, -1)
        x_biased = x + bias_broadcast
        
        # Concatenate Input + State
        combined = torch.cat([x_biased, h_prev], dim=-1) # (B, N, 2C)
        
        # Calculate Derivative components
        f_val = self.f_neural(combined)
        g_val = self.g_gate(combined)
        
        # Calculate Tau (Time Constant)
        # We want tau to be positive. Softplus or Sigmoid.
        raw_tau = self.tau_system(f_val)
        tau = self._tau_from_raw(raw_tau)
        
        # Neural ODE Update Step (Liquid Time Constant variant)
        # dh/dt = -[h(t) - f(x)] / tau
        # Discrete approximation: h_new = (h_old + f(x) * dt/tau) / (1 + dt/tau)
        
        # Usually CfC uses a gating mechanism:
        # h_new = g * (h_old + f * dt) + (1-g) * h_old ... many variants.
        # Let's use the explicit Closed-form solution approximation:
        
        w = torch.exp(-timespan.unsqueeze(-1) / tau)
        h_new = w * h_prev + (1 - w) * f_val
        
        # Apply Gating
        h_final = h_new * g_val + h_prev * (1 - g_val)
        
        # Store diagnostics for biological regularization.
        # These are not detached because the training loss may use them.
        self.last_tau = tau
        self.last_gate = g_val
        self.last_f_val = f_val
        self.last_h_delta = h_final - h_prev        
        
        return h_final, tau
        
    def get_tau_values_avg(self):
        # Return average tau per node for visualization
        return torch.sigmoid(self.tau_system.weight).mean(dim=1).detach()
     
        # In models.py -> class CfCLayer

    def get_tau_values(self):
        """
        Returns the effective Time Constant (Tau) in seconds for every node.
        Calculated by simulating a 'Resting State' (Zero Input, Zero History).
        Returns Shape: (Num_Nodes,)
        """
        # 1. Simulate Zero Input and Zero History
        # node_bias shape: (Num_Nodes, Hidden)
        # We use the node_bias as the "resting input" because x and h_prev are zero.
        
        # In forward(): x_biased = x + node_bias. If x=0, x_biased = node_bias.
        x_biased = self.node_bias 
        
        # h_prev is zeros
        h_prev = torch.zeros_like(x_biased)
        
        # Concatenate: (Num_Nodes, 2*Hidden)
        combined = torch.cat([x_biased, h_prev], dim=-1) 
        
        # 2. Pass through the Neural Network
        f_val = self.f_neural(combined)
        
        # 3. Calculate Tau (matching the forward pass logic)
        # forward(): tau = F.softplus(self.tau_system(f_val)) + 0.1
        raw_output = self.tau_system(f_val) # (Num_Nodes, 1)
        tau = self._tau_from_raw(raw_output)
        
        # Flatten to (Num_Nodes,) to match the visualization expectation
        return tau.squeeze(-1)
    
    def get_tau_secs(self):
        """
        Returns the effective Time Constant (Tau) in seconds for every node.
        Calculated by simulating a 'Resting State' (Zero Input, Zero History).
        Required for Training Loss and Visualization.
        """
        # 1. Simulate Zero Input and Zero History
        # node_bias shape: (Num_Nodes, Hidden)
        # We use the node_bias as the "resting input" because x and h_prev are zero.
        x_biased = self.node_bias 
        
        # h_prev is zeros
        h_prev = torch.zeros_like(x_biased)
        
        # Concatenate: (Num_Nodes, 2*Hidden)
        combined = torch.cat([x_biased, h_prev], dim=-1) 
        
        # 2. Pass through the Neural Network (f_neural is shared)
        f_val = self.f_neural(combined)
        
        # 3. Calculate Tau (matching the forward pass logic)
        raw_output = self.tau_system(f_val) # (Num_Nodes, 1)
        
        # Apply the same non-linearity as in forward()
        tau = self._tau_from_raw(raw_output)
        
        # Flatten to (Num_Nodes,)
        return tau.squeeze(-1)
    
    def get_causal_drive(self):
        """
        Returns the RAW, SIGNED Causal Drive (Activity) for each node.
        
        Why this vs get_tau_values?
        - get_tau_values returns Time Constants (Seconds), which are strictly POSITIVE.
        - get_causal_drive returns the internal Activation (Logits), which can be 
          NEGATIVE (Inhibition) or POSITIVE (Excitation).
          
        This allows Group Lasso/Elastic Net to create the 'Negative Sea' (Inhibition)
        and 'Positive Islands' (Drivers) you see in the maps.
        """
        # 1. Resting State Simulation (Same as before)
        x_biased = self.node_bias 
        h_prev = torch.zeros_like(x_biased)
        combined = torch.cat([x_biased, h_prev], dim=-1) 
        f_val = self.f_neural(combined)
        
        # 2. Return RAW output (Before Softplus/Sigmoid)
        # This is the signed intrinsic drive (independent from tau head).
        raw_drive = self.drive_head(f_val) # (Num_Nodes, 1)
        
        return raw_drive.squeeze(-1) # (Num_Nodes,)
    
    def _tau_from_raw(self, raw_tau):
        if getattr(config, "TAU_BOUND_MODE", "sigmoid") == "sigmoid":
            tau_min = float(getattr(config, "TAU_MIN_PHYS", 1.5))
            tau_max = float(getattr(config, "TAU_MAX_PHYS", 12.0))
            return tau_min + (tau_max - tau_min) * torch.sigmoid(raw_tau)
    
        # fallback old behavior
        return F.softplus(raw_tau) + 0.1


class StimulusTemporalEncoder(nn.Module):
    """
    Encodes the stimulus window into a per-TR hidden drive sequence.
    Input:  (B, T, C)
    Output: (B, T, H)
    """
    def __init__(self, in_channels, hidden_dim, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        mid = max(hidden_dim // 2, 8)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, mid, kernel_size=kernel_size, padding=pad),
            nn.GELU(),
            nn.Conv1d(mid, hidden_dim, kernel_size=1)
        )

    def forward(self, stim):
        # stim: (B, T, C) -> Conv1d expects (B, C, T)
        x = stim.transpose(1, 2)
        y = self.net(x)
        return y.transpose(1, 2)  # (B, T, H)
  
    
class HemodynamicObservationHead(nn.Module):
    """
    Lightweight BOLD/CBV observation model.

    Input:  neural_pred (B, Horizon, N)
    Output: fmri_pred   (B, Horizon, N)

    Supports:
    - global causal temporal kernel
    - global gain/bias
    - optional nodewise gain/bias for region-specific HRF amplitude shifts
    """
    def __init__(self, kernel_size=5, init_mode="hrf", num_nodes=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_nodes = num_nodes

        if init_mode == "hrf":
            if kernel_size == 3:
                init_kernel = torch.tensor([0.10, 0.35, 0.55], dtype=torch.float32)
            elif kernel_size == 5:
                init_kernel = torch.tensor([0.05, 0.15, 0.35, 0.30, 0.15], dtype=torch.float32)
            elif kernel_size == 7:
                init_kernel = torch.tensor([0.03, 0.07, 0.15, 0.30, 0.25, 0.14, 0.06], dtype=torch.float32)
            else:
                init_kernel = torch.ones(kernel_size, dtype=torch.float32) / kernel_size

        elif init_mode == "cbv":
            if kernel_size == 3:
                init_kernel = torch.tensor([0.15, 0.55, 0.30], dtype=torch.float32)
            elif kernel_size == 5:
                init_kernel = torch.tensor([0.05, 0.15, 0.40, 0.25, 0.15], dtype=torch.float32)
            else:
                init_kernel = torch.ones(kernel_size, dtype=torch.float32) / kernel_size

        else:
            init_kernel = torch.ones(kernel_size, dtype=torch.float32) / kernel_size

        init_kernel = init_kernel / (init_kernel.sum() + 1e-8)

        self.kernel_logits = nn.Parameter(torch.log(init_kernel + 1e-6))
        self.log_gain = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

        if num_nodes is not None:
            self.node_log_gain = nn.Parameter(torch.zeros(1, 1, num_nodes))
            self.node_bias = nn.Parameter(torch.zeros(1, 1, num_nodes))
        else:
            self.node_log_gain = None
            self.node_bias = None

    def forward(self, neural_pred):
        # neural_pred: (B, Horizon, N)
        B, H, N = neural_pred.shape

        x = neural_pred.permute(0, 2, 1).reshape(B * N, 1, H)
        kernel = torch.softmax(self.kernel_logits, dim=0).view(1, 1, -1)

        x_pad = F.pad(x, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, kernel)[:, :, :H]

        y = y.reshape(B, N, H).permute(0, 2, 1)

        global_gain = torch.exp(self.log_gain)
        out = global_gain * y + self.bias

        if self.node_log_gain is not None:
            if self.node_log_gain.shape[-1] != N:
                raise ValueError(
                    f"HemodynamicObservationHead expected {self.node_log_gain.shape[-1]} nodes, got {N}"
                )
            out = torch.exp(self.node_log_gain) * out + self.node_bias

        return out
    
    
# ==========================================
# 5. Main Model (MesocortGBB)
# ==========================================

class MesocortGBB(nn.Module):
    def __init__(self, num_nodes, time_points, freeze_extractor=False, sensory_mask=None):
        super().__init__()
        self.freeze_extractor = freeze_extractor
        h_dim = config.FEAT_EXT_HIDDEN
        drop = config.FEAT_EXT_DROPOUT
        # In __init__
        # --- 1. Siamese Feature Extractor ---
        # Note: We pass "input_channels=1" implicitly by how we initialize these classes.
        # The 'time_points' arg is used for linear layer sizing.
        # We pass layer counts from config.py where applicable.
        
        if config.MODEL_TYPE == "H1":
            self.extractor = H1_CNN_Extractor(time_points, h_dim, config.CNN_LAYERS, drop, config.CNN_STRIDE)
        elif config.MODEL_TYPE == "H2":
            self.extractor = H2_MLP_Extractor(time_points, h_dim, config.MLP_LAYERS, drop)
        elif config.MODEL_TYPE == "H3":
            self.extractor = H3_CNNLSTM_Extractor(time_points, h_dim, drop, config.CNN_LAYERS, config.RNN_LAYERS, config.CNN_STRIDE)
        elif config.MODEL_TYPE == "H4":
            self.extractor = H4_LSTM_Extractor(time_points, h_dim, drop, config.RNN_LAYERS)
        elif config.MODEL_TYPE == "H5":
            self.extractor = H5_Transformer_Extractor(time_points, h_dim, config.TRANSFORMER_LAYERS, drop)
        elif config.MODEL_TYPE == "H6":
            self.extractor = H6_LocalTransformer_Extractor(time_points, h_dim, drop, config.TRANSFORMER_LAYERS)
        else:
            raise ValueError(f"Unknown MODEL_TYPE: {config.MODEL_TYPE}")

        # --- 2. Feature Adapter ---
        target_dim = config.CFC_BACKBONE_UNITS # e.g. 128

        # Infer Dimension (using single channel input)
        # Dummy: (Batch=1, Time, Channels=1)
        # We test this to ensure dimension matching logic works with the new Siamese extractors.
        dummy_input = torch.zeros(1, time_points, 1) 
        feat_dim = 64
        with torch.no_grad():
            try:
                # The forward pass in MesocortGBB handles the reshape, 
                # but here we test the extractor directly.
                dummy_out = self.extractor(dummy_input) 
                feat_dim = dummy_out.shape[-1]
                print(f"✅ Inferred Feature Extractor Dimension: {feat_dim}")
            except Exception as e:
                print(f"⚠️ Could not infer extractor output shape: {e}. Defaulting to {feat_dim}")
        
        if feat_dim != target_dim:
            print(f"⚠️ Resizing Feature Extractor ({feat_dim}) -> Backbone ({target_dim})")
            self.adapter = nn.Linear(feat_dim, target_dim)
        else:
            self.adapter = nn.Identity()

        # --- 3. Graph Layers (FastKAN) ---
        self.kan_layers = nn.ModuleList([
            MultiHeadFastKANLayer(num_nodes=num_nodes,
                in_dim=target_dim,   
                out_dim=target_dim,  
                num_heads=config.KAN_HEADS,
                num_basis=config.KAN_BASIS_FUNCTIONS
            ) for _ in range(config.KAN_LAYERS)
        ])
            
        # --- 4. CfC Dynamics ---
        self.cfc = CfCLayer(input_size=target_dim, hidden_size=target_dim, num_nodes=num_nodes)
        
        # --- 5. Temporal fmri drive + stimulus + decoder ---
        input_dim = config.STIMULUS_INPUT_CHANNELS
        
        # Per-TR local node drive from the raw fMRI value at each time step
        self.fmri_step_proj = nn.Linear(1, target_dim)
        
        # Temporal stimulus encoder (preferred over stim_window.mean(...))
        self.stim_encoder = StimulusTemporalEncoder(
            in_channels=input_dim,
            hidden_dim=target_dim,
            kernel_size=config.STIM_ENCODER_KERNEL
        )
        
        self.dropout = nn.Dropout(p=config.DROPOUT)
        self.norm_decoder = nn.LayerNorm(target_dim)
        
        # Decode a future neural prediction (still scalar per horizon per node)
        self.decoder = nn.Linear(target_dim, config.PREDICTION_HORIZON)
        
        # Small hemodynamic observation model
        self.hemo_head = HemodynamicObservationHead(
            kernel_size=config.HEMO_KERNEL_SIZE,
            init_mode=config.HEMO_INIT,
            num_nodes=num_nodes
        )
        
        if sensory_mask is None:
            sensory_mask = torch.ones(num_nodes, 1)
        self.register_buffer('sensory_mask', sensory_mask)        
    
    
    def forward(self, fmri_window, stim_window, adj, return_head_weights=False):
        """
        fmri_window: (B, T, N)
        stim_window: (B, T, C)
        adj:         (N, N) or (B, N, N)
        """
        B, T, N = fmri_window.shape
    
        # ==========================================
        # 1. Context prior from the existing extractor
        # ==========================================
        x_flat = fmri_window.permute(0, 2, 1).contiguous().view(B * N, T, 1)
        context = self.extractor(x_flat)
        context = self.adapter(context).view(B, N, -1)   # (B, N, H)
    
        # ==========================================
        # 2. Per-TR latent drives
        # ==========================================
        # raw fMRI step drive: (B,T,N,1) -> (B,T,N,H)
        fmri_step_drive = self.fmri_step_proj(fmri_window.unsqueeze(-1))
    
        # Broadcast context over time
        drives = fmri_step_drive + context.unsqueeze(1)  # (B,T,N,H)
    
        # ==========================================
        # 3. Preserve temporal structure of stimulus
        # ==========================================
        if stim_window is not None and getattr(config, "USE_TEMPORAL_STIM_ENCODER", True):
            stim_seq = self.stim_encoder(stim_window)                  # (B,T,H)
            stim_seq = stim_seq.unsqueeze(2).expand(-1, -1, N, -1)    # (B,T,N,H)
    
            mask_broadcast = self.sensory_mask.reshape(1, 1, N, 1)
            if mask_broadcast.shape[2] != N:
                raise ValueError(f"sensory_mask has {mask_broadcast.shape[2]} nodes but input has {N}")
    
            drives = drives + stim_seq * mask_broadcast
        elif stim_window is not None:
            stim_val = stim_window.mean(dim=1)
            stim_static = self.stim_proj(stim_val).unsqueeze(1).expand(-1, N, -1)
            mask_broadcast = self.sensory_mask.reshape(1, N, 1)
            drives = drives + stim_static.unsqueeze(1) * mask_broadcast.unsqueeze(1)
    
        self.last_kan_input = drives[:, -1].detach()
    
        # ==========================================
        # 4. Step graph + CfC across the full 30-TR history
        # ==========================================
        h_prev = context
        total_attn = 0.0
        all_head_weights = [] if return_head_weights else None
    
        ts = torch.full((B, 1), float(config.TR), device=fmri_window.device)
    
        for t in range(T):
            x_t = drives[:, t]  # (B,N,H)
            x_graph = x_t
    
            for layer in self.kan_layers:
                neighbor_input, attn_map, head_weights = layer(x_graph, adj)
                x_graph = x_graph + neighbor_input
                total_attn = total_attn + attn_map
                
                if return_head_weights:
                    all_head_weights.append(head_weights)
                        # Keep a direct local path so the graph refines rather than overwrites node evidence.
            
            graph_mix = getattr(config, "GRAPH_MIX", 0.70)
            x_graph = graph_mix * x_graph + (1.0 - graph_mix) * x_t        
            h_prev, _ = self.cfc(x=x_graph, h_prev=h_prev, timespan=ts)
    
        if len(self.kan_layers) > 0:
            avg_attn = total_attn / (T * len(self.kan_layers))
        else:
            avg_attn = None
    
        # final neural state after processing the full history
        h_dynamic = self.norm_decoder(self.dropout(h_prev))
    
        # ==========================================
        # 5. Decode future neural prediction
        # ==========================================
        neural_delta = self.decoder(h_dynamic)                           # (B,N,Horizon)
        neural_delta = F.hardtanh(neural_delta, min_val=-4.0, max_val=4.0)
    
        last_observation = fmri_window[:, -1, :].unsqueeze(-1)          # (B,N,1)
        neural_pred = last_observation + neural_delta                   # (B,N,Horizon)
        neural_pred = neural_pred.permute(0, 2, 1).contiguous()         # (B,Horizon,N)
    
        # ==========================================
        # 6. Hemodynamic observation head
        # ==========================================
        if config.USE_HEMODYNAMIC_HEAD:
            pred = self.hemo_head(neural_pred)                          # (B,Horizon,N)
        else:
            pred = neural_pred
    
        return pred, avg_attn, h_dynamic, all_head_weights
    
    @property
    def tau_values(self):
        return self.cfc.get_tau_values()
    
    @property
    def causal_drive(self):
        return self.cfc.get_causal_drive()