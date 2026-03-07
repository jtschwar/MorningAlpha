"""
Set Transformer implementation for stock prediction.
Based on "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks"
by Lee et al., 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for set transformer."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head attention
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.W_o(context)
        return output, attention_weights


class SetAttentionBlock(nn.Module):
    """Set Attention Block (SAB) for set transformer."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class PoolingByMultiHeadAttention(nn.Module):
    """Pooling layer using multi-head attention (PMA)."""
    
    def __init__(self, d_model, num_heads, num_seeds=1, dropout=0.1):
        super().__init__()
        self.num_seeds = num_seeds
        self.seeds = nn.Parameter(torch.randn(1, num_seeds, d_model))
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: [batch_size, set_size, d_model]
        batch_size = x.size(0)
        seeds = self.seeds.repeat(batch_size, 1, 1)  # [batch_size, num_seeds, d_model]
        
        # Attention from seeds to input set
        pooled, _ = self.attention(seeds, x, x)
        pooled = self.norm(pooled)
        
        return pooled


class SetTransformer(nn.Module):
    """
    Set Transformer for processing variable-length sets of features.
    
    Args:
        dim_input: Input feature dimension
        dim_output: Output embedding dimension
        num_heads: Number of attention heads
        num_blocks: Number of Set Attention Blocks
        num_seeds: Number of seed vectors for pooling
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        dim_input,
        dim_output,
        num_heads=4,
        num_blocks=2,
        num_seeds=1,
        dropout=0.1
    ):
        super().__init__()
        
        # Input embedding
        self.input_embedding = nn.Linear(dim_input, dim_output)
        
        # Set Attention Blocks
        self.blocks = nn.ModuleList([
            SetAttentionBlock(dim_output, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        
        # Pooling layer
        self.pooling = PoolingByMultiHeadAttention(
            dim_output, num_heads, num_seeds, dropout
        )
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, set_size, dim_input]
        
        Returns:
            Output tensor [batch_size, num_seeds, dim_output]
        """
        # Embed input
        x = self.input_embedding(x)
        
        # Apply Set Attention Blocks
        for block in self.blocks:
            x = block(x)
        
        # Pool to fixed-size representation
        x = self.pooling(x)
        
        return x


class StockTower(nn.Module):
    """
    Tower 1: Stock-specific features.
    Processes time-series data (prices, volumes, returns).
    """
    
    def __init__(self, dim_output=128, num_heads=8, num_blocks=3):
        super().__init__()
        
        # Time-series features: price, volume, returns, high, low
        self.time_series_encoder = SetTransformer(
            dim_input=5,
            dim_output=dim_output,
            num_heads=num_heads,
            num_blocks=num_blocks
        )
        
        # Scalar features: RSI, Entry Score, Quality Score, etc.
        self.scalar_encoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64)
        )
        
        # Combine time-series and scalar features
        self.combine = nn.Sequential(
            nn.Linear(dim_output + 64, dim_output),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, time_series, scalar_features):
        """
        Args:
            time_series: [batch_size, days, 5] (price, volume, returns, high, low)
            scalar_features: [batch_size, 10] (RSI, Entry Score, etc.)
        
        Returns:
            Stock embedding [batch_size, dim_output]
        """
        # Process time-series
        ts_emb = self.time_series_encoder(time_series)  # [batch, num_seeds, dim_output]
        ts_emb = ts_emb.mean(dim=1)  # Average over seeds
        
        # Process scalar features
        scalar_emb = self.scalar_encoder(scalar_features)
        
        # Combine
        combined = torch.cat([ts_emb, scalar_emb], dim=1)
        output = self.combine(combined)
        
        return output


class MarketTower(nn.Module):
    """
    Tower 2: Market context features.
    Processes market-level information (SPY returns, sector returns, VIX).
    """
    
    def __init__(self, dim_output=128, num_heads=8, num_blocks=2):
        super().__init__()
        
        # Market time-series: market_return, sector_return, vix
        self.market_encoder = SetTransformer(
            dim_input=3,
            dim_output=dim_output,
            num_heads=num_heads,
            num_blocks=num_blocks
        )
        
        # Market regime features: beta, relative_strength, etc.
        self.regime_encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64)
        )
        
        # Combine
        self.combine = nn.Sequential(
            nn.Linear(dim_output + 64, dim_output),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, market_series, regime_features):
        """
        Args:
            market_series: [batch_size, days, 3] (market_return, sector_return, vix)
            regime_features: [batch_size, 5] (beta, relative_strength, etc.)
        
        Returns:
            Market embedding [batch_size, dim_output]
        """
        # Process market time-series
        market_emb = self.market_encoder(market_series)
        market_emb = market_emb.mean(dim=1)
        
        # Process regime features
        regime_emb = self.regime_encoder(regime_features)
        
        # Combine
        combined = torch.cat([market_emb, regime_emb], dim=1)
        output = self.combine(combined)
        
        return output


class TwoTowerStockPredictor(nn.Module):
    """
    Two-tower model for stock prediction.
    Combines stock-specific and market context features.
    """
    
    def __init__(
        self,
        stock_dim_output=128,
        market_dim_output=128,
        fusion_dim=256,
        num_heads=8,
        dropout=0.2
    ):
        super().__init__()
        
        # Two towers
        self.stock_tower = StockTower(
            dim_output=stock_dim_output,
            num_heads=num_heads
        )
        
        self.market_tower = MarketTower(
            dim_output=market_dim_output,
            num_heads=num_heads
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(stock_dim_output + market_dim_output, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output heads
        self.return_head = nn.Linear(fusion_dim // 2, 1)  # 3-month return
        self.entry_head = nn.Linear(fusion_dim // 2, 1)    # Entry score (0-100)
        self.risk_head = nn.Linear(fusion_dim // 2, 1)     # Max drawdown
        
    def forward(self, stock_time_series, stock_scalar, market_series, market_regime):
        """
        Forward pass.
        
        Args:
            stock_time_series: [batch, days, 5] (price, volume, returns, high, low)
            stock_scalar: [batch, 10] (RSI, Entry Score, etc.)
            market_series: [batch, days, 3] (market_return, sector_return, vix)
            market_regime: [batch, 5] (beta, relative_strength, etc.)
        
        Returns:
            Dictionary with predictions
        """
        # Process through towers
        stock_emb = self.stock_tower(stock_time_series, stock_scalar)
        market_emb = self.market_tower(market_series, market_regime)
        
        # Fusion
        fused = torch.cat([stock_emb, market_emb], dim=1)
        fused = self.fusion(fused)
        
        # Predictions
        return_pred = self.return_head(fused)
        entry_pred = torch.sigmoid(self.entry_head(fused)) * 100  # Scale to 0-100
        risk_pred = self.risk_head(fused)
        
        return {
            'return_3m': return_pred,
            'entry_score': entry_pred,
            'max_drawdown': risk_pred
        }


# Example usage
if __name__ == "__main__":
    # Create model
    model = TwoTowerStockPredictor()
    
    # Example inputs
    batch_size = 32
    days = 60
    
    stock_ts = torch.randn(batch_size, days, 5)  # Price, volume, returns, high, low
    stock_scalar = torch.randn(batch_size, 10)   # RSI, Entry Score, etc.
    market_ts = torch.randn(batch_size, days, 3)  # Market return, sector return, VIX
    market_regime = torch.randn(batch_size, 5)   # Beta, relative strength, etc.
    
    # Forward pass
    predictions = model(stock_ts, stock_scalar, market_ts, market_regime)
    
    print("Predictions:")
    print(f"  Return 3m: {predictions['return_3m'].shape}")
    print(f"  Entry Score: {predictions['entry_score'].shape}")
    print(f"  Max Drawdown: {predictions['max_drawdown'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

