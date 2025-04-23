import torch
from torch import jit
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Union, Tuple, Any, Literal
from enum import Enum
from dataclasses import dataclass


@dataclass
class HuberConfig:
    """Configuration for Huber loss parameters"""
    k: float = 1.5
    reduction: str = "mean"
    device: str = "cpu"
    learn_k: bool = False


@dataclass
class AsymmetryConfig:
    """Configuration for asymmetric loss"""
    alpha: float = 0.5
    learn_alpha: bool = False


@dataclass
class TemporalAttentionConfig:
    """Configuration for temporal attention"""
    enabled: bool = False
    lookback: int = 10
    hidden_dim: int = 64
    max_features: int = 50  # Maximum number of features to support
    normalize_attention: bool = True  # Whether to normalize per-timestep loss by lookback


@dataclass
class RobustConfig:
    """Configuration for robust outlier handling"""
    beta: float = 0.1
    soft_winsorizing: bool = True
    smooth_factor: float = 1.0  # Renamed from beta to avoid confusion
    buffer_dtype: torch.dtype = torch.float32  # Buffer precision


@dataclass
class MetadataSpec:
    """Specification for required and optional metadata keys"""
    # Required keys
    ROLLING_VOL: str = "atr14"
    # Optional keys
    VOLUME: str = "volume"
    HISTORY: str = "history_features"
    REGIME: str = "market_regime"
    VOL_MODEL: str = "volatility_estimate"


class MarketRegime(Enum):
    """Enumeration of market regimes for consistent reference"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"
    VOLATILE = "volatile"
    LOW_LIQUIDITY = "low_liquidity"
    NEWS_EVENT = "news_event"


class RegimeMapper(nn.Module):
    """Maps market regimes to k multipliers"""
    def __init__(self, device: str = "cpu"):
        super().__init__()
        
        # Define regime multipliers
        regimes = [r.value for r in MarketRegime]
        self.regime_to_idx = {regime: i for i, regime in enumerate(regimes)}
        
        # Initial multiplier values
        initial_values = torch.tensor([
            1.2,  # TRENDING_UP
            1.2,  # TRENDING_DOWN
            1.5,  # RANGE_BOUND
            2.0,  # VOLATILE
            1.8,  # LOW_LIQUIDITY
            2.5   # NEWS_EVENT
        ], device=device)
        
        # Create as parameter for potential learning
        self.multipliers = nn.Parameter(initial_values)
    
    @jit.export
    def get_regime_idx(self, regime: Optional[str]) -> int:
        """Get index for regime, with fallback"""
        if regime is not None and regime in self.regime_to_idx:
            return self.regime_to_idx[regime]
        return -1  # Invalid index as fallback
        
    def forward(self, regime: Optional[str]) -> torch.Tensor:
        """Get k multiplier for the given regime"""
        # Get index for the regime
        regime_idx = self.get_regime_idx(regime)
        
        # Return multiplier or default
        if regime_idx >= 0:
            return self.multipliers[regime_idx].unsqueeze(0)
        else:
            return torch.ones(1, device=self.multipliers.device)


class TemporalAttention(nn.Module):
    """Attention mechanism for time series data"""
    def __init__(self, config: TemporalAttentionConfig, device: str = "cpu"):
        super().__init__()
        self.config = config
        self.device = device
        
        # Separate projections for 2D and 3D inputs
        self.proj_2d = nn.Linear(config.lookback, config.hidden_dim)
        
        # Support variable feature count with a maximum limit
        max_input_size = config.lookback * config.max_features
        self.feat_proj = nn.Linear(max_input_size, config.hidden_dim)
        
        self.attention_net = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.lookback),
            nn.Softmax(dim=1)
        ).to(device)
    
    def forward(self, history_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply attention to history tensor
        
        Parameters:
        -----------
        history_tensor : torch.Tensor
            Historical data with shape [batch_size, lookback] or [batch_size, lookback, features]
            
        Returns:
        --------
        torch.Tensor
            Attention weights [batch_size, lookback]
        """
        batch_size = history_tensor.shape[0]
        
        # Check tensor shape matches expected lookback
        if history_tensor.shape[1] != self.config.lookback:
            raise ValueError(
                f"Expected history_tensor with lookback {self.config.lookback}, "
                f"but got shape {history_tensor.shape}"
            )
            
        # Handle different input shapes
        if history_tensor.dim() == 2:
            # [batch_size, lookback]
            projected = self.proj_2d(history_tensor)
        elif history_tensor.dim() == 3:
            # [batch_size, lookback, features]
            lookback, features = history_tensor.shape[1:]
            
            # Verify we're not silently truncating too many features
            if features > self.config.max_features:
                raise ValueError(
                    f"Input has {features} features, which exceeds max_features={self.config.max_features}. "
                    f"Increase max_features or reduce input dimensionality."
                )
                
            flattened = history_tensor.reshape(batch_size, lookback * features)
            
            # Pad to fixed size for consistent projections
            if flattened.shape[1] < self.feat_proj.in_features:
                padding = torch.zeros(
                    batch_size, 
                    self.feat_proj.in_features - flattened.shape[1],
                    device=flattened.device, 
                    dtype=flattened.dtype
                )
                flattened = torch.cat([flattened, padding], dim=1)
            
            projected = self.feat_proj(flattened)
        else:
            raise ValueError(f"Unexpected shape for history_tensor: {history_tensor.shape}")
        
        # Calculate attention weights
        attention_weights = self.attention_net(projected)
        
        return attention_weights


class StreamingQuantileEstimator:
    """
    Approximate streaming quantile estimator to avoid repeated torch.quantile calls
    Uses a tensor-based buffer for better performance
    """
    def __init__(self, window_size: int = 1000, quantiles: List[float] = [0.1, 0.9], 
                 dtype: torch.dtype = torch.float32):
        self.window_size = window_size
        self.quantiles = quantiles
        
        # Use tensor-based circular buffer with specified dtype
        self.buffer = torch.zeros(window_size, dtype=dtype)
        self.buffer_pos = 0
        self.buffer_full = False
        
        self.current_estimates = {q: None for q in quantiles}
        self.update_frequency = 100
        self.updates_since_recalc = 0
    
    @jit.ignore
    def update(self, values: torch.Tensor) -> None:
        """
        Update the estimator with new values
        
        Parameters:
        -----------
        values : torch.Tensor
            New values to incorporate
        """
        # Convert to CPU tensor with preserved precision
        cpu_values = values.detach().cpu().to(self.buffer.dtype)
        flat_values = cpu_values.flatten()
        
        # Add values to circular buffer
        n_values = len(flat_values)
        
        # Handle case where incoming batch exceeds buffer size
        if n_values >= self.window_size:
            # If batch is larger than buffer, just use the most recent values
            self.buffer = flat_values[-self.window_size:].clone()
            self.buffer_pos = 0
            self.buffer_full = True
        else:
            # Add values to buffer at current position
            space_left = self.window_size - self.buffer_pos
            if n_values <= space_left:
                # Can fit all values in remaining space
                self.buffer[self.buffer_pos:self.buffer_pos + n_values] = flat_values
                self.buffer_pos += n_values
                if self.buffer_pos >= self.window_size:
                    self.buffer_full = True
                    self.buffer_pos = 0
            else:
                # Need to wrap around
                first_part = space_left
                second_part = n_values - space_left
                
                self.buffer[self.buffer_pos:] = flat_values[:first_part]
                self.buffer[:second_part] = flat_values[first_part:]
                
                self.buffer_pos = second_part
                self.buffer_full = True
        
        # Recalculate quantiles periodically
        self.updates_since_recalc += 1
        if self.updates_since_recalc >= self.update_frequency or any(v is None for v in self.current_estimates.values()):
            self._recalculate_quantiles()
            self.updates_since_recalc = 0
    
    @jit.ignore
    def _recalculate_quantiles(self) -> None:
        """Recalculate quantile estimates from current buffer"""
        if not self.buffer_full and self.buffer_pos == 0:
            # Empty buffer
            return
            
        # Get valid elements from buffer
        if self.buffer_full:
            valid_values = self.buffer
        else:
            valid_values = self.buffer[:self.buffer_pos]
        
        # Sort values for quantile calculation
        sorted_values, _ = torch.sort(valid_values)
        
        # Calculate each quantile
        for q in self.quantiles:
            idx = int(q * (len(sorted_values) - 1))
            self.current_estimates[q] = sorted_values[idx].item()
    
    @jit.ignore
    def get_quantiles(self, device: str) -> Dict[float, torch.Tensor]:
        """
        Get current quantile estimates
        
        Returns:
        --------
        Dict[float, torch.Tensor]
            Dictionary mapping quantiles to their estimates
        """
        if any(v is None for v in self.current_estimates.values()):
            self._recalculate_quantiles()
            
        return {q: torch.tensor([v], device=device) for q, v in self.current_estimates.items()}


# Base Huber loss implementation
class HuberBase(nn.Module):
    """
    Base Huber loss implementation
    
    Parameters:
    -----------
    config : HuberConfig
        Configuration for the Huber loss
    """
    def __init__(self, config: HuberConfig):
        super().__init__()
        self.config = config
        
        # Learnable parameter if enabled
        if config.learn_k:
            self.k = nn.Parameter(torch.tensor([config.k], device=config.device))
        else:
            self.register_buffer('k', torch.tensor([config.k], device=config.device))
    
    def standard_huber(self, 
                     residuals: torch.Tensor, 
                     delta: torch.Tensor) -> torch.Tensor:
        """
        Calculate standard Huber loss
        
        Parameters:
        -----------
        residuals : torch.Tensor
            Prediction residuals [batch_size, 1]
        delta : torch.Tensor
            Delta parameter for Huber transition point [batch_size, 1]
            
        Returns:
        --------
        torch.Tensor
            Huber loss values [batch_size, 1]
        """
        abs_residuals = residuals.abs()
        quadratic_mask = abs_residuals <= delta
        
        # Use torch.where instead of masked_scatter for better performance
        quadratic = 0.5 * residuals.pow(2)
        linear = delta * (abs_residuals - 0.5 * delta)
        
        loss = torch.where(quadratic_mask, quadratic, linear)
        return loss
    
    def forward(self, 
               residuals: torch.Tensor, 
               delta: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Parameters:
        -----------
        residuals : torch.Tensor
            Prediction residuals [batch_size, 1]
        delta : torch.Tensor
            Delta parameter for Huber transition point [batch_size, 1]
            
        Returns:
        --------
        torch.Tensor
            Huber loss values [batch_size, 1]
        """
        return self.standard_huber(residuals, delta)


# Asymmetric loss mixin
class AsymmetricLoss(nn.Module):
    """
    Asymmetric loss mixin for differential penalization of over/under predictions
    
    Parameters:
    -----------
    config : AsymmetryConfig
        Configuration for asymmetric loss
    device : str
        Device to use for computations
    """
    def __init__(self, config: AsymmetryConfig, device: str = "cpu"):
        super().__init__()
        self.config = config
        
        # Learnable parameter if enabled
        if config.learn_alpha:
            # Use logit space for stable optimization
            alpha_logit = torch.log(torch.tensor([config.alpha / (1 - config.alpha)]))
            self.alpha_logit = nn.Parameter(alpha_logit.to(device))
        else:
            self.register_buffer('alpha', torch.tensor([config.alpha], device=device))
    
    def get_alpha(self) -> torch.Tensor:
        """Get current asymmetry parameter value"""
        if hasattr(self, 'alpha_logit'):
            # Convert from logit space
            return torch.sigmoid(self.alpha_logit)
        else:
            return self.alpha
    
    def apply_asymmetry(self, 
                       loss: torch.Tensor, 
                       residuals: torch.Tensor) -> torch.Tensor:
        """
        Apply asymmetric weighting to the loss
        
        Parameters:
        -----------
        loss : torch.Tensor
            Calculated loss values [batch_size, 1]
        residuals : torch.Tensor
            Prediction residuals [batch_size, 1]
            
        Returns:
        --------
        torch.Tensor
            Asymmetrically weighted loss [batch_size, 1]
        """
        alpha = self.get_alpha()
        
        # Calculate weights directly with torch.where for better performance
        n_total = residuals.numel()
        n_neg = (residuals < 0).sum().item()
        n_pos = n_total - n_neg
        
        # Scale to ensure the mean weight remains 1.0
        if n_total > 0:
            if n_neg > 0:
                w_neg = alpha * n_total / n_neg
                w_neg = torch.clamp(w_neg, 0.1, 10.0)
            else:
                w_neg = torch.tensor(0.1, device=residuals.device)
                
            if n_pos > 0:
                w_pos = (1 - alpha) * n_total / n_pos
                w_pos = torch.clamp(w_pos, 0.1, 10.0)
            else:
                w_pos = torch.tensor(0.1, device=residuals.device)
        else:
            w_neg = w_pos = torch.tensor(1.0, device=residuals.device)
        
        # Apply weights
        asymmetry = torch.where(residuals < 0, 
                              torch.ones_like(residuals) * w_neg, 
                              torch.ones_like(residuals) * w_pos)
        
        return loss * asymmetry
    
    def forward(self, loss: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying asymmetry to loss
        
        Parameters:
        -----------
        loss : torch.Tensor
            Base loss values [batch_size, 1]
        residuals : torch.Tensor
            Prediction residuals [batch_size, 1]
            
        Returns:
        --------
        torch.Tensor
            Asymmetrically weighted loss [batch_size, 1]
        """
        return self.apply_asymmetry(loss, residuals)


# Quantile Huber loss mixin
class QuantileHuberLoss(nn.Module):
    """
    Quantile Huber loss implementation
    
    Parameters:
    -----------
    quantile : float
        Quantile to estimate (0.5 = median)
    config : HuberConfig
        Configuration for the Huber loss
    """
    def __init__(self, quantile: float = 0.5, config: HuberConfig = None):
        super().__init__()
        self.quantile = quantile
        
        # Store k parameter even in quantile mode for consistent access
        if config:
            if config.learn_k:
                self.k = nn.Parameter(torch.tensor([config.k], device=config.device))
            else:
                self.register_buffer('k', torch.tensor([config.k], device=config.device))
    
    def forward(self, 
              residuals: torch.Tensor, 
              delta: torch.Tensor) -> torch.Tensor:
        """
        Calculate quantile Huber loss
        
        Parameters:
        -----------
        residuals : torch.Tensor
            Prediction residuals [batch_size, 1]
        delta : torch.Tensor
            Delta parameter for Huber transition point [batch_size, 1]
            
        Returns:
        --------
        torch.Tensor
            Quantile Huber loss values [batch_size, 1]
        """
        abs_residuals = residuals.abs()
        q = self.quantile
        
        # Calculate quantile pinball loss component with torch.where for performance
        quantile_error = torch.where(
            residuals >= 0,
            q * residuals,
            (q - 1) * residuals
        )
        
        # Apply Huber modification to quantile loss 
        huber_quantile_loss = torch.where(
            abs_residuals <= delta,
            0.5 * quantile_error.pow(2) / delta,
            quantile_error * (abs_residuals - 0.5 * delta)
        )
        
        return huber_quantile_loss


# Robust outlier handling mixin
class RobustLoss(nn.Module):
    """
    Robust loss with outlier handling
    
    Parameters:
    -----------
    config : RobustConfig
        Configuration for robust outlier handling
    """
    def __init__(self, config: RobustConfig):
        super().__init__()
        self.config = config
        self.quantile_estimator = StreamingQuantileEstimator(
            quantiles=[config.beta, 1.0 - config.beta],
            dtype=config.buffer_dtype
        )
    
    @jit.ignore
    def soft_clamp(self, 
                 x: torch.Tensor, 
                 low: torch.Tensor, 
                 high: torch.Tensor, 
                 smooth_factor: float = 1.0) -> torch.Tensor:
        """
        Soft clamping function that preserves gradients
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
        low : torch.Tensor
            Lower bound
        high : torch.Tensor
            Upper bound
        smooth_factor : float
            Smoothing parameter
            
        Returns:
        --------
        torch.Tensor
            Soft-clamped tensor
        """
        # Smooth lower bound
        x_low = x - low
        soft_lower = low + F.softplus(x_low, beta=smooth_factor, threshold=20)
        
        # Smooth upper bound
        soft_high_x = high - soft_lower
        soft_clamped = high - F.softplus(soft_high_x, beta=smooth_factor, threshold=20)
        
        return soft_clamped
    
    @jit.ignore
    def detect_outliers(self, 
                      residuals: torch.Tensor, 
                      rolling_vol: torch.Tensor) -> torch.Tensor:
        """
        Detect and handle outliers using Winsorization
        
        Parameters:
        -----------
        residuals : torch.Tensor
            Prediction residuals [batch_size, 1]
        rolling_vol : torch.Tensor
            Per-sample volatility estimate [batch_size, 1]
            
        Returns:
        --------
        torch.Tensor
            Winsorized residuals [batch_size, 1]
        """
        # Normalize residuals by volatility for comparable thresholds
        norm_residuals = residuals / rolling_vol.clamp(min=1e-8)
        
        # Update streaming quantile estimator
        self.quantile_estimator.update(norm_residuals)
        
        # Get quantile estimates
        quantiles = self.quantile_estimator.get_quantiles(residuals.device)
        q_low = quantiles[self.config.beta]
        q_high = quantiles[1.0 - self.config.beta]
        
        # Apply Winsorization (either hard or soft)
        if self.config.soft_winsorizing:
            # Soft winsorization preserves gradients
            winsorized_norm_residuals = self.soft_clamp(
                norm_residuals, q_low, q_high, self.config.smooth_factor
            )
        else:
            # Hard winsorization (gradient is zero at the clipping points)
            winsorized_norm_residuals = torch.clamp(norm_residuals, q_low, q_high)
        
        # Convert back to original scale
        return winsorized_norm_residuals * rolling_vol
    
    def forward(self, 
              residuals: torch.Tensor, 
              rolling_vol: torch.Tensor) -> torch.Tensor:
        """
        Apply robust outlier handling
        
        Parameters:
        -----------
        residuals : torch.Tensor
            Prediction residuals [batch_size, 1]
        rolling_vol : torch.Tensor
            Per-sample volatility estimate [batch_size, 1]
            
        Returns:
        --------
        torch.Tensor
            Robust residuals [batch_size, 1]
        """
        return self.detect_outliers(residuals, rolling_vol)


# Heteroscedastic scaling mixin
class HeteroscedasticLoss(nn.Module):
    """
    Heteroscedastic scaling for varying noise levels
    """
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.register_buffer('residual_var', torch.tensor([1.0], device=device))
        self.register_buffer('update_count', torch.tensor([0], dtype=torch.long, device=device))
        self.ema_alpha = 0.05
    
    def update_statistics(self, residuals: torch.Tensor) -> None:
        """
        Update running statistics of residuals
        
        Parameters:
        -----------
        residuals : torch.Tensor
            Prediction residuals
        """
        if not self.training:
            return
            
        with torch.no_grad():
            batch_var = residuals.var()
            
            # Update count and reset if too large to maintain time-constant
            self.update_count += 1
            if self.update_count >= 10_000:
                self.update_count.zero_()
            
            # Exponential moving average update
            alpha = self.ema_alpha
            self.residual_var = (1 - alpha) * self.residual_var + alpha * batch_var
    
    def apply_heteroscedastic_scaling(self, 
                                   residuals: torch.Tensor,
                                   delta: torch.Tensor,
                                   vol_model: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply heteroscedastic scaling to residuals only, not delta
        
        Parameters:
        -----------
        residuals : torch.Tensor
            Prediction residuals [batch_size, 1]
        delta : torch.Tensor
            Delta parameter for Huber transition point [batch_size, 1]
        vol_model : Optional[torch.Tensor]
            Model-estimated volatility if available [batch_size, 1]
            
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            Scaled residuals and original delta
        """
        if vol_model is not None:
            # Use model-provided volatility estimate
            scaling_factor = vol_model.clamp(min=1e-8)
        else:
            # Use simple exponential scaling based on residual statistics
            scaling_factor = torch.sqrt(self.residual_var).clamp(min=1e-8)
            
        # Scale residuals only, not delta (fixes the nullification issue)
        scaled_residuals = residuals / scaling_factor
        
        return scaled_residuals, delta
    
    def forward(self, 
              residuals: torch.Tensor,
              delta: torch.Tensor,
              vol_model: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply heteroscedastic scaling
        
        Parameters:
        -----------
        residuals : torch.Tensor
            Prediction residuals [batch_size, 1]
        delta : torch.Tensor
            Delta parameter for Huber transition point [batch_size, 1]
        vol_model : Optional[torch.Tensor]
            Model-estimated volatility if available [batch_size, 1]
            
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            Scaled residuals and delta
        """
        self.update_statistics(residuals)
        return self.apply_heteroscedastic_scaling(residuals, delta, vol_model)


# Volume weighting mixin
class VolumeWeightedLoss(nn.Module):
    """
    Volume weighting for economic significance
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, loss: torch.Tensor, vol: torch.Tensor) -> torch.Tensor:
        """
        Apply volume weighting to loss
        
        Parameters:
        -----------
        loss : torch.Tensor
            Base loss values [batch_size, 1]
        vol : torch.Tensor
            Volume data [batch_size, 1]
            
        Returns:
        --------
        torch.Tensor
            Volume-weighted loss [batch_size, 1]
        """
        # Normalize weights to preserve the scale of the loss
        w = vol / vol.mean()
        return loss * w


# Main Advanced Adaptive Huber Loss that composes the components
class AdvancedAdaptiveHuberLoss(nn.Module):
    """
    Advanced Huber loss implementation with multiple customizations, composed
    from smaller, specialized components.
    
    Parameters:
    -----------
    k : float, default=1.5
        Base multiplier for delta calculation
    lookback : int, default=10
        Number of time steps to consider for temporal attention
    asymmetry_alpha : float, default=0.5
        Asymmetry factor (0.5 = symmetric, <0.5 = higher penalty for underestimation)
    reduction : str, default="mean"
        Reduction method for batch losses ("mean", "sum", or "none")
    use_temporal_attention : bool, default=True
        Whether to use temporal attention mechanism
    learn_asymmetry : bool, default=False
        Whether to learn asymmetry parameter based on data
    learn_k : bool, default=False
        Whether to learn k parameter based on data
    quantile : Optional[float], default=None
        If provided, use quantile regression version of Huber loss
    robust_beta : float, default=0.1
        Parameter for robust outlier detection (beta value for Winsorizing)
    soft_winsorizing : bool, default=True
        Whether to use soft winsorizing (gradient-preserving)
    attn_hidden_dim : int, default=64
        Hidden dimension for attention network
    max_features : int, default=50
        Maximum number of features supported in history tensor
    normalize_attention : bool, default=True
        Whether to normalize per-timestep loss by lookback
    buffer_dtype : torch.dtype, default=torch.float32
        Precision for quantile estimator buffer
    device : str, default="cpu"
        Device to run computations on
    """
    def __init__(
        self,
        k: float = 1.5,
        lookback: int = 10,
        asymmetry_alpha: float = 0.5,
        reduction: str = "mean",
        use_temporal_attention: bool = True,
        learn_asymmetry: bool = False,
        learn_k: bool = False,
        quantile: Optional[float] = None,
        robust_beta: float = 0.1,
        soft_winsorizing: bool = True,
        attn_hidden_dim: int = 64,
        max_features: int = 50,
        normalize_attention: bool = True,
        buffer_dtype: torch.dtype = torch.float32,
        device: str = "cpu"
    ):
        super().__init__()
        self.reduction = reduction
        self.use_temporal_attention = use_temporal_attention
        self.quantile_mode = quantile is not None
        self.metadata_spec = MetadataSpec()
        self.device = device
        self.lookback = lookback
        self.normalize_attention = normalize_attention
        
        # Create configuration objects
        huber_config = HuberConfig(k=k, reduction="none", device=device, learn_k=learn_k)
        
        asymmetry_config = AsymmetryConfig(alpha=asymmetry_alpha, learn_alpha=learn_asymmetry)
        
        temporal_config = TemporalAttentionConfig(
            enabled=use_temporal_attention,
            lookback=lookback,
            hidden_dim=attn_hidden_dim,
            max_features=max_features,
            normalize_attention=normalize_attention
        )
        
        robust_config = RobustConfig(
            beta=robust_beta, 
            soft_winsorizing=soft_winsorizing,
            smooth_factor=1.0,
            buffer_dtype=buffer_dtype
        )
        
        # Always create a HuberBase for consistent k access
        self.huber_base = HuberBase(huber_config)
        
        # Initialize components
        # Base loss (either standard Huber or quantile Huber)
        if self.quantile_mode:
            self.base_loss = QuantileHuberLoss(quantile=quantile, config=huber_config)
        else:
            self.base_loss = self.huber_base  # Use the same instance
            self.asymmetric_loss = AsymmetricLoss(asymmetry_config, device)
        
        # Other components
        self.robust_loss = RobustLoss(robust_config)
        self.heteroscedastic_loss = HeteroscedasticLoss(device)
        self.volume_weighted_loss = VolumeWeightedLoss()
        self.regime_mapper = RegimeMapper(device)
        
        # Temporal attention (conditional)
        if use_temporal_attention:
            self.temporal_attention = TemporalAttention(temporal_config, device)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AdvancedAdaptiveHuberLoss':
        """
        Create loss function from configuration dictionary
        
        Parameters:
        -----------
        config : Dict[str, Any]
            Configuration dictionary
            
        Returns:
        --------
        AdvancedAdaptiveHuberLoss
            Configured loss function
        """
        return cls(**config)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration dictionary
        
        Returns:
        --------
        Dict[str, Any]
            Configuration dictionary
        """
        # Return all constructor parameters for idempotent round-trip
        config = {
            'k': self.huber_base.k.item(),
            'lookback': self.lookback,
            'asymmetry_alpha': (
                self.asymmetric_loss.get_alpha().item() 
                if hasattr(self, 'asymmetric_loss') else 0.5
            ),
            'reduction': self.reduction,
            'use_temporal_attention': self.use_temporal_attention,
            'learn_asymmetry': hasattr(self, 'asymmetric_loss') and self.asymmetric_loss.config.learn_alpha,
            'learn_k': self.huber_base.config.learn_k,
            'quantile': getattr(self.base_loss, 'quantile', None) if self.quantile_mode else None,
            'robust_beta': self.robust_loss.config.beta,
            'soft_winsorizing': self.robust_loss.config.soft_winsorizing,
            'attn_hidden_dim': (
                self.temporal_attention.config.hidden_dim 
                if self.use_temporal_attention else 64
            ),
            'max_features': (
                self.temporal_attention.config.max_features 
                if self.use_temporal_attention else 50
            ),
            'normalize_attention': self.normalize_attention,
            'buffer_dtype': str(self.robust_loss.quantile_estimator.buffer.dtype).split('.')[-1],
            'device': self.device
        }
        return config
    
    def forward(self, 
              y_pred: torch.Tensor, 
              y_true: torch.Tensor, 
              rolling_vol: torch.Tensor,
              vol: Optional[torch.Tensor] = None, 
              history_tensor: Optional[torch.Tensor] = None,
              regime: Optional[str] = None,
              vol_model: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass calculating the advanced Huber loss
        
        Parameters:
        -----------
        y_pred : torch.Tensor
            Predicted values [batch_size, ...] 
            Can be any shape as long as it broadcasts with y_true
        y_true : torch.Tensor
            True values [batch_size, ...]
            Can be any shape as long as it broadcasts with y_pred
        rolling_vol : torch.Tensor
            Rolling volatility for each sample [batch_size, 1]
        vol : Optional[torch.Tensor]
            Volume data for volume weighting [batch_size, 1]
        history_tensor : Optional[torch.Tensor]
            Historical data for temporal attention [batch_size, lookback] or [batch_size, lookback, features]
        regime : Optional[str]
            Current market regime
        vol_model : Optional[torch.Tensor]
            Model-estimated volatility for heteroscedastic scaling [batch_size, 1]
            
        Returns:
        --------
        torch.Tensor
            Final calculated loss (scalar or [batch_size, 1] depending on reduction)
        """
        # Add fail-fast guards
        if not torch.all(rolling_vol > 0):
            raise ValueError("rolling_vol must be positive")
        if vol is not None and not torch.all(vol >= 0):
            raise ValueError("vol must be non-negative")
        if history_tensor is not None and history_tensor.dim() == 2:
            if history_tensor.shape[1] != self.lookback:
                raise ValueError(
                    f"Expected history_tensor with lookback {self.lookback}, "
                    f"but got shape {history_tensor.shape}"
                )
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Apply robust outlier handling
        robust_residuals = self.robust_loss(residuals, rolling_vol)
        
        # Get adaptive k value based on regime
        k_multiplier = self.regime_mapper(regime)
        
        # Safely access k parameter (works in both standard and quantile modes)
        k_tensor = self.huber_base.k
        
        # Calculate delta with adaptive k value
        delta = k_multiplier * k_tensor * rolling_vol.clamp(min=1e-8)
        
        # Apply heteroscedastic scaling if model volatility provided
        if vol_model is not None:
            # Scale residuals only, not delta
            scaled_residuals, scaled_delta = self.heteroscedastic_loss(robust_residuals, delta, vol_model)
        else:
            scaled_residuals, scaled_delta = robust_residuals, delta
        
        # Calculate base loss (Huber or quantile Huber)
        loss = self.base_loss(scaled_residuals, scaled_delta)
        
        # Apply asymmetric weighting if not in quantile mode
        if not self.quantile_mode:
            loss = self.asymmetric_loss(loss, scaled_residuals)
        
        # Apply volume weighting if provided
        if vol is not None:
            loss = self.volume_weighted_loss(loss, vol)
        
        # Apply temporal attention if provided and enabled
        if self.use_temporal_attention and history_tensor is not None:
            # Calculate attention weights for the sequence
            attention_weights = self.temporal_attention(history_tensor)  # [B, L]
            
            # Proper temporal attention requires per-timestep losses
            # Since we only have per-sample loss, create per-timestep loss by distributing
            # equally across timesteps, then apply attention weights, and sum back
            per_ts_loss = loss.repeat(1, attention_weights.size(1))
            
            # Optionally normalize by lookback to maintain consistent gradient magnitude
            if self.normalize_attention:
                per_ts_loss = per_ts_loss / attention_weights.size(1)
                
            loss = (per_ts_loss * attention_weights).sum(dim=1, keepdim=True)
        
        # Apply reduction if needed
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss
    
    def extra_repr(self) -> str:
        """Extra representation string for printing"""
        config = self.get_config()
        return (f'k={config["k"]}, '
                f'asymmetry_alpha={config["asymmetry_alpha"]}, '
                f'reduction={config["reduction"]}, '
                f'use_temporal_attention={config["use_temporal_attention"]}, '
                f'quantile={config["quantile"]}')


# Example of using the loss function
def example_usage():
    # Initialize the loss function
    criterion = AdvancedAdaptiveHuberLoss(
        k=1.3,
        asymmetry_alpha=0.4,  # Penalize underestimation more
        use_temporal_attention=True,
        lookback=10,
        learn_asymmetry=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Create sample data
    batch_size = 32
    lookback = 10
    
    y_pred = torch.randn(batch_size, 1)
    y_true = torch.randn(batch_size, 1)
    rolling_vol = torch.abs(torch.randn(batch_size, 1)) + 0.5  # Simulated volatility
    vol = torch.abs(torch.randn(batch_size, 1)) + 1.0  # Simulated volume
    
    # Historical data for attention mechanism (e.g., past returns)
    history_tensor = torch.randn(batch_size, lookback)
    
    # Optional volatility model estimates
    vol_model = torch.abs(torch.randn(batch_size, 1)) + 0.3
    
    # Current market regime
    regime = "volatile"
    
    # Calculate loss
    loss = criterion(
        y_pred=y_pred,
        y_true=y_true,
        rolling_vol=rolling_vol,
        vol=vol,
        history_tensor=history_tensor,
        regime=regime,
        vol_model=vol_model
    )
    
    print(f"Loss: {loss.item()}")
    
    # Test with 3D history tensor
    features = 3
    history_tensor_3d = torch.randn(batch_size, lookback, features)
    
    loss_3d = criterion(
        y_pred=y_pred,
        y_true=y_true,
        rolling_vol=rolling_vol,
        vol=vol,
        history_tensor=history_tensor_3d,
        regime=regime,
        vol_model=vol_model
    )
    
    print(f"Loss with 3D history: {loss_3d.item()}")
    
    # Test config round-trip
    config = criterion.get_config()
    print(f"Config: {config}")
    
    criterion2 = AdvancedAdaptiveHuberLoss.from_config(config)
    print(f"Recreated loss: {criterion2}")


# Example training function with the advanced loss
def train_model(model, train_loader, optimizer, epochs=10):
    """
    Example training function using the advanced Huber loss
    
    Parameters:
    -----------
    model : torch.nn.Module
        PyTorch model to train
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data
    optimizer : torch.optim.Optimizer
        Optimizer for training
    epochs : int, default=10
        Number of training epochs
    """
    # Initialize loss function
    criterion = AdvancedAdaptiveHuberLoss(
        k=1.3,
        asymmetry_alpha=0.4,
        use_temporal_attention=True,
        learn_asymmetry=True,
        learn_k=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for X_batch, y_batch, meta in train_loader:
            # Forward pass
            y_pred, vol_pred = model(X_batch)
            
            # Extract metadata
            rolling_vol = meta[criterion.metadata_spec.ROLLING_VOL]
            vol = meta.get(criterion.metadata_spec.VOLUME, None)
            history = meta.get(criterion.metadata_spec.HISTORY, None)
            regime = meta.get(criterion.metadata_spec.REGIME, None)
            
            # Calculate loss
            loss = criterion(
                y_pred=y_pred,
                y_true=y_batch,
                rolling_vol=rolling_vol,
                vol=vol,
                history_tensor=history,
                regime=regime,
                vol_model=vol_pred
            )
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.6f}")
        
        # Check learned parameters if applicable
        if isinstance(criterion.huber_base.k, nn.Parameter):
            print(f"  Learned k: {criterion.huber_base.k.item():.4f}")
        
        if hasattr(criterion, 'asymmetric_loss') and criterion.asymmetric_loss.config.learn_alpha:
            alpha = criterion.asymmetric_loss.get_alpha().item()
            print(f"  Learned asymmetry: {alpha:.4f}")


# Unit tests for the loss function
def run_unit_tests():
    """Run basic unit tests for the loss function"""
    # Enable this to run more extensive tests
    import os
    FULL_TEST_SUITE = os.environ.get('LOSS_FULL_TEST', '0') == '1'  
    
    print("Running unit tests...")
    
    # Test 1: Basic initialization
    try:
        loss = AdvancedAdaptiveHuberLoss()
        print("✓ Basic initialization")
    except Exception as e:
        print(f"✗ Basic initialization failed: {e}")
    
    # Test 2: Gradient check
    try:
        loss = AdvancedAdaptiveHuberLoss(use_temporal_attention=False)
        y_pred = torch.randn(8, 1, requires_grad=True)
        y_true = torch.randn(8, 1)
        rolling_vol = torch.ones_like(y_true) + 0.1  # Ensure positive
        
        l = loss(y_pred, y_true, rolling_vol)
        l.backward()  # Check if gradients flow
        assert y_pred.grad is not None
        print("✓ Gradient flow")
    except Exception as e:
        print(f"✗ Gradient check failed: {e}")
    
    # Test 3: Different input shapes
    try:
        loss = AdvancedAdaptiveHuberLoss(use_temporal_attention=True)
        batch_size = 4
        lookback = 10
        features = 3
        
        # Test with 2D history tensor
        y_pred = torch.randn(batch_size, 1)
        y_true = torch.randn(batch_size, 1)
        rolling_vol = torch.ones_like(y_true) + 0.1
        history_2d = torch.randn(batch_size, lookback)
        
        l1 = loss(y_pred, y_true, rolling_vol, history_tensor=history_2d)
        
        # Test with 3D history tensor
        history_3d = torch.randn(batch_size, lookback, features)
        l2 = loss(y_pred, y_true, rolling_vol, history_tensor=history_3d)
        
        print("✓ Different input shapes")
    except Exception as e:
        print(f"✗ Input shape test failed: {e}")
    
    # Test 4: Configuration save/load
    try:
        loss1 = AdvancedAdaptiveHuberLoss(k=1.5, asymmetry_alpha=0.4)
        config = loss1.get_config()
        loss2 = AdvancedAdaptiveHuberLoss.from_config(config)
        
        # Test that all key parameters are preserved
        assert abs(loss1.huber_base.k.item() - loss2.huber_base.k.item()) < 1e-6
        assert loss1.reduction == loss2.reduction
        assert loss1.use_temporal_attention == loss2.use_temporal_attention
        
        print("✓ Configuration save/load")
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
    
    # Test 5: Quantile mode
    try:
        loss = AdvancedAdaptiveHuberLoss(quantile=0.75)
        y_pred = torch.randn(8, 1, requires_grad=True)
        y_true = torch.randn(8, 1)
        rolling_vol = torch.ones_like(y_true) + 0.1
        
        l = loss(y_pred, y_true, rolling_vol)
        l.backward()  # Check if gradients flow
        assert y_pred.grad is not None
        print("✓ Quantile mode")
    except Exception as e:
        print(f"✗ Quantile mode test failed: {e}")
    
    # Only run more intensive tests if enabled
    if FULL_TEST_SUITE:
        # Test 6: Feature count change in attention
        try:
            loss = AdvancedAdaptiveHuberLoss(use_temporal_attention=True)
            batch_size = 4
            lookback = 10
            
            y_pred = torch.randn(batch_size, 1)
            y_true = torch.randn(batch_size, 1)
            rolling_vol = torch.ones_like(y_true) + 0.1
            
            # First with 2 features
            history_1 = torch.randn(batch_size, lookback, 2)
            l1 = loss(y_pred, y_true, rolling_vol, history_tensor=history_1)
            
            # Then with 5 features
            history_2 = torch.randn(batch_size, lookback, 5)
            l2 = loss(y_pred, y_true, rolling_vol, history_tensor=history_2)
            
            print("✓ Feature count change in attention")
        except Exception as e:
            print(f"✗ Feature count change test failed: {e}")
        
        # Test 7: Error on feature count exceeding max_features
        try:
            loss = AdvancedAdaptiveHuberLoss(use_temporal_attention=True, max_features=3)
            batch_size = 4
            lookback = 10
            
            y_pred = torch.randn(batch_size, 1)
            y_true = torch.randn(batch_size, 1)
            rolling_vol = torch.ones_like(y_true) + 0.1
            
            # Try with 5 features > max_features=3
            history = torch.randn(batch_size, lookback, 5)
            
            try:
                l = loss(y_pred, y_true, rolling_vol, history_tensor=history)
                print("✗ Should have raised error on max_features exceeded")
            except ValueError:
                print("✓ Properly raises error when feature count exceeds max_features")
        except Exception as e:
            print(f"✗ Feature count limit test failed: {e}")
        
        # Test 8: Multi-GPU synchronization (if available)
        if torch.cuda.device_count() > 1:
            try:
                device = torch.device("cuda:0")
                loss = AdvancedAdaptiveHuberLoss(device=device)
                loss = nn.DataParallel(loss)
                
                batch_size = 8
                y_pred = torch.randn(batch_size, 1, requires_grad=True, device=device)
                y_true = torch.randn(batch_size, 1, device=device)
                rolling_vol = torch.ones_like(y_true, device=device) + 0.1
                
                l = loss(y_pred, y_true, rolling_vol)
                l.backward()
                
                print("✓ Multi-GPU synchronization")
            except Exception as e:
                print(f"✗ Multi-GPU test failed: {e}")
    
    print("Unit tests complete!")


if __name__ == "__main__":
    # Run example
    example_usage()
    
    # Run unit tests
    run_unit_tests()