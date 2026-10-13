from __future__ import annotations

import copy
import inspect

import torch
from torch import nn

from holosoma.agents.modules.vision_transformer import VisionTransformerEncoder
from holosoma.config_types.algo import LayerConfig, ModuleConfig


class ImgChLayerNorm(nn.Module):
    """Image channel-wise layer normalization."""

    def __init__(self, num_channels, eps: float = 1e-5):
        """Initialize ImgChLayerNorm module.

        Parameters
        ----------
        num_channels: int
            Number of channels in the input tensor
        eps: float, optional
            Small value to prevent division by zero, by default 1e-5
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        """Forward pass for image channel-wise layer normalization.

        Normalizes each channel of the input tensor independently.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape [B, C, H, W]

        Returns
        -------
        torch.Tensor
            Output tensor of shape [B, C, H, W]
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class CameraVisionMask(nn.Module):
    """Velocity-conditioned scalar masking of per-camera features.

    Supports 1, 2, or 4 cameras:
      - 1 camera: pass-through (no masking)
      - 2 cameras [front, back]: masked by forward/backward velocity
      - 4 cameras [front, back, left, right]: also masked by lateral velocity
    """

    def __init__(
        self,
        num_cameras: int = 2,
        k: float = 4.0,
        v_th: float = 0.2,
        hard_cutoff: float = 0.0,
    ):
        super().__init__()
        self.num_cameras = num_cameras
        self.k = k
        self.v_th = v_th
        self.hard_cutoff = hard_cutoff

    def forward(
        self,
        camera_features: list[torch.Tensor],
        v_cmd: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Parameters
        ----------
        camera_features : list[Tensor]
            Per-camera tensors of shape ``[B, C_out, feat_dim]``.
        v_cmd : Tensor
            Velocity command ``[B, D]`` where ``D >= 1`` (or ``D >= 2`` for 4 cameras).

        Returns
        -------
        list[Tensor]
            Scaled camera features.
        """
        num_cams = len(camera_features)
        if num_cams == 1:
            return camera_features

        v_x = v_cmd[:, 0:1]  # [B, 1]
        scale_front = 1.0 - torch.sigmoid(self.k * (-v_x - self.v_th))
        scale_back = 1.0 - torch.sigmoid(self.k * (v_x - self.v_th))

        if self.hard_cutoff > 0:
            scale_front = torch.where(scale_front < self.hard_cutoff, torch.zeros_like(scale_front), scale_front)
            scale_back = torch.where(scale_back < self.hard_cutoff, torch.zeros_like(scale_back), scale_back)

        scales: list[torch.Tensor] = [scale_front, scale_back]

        if num_cams == 4:
            v_y = v_cmd[:, 1:2]
            scale_left = 1.0 - torch.sigmoid(self.k * (-v_y - self.v_th))
            scale_right = 1.0 - torch.sigmoid(self.k * (v_y - self.v_th))
            if self.hard_cutoff > 0:
                scale_left = torch.where(scale_left < self.hard_cutoff, torch.zeros_like(scale_left), scale_left)
                scale_right = torch.where(scale_right < self.hard_cutoff, torch.zeros_like(scale_right), scale_right)
            scales.extend([scale_left, scale_right])

        masked = []
        for cam_idx, cam_feat in enumerate(camera_features):
            if cam_idx < len(scales):
                masked.append(cam_feat * scales[cam_idx].unsqueeze(-1))
            else:
                masked.append(cam_feat)
        return masked


class CNNWrapper(nn.Module):
    """Wrapper module that handles reshaping for CNN layers when working with flattened inputs.

    Supports single-camera and multi-camera modes.  In multi-camera mode each
    camera gets its own CNN branch and outputs are concatenated along the
    feature dimension.  An optional :class:`CameraVisionMask` can scale
    per-camera features based on a velocity command before concatenation.
    """

    def __init__(
        self,
        cnn_layers,
        input_channels,
        input_height,
        input_width,
        flatten_output=True,
        num_cameras: int = 1,
        vision_mask: CameraVisionMask | None = None,
        vision_tokens: str = "spatial",
    ):
        """Initialize CNNWrapper module.

        Parameters
        ----------
        cnn_layers: nn.Module or nn.ModuleList
            CNN layers.  For multi-camera, pass an ``nn.ModuleList`` of per-camera CNNs.
        input_channels: int
            Number of input channels **per camera** (typically ``history_length``).
        input_height: int
            Height of input feature maps.
        input_width: int
            Width of input feature maps.
        flatten_output: bool
            Whether to flatten the output.
        num_cameras: int
            Number of camera views (1 for single-camera).
        vision_mask: CameraVisionMask | None
            Optional velocity-conditioned vision mask.
        vision_tokens: str
            Unflattened output layout: ``"spatial"`` -> ``[B, cells, C_out]``,
            ``"channel"`` -> ``[B, C_out, cells]`` (FAR-FALCON-Terrains).
        """
        super().__init__()
        self.cnn_layers = cnn_layers
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.num_cameras = num_cameras
        self.is_multi_camera = num_cameras > 1
        self.vision_mask = vision_mask
        self.flatten_output = flatten_output
        if vision_tokens not in ("spatial", "channel"):
            raise ValueError(f"vision_tokens must be 'spatial' or 'channel', got {vision_tokens!r}")
        self.vision_tokens = vision_tokens

        self.expected_input_size = num_cameras * input_channels * input_height * input_width

        # Precompute output_channels for multi-camera reshape
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_height, input_width)
            if self.is_multi_camera:
                out = self.cnn_layers[0](dummy)
            else:
                out = self.cnn_layers(dummy)
            self.output_channels = out.shape[1]

    @property
    def output_size(self):
        """Computes the output size of the CNN layers by doing a forward pass with dummy data."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.expected_input_size)
            dummy_output = self.forward(dummy_input)
            return dummy_output.shape[-1]

    def forward(self, x, v_cmd: torch.Tensor | None = None):
        """Forward pass for CNNWrapper module.

        Parameters
        ----------
        x: torch.Tensor
            Flattened input ``[B, num_cameras * C * H * W]``.
        v_cmd: torch.Tensor | None
            Velocity command ``[B, D]`` for vision masking (only used when
            ``vision_mask`` is not None).

        Returns
        -------
        torch.Tensor
            ``[B, -1]`` if ``flatten_output`` else ``[B, tokens, feat_dim]``.
        """
        batch_size = x.shape[0]
        if x.shape[1] != self.expected_input_size:
            raise ValueError(
                f"Input size mismatch: expected {self.expected_input_size} "
                f"(cameras={self.num_cameras}, channels={self.input_channels}, "
                f"height={self.input_height}, width={self.input_width}), "
                f"but got {x.shape[1]}"
            )

        if self.is_multi_camera:
            x = x.view(batch_size, self.input_channels, self.num_cameras, self.input_height, self.input_width)
            camera_outputs: list[torch.Tensor] = []
            for cam_idx in range(self.num_cameras):
                cam_x = x[:, :, cam_idx, :, :]  # [B, C, H, W]
                cam_x = self.cnn_layers[cam_idx](cam_x)
                cam_x = cam_x.view(batch_size, self.output_channels, -1)
                camera_outputs.append(cam_x)

            if self.vision_mask is not None:
                if v_cmd is None:
                    raise ValueError("vision_mask is enabled but v_cmd was not provided")
                camera_outputs = self.vision_mask(camera_outputs, v_cmd)

            x = torch.cat(camera_outputs, dim=-1)  # [B, C_out, feat*num_cameras]
        else:
            x = x.view(batch_size, self.input_channels, self.input_height, self.input_width)
            x = self.cnn_layers(x)
            x = x.view(batch_size, self.output_channels, -1)

        if self.flatten_output:
            x = x.view(batch_size, -1)
        elif self.vision_tokens == "spatial":
            x = x.permute(0, 2, 1)  # [B, tokens, C_out]
        # "channel": keep [B, C_out, feat*num_cameras] (one token per CNN channel)

        return x


def build_mlp_layer(
    input_dim,
    hidden_dims,
    output_dim,
    layer_config,
):
    """Builds a multi-layer perceptron (MLP) layer.

    Parameters
    ----------
    input_dim: int
        Number of input dimensions
    hidden_dims: tuple[int, ...]
        Tuple of hidden dimensions
    output_dim: int
        Number of output dimensions
    layer_config: dict
        Dictionary containing:
        - activation: Activation function name (e.g., "ReLU")
        - dropout_prob: Dropout probability (default: 0)

    Returns
    -------
    nn.Sequential
        The constructed MLP layer
    """
    if hidden_dims is None:
        return None

    layers = []
    activation = getattr(nn, layer_config.activation)()
    dropout = layer_config.dropout_prob

    if len(hidden_dims) == 0:
        # No hidden layer, just one linear layer
        layers.append(nn.Linear(input_dim, output_dim))
    else:
        # First hidden layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(activation)
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

        # Additional hidden layers
        for layer_idx in range(len(hidden_dims)):
            if layer_idx == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[layer_idx], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[layer_idx], hidden_dims[layer_idx + 1]))
                layers.append(activation)
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))

    return nn.Sequential(*layers)


def build_cnn_layer(
    input_channels: int,
    input_height: int,
    input_width: int,
    hidden_channels: tuple[int, ...] | None,
    kernel_size: int | tuple[int, ...],
    stride: int | tuple[int, ...],
    padding: str | int | tuple[str | int, ...],
    layer_config: LayerConfig,
    flatten_output: bool = True,
):
    """Builds a convolutional neural network layer that works with flattened inputs.

    Parameters
    ----------
        input_channels: int
            Number of input channels
        input_height: int
            Height of input feature maps
        input_width: int
            Width of input feature maps
        hidden_channels: tuple[int, ...]
            Tuple of channel dimensions (last value becomes output channels)
        kernel_size: int or tuple[int, ...]
            Kernel size for convolutions (int or tuple for per-layer values)
        stride: int or tuple[int, ...]
            Stride for convolutions (int or tuple for per-layer values)
        padding: str | int | tuple[str | int, ...]
            Padding mode (int, "same", "valid", or tuple for per-layer values)
        layer_config: dict
            Dictionary containing:
            - activation: Activation function name (e.g., "ReLU")
            - dropout_prob: Dropout probability (default: 0)
            - use_layer_norm: Whether to use layer normalization (default: False)

    Returns
    -------
        CNNWrapper
            The constructed CNN layer wrapped to handle flattened inputs/outputs
    """
    if hidden_channels is None:
        return None
    assert len(hidden_channels) > 0, "hidden_channels must be a non-empty tuple"

    layers: list[nn.Module] = []
    activation = getattr(nn, layer_config.encoder_activation)()
    dropout = layer_config.dropout_prob
    use_layer_norm = layer_config.use_layer_norm

    num_layers = len(hidden_channels)
    # Convert single values to tuples if needed
    if isinstance(kernel_size, int):
        kernel_sizes = (kernel_size,) * num_layers
    else:
        kernel_sizes = kernel_size
        if len(kernel_sizes) != num_layers:
            raise ValueError(
                f"kernel_size tuple length ({len(kernel_sizes)}) must match number of layers ({num_layers})"
            )

    if isinstance(stride, int):
        strides = (stride,) * num_layers
    else:
        strides = stride
        if len(strides) != num_layers:
            raise ValueError(f"stride tuple length ({len(strides)}) must match number of layers ({num_layers})")

    if isinstance(padding, (str, int)):
        paddings = (padding,) * num_layers
    else:
        paddings = padding
        if len(paddings) != num_layers:
            raise ValueError(f"padding tuple length ({len(paddings)}) must match number of layers ({num_layers})")

    # Helper function to get padding value
    def get_padding_value(padding_spec, kernel_size_val):
        if padding_spec == "same":
            return kernel_size_val // 2
        if padding_spec == "valid":
            return 0
        return padding_spec

    # Build layers
    current_in_channels = input_channels
    for layer_idx in range(num_layers):
        current_out_channels = hidden_channels[layer_idx]
        current_kernel_size = kernel_sizes[layer_idx]
        current_stride = strides[layer_idx]
        current_padding = get_padding_value(paddings[layer_idx], current_kernel_size)

        # Add convolution layer
        layers.append(
            nn.Conv2d(
                current_in_channels,
                current_out_channels,
                kernel_size=current_kernel_size,
                stride=current_stride,
                padding=current_padding,
            )
        )

        # Add layer norm, activation and dropout for all layers except the last one
        if layer_idx < num_layers - 1:
            if use_layer_norm:
                layers.append(ImgChLayerNorm(current_out_channels))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout2d(p=dropout))

        current_in_channels = current_out_channels

    cnn_sequential = nn.Sequential(*layers)

    def _fresh_sequential() -> nn.Sequential:
        # Same architecture, new parameters (activation modules are stateless and shared).
        fresh = copy.deepcopy(cnn_sequential)
        for module in fresh.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        return fresh

    num_cameras = getattr(layer_config, "num_cameras", 1)
    vision_mask = None

    if num_cameras > 1:
        if getattr(layer_config, "independent_camera_init", False):
            extra_branches = [_fresh_sequential() for _ in range(num_cameras - 1)]
        else:
            extra_branches = [_clone_sequential(cnn_sequential) for _ in range(num_cameras - 1)]
        cnn_layers = nn.ModuleList([cnn_sequential] + extra_branches)
        if getattr(layer_config, "use_vision_mask", False):
            vision_mask = CameraVisionMask(
                num_cameras=num_cameras,
                k=getattr(layer_config, "vision_mask_k", 4.0),
                v_th=getattr(layer_config, "vision_mask_v_th", 0.2),
                hard_cutoff=getattr(layer_config, "vision_mask_hard_cutoff", 0.0),
            )
    else:
        cnn_layers = cnn_sequential

    return CNNWrapper(
        cnn_layers,
        input_channels,
        input_height,
        input_width,
        flatten_output,
        num_cameras=num_cameras,
        vision_mask=vision_mask,
        vision_tokens=getattr(layer_config, "vision_tokens", "spatial"),
    )


def _clone_sequential(seq: nn.Sequential) -> nn.Sequential:
    """Deep-copy an ``nn.Sequential`` so each camera gets independent weights."""
    return copy.deepcopy(seq)


class BaseModule(nn.Module):
    def __init__(self, obs_dim_dict, module_config_dict, history_length: dict[str, int]):
        super().__init__()
        self.obs_dim_dict = obs_dim_dict
        self.module_config_dict = module_config_dict
        self.history_length = history_length
        self._calculate_input_dim()
        self._calculate_output_dim()
        self._build_network_layer(self.module_config_dict)

    def _calculate_input_dim(self):
        # calculate input dimension and input slices
        self.input_dim = 0
        self.input_dim_dict = {}
        self.input_indices_dict = {}

        current_index = 0
        for each_input in self.module_config_dict.input_dim:
            if each_input in self.obs_dim_dict:
                # atomic observation type
                # Note: obs_dim_dict already includes history, so we don't multiply by history_length
                input_dim = self.obs_dim_dict[each_input]
                self.input_dim += input_dim
                self.input_dim_dict[each_input] = input_dim
                self.input_indices_dict[each_input] = slice(current_index, current_index + input_dim)
                current_index += input_dim

            elif isinstance(each_input, (int, float)):
                # direct numeric input
                input_dim = int(each_input)
                self.input_dim += input_dim
                self.input_dim_dict[each_input] = input_dim
                self.input_indices_dict[each_input] = slice(current_index, current_index + input_dim)
                current_index += input_dim

            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown input type: {each_input}")

    def _calculate_output_dim(self):
        # calculate output dimension based on the output specifications
        self.output_dim = 0
        for each_output in self.module_config_dict.output_dim:
            if isinstance(each_output, (int, float)):
                self.output_dim += each_output
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown output type: {each_output}")

    def _build_network_layer(self, module_config: ModuleConfig):
        layer_type = module_config.type
        layer_config = module_config.layer_config
        if layer_type == "MLP":
            self.module = build_mlp_layer(
                self.input_dim,
                layer_config.hidden_dims,
                self.output_dim,
                layer_config,
            )
        elif layer_type == "CNNEncoder":
            self.encoder = build_cnn_layer(
                layer_config.input_channels,
                layer_config.input_height,
                layer_config.input_width,
                layer_config.hidden_channels,
                layer_config.kernel_size,
                layer_config.stride,
                layer_config.padding,
                layer_config,
                flatten_output=True,
            )
            self.encoder_output_dim = self.encoder.output_size
            mlp_input_dim = sum(self.input_dim_dict[each_input] for each_input in layer_config.module_input_name)
            self.module = build_mlp_layer(
                mlp_input_dim + self.encoder_output_dim,
                layer_config.hidden_dims,
                self.output_dim,
                layer_config,
            )
        elif layer_type == "MLPEncoder":
            self.encoder_output_dim = (
                layer_config.encoder_output_dim
                if layer_config.encoder_hidden_dims is not None
                else self.input_dim_dict[layer_config.encoder_input_name]
            )
            self.encoder = build_mlp_layer(
                self.input_dim_dict[layer_config.encoder_input_name],
                layer_config.encoder_hidden_dims,
                self.encoder_output_dim,
                layer_config,
            )
            mlp_input_dim = sum(self.input_dim_dict[each_input] for each_input in layer_config.module_input_name)
            self.module = build_mlp_layer(
                mlp_input_dim + self.encoder_output_dim,
                layer_config.hidden_dims,
                self.output_dim,
                layer_config,
            )
        elif layer_type == "VisionTransformerEncoder":
            cnn_encoder = build_cnn_layer(
                layer_config.input_channels,
                layer_config.input_height,
                layer_config.input_width,
                layer_config.hidden_channels,
                layer_config.kernel_size,
                layer_config.stride,
                layer_config.padding,
                layer_config,
                flatten_output=False,
            )
            first_input = layer_config.module_input_name[0]
            obs_dim_for_vit = self.input_dim_dict[first_input]
            hl = self.history_length.get(first_input, 1)
            self.encoder = VisionTransformerEncoder(
                cnn_encoder,
                obs_dim_for_vit,
                hl,
                layer_config.embed_dim,
                layer_config.num_heads,
                layer_config,
            )
            self.vision_mask_input_name = (
                layer_config.vision_mask_input_name
                if layer_config.use_vision_mask and layer_config.vision_mask_input_name
                else ""
            )
            mlp_input_dim = sum(self.input_dim_dict[n] for n in layer_config.module_input_name)
            self.module = build_mlp_layer(
                mlp_input_dim + self.encoder.output_size,
                layer_config.hidden_dims,
                self.output_dim,
                layer_config,
            )
        else:
            raise NotImplementedError(f"Unsupported layer type: {layer_type}")

    def forward(self, policy_input):
        return self.module(policy_input)
