import sys
sys.path.append("./uni2ts/src")

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from pmdarima import auto_arima
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai.module import decode_distr_output

try:
    from safetensors.torch import load_file as load_safetensors_file
except ImportError:  # pragma: no cover - safetensors is expected but keep a fallback
    load_safetensors_file = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _extract_moirai_module_kwargs(raw_config):
    candidates = [
        raw_config,
        raw_config.get("module_kwargs") if isinstance(raw_config, dict) else None,
        raw_config.get("model_config") if isinstance(raw_config, dict) else None,
        raw_config.get("module_config") if isinstance(raw_config, dict) else None,
    ]
    required = {
        "distr_output",
        "d_model",
        "num_layers",
        "patch_sizes",
        "max_seq_len",
        "attn_dropout_p",
        "dropout_p",
    }

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if required.issubset(candidate.keys()):
            module_kwargs = dict(candidate)
            if isinstance(module_kwargs.get("distr_output"), dict):
                module_kwargs["distr_output"] = decode_distr_output(module_kwargs["distr_output"])
            if isinstance(module_kwargs.get("patch_sizes"), list):
                module_kwargs["patch_sizes"] = tuple(module_kwargs["patch_sizes"])
            return module_kwargs

    raise KeyError(f"Could not find MoiraiModule kwargs in config keys: {list(raw_config.keys()) if isinstance(raw_config, dict) else type(raw_config)}")


def _resolve_moirai_source(size):
    local_path = PROJECT_ROOT / f"moirai-1.1-R-{size}"
    if local_path.exists():
        print(f"[Moirai][LOAD] using local model path: {local_path}")
        return str(local_path)
    return f"Salesforce/moirai-1.1-R-{size}"


def _load_moirai_module(repo_or_path):
    repo_or_path = str(repo_or_path)
    try:
        return MoiraiModule.from_pretrained(repo_or_path)
    except TypeError as exc:
        # Newer hub/code combinations sometimes fail to hydrate init kwargs for MoiraiModule.
        msg = str(exc)
        required_markers = [
            "distr_output",
            "d_model",
            "num_layers",
            "patch_sizes",
            "max_seq_len",
            "attn_dropout_p",
            "dropout_p",
        ]
        if not all(marker in msg for marker in required_markers):
            raise

        print(f"[Moirai][LOAD] direct from_pretrained failed for {repo_or_path}; trying manual config+weights fallback")

        if Path(repo_or_path).exists():
            config_path = Path(repo_or_path) / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Missing config.json under local Moirai path: {repo_or_path}")

            with open(config_path, "r", encoding="utf-8") as f:
                raw_config = json.load(f)
            module_kwargs = _extract_moirai_module_kwargs(raw_config)
            module = MoiraiModule(**module_kwargs)

            safetensor_path = Path(repo_or_path) / "model.safetensors"
            pytorch_bin_path = Path(repo_or_path) / "pytorch_model.bin"

            if safetensor_path.exists():
                if load_safetensors_file is None:
                    raise ImportError("safetensors is not available but model.safetensors is present")
                state_dict = load_safetensors_file(str(safetensor_path), device="cpu")
            elif pytorch_bin_path.exists():
                state_dict = torch.load(str(pytorch_bin_path), map_location="cpu")
            else:
                raise FileNotFoundError(
                    f"Neither model.safetensors nor pytorch_model.bin found under {repo_or_path}"
                )

            module.load_state_dict(state_dict, strict=True)
            return module

        config_path = hf_hub_download(repo_or_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)
        module_kwargs = _extract_moirai_module_kwargs(raw_config)
        module = MoiraiModule(**module_kwargs)

        state_dict = None
        safetensor_error = None
        try:
            weight_path = hf_hub_download(repo_or_path, "model.safetensors")
            if load_safetensors_file is None:
                raise ImportError("safetensors is not available")
            state_dict = load_safetensors_file(weight_path, device="cpu")
        except Exception as inner_exc:
            safetensor_error = inner_exc

        if state_dict is None:
            try:
                weight_path = hf_hub_download(repo_or_path, "pytorch_model.bin")
                state_dict = torch.load(weight_path, map_location="cpu")
            except Exception:
                raise RuntimeError(
                    f"Failed to load Moirai weights for {repo_or_path} via both safetensors and pytorch_model.bin"
                ) from safetensor_error

        module.load_state_dict(state_dict, strict=True)
        return module


def _avg_history_np(history, k):
    window = history[-min(len(history), max(1, int(k))):]
    return sum(window) / float(len(window))


def _flatten_grad_list_to_np(global_gradient):
    return torch.cat([g.reshape(-1).detach().cpu() for g in global_gradient]).numpy().astype(np.float32)


def _log_fallback(tag, round_num, mode, reason, extra=""):
    suffix = f" {extra}" if extra else ""
    print(f"[{tag}][FALLBACK] round={round_num} mode={mode} reason={reason}{suffix}")


def _is_constant_series(series, atol=1e-12):
    arr = np.asarray(series, dtype=np.float32)
    if arr.size <= 1:
        return True
    return bool(np.all(np.abs(arr - arr[0]) <= float(atol)))


def _constant_series_model(series):
    arr = np.asarray(series, dtype=np.float32)
    if arr.size == 0:
        return ("constant", 0.0)
    return ("constant", float(np.mean(arr)))


def zeros_hessian_provider(model, device):
    return [torch.zeros_like(p, device=device) for p in model.parameters()]


@torch.no_grad()
def frobenius_scaled_identity_provider(model, device, scale=1e-3):
    return [scale * torch.ones_like(p, device=device) for p in model.parameters()]


def hutchinson_hvp_provider(model, device, loss_fn, batch):
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model, batch)
    grad1 = torch.autograd.grad(loss, list(model.parameters()), create_graph=True, retain_graph=True)
    v = [torch.sign(torch.randn_like(g)) for g in grad1]
    dot = sum((g * vv).sum() for g, vv in zip(grad1, v))
    hvp = torch.autograd.grad(dot, list(model.parameters()), retain_graph=False, create_graph=False)
    return [h for h in hvp]

# Attacking Method 1: Random
class RandomAttack:
    def __init__(self):
        self.global_gradient_history = []

    def get_fake_gradient(self, round_num, device, model):
        _log_fallback("RANDOM", round_num, "random", "always_random")
        return [torch.randn_like(param, device=device) for param in model.parameters()]

    def record_global_gradient(self, global_gradient):
        pass

# Attacking Method 2: AVG
class GradientReplayAttack:
    def __init__(self, k, start_attack_round):
        self.k = int(k)
        self.start_attack_round = int(start_attack_round)
        self.global_gradient_history = []

    def get_fake_gradient(self, round_num, device, model):
        if len(self.global_gradient_history) == 0:
            _log_fallback("REPLAY", round_num, "random", "no_history")
            return [torch.randn_like(param, device=device) for param in model.parameters()]
        used = self.global_gradient_history[-min(len(self.global_gradient_history), self.k):]
        denom = float(len(used))
        _log_fallback("REPLAY", round_num, "avg", "history_average", extra=f"used={len(used)}")
        return [sum(t[i] for t in used) / denom for i in range(len(self.global_gradient_history[0]))]

    def record_global_gradient(self, global_gradient):
        self.global_gradient_history.append([g.detach().clone().to(g.device) for g in global_gradient])

# Attacking Method 3: Offline ARIMA
class OfflineARIMAAttack:
    def __init__(self, k, random_round, model_params, var_window=3, var_threshold=5e-3,
                 lambda_h=0.9, hessian_provider=None):
        self.k = int(k)
        self.random_round = int(random_round)
        self.var_window = int(var_window)
        self.var_threshold = float(var_threshold)
        self.global_gradient_history = []

        self.param_shapes = [p.shape for p in model_params]
        self.param_sizes = [p.numel() for p in model_params]
        self.param_indices = self._build_param_indices()

        self.lambda_h = float(lambda_h)
        self.hessian_provider = hessian_provider

        self.arima_started = False
        self.arima_start_round = None
        self.offline_model_trained = False
        self.offline_model = {}

    def _build_param_indices(self):
        idx = []
        s = 0
        for size in self.param_sizes:
            e = s + size
            idx.append((s, e))
            s = e
        return idx

    def record_gradient_history(self, flat_gradient):
        self.global_gradient_history.append(flat_gradient)

    def record_global_gradient(self, global_gradient):
        self.record_gradient_history(_flatten_grad_list_to_np(global_gradient))

    def _avg_fallback(self, device, model):
        avg_vec = _avg_history_np(self.global_gradient_history, self.k)
        avg_vec = torch.as_tensor(avg_vec, device=device)
        fake_gradient, start = [], 0
        for p in model.parameters():
            n = p.numel()
            fake_gradient.append(avg_vec[start:start + n].view_as(p).to(dtype=p.dtype))
            start += n
        return fake_gradient

    def _hessian_fuse(self, predicted_gradient, model, device):
        provider = self.hessian_provider or (lambda m, d: [torch.zeros_like(p, device=d) for p in m.parameters()])
        h_term = provider(model, device)
        return [self.lambda_h * pg + (1.0 - self.lambda_h) * hh for pg, hh in zip(predicted_gradient, h_term)]

    def check_arima_ready(self):
        if len(self.global_gradient_history) < self.k + self.var_window:
            return False
        var_list = []
        start = 0
        for size, shape in zip(self.param_sizes, self.param_shapes):
            series = []
            for round_grad in self.global_gradient_history[-(self.k + self.var_window):]:
                param_grad = round_grad[start:start + size].reshape(shape)
                series.append(param_grad)
            start += size
            window_vars = []
            for i in range(self.var_window):
                window = series[i:i + self.k]
                window_vars.append(np.var(np.stack(window)))
            var_list.append(window_vars)
        var_diff = np.abs(np.diff(np.mean(var_list, axis=0)))
        print(f"[ARIMA][OFFLINE] var diff: {var_diff}")
        return bool(np.all(var_diff < self.var_threshold))

    def train_offline_arima_model(self):
        print("[ARIMA][OFFLINE] Starting offline auto_arima training...")
        self.offline_model = {}
        constant_dims = 0
        for param_index, size in enumerate(self.param_sizes):
            st, _ = self.param_indices[param_index]
            param_series_list = []
            for i in range(size):
                series = np.array([g[st + i] for g in self.global_gradient_history], dtype=np.float32)
                if _is_constant_series(series):
                    param_series_list.append(_constant_series_model(series))
                    constant_dims += 1
                    continue
                try:
                    m = auto_arima(
                        series,
                        start_p=0, start_q=0, max_p=2, max_q=2,
                        d=None, max_d=1, seasonal=False, stationary=False,
                        error_action='ignore', suppress_warnings=True, stepwise=True, maxiter=20
                    )
                    param_series_list.append(m)
                except Exception:
                    param_series_list.append(None)
            self.offline_model[param_index] = param_series_list
        self.offline_model_trained = True
        if constant_dims > 0:
            print(f"[ARIMA][OFFLINE] constant series detected in {constant_dims} dims -> using mean-value fallback")
        print("[ARIMA][OFFLINE] Offline model training complete.")

    def get_fake_gradient(self, round_num, device, model):
        if len(self.global_gradient_history) == 0:
            _log_fallback("ARIMA][OFFLINE", round_num, "random", "no_history")
            return [torch.randn_like(p, device=device) for p in model.parameters()]

        if (not self.arima_started) and self.check_arima_ready():
            self.arima_started = True
            self.arima_start_round = int(round_num)
            self.train_offline_arima_model()
            print(f"[ARIMA][OFFLINE] round={round_num} SWITCHED TO AUTO-ARIMA")

        if self.arima_started and self.offline_model_trained:
            predicted_gradient = []
            for param_index, param in enumerate(model.parameters()):
                predicted_param = torch.zeros_like(param, device=device)
                model_list = self.offline_model.get(param_index, [])
                for i in range(param.numel()):
                    m = model_list[i] if i < len(model_list) else None
                    if isinstance(m, tuple) and len(m) == 2 and m[0] == "constant":
                        predicted_param.view(-1)[i] = torch.as_tensor(m[1], device=device, dtype=param.dtype)
                    elif m is not None:
                        try:
                            forecast = m.predict(n_periods=1)
                            predicted_param.view(-1)[i] = torch.as_tensor(forecast[0], device=device, dtype=param.dtype)
                        except Exception:
                            predicted_param.view(-1)[i] = torch.zeros((), device=device, dtype=param.dtype)
                predicted_gradient.append(predicted_param)
            return self._hessian_fuse(predicted_gradient, model, device)

        if round_num < self.random_round:
            _log_fallback("ARIMA][OFFLINE", round_num, "avg", "warmup", extra=f"used={min(len(self.global_gradient_history), self.k)}")
            return self._avg_fallback(device, model)

        _log_fallback("ARIMA][OFFLINE", round_num, "avg", "not_ready", extra=f"used={min(len(self.global_gradient_history), self.k)}")
        return self._avg_fallback(device, model)

# Attacking Method 4: Adaptive ARIMA
class AdaptiveARIMAAttack:
    def __init__(self, k, random_round, model_params,
                 var_window=3, var_threshold=5e-3,
                 lambda_h=0.9, hessian_provider=None,
                 err_ema_beta=0.9, err_threshold=5e-3,
                 min_rounds_after_start=2, retrain_interval=10,
                 trim_history=True):
        self.k = int(k)
        self.random_round = int(random_round)
        self.var_window = int(var_window)
        self.var_threshold = float(var_threshold)
        self.global_gradient_history = []

        self.param_shapes = [p.shape for p in model_params]
        self.param_sizes = [p.numel() for p in model_params]
        self.param_indices = self._build_param_indices()

        self.lambda_h = float(lambda_h)
        self.hessian_provider = hessian_provider

        self.arima_started = False
        self.arima_start_round = None
        self.offline_model_trained = False
        self.offline_model = {}

        self.err_ema_beta = float(err_ema_beta)
        self.err_threshold = float(err_threshold)
        self.min_rounds_after_start = int(min_rounds_after_start)
        self.retrain_interval = int(retrain_interval)
        self.trim_history = bool(trim_history)

        self._err_ema = None
        self._since_start = 0
        self._last_pred_flat = None
        self._last_pred_round = None

    def _build_param_indices(self):
        idx = []
        s = 0
        for size in self.param_sizes:
            e = s + size
            idx.append((s, e))
            s = e
        return idx

    def _avg_fallback(self, device, model):
        if len(self.global_gradient_history) == 0:
            return [torch.randn_like(p, device=device) for p in model.parameters()]
        avg_vec = _avg_history_np(self.global_gradient_history, self.k)
        avg_vec = torch.as_tensor(avg_vec, device=device)
        fake_gradient, start = [], 0
        for p in model.parameters():
            n = p.numel()
            fake_gradient.append(avg_vec[start:start + n].view_as(p).to(dtype=p.dtype))
            start += n
        return fake_gradient

    def _hessian_fuse(self, predicted_gradient, model, device):
        provider = self.hessian_provider or (lambda m, d: [torch.zeros_like(p, device=d) for p in m.parameters()])
        h_term = provider(model, device)
        return [self.lambda_h * pg + (1.0 - self.lambda_h) * hh for pg, hh in zip(predicted_gradient, h_term)]

    def check_arima_ready(self):
        if len(self.global_gradient_history) < self.k + self.var_window:
            return False
        var_list = []
        start = 0
        for size, shape in zip(self.param_sizes, self.param_shapes):
            series = []
            for round_grad in self.global_gradient_history[-(self.k + self.var_window):]:
                param_grad = round_grad[start:start + size].reshape(shape)
                series.append(param_grad)
            start += size
            window_vars = []
            for i in range(self.var_window):
                window = series[i:i + self.k]
                window_vars.append(np.var(np.stack(window)))
            var_list.append(window_vars)
        var_diff = np.abs(np.diff(np.mean(var_list, axis=0)))
        print(f"[ARIMA][ADAPT] var diff: {var_diff}")
        return bool(np.all(var_diff < self.var_threshold))

    def train_offline_arima_model(self):
        print("[ARIMA][ADAPT] Starting offline auto_arima training...")
        self.offline_model = {}
        constant_dims = 0
        for param_index, size in enumerate(self.param_sizes):
            st, _ = self.param_indices[param_index]
            param_series_list = []
            for i in range(size):
                series = np.array([g[st + i] for g in self.global_gradient_history], dtype=np.float32)
                if _is_constant_series(series):
                    param_series_list.append(_constant_series_model(series))
                    constant_dims += 1
                    continue
                try:
                    m = auto_arima(
                        series,
                        start_p=0, start_q=0, max_p=2, max_q=2,
                        d=None, max_d=1, seasonal=False, stationary=False,
                        error_action='ignore', suppress_warnings=True, stepwise=True, maxiter=20
                    )
                    param_series_list.append(m)
                except Exception:
                    param_series_list.append(None)
            self.offline_model[param_index] = param_series_list
        self.offline_model_trained = True
        if constant_dims > 0:
            print(f"[ARIMA][ADAPT] constant series detected in {constant_dims} dims -> using mean-value fallback")
        print("[ARIMA][ADAPT] Offline model training complete.")

    def _reset_arima_state(self):
        self.arima_started = False
        self.arima_start_round = None
        self.offline_model_trained = False
        self.offline_model = {}
        self._last_pred_flat = None
        self._last_pred_round = None
        self._err_ema = None
        self._since_start = 0
        if self.trim_history:
            keep = max(self.k + self.var_window, self.k)
            if len(self.global_gradient_history) > keep:
                self.global_gradient_history = self.global_gradient_history[-keep:]
        print("[ARIMA][ADAPT] RESET completed.")

    def record_gradient_history(self, flat_gradient):
        self.global_gradient_history.append(flat_gradient)

    def record_global_gradient(self, global_gradient):
        flat = _flatten_grad_list_to_np(global_gradient)
        self.record_gradient_history(flat)

        if not (self.arima_started and self.offline_model_trained):
            return
        if self._last_pred_flat is None or self._last_pred_round is None:
            return

        self._since_start += 1
        mae = float(np.mean(np.abs(flat - self._last_pred_flat)))
        if self._err_ema is None:
            self._err_ema = mae
        else:
            b = self.err_ema_beta
            self._err_ema = b * self._err_ema + (1.0 - b) * mae

        print(f"[ARIMA][ADAPT] pred_round={self._last_pred_round} mae={mae:.6e} ema={self._err_ema:.6e}")
        if self._since_start >= self.min_rounds_after_start and self._err_ema > self.err_threshold:
            print(f"[ARIMA][ADAPT] ema_err>{self.err_threshold:.2e} -> RESET")
            self._reset_arima_state()

    def get_fake_gradient(self, round_num, device, model):
        if len(self.global_gradient_history) == 0:
            _log_fallback("ARIMA][ADAPT", round_num, "random", "no_history")
            return [torch.randn_like(p, device=device) for p in model.parameters()]

        if (not self.arima_started) and self.check_arima_ready():
            self.arima_started = True
            self.arima_start_round = int(round_num)
            self.train_offline_arima_model()
            print(f"[ARIMA][ADAPT] round={round_num} SWITCHED TO ADAPTIVE")

        if self.arima_started and self.offline_model_trained and self.retrain_interval > 0:
            base = self.arima_start_round or 0
            if (round_num - base) > 0 and ((round_num - base) % self.retrain_interval == 0):
                print(f"[ARIMA][ADAPT] round={round_num} periodic retrain")
                self.train_offline_arima_model()

        if self.arima_started and self.offline_model_trained:
            predicted_gradient = []
            for param_index, param in enumerate(model.parameters()):
                predicted_param = torch.zeros_like(param, device=device)
                model_list = self.offline_model.get(param_index, [])
                for i in range(param.numel()):
                    m = model_list[i] if i < len(model_list) else None
                    if isinstance(m, tuple) and len(m) == 2 and m[0] == "constant":
                        predicted_param.view(-1)[i] = torch.as_tensor(m[1], device=device, dtype=param.dtype)
                    elif m is not None:
                        try:
                            forecast = m.predict(n_periods=1)
                            predicted_param.view(-1)[i] = torch.as_tensor(forecast[0], device=device, dtype=param.dtype)
                        except Exception:
                            predicted_param.view(-1)[i] = torch.zeros((), device=device, dtype=param.dtype)
                predicted_gradient.append(predicted_param)

            self._last_pred_round = int(round_num)
            self._last_pred_flat = torch.cat([g.reshape(-1).detach().cpu() for g in predicted_gradient]).numpy().astype(np.float32)
            return self._hessian_fuse(predicted_gradient, model, device)

        if round_num < self.random_round:
            _log_fallback("ARIMA][ADAPT", round_num, "avg", "warmup", extra=f"used={min(len(self.global_gradient_history), self.k)}")
            return self._avg_fallback(device, model)

        _log_fallback("ARIMA][ADAPT", round_num, "avg", "not_ready", extra=f"used={min(len(self.global_gradient_history), self.k)}")
        return self._avg_fallback(device, model)

# Attacking Method 5: Offline MOIRAI
class OfflineMoiraiAttack:
    def __init__(self, k, random_round, history_length=5, model_type="moirai",
                 size="small", patch_size="auto",
                 lambda_h=0.9, hessian_provider=None):
        self.history_length = int(history_length)
        self.k = int(k)
        self.model_type = model_type
        self.size = size
        self.patch_size = patch_size
        self.random_round = int(random_round)
        self.global_gradient_history = []

        self.lambda_h = float(lambda_h)
        self.hessian_provider = hessian_provider

        self.param_shapes = None
        self.param_sizes = None
        self.D = None
        self.model = None
        self._predictor = None

        assert self.model_type == "moirai", "Only 'moirai' type is supported."

    def record_gradient_history(self, flat_gradient):
        self.global_gradient_history.append(flat_gradient)

    def record_global_gradient(self, global_gradient):
        self.record_gradient_history(_flatten_grad_list_to_np(global_gradient))

    def _lazy_init(self, model):
        if self.D is not None:
            return
        self.param_shapes = [p.shape for p in model.parameters()]
        self.param_sizes = [p.numel() for p in model.parameters()]
        self.D = int(sum(self.param_sizes))
        self.model = MoiraiForecast(
            module=_load_moirai_module(_resolve_moirai_source(self.size)),
            prediction_length=1,
            context_length=self.history_length,
            patch_size=self.patch_size,
            num_samples=1,
            target_dim=self.D,                                                                                                                                                                                                                                                             
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        self._predictor = self.model.create_predictor(batch_size=1)

    def _avg_fallback(self, device, model):
        if len(self.global_gradient_history) == 0:
            return [torch.randn_like(p, device=device) for p in model.parameters()]
        avg_vec = _avg_history_np(self.global_gradient_history, self.k)
        avg_vec = torch.as_tensor(avg_vec, device=device)
        fake_gradient, start = [], 0
        for p in model.parameters():
            n = p.numel()
            fake_gradient.append(avg_vec[start:start + n].view_as(p).to(dtype=p.dtype))
            start += n
        return fake_gradient

    def _hessian_fuse(self, predicted_gradient, model, device):
        provider = self.hessian_provider or (lambda m, d: [torch.zeros_like(pp, device=d) for pp in m.parameters()])
        h_term = provider(model, device)
        return [self.lambda_h * pg + (1.0 - self.lambda_h) * hh for pg, hh in zip(predicted_gradient, h_term)]

    def get_fake_gradient(self, round_num, device, model):
        self._lazy_init(model)

        if len(self.global_gradient_history) == 0:
            _log_fallback("Moirai][OFFLINE", round_num, "random", "no_history")
            return [torch.randn_like(p, device=device) for p in model.parameters()]

        if len(self.global_gradient_history) < self.history_length or round_num < self.random_round:
            _log_fallback("Moirai][OFFLINE", round_num, "avg", "warmup_or_short_history", extra=f"used={min(len(self.global_gradient_history), self.k)}")
            return self._avg_fallback(device, model)

        print(f"[Moirai][OFFLINE] round={round_num} USING MOIRAI PREDICTION")
        gradient_series = np.stack(self.global_gradient_history[-self.history_length:], axis=0)
        input_data = [{
            "start": pd.Period("2000-01-01", freq="D"),
            "target": gradient_series.T,
        }]
        forecast = next(self._predictor.predict(input_data))
        pred_vec = torch.as_tensor(forecast.samples[0, 0], device=device)

        fake_gradient, start = [], 0
        for p in model.parameters():
            n = p.numel()
            fake_gradient.append(pred_vec[start:start + n].view_as(p).to(dtype=p.dtype))
            start += n
        return self._hessian_fuse(fake_gradient, model, device)

# Attacking Method 6: Online MOIRAI
class OnlineMoiraiAttack:
    def __init__(self, k, random_round, history_length=5, model_type="moirai",
                 size="small", patch_size="auto",
                 lambda_h=0.9, hessian_provider=None,
                 online_tuning=True, tune_lr=1e-3, tune_steps=1, tune_start_round=None,
                 residual_clip=5.0, adapt_interval=10, residual_decay=0.9,
                 err_ema_beta=0.9, err_threshold=5e-3, min_rounds_after_start=2,
                 trim_history=True):
        self.history_length = int(history_length)
        self.k = int(k)
        self.model_type = model_type
        self.size = size
        self.patch_size = patch_size
        self.random_round = int(random_round)
        self.global_gradient_history = []
        self.lambda_h = float(lambda_h)
        self.hessian_provider = hessian_provider

        self.param_shapes = None
        self.param_sizes = None
        self.D = None
        self.model = None
        self._predictor = None

        self.online_tuning = bool(online_tuning)
        self.tune_lr = float(tune_lr)
        self.tune_steps = int(max(1, tune_steps))
        self.tune_start_round = self.random_round if tune_start_round is None else int(tune_start_round)
        self.residual_clip = residual_clip
        self.adapt_interval = int(adapt_interval)
        self.residual_decay = float(residual_decay)
        self.trim_history = bool(trim_history)

        self.err_ema_beta = float(err_ema_beta)
        self.err_threshold = float(err_threshold)
        self.min_rounds_after_start = int(min_rounds_after_start)
        self._err_ema = None
        self._since_start = 0

        self.residual = None
        self._last_pred_round = None
        self._last_pred_flat = None

        assert self.model_type == "moirai", "Only 'moirai' type is supported."

    def _lazy_init(self, model):
        if self.D is not None:
            return
        self.param_shapes = [p.shape for p in model.parameters()]
        self.param_sizes = [p.numel() for p in model.parameters()]
        self.D = int(sum(self.param_sizes))
        self.model = MoiraiForecast(
            module=_load_moirai_module(_resolve_moirai_source(self.size)),
            prediction_length=1,
            context_length=self.history_length,
            patch_size=self.patch_size,
            num_samples=1,
            target_dim=self.D,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        self._predictor = self.model.create_predictor(batch_size=1)
        if self.residual is None:
            self.residual = np.zeros((self.D,), dtype=np.float32)

    def _lazy_init_from_global_grad(self, global_gradient):
        if self.D is not None:
            return
        self.param_shapes = [g.shape for g in global_gradient]
        self.param_sizes = [g.numel() for g in global_gradient]
        self.D = int(sum(self.param_sizes))
        self.model = MoiraiForecast(
            module=_load_moirai_module(_resolve_moirai_source(self.size)),
            prediction_length=1,
            context_length=self.history_length,
            patch_size=self.patch_size,
            num_samples=1,
            target_dim=self.D,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        self._predictor = self.model.create_predictor(batch_size=1)
        if self.residual is None:
            self.residual = np.zeros((self.D,), dtype=np.float32)

    def _avg_fallback(self, device, model):
        if len(self.global_gradient_history) == 0:
            return [torch.randn_like(p, device=device) for p in model.parameters()]
        avg_vec = _avg_history_np(self.global_gradient_history, self.k)
        avg_vec = torch.as_tensor(avg_vec, device=device)
        fake_gradient, start = [], 0
        for p in model.parameters():
            n = p.numel()
            fake_gradient.append(avg_vec[start:start + n].view_as(p).to(dtype=p.dtype))
            start += n
        return fake_gradient

    def _hessian_fuse(self, predicted_gradient, model, device):
        provider = self.hessian_provider or (lambda m, d: [torch.zeros_like(pp, device=d) for pp in m.parameters()])
        h_term = provider(model, device)
        return [self.lambda_h * pg + (1.0 - self.lambda_h) * hh for pg, hh in zip(predicted_gradient, h_term)]

    def _soft_adjust(self):
        if self.residual is not None:
            self.residual *= self.residual_decay

    def _hard_reset(self):
        if self.residual is not None:
            self.residual.fill(0.0)
        self._err_ema = None
        self._since_start = 0
        self._last_pred_round = None
        self._last_pred_flat = None
        if self.trim_history:
            keep = max(self.k + self.history_length, self.k)
            if len(self.global_gradient_history) > keep:
                self.global_gradient_history = self.global_gradient_history[-keep:]
        print("[Moirai][OnlineTune] RESET completed.")

    def record_gradient_history(self, flat_gradient):
        self.global_gradient_history.append(flat_gradient)

    def record_global_gradient(self, global_gradient):
        flat = _flatten_grad_list_to_np(global_gradient)
        self.record_gradient_history(flat)

        if self.model is None:
            self._lazy_init_from_global_grad(global_gradient)

        cur_round = len(self.global_gradient_history) - 1
        if self.adapt_interval > 0 and cur_round > 0 and (cur_round % self.adapt_interval == 0):
            self._soft_adjust()

        if self._last_pred_flat is not None and self._last_pred_round is not None:
            self._since_start += 1
            mae = float(np.mean(np.abs(flat - self._last_pred_flat)))
            if self._err_ema is None:
                self._err_ema = mae
            else:
                b = self.err_ema_beta
                self._err_ema = b * self._err_ema + (1.0 - b) * mae
            print(f"[Moirai][OnlineTune] pred_round={self._last_pred_round} mae={mae:.6e} ema={self._err_ema:.6e}")
            if self._since_start >= self.min_rounds_after_start and self._err_ema > self.err_threshold:
                print(f"[Moirai][OnlineTune] ema_err>{self.err_threshold:.2e} -> RESET")
                self._hard_reset()

        if not self.online_tuning:
            return
        if cur_round < self.tune_start_round:
            return
        if len(self.global_gradient_history) < self.history_length + 1:
            return
        if self.residual is None:
            self.residual = np.zeros((self.D,), dtype=np.float32)

        target = self.global_gradient_history[-1]
        hist = np.stack(self.global_gradient_history[-(self.history_length + 1):-1], axis=0)
        input_data = [{
            "start": pd.Period("2000-01-01", freq="D"),
            "target": hist.T,
        }]
        try:
            forecast = next(self._predictor.predict(input_data))
            pred = forecast.samples[0, 0].astype(np.float32)
        except Exception as e:
            print(f"[Moirai][OnlineTune] skip residual update due to predict error: {e}")
            return

        for _ in range(self.tune_steps):
            err = (pred + self.residual) - target
            self.residual -= self.tune_lr * err
            if self.residual_clip is not None:
                np.clip(self.residual, -float(self.residual_clip), float(self.residual_clip), out=self.residual)

    def get_fake_gradient(self, round_num, device, model):
        self._lazy_init(model)

        if len(self.global_gradient_history) == 0:
            _log_fallback("Moirai][OnlineTune", round_num, "random", "no_history")
            return [torch.randn_like(p, device=device) for p in model.parameters()]

        if len(self.global_gradient_history) < self.history_length or round_num < self.random_round:
            _log_fallback("Moirai][OnlineTune", round_num, "avg", "warmup_or_short_history", extra=f"used={min(len(self.global_gradient_history), self.k)}")
            return self._avg_fallback(device, model)

        print(f"[Moirai][OnlineTune] round={round_num} USING MOIRAI PREDICTION")
        gradient_series = np.stack(self.global_gradient_history[-self.history_length:], axis=0)
        input_data = [{
            "start": pd.Period("2000-01-01", freq="D"),
            "target": gradient_series.T,
        }]
        forecast = next(self._predictor.predict(input_data))
        pred_np = forecast.samples[0, 0].astype(np.float32)
        if self.residual is not None:
            pred_np = pred_np + self.residual

        self._last_pred_round = int(round_num)
        self._last_pred_flat = pred_np.copy()

        pred_vec = torch.as_tensor(pred_np, device=device)
        fake_gradient, start = [], 0
        for p in model.parameters():
            n = p.numel()
            fake_gradient.append(pred_vec[start:start + n].view_as(p).to(dtype=p.dtype))
            start += n
        return self._hessian_fuse(fake_gradient, model, device)


# Backward compatibility for existing entry scripts.
ARIMAAttack = OfflineARIMAAttack
MoiraiAttack = OfflineMoiraiAttack
