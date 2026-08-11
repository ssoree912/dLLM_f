import logging
import os
import json
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union,Type,TypeVar

import jinja2
import torch
import torch.nn.functional as F
import transformers
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
    find_executable_batch_size,
)
from datasets import Dataset
from accelerate.utils import get_max_memory
from huggingface_hub import HfApi
from packaging import version
try:
    from peft import PeftModel
    from peft import __version__ as PEFT_VERSION
except ModuleNotFoundError:
    PeftModel = None
    PEFT_VERSION = "0"
from tqdm import tqdm
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,

)

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import configure_pad_token
from eval_model.lm_eval_compat import get_dtype

eval_logger = logging.getLogger(__name__)
from utils import  generate
from dllm_cache.cache import  dLLMCacheConfig,dLLMCache
from dllm_cache.hooks import  register_cache_LLaDA
from dataclasses import asdict
T = TypeVar("T", bound="LM")
from lm_eval.api.model import LM

@register_model("LLaDA")
class LLaDA(TemplateLM):
    AUTO_MODEL_CLASS = None
    _DEFAULT_MAX_LENGTH = 20480

    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        backend: Literal["default", "causal", "seq2seq"] = "causal",
        revision: Optional[str] = "main",
        subfolder: Optional[str] = None,
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ] = None,
        truncation: Optional[bool] = False,
        truncation_strategy: str = "left",
        logits_cache: bool = True,
        max_length: Optional[int] = None,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int]] = 1,
        max_batch_size: Optional[int] = 64,
        trust_remote_code: Optional[bool] = True,
        use_fast_tokenizer: Optional[bool] = True,
        add_bos_token: Optional[bool] = False,
        escape_until:Optional[bool] = False,
        prefix_token_id: Optional[int] = None,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        parallelize: Optional[bool] = False,
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[Union[str, os.PathLike]] = "./offload",
        # PEFT, delta weights and quantization options
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        gptqmodel: Optional[bool] = False,
        gguf_file: Optional[str] = None,
        is_feature_cache: bool = False,
        is_cfg_cache: bool = False,
        prompt_interval_steps: int = 1,
        gen_interval_steps: int = 1,
        cfg_interval_steps: int = 1,
        transfer_ratio:float = 0.0,
        mc_num: int = 1024,
        remasking: str = "low_confidence",
        mask_id: int = 126336,
        is_check_greedy : bool =True,
        student_path: Optional[str] = None,
        student_prompt_kv_cache: bool = False,
        student_prompt_prune: bool = False,
        student_prompt_pool_active: bool = False,
        student_prompt_dynamic_kv: bool = False,
        student_prompt_layer_split: bool = False,
        student_prompt_drift_refresh: bool = False,
        maskkv_student_scores: bool = False,
        student_drift_mode: str = "oracle",
        student_delta_select_once: bool = False,
        student_drift_ckpt: Optional[str] = None,
        student_drift_frozen_layers: int = 0,
        student_frozen_layers: int = 16,
        student_refresh_tokens: int = 0,
        student_refresh_path: Optional[str] = None,
        student_refresh_rotate: bool = False,
        student_measured_tokens: int = 0,
        student_full_refresh_interval: int = 0,
        student_measured_baseline: str = "refresh",
        student_random_scores: bool = False,
        student_selection_mode: str = "global",
        student_refresh_interval: int = 1,
        student_budget: int = 128,
        student_pool_budget: int = 512,
        student_min_pool_budget: int = 1,
        student_pool_budget_mode: str = "fixed",
        student_pool_budget_scale: float = 1.0,
        student_question_window: int = 128,
        student_score_activation: str = "softmax",
        **kwargs,
    ) -> None:
        super().__init__()
        self.mc_num = mc_num
        self.mask_id = mask_id
        self.remasking = remasking
        self.pretrained = pretrained
        self.prompt_interval_steps = prompt_interval_steps
        self.gen_interval_steps = gen_interval_steps
        self.cfg_interval_steps = cfg_interval_steps
        self.transfer_ratio = transfer_ratio
        self.is_check_greedy = is_check_greedy
        self.student_path = student_path
        self.student_prompt_kv_cache = self._coerce_bool(student_prompt_kv_cache)
        self.student_prompt_prune = self._coerce_bool(student_prompt_prune)
        self.student_prompt_pool_active = self._coerce_bool(student_prompt_pool_active)
        self.student_prompt_dynamic_kv = self._coerce_bool(student_prompt_dynamic_kv)
        self.student_prompt_layer_split = self._coerce_bool(student_prompt_layer_split)
        self.student_prompt_drift_refresh = self._coerce_bool(student_prompt_drift_refresh)
        # MaskKV's budget split with our ranking: needs the student loaded but no
        # student_prompt_* generation mode.
        self.maskkv_student_scores = self._coerce_bool(maskkv_student_scores)
        self.student_drift_mode = str(student_drift_mode)
        # delta_student mode only: freeze the refresh set at the first selection
        # instead of re-ranking every step.
        self.student_delta_select_once = self._coerce_bool(student_delta_select_once)
        if self.student_drift_mode not in {"oracle", "student", "random", "delta_student"}:
            raise RuntimeError(
                "student_drift_mode must be 'oracle', 'student', 'random' or 'delta_student'"
            )
        self.student_drift_ckpt = student_drift_ckpt
        # Drift says the shallow layers barely move, so refreshing them is wasted work.
        self.student_drift_frozen_layers = int(student_drift_frozen_layers)
        if self.student_drift_frozen_layers < 0:
            raise RuntimeError("student_drift_frozen_layers must be non-negative")
        self.drift_refresh_student = None
        self.student_frozen_layers = int(student_frozen_layers)
        if self.student_frozen_layers < 0:
            raise RuntimeError("student_frozen_layers must be non-negative")
        self.student_refresh_tokens = int(student_refresh_tokens)
        if self.student_refresh_tokens < 0:
            raise RuntimeError("student_refresh_tokens must be non-negative")
        # Keeping and refreshing are different questions, so they can be ranked by
        # different students: importance decides what survives, drift decides what
        # stays fresh.
        self.student_refresh_path = student_refresh_path
        self.refresh_student = None
        # Rotate the refresh set instead of freezing the complement forever.
        self.student_refresh_rotate = self._coerce_bool(student_refresh_rotate)
        # Measure movement at runtime instead of scheduling it ahead of time.
        self.student_measured_tokens = int(student_measured_tokens)
        if self.student_measured_tokens < 0:
            raise RuntimeError("student_measured_tokens must be non-negative")
        # Periodic full recompute on top of the per-step top-K, the way dLLM-Cache
        # pairs partial updates with a full refresh every `refresh_interval` steps.
        self.student_full_refresh_interval = int(student_full_refresh_interval)
        if self.student_full_refresh_interval < 0:
            raise RuntimeError("student_full_refresh_interval must be non-negative")
        # "refresh" ranks by error accumulated since a position was last written;
        # "step" ranks by how fast it is moving right now.
        self.student_measured_baseline = str(student_measured_baseline).strip().lower()
        if self.student_measured_baseline not in {"refresh", "step"}:
            raise RuntimeError("student_measured_baseline must be 'refresh' or 'step'")
        # Control: keep the same budget but choose the kept positions at random, to
        # separate "this selector is wrong" from "dropping tokens is wrong".
        self.student_random_scores = self._coerce_bool(student_random_scores)
        self.student_selection_mode = str(student_selection_mode).strip().lower()
        if self.student_selection_mode not in {"layer_union", "global"}:
            raise RuntimeError("student_selection_mode must be one of: layer_union, global")
        self.student_refresh_interval = int(student_refresh_interval)
        if self.student_refresh_interval <= 0:
            raise RuntimeError("student_refresh_interval must be positive")
        self.student_budget = int(student_budget)
        self.student_pool_budget = int(student_pool_budget)
        self.student_min_pool_budget = int(student_min_pool_budget)
        self.student_pool_budget_mode = str(student_pool_budget_mode).strip().lower()
        if self.student_pool_budget_mode not in {"fixed", "predicted_mass"}:
            raise RuntimeError("student_pool_budget_mode must be one of: fixed, predicted_mass")
        self.student_pool_budget_scale = float(student_pool_budget_scale)
        self.student_question_window = int(student_question_window)
        self.student_score_activation = str(student_score_activation).strip().lower()
        if self.student_score_activation not in {"softmax", "sigmoid", "raw"}:
            raise RuntimeError("student_score_activation must be one of: softmax, sigmoid, raw")
        self.student = None
        self.add_bos_token = add_bos_token
        self.escape_until = escape_until
        active_student_modes = sum(
            int(enabled)
            for enabled in (
                self.student_prompt_kv_cache,
                self.student_prompt_prune,
                self.student_prompt_pool_active,
                self.student_prompt_dynamic_kv,
                self.student_prompt_layer_split,
                self.student_prompt_drift_refresh,
            )
        )
        if active_student_modes > 1:
            raise RuntimeError("student prompt compression modes are mutually exclusive")
        if not isinstance(pretrained, str):
            eval_logger.warning(
                "`pretrained` model kwarg is not of type `str`. Many other model arguments may be ignored. Please do not launch via accelerate or use `parallelize=True` if passing an existing model this way."
            )
            assert not parallelize, (
                "`parallelize=True` is not compatible with passing pre-initialized model to `pretrained`"
            )
            self._model = pretrained
            self._device = self._model.device
            self._config = self._model.config
            gpus = 0

        else:
            assert isinstance(device, str)
            assert isinstance(pretrained, str)
            assert isinstance(batch_size, (int, str))

            gpus = torch.cuda.device_count()
            accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
            accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
            if accelerator.num_processes > 1:
                self.accelerator = accelerator

            if "npu" in accelerator.device.type:
                gpus = torch.npu.device_count()

            # using one process with no model parallelism
            if not (parallelize or accelerator.num_processes > 1):
                # use user-passed device
                device_list = set(
                    ["cuda", "cpu"]
                    + [f"cuda:{i}" for i in range(gpus)]
                    + ["mps", "mps:0"]
                    + [f"npu:{i}" for i in range(gpus)]
                )
                if device and device in device_list:
                    self._device = torch.device(device)
                    eval_logger.info(f"Using device '{device}'")
                    if device in ("mps", "mps:0") and version.parse(
                        torch.__version__
                    ) < version.parse("2.1"):
                        raise RuntimeError(
                            f"mps requires torch >= 2.1. You have {torch.__version__}"
                        )
                else:
                    eval_logger.info("Device not specified")
                    eval_logger.info(f"Cuda Available? {torch.cuda.is_available()}")
                    self._device = (
                        torch.device("cuda")
                        if torch.cuda.is_available()
                        else torch.device("cpu")
                    )
            else:  # Parallelism managed by accelerate
                if device != "cuda":
                    eval_logger.info(
                        f"Using `accelerate launch` or `parallelize=True`, device '{device}' will be overridden when placing model."
                    )
                # TODO: include in warning that `load_in_8bit` etc. affect this too
                self._device = (
                    self.accelerator.device
                    if hasattr(self, "accelerator")
                    else torch.device(device)
                )

            revision = str(revision)  # cast to string if not already one
            # TODO: update this to be less of a hack once subfolder is fixed in HF
            revision = revision + ("/" + subfolder if subfolder is not None else "")

            self._get_config(
                pretrained,
                revision=revision,
                trust_remote_code=trust_remote_code,
                gguf_file=gguf_file,
            )
            # determine which of 'causal' and 'seq2seq' backends to use for HF models
        self._get_backend(
            config=self.config, backend=backend, trust_remote_code=trust_remote_code
        )
        self._create_tokenizer(
            pretrained,
            tokenizer,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_fast_tokenizer=use_fast_tokenizer,
            gguf_file=gguf_file,
            add_bos_token=add_bos_token,
        )
        # load tokenizer so we know tokenizer vocabulary size before loading model and PEFT
        if isinstance(pretrained, str):
            self._create_model(
                pretrained=pretrained,
                revision=revision,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                parallelize=parallelize,
                gpus=gpus,
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                peft=peft,
                delta=delta,
                autogptq=autogptq,
                gptqmodel=gptqmodel,
                gguf_file=gguf_file,
                **kwargs,
            )

        # access self._model through self.model property outside this method
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            self.model.tie_weights()

        self.truncation = truncation
        self.truncation_strategy = str(truncation_strategy).strip().lower()
        if self.truncation_strategy not in {"left", "middle"}:
            raise ValueError("truncation_strategy must be 'left' or 'middle'")
        self.logits_cache = logits_cache
        self.vocab_size = self.tokenizer.vocab_size
        # select (or create) a pad token to use
        self.tokenizer = configure_pad_token(self.tokenizer, model_config=self.config)

        self.add_bos_token = add_bos_token
        if "gemma" in getattr(self.config, "model_type", ""):
            self.add_bos_token = True
            eval_logger.info(
                f"Model type is '{self.config.model_type}', part of the Gemma family--a BOS token will be used as Gemma underperforms without it."
            )

        self._max_length = max_length
        self.pretrained = pretrained
        self.delta = delta
        self.peft = peft
        self.revision = revision
        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = max_batch_size

        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)

        if isinstance(pretrained, str):
            if gpus >= 1 or str(self.device) == "mps":
                # TODO: can remove this whole snippet except in the mps case, perhaps?
                if not (parallelize or autogptq or hasattr(self, "accelerator")):
                    # place model onto device requested manually,
                    # if not using HF Accelerate or device_map
                    # or any other option that preloads model onto device
                    try:
                        self.model.to(self.device)
                    except ValueError:
                        eval_logger.debug(
                            "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes` or `device_map` is provided. If the desired GPU is being used, this message is safe to ignore."
                        )
            # multigpu data-parallel support when launched with accelerate
            if gpus > 1:
                if accelerator.num_processes > 1:
                    if parallelize:
                        eval_logger.warning(
                            "You are both using a HF Accelerate `device_map` (`--model_args parallelize=True`) and launching via `accelerate launch`. This will attempt to do model and data parallelism depending on the resources available."
                        )
                    elif gpus > accelerator.num_processes:
                        eval_logger.warning(
                            "WARNING: The number of total system GPUs does not match the number of spawned processes. "
                            "If you would like to use data parallelism, please launch the script "
                            "with 'accelerate launch *script*'. "
                            f"Current run will proceed with {accelerator.num_processes} devices."
                        )
                        if self.accelerator.is_local_main_process:
                            eval_logger.info(
                                f"Using {gpus} devices with data parallelism"
                            )

                    self._device = torch.device(f"{accelerator.device}")
                    self.accelerator = accelerator
                    self._rank = self.accelerator.local_process_index
                    self._world_size = self.accelerator.num_processes
                else:
                    # if we aren't launching via accelerate, ditch
                    self._rank = 0
                    self._world_size = 1
        else:
            # if a PreTrainedModel was passed into HFLM, we forgo distributed setup.
            eval_logger.warning(
                "Passed an already-initialized model through `pretrained`, assuming single-process call to evaluate() or custom distributed integration"
            )
            self._rank = 0
            self._world_size = 1
        self.custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            eval_logger.info(
                f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}"
            )
        if is_feature_cache:
            dLLMCache.new_instance(**asdict(dLLMCacheConfig(
                    prompt_interval_steps=prompt_interval_steps,
                    gen_interval_steps=gen_interval_steps,
                    transfer_ratio=transfer_ratio,
                    cfg_interval_steps=cfg_interval_steps if is_cfg_cache else 1,
                )))
            register_cache_LLaDA(self.model,"model.transformer.blocks")
        else:
            dLLMCache.new_instance(**asdict(dLLMCacheConfig(
                    prompt_interval_steps=1,
                    gen_interval_steps=1,
                    transfer_ratio=0,
                    cfg_interval_steps=cfg_interval_steps if is_cfg_cache else 1,
                )))

        if (
            self.student_prompt_kv_cache
            or self.student_prompt_prune
            or self.student_prompt_pool_active
            or self.student_prompt_dynamic_kv
            or self.student_prompt_layer_split
            or self.student_prompt_drift_refresh
            or self.maskkv_student_scores
        ):
            if self.student_path is None:
                raise RuntimeError("student prompt compression requires student_path")
            self.student = self._load_prompt_utility_student(self.student_path)
            if self.student_refresh_path is not None:
                self.refresh_student = self._load_prompt_utility_student(
                    self.student_refresh_path
                )
        if self.student_prompt_drift_refresh and self.student_drift_mode == "student":
            if self.student_drift_ckpt is None:
                raise RuntimeError("student_drift_mode=student requires student_drift_ckpt")
            from dllm_cache.budget.drift_refresh_kv import load_refresh_student

            self.drift_refresh_student = load_refresh_student(
                self.student_drift_ckpt, self.device
            )

        if self.rank == 0:
                print(f"Feature Cache is {is_feature_cache}.CFG Cache is {is_cfg_cache},prompt_interval_steps={prompt_interval_steps}, gen_interval_steps={gen_interval_steps}, cfg_interval_steps={cfg_interval_steps},transfer_ratio={transfer_ratio}")

    @staticmethod
    def _coerce_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _load_prompt_utility_student(self, checkpoint_dir: Union[str, os.PathLike]):
        from dllm_cache.budget.student_model import PromptUtilityStudent, StudentConfig

        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
        config_path = checkpoint_path / "config.json"
        state_path = checkpoint_path / "pytorch_model.bin"
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        student = PromptUtilityStudent(StudentConfig(**raw_config))
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        student.load_state_dict(state)
        student.to(self.device)
        student.eval()
        if self.rank == 0:
            if self.student_prompt_pool_active:
                mode = "pool-active KV cache"
                budget_text = (
                    f"pool_budget={self.student_pool_budget}, active_budget={self.student_budget}, "
                    f"pool_budget_mode={self.student_pool_budget_mode}, score_activation={self.student_score_activation}"
                )
            elif self.student_prompt_prune:
                mode = "prune"
                budget_text = f"budget={self.student_budget}"
            elif self.student_prompt_drift_refresh:
                mode = f"drift refresh ({self.student_drift_mode})"
                budget_text = (
                    f"budget={self.student_budget}, "
                    f"refresh_tokens={self.student_refresh_tokens or self.student_budget // 4}"
                )
            elif self.student_prompt_layer_split:
                mode = "layer-split prompt cache"
                budget_text = (
                    f"budget={self.student_budget}, "
                    f"frozen_layers={self.student_frozen_layers}, "
                    f"refresh_tokens={self.student_refresh_tokens or self.student_budget}"
                )
            elif self.student_prompt_dynamic_kv:
                mode = "dynamic reduced sequence"
                budget_text = (
                    f"budget={self.student_budget}, selection_mode={self.student_selection_mode}, "
                    f"refresh_interval={self.student_refresh_interval}"
                )
            else:
                mode = "KV cache"
                budget_text = f"budget={self.student_budget}"
            print(
                f"Student prompt {mode} is enabled. checkpoint={checkpoint_path}, "
                f"{budget_text}, question_window={self.student_question_window}",
                flush=True,
            )
        return student

    @torch.inference_mode()
    def _predict_student_scores(self, input_ids: torch.Tensor, student=None) -> torch.Tensor:
        student = self.student if student is None else student
        if student is None:
            raise RuntimeError("student model is not loaded")
        if self.student_random_scores:
            layers = len(self.model.model.transformer.blocks)
            return torch.rand(
                (layers, int(input_ids.shape[1])), device=input_ids.device
            )
        if input_ids.shape[0] != 1:
            raise RuntimeError("student prompt compression currently requires batch_size=1")
        out = self.model(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        prompt_length = int(input_ids.shape[1])
        question_count = min(max(1, self.student_question_window), prompt_length)
        prompt_indices = torch.arange(prompt_length, dtype=torch.long, device=input_ids.device)
        question_indices = torch.arange(
            prompt_length - question_count,
            prompt_length,
            dtype=torch.long,
            device=input_ids.device,
        )
        scores = []
        for layer_id in student.layer_indices:
            layer_scores = student.forward_layer(
                layer_id,
                out.hidden_states[layer_id].float(),
                prompt_indices,
                question_indices,
            )
            layer_scores = layer_scores.float()
            match self.student_score_activation:
                case "softmax":
                    layer_scores = torch.softmax(layer_scores, dim=-1)
                case "sigmoid":
                    layer_scores = torch.sigmoid(layer_scores)
                case "raw":
                    pass
                case _:
                    raise RuntimeError(f"unsupported student_score_activation: {self.student_score_activation}")
            scores.append(layer_scores.squeeze(0).cpu())
        return torch.stack(scores)

    @torch.inference_mode()
    def _generate_with_student_prompt_kv(self, input_ids: torch.Tensor, gen_kwargs: dict) -> torch.Tensor:
        from dllm_cache.budget.prompt_kv_cache import build_prompt_kv_cache
        from dllm_cache.budget.prompt_kv_generate import generate_with_prompt_kv

        student_scores = self._predict_student_scores(input_ids)
        prompt_cache = build_prompt_kv_cache(
            self.model,
            input_ids,
            budget=self.student_budget,
            teacher_scores=student_scores,
        )
        return generate_with_prompt_kv(
            input_ids=input_ids,
            model=self.model,
            prompt_cache=prompt_cache,
            steps=int(gen_kwargs.get("steps")),
            gen_length=int(gen_kwargs.get("gen_length")),
            block_length=int(gen_kwargs.get("block_length")),
            cfg_scale=float(gen_kwargs.get("cfg_scale", 0.0) or 0.0),
            remasking=gen_kwargs.get("remasking", None)
            if gen_kwargs.get("remasking", None)
            else "low_confidence",
            mask_id=self.mask_id,
        )

    @torch.inference_mode()
    def _generate_with_student_layer_split(self, input_ids: torch.Tensor, gen_kwargs: dict) -> torch.Tensor:
        """Freeze the prompt through the shallow layers, keep refreshing it deeper.

        Prompt value vectors barely move in the shallow half (0.04 relative drift by
        layer 8) and move a lot in the deep half (0.37 by layer 24), so the shallow
        layers can serve their prefill K/V for the whole trajectory.
        """
        from dllm_cache.budget.layer_split_prompt_kv import (
            build_layer_split_prompt_cache,
            generate_with_layer_split_prompt_kv,
        )

        student_scores = self._predict_student_scores(input_ids)
        prompt_cache = build_layer_split_prompt_cache(
            self.model,
            input_ids,
            budget=self.student_budget,
            teacher_scores=student_scores,
            frozen_layers=self.student_frozen_layers,
            refresh_tokens=self.student_refresh_tokens,
            refresh_scores=(
                None
                if self.refresh_student is None
                else self._predict_student_scores(input_ids, self.refresh_student)
            ),
            rotate_steps=(
                int(gen_kwargs.get("steps")) if self.student_refresh_rotate else 0
            ),
            measured_tokens=self.student_measured_tokens,
            full_refresh_interval=self.student_full_refresh_interval,
            measured_baseline=self.student_measured_baseline,
        )
        return generate_with_layer_split_prompt_kv(
            input_ids=input_ids,
            model=self.model,
            prompt_cache=prompt_cache,
            steps=int(gen_kwargs.get("steps")),
            gen_length=int(gen_kwargs.get("gen_length")),
            block_length=int(gen_kwargs.get("block_length")),
            cfg_scale=float(gen_kwargs.get("cfg_scale", 0.0) or 0.0),
            remasking=gen_kwargs.get("remasking") or "low_confidence",
            mask_id=self.mask_id,
        )

    @torch.inference_mode()
    def _generate_with_student_dynamic_kv(self, input_ids: torch.Tensor, gen_kwargs: dict) -> torch.Tensor:
        """Drop the unselected prompt tokens and re-forward the shortened sequence.

        Unlike the frozen prompt-KV cache, the retained prompt tokens keep their
        representations refreshed, so bidirectional attention still lets them see the
        suffix. refresh_interval=1 recomputes them at every denoising step.
        """
        from dllm_cache.budget.dynamic_prompt_kv import (
            build_dynamic_prompt_kv_cache,
            generate_with_dynamic_prompt_kv,
        )

        student_scores = self._predict_student_scores(input_ids)
        prompt_cache = build_dynamic_prompt_kv_cache(
            self.model,
            input_ids,
            budget=self.student_budget,
            teacher_scores=student_scores,
            selection_mode=self.student_selection_mode,
        )
        return generate_with_dynamic_prompt_kv(
            input_ids=input_ids,
            model=self.model,
            prompt_cache=prompt_cache,
            steps=int(gen_kwargs.get("steps")),
            gen_length=int(gen_kwargs.get("gen_length")),
            block_length=int(gen_kwargs.get("block_length")),
            refresh_interval=self.student_refresh_interval,
            cfg_scale=float(gen_kwargs.get("cfg_scale", 0.0) or 0.0),
            remasking=gen_kwargs.get("remasking") or "low_confidence",
            mask_id=self.mask_id,
        )

    @torch.inference_mode()
    def _generate_with_drift_refresh(self, input_ids: torch.Tensor, gen_kwargs: dict) -> torch.Tensor:
        """Keep set from the importance student; refresh set chosen per step by drift.

        `student_drift_mode=oracle` ranks by the measured r* = attention x value staleness
        (an upper bound, it needs the fresh values); `delta_student` re-scores the current
        served prompt states with the offline delta student every step; `student` uses the
        separately trained online refresh student.
        """
        from dllm_cache.budget.drift_refresh_kv import generate_with_drift_refresh

        scores = self._predict_student_scores(input_ids)
        budget = min(self.student_budget, int(input_ids.shape[1]))
        keep = torch.topk(scores.mean(dim=0), k=budget, largest=True).indices.sort().values
        refresh_tokens = self.student_refresh_tokens or max(1, budget // 4)
        return generate_with_drift_refresh(
            input_ids=input_ids,
            model=self.model,
            keep_indices=keep,
            refresh_tokens=refresh_tokens,
            mode=self.student_drift_mode,
            refresh_student=self.drift_refresh_student,
            delta_student=self.refresh_student,
            frozen_layers=self.student_drift_frozen_layers,
            question_window=self.student_question_window,
            refresh_interval=self.student_refresh_interval,
            delta_select_once=self.student_delta_select_once,
            steps=int(gen_kwargs.get("steps")),
            gen_length=int(gen_kwargs.get("gen_length")),
            block_length=int(gen_kwargs.get("block_length")),
            temperature=float(gen_kwargs.get("temperature", 0.0) or 0.0),
            cfg_scale=float(gen_kwargs.get("cfg_scale", 0.0) or 0.0),
            remasking=gen_kwargs.get("remasking") or "low_confidence",
            mask_id=self.mask_id,
        )

    @torch.inference_mode()
    def _generate_with_student_prompt_prune(self, input_ids: torch.Tensor, gen_kwargs: dict) -> torch.Tensor:
        from dllm_cache.budget.oracle_prune import install_oracle_pruner

        student_scores = self._predict_student_scores(input_ids)
        controller = install_oracle_pruner(
            self.model,
            prompt_length=int(input_ids.shape[1]),
            budget=self.student_budget,
            teacher_scores=student_scores,
        )
        try:
            output = generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                model=self.model,
                steps=int(gen_kwargs.get("steps")),
                gen_length=int(gen_kwargs.get("gen_length")),
                block_length=int(gen_kwargs.get("block_length")),
                cfg_scale=float(gen_kwargs.get("cfg_scale", 0.0) or 0.0),
                remasking=gen_kwargs.get("remasking", None)
                if gen_kwargs.get("remasking", None)
                else "low_confidence",
                mask_id=self.mask_id,
            )
            if controller.pruned_attention_calls == 0:
                raise RuntimeError("student prompt pruning did not intercept any attention calls")
            return output
        finally:
            controller.restore()

    @torch.inference_mode()
    def _generate_with_student_prompt_pool_active(self, input_ids: torch.Tensor, gen_kwargs: dict) -> torch.Tensor:
        from dllm_cache.budget.pool_active_prompt_kv import (
            build_pool_active_prompt_kv_cache,
            generate_with_pool_active_prompt_kv,
        )

        cfg_scale = float(gen_kwargs.get("cfg_scale", 0.0) or 0.0)
        if cfg_scale != 0.0:
            raise RuntimeError("student prompt pool-active generation does not support cfg_scale")
        student_scores = self._predict_student_scores(input_ids)
        prompt_cache = build_pool_active_prompt_kv_cache(
            self.model,
            input_ids,
            pool_budget=self.student_pool_budget,
            active_budget=self.student_budget,
            teacher_scores=student_scores,
            budget_mode=self.student_pool_budget_mode,
            min_pool_budget=self.student_min_pool_budget,
            pool_budget_scale=self.student_pool_budget_scale,
        )
        return generate_with_pool_active_prompt_kv(
            input_ids=input_ids,
            model=self.model,
            prompt_cache=prompt_cache,
            steps=int(gen_kwargs.get("steps")),
            gen_length=int(gen_kwargs.get("gen_length")),
            block_length=int(gen_kwargs.get("block_length")),
            refresh_interval=self.student_refresh_interval,
            mask_id=self.mask_id,
        )

    def _get_accelerate_args(
        self,
        parallelize: Optional[bool] = None,
        device_map: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        gpus: Optional[int] = None,
    ) -> dict:
        """Returns the kwargs needed to apply `accelerate` in `AutoModel.from_pretrained`."""
        num_local_processes = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        num_machines = int(os.environ.get("WORLD_SIZE", 0)) // num_local_processes
        if (
            num_machines == 0
            and hasattr(self, "accelerator")
            and self.accelerator is not None
        ):
            eval_logger.info("We are not in a distributed setting for accelerate. Setting model_parallel to False.")
            parallelize = False

        if parallelize is None:
            # If parallelism is unset by the user, we automatically assign model parallelism
            # if enough extra GPUs are available
            max_memory_all_gpus = get_max_memory()
            # We just want gpu, not cpu, max memory
            if "cpu" in max_memory_all_gpus:
                del max_memory_all_gpus["cpu"]
            parallelize = bool(num_local_processes < len(max_memory_all_gpus))
            eval_logger.info(
                f"Setting model parallel to {parallelize} since "
                f"the number of local processes is {num_local_processes} "
                f"and the number of GPUs is {len(max_memory_all_gpus)}"
            )

        args = {}
        if parallelize:  # Model parallelism will be used
            max_memory = {}
            if max_memory_per_gpu is not None:  # Using the provided memory requirements
                max_memory_per_gpu_map = {
                    device_idx: max_memory_per_gpu for device_idx in range(gpus)
                }
            else:  # Estimating the possible memory requirements
                max_memory_all_gpus = get_max_memory()
                if "cpu" in max_memory_all_gpus:
                    del max_memory_all_gpus["cpu"]
                if not hasattr(self, "accelerator"):
                    max_memory_per_gpu_map = {
                        k: v for k, v in max_memory_all_gpus.items()
                    }
                else:
                    # use only 1 / num_processes of the GPUs if we are running under accelerate launch
                    max_memory_per_gpu_map = {
                        k: v
                        for k, v in max_memory_all_gpus.items()
                        if k % num_local_processes
                        == (self.accelerator.process_index % num_local_processes)
                    }
            args["max_memory"] = max_memory_per_gpu_map
            args["device_map"] = "auto" if device_map is None else device_map
            eval_logger.info(
                f"Model parallel was set to True, setting max memory per GPU to {max_memory_per_gpu_map} and device map to {args.get('device_map')}"
            )

            if max_cpu_memory is not None:
                max_memory["cpu"] = max_cpu_memory

            args["offload_folder"] = offload_folder
        elif (
            device_map is None
        ):  # No model parallelism, we use the default provided device for our model
            if hasattr(self, "accelerator"):
                device_map = {"": f"{self.accelerator.device}"}
            else:
                device_map = {"": str(self.device)}
            args["max_memory"] = None
            args["device_map"] = device_map
            eval_logger.info(
                f"Model parallel was set to False, max memory was not set, and device map was set to {device_map}"
            )
        else:
            args["max_memory"] = None
            args["device_map"] = None
            eval_logger.info("Model parallel was set to False.")

        return args

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self): 
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        # it is used as prefix for loglikelihood
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        
        return self._device
    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def _get_backend(
        self,
        config: Union[transformers.PretrainedConfig, transformers.AutoConfig],
        backend: Literal["default", "causal", "seq2seq"] = "default",
        trust_remote_code: Optional[bool] = False,
    ) -> None:
        """
        Helper method during initialization.
        Determines the backend ("causal" (decoder-only) or "seq2seq" (encoder-decoder)) model type to be used.
        sets `self.AUTO_MODEL_CLASS` appropriately if not already set.

        **If not calling HFLM.__init__() or HFLM._get_backend() within a subclass of HFLM,
        user must set `self.backend` to be either "causal" or "seq2seq" manually!**
        """

        assert backend in ["default", "causal", "seq2seq"]

        if backend != "default":
            # if we've settled on non-default backend, use that manually
            if backend == "causal":
                self.backend = backend
            elif backend == "seq2seq":
                self.backend = backend
            eval_logger.info(
                f"Overrode HF model backend type, and using type '{self.backend}'"
            )
        else:
            # determine and use the default HF backend for this model, based on its config + metadata.
            if (
                getattr(config, "model_type")
                in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
            ):
                # first check if model type is listed under seq2seq models, since some
                # models like MBart are listed in both seq2seq and causal mistakenly in HF transformers.
                # these special cases should be treated as seq2seq models.
                self.backend = "seq2seq"
                eval_logger.debug(f"Using model type '{self.backend}'")
            elif (
                getattr(self.config, "model_type") in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
            ):
                self.backend = "causal"
                eval_logger.debug(f"Using model type '{self.backend}'")
            else:
                if not trust_remote_code:
                    eval_logger.warning(
                        "HF model type is neither marked as CausalLM or Seq2SeqLM. \
                    This is expected if your model requires `trust_remote_code=True` but may be an error otherwise."
                        "Setting backend to causal"
                    )
                # if model type is neither in HF transformers causal or seq2seq model registries
                # then we default to assuming AutoModelForCausalLM
                self.backend = "causal"
                eval_logger.info(
                    f"Model type cannot be determined. Using default model type '{self.backend}'"
                )

        if self.AUTO_MODEL_CLASS is None:
            if self.backend == "causal":
                self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
            elif self.backend == "seq2seq":
                self.AUTO_MODEL_CLASS = transformers.AutoModelForSeq2SeqLM

    def _get_config(
        self,
        pretrained: str,
        revision: str = "main",
        trust_remote_code: bool = False,
        gguf_file: Optional[str] = None,
    ) -> None:
        """Return the model config for HuggingFace models"""
        self._config = transformers.AutoConfig.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
            gguf_file=gguf_file,
        )

    def _create_model(
        self,
        pretrained: str,
        revision: Optional[str] = "main",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        trust_remote_code: Optional[bool] = False,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        # (accelerate naive PP (device_map) options)
        parallelize: Optional[bool] = False,
        gpus: Optional[int] = None,
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        # PEFT, delta weights and quantization options
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        gptqmodel: Optional[bool] = False,
        gguf_file: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initializes an HF or HF-compatible PreTrainedModel from scratch
        inside HFLM, using the kwargs passed into self.__init__().

        Also handles functionality such as AutoGPTQ usage and PEFT wrapping.

        For future similar extensions to AutoGPTQ that are not core to HF's ecosystem,
        (such as PyTorch models that are nearly, but not quite, fully mirroring
        HF's public interface relied on in this HFLM class)
        please consider subclassing HFLM and overriding this and other methods as needed.
        """
        if autogptq or gptqmodel:
            raise ValueError("'autogptq' and 'gptqmodel' are not supported yet")
        if peft or delta:
            raise ValueError("'peft' and 'delta' are not supported yet")
        model_kwargs = kwargs if kwargs else {}

        model_kwargs.update(
            self._get_accelerate_args(
                parallelize=parallelize,
                device_map=kwargs.get("device_map", None),
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                gpus=gpus,
            )
        )
            
        self._model = transformers.AutoModel.from_pretrained(
            pretrained,
            revision=revision,
            torch_dtype=get_dtype(dtype),
            trust_remote_code=trust_remote_code
        )
        self._model = self._model.to(self.device).eval()
            


    def _create_tokenizer(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ],
        revision: Optional[str] = "main",
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
        gguf_file: Optional[str] = None,
        add_bos_token: Optional[bool] = False,
    ) -> None:
        """
        Helper method during initialization.

        Create a tokenizer object corresponding to the correct
        tokenizer for value of `pretrained`, or use the pre-initialized tokenizer passed.
        """
        kwargs = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
        }

        # gguf format embeds tokenizer and is not compatible with hf tokenizer `use_fast` param
        if gguf_file is not None:
            kwargs["gguf_file"] = gguf_file
        else:
            kwargs["use_fast"] = use_fast_tokenizer

        if add_bos_token:
            kwargs["add_bos_token"] = True

        if tokenizer:
            if isinstance(tokenizer, str):
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    tokenizer, **kwargs
                )
            else:
                assert isinstance(
                    tokenizer, transformers.PreTrainedTokenizer
                ) or isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
                self.tokenizer = tokenizer
        else:
            # Get tokenizer based on 'pretrained'
            if isinstance(pretrained, str):
                model_name = pretrained
            else:
                # get the HF hub name via accessor on model
                model_name = self.model.name_or_path
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name, **kwargs
            )
        return None


    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        """ """
        # default for None - empty dict, use predefined tokenizer param
        # used for all models except for CausalLM or predefined value
        special_tokens_kwargs = {}

        # by default for CausalLM - false or self.add_bos_token is set
        if add_special_tokens is None:
            if self.backend == "causal":
                special_tokens_kwargs = {
                    "add_special_tokens": False or self.add_bos_token
                }
        # otherwise the method explicitly defines the value
        else:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
        truncation_strategy: str = "left",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        add_special_tokens = {}
        if self.backend == "causal":
            add_special_tokens = {"add_special_tokens": False or self.add_bos_token}

        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            **add_special_tokens,
        )
        if left_truncate_len and truncation_strategy == "middle":
            original_lengths = encoding["attention_mask"].sum(dim=1).tolist()
            if any(length > left_truncate_len for length in original_lengths):
                eval_logger.warn(
                    f"Middle truncation applied. Original sequence lengths were {original_lengths}, "
                    f"truncating to first/last {left_truncate_len} tokens. Some middle content will be lost.",
                )
            truncated_rows = []
            masks = []
            pad_id = self.tokenizer.pad_token_id
            for input_ids, attention_mask in zip(encoding["input_ids"], encoding["attention_mask"]):
                valid_tokens = input_ids[attention_mask.bool()]
                if valid_tokens.numel() > left_truncate_len:
                    head_len = left_truncate_len // 2
                    tail_len = left_truncate_len - head_len
                    valid_tokens = torch.cat([valid_tokens[:head_len], valid_tokens[-tail_len:]], dim=0)
                truncated_rows.append(valid_tokens)
            max_row_len = max(row.numel() for row in truncated_rows)
            rows = []
            for row in truncated_rows:
                pad_len = max_row_len - row.numel()
                pad = torch.full((pad_len,), pad_id, dtype=row.dtype)
                if padding_side == "left":
                    row_ids = torch.cat([pad, row], dim=0)
                    row_mask = torch.cat(
                        [torch.zeros(pad_len, dtype=encoding["attention_mask"].dtype), torch.ones(row.numel(), dtype=encoding["attention_mask"].dtype)],
                        dim=0,
                    )
                else:
                    row_ids = torch.cat([row, pad], dim=0)
                    row_mask = torch.cat(
                        [torch.ones(row.numel(), dtype=encoding["attention_mask"].dtype), torch.zeros(pad_len, dtype=encoding["attention_mask"].dtype)],
                        dim=0,
                    )
                rows.append(row_ids)
                masks.append(row_mask)
            encoding["input_ids"] = torch.stack(rows)
            encoding["attention_mask"] = torch.stack(masks)
        elif left_truncate_len:
            original_lengths = encoding["input_ids"].size(1)
            if original_lengths > left_truncate_len:
                eval_logger.warn(
                    f"Left truncation applied. Original sequence length was {original_lengths}, "
                    f"truncating to last {left_truncate_len} tokens. Some content will be lost.",
                )
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][
                :, -left_truncate_len:
            ]
        self.tokenizer.padding_side = old_padding_side

        return encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _model_call(self, inps, attn_mask=None, labels=None):
        """
        :param inps: torch.Tensor
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
            [batch, sequence_ctx]. the size of sequence may vary from call to call
        :param attn_mask: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :param labels: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model's decoder
        """
        with torch.no_grad():
            if attn_mask is not None or labels is not None:
                assert attn_mask is not None and labels is not None
                assert self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM
                return self.model(
                    input_ids=inps, attention_mask=attn_mask, labels=labels
                ).logits
            else:
                assert self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
                return self.model(inps).logits

    def _loglikelihood_tokens(self, requests, **kwargs) -> List[Tuple[float, bool]]:
        raise NotImplementedError
    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        raise NotImplementedError


    def loglikelihood(self, requests):
        raise NotImplementedError
    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        req = []
        bar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Running generate_until requests")
        ds = [{"text": req.args[0]} for req in requests]
        ds = Dataset.from_list(ds)
        gen_kwargs = requests[0].args[1]
        gen_length = int(gen_kwargs.get("gen_length"))
        left_truncate_len = max(1, self.max_length - gen_length)
        for batch in ds.iter(self.batch_size):
            contexts = batch["text"]
            if self.add_bos_token:
                contexts = [self.tokenizer.bos_token + p for p in contexts]
            context_enc, attn_masks = self.tok_batch_encode(
                contexts,
                truncation=self.truncation,
                left_truncate_len=left_truncate_len,
                truncation_strategy=self.truncation_strategy,
            )
            if self.maskkv_student_scores:
                # MaskKV keeps its layer/head budget split; only the ranking it selects
                # with becomes the student's importance prediction. The scoring forward
                # runs through the cache hooks, so the cache has to be initialised first;
                # generate() resets it again, and the scores survive that reset.
                feature_cache = dLLMCache()
                feature_cache.reset_cache(int(context_enc.shape[1]))
                feature_cache.set_prompt_scores(self._predict_student_scores(context_enc))
            if self.student_prompt_pool_active:
                out = self._generate_with_student_prompt_pool_active(context_enc, gen_kwargs)
            elif self.student_prompt_drift_refresh:
                out = self._generate_with_drift_refresh(context_enc, gen_kwargs)
            elif self.student_prompt_layer_split:
                out = self._generate_with_student_layer_split(context_enc, gen_kwargs)
            elif self.student_prompt_dynamic_kv:
                out = self._generate_with_student_dynamic_kv(context_enc, gen_kwargs)
            elif self.student_prompt_prune:
                out = self._generate_with_student_prompt_prune(context_enc, gen_kwargs)
            elif self.student_prompt_kv_cache:
                out = self._generate_with_student_prompt_kv(context_enc, gen_kwargs)
            else:
                out = generate(
                    input_ids=context_enc,
                    attention_mask=attn_masks,
                    model=self.model,
                    steps=gen_kwargs.get("steps"),
                    gen_length=gen_length,
                    block_length=gen_kwargs.get("block_length"),
                    cfg_scale=gen_kwargs.get("cfg_scale"),
                    remasking=gen_kwargs.get("remasking",None) if gen_kwargs.get("remasking",None) else "low_confidence"
                )
            cont_toks_list = self.tokenizer.batch_decode(out, skip_special_tokens=True)
            for s in cont_toks_list:
                if not self.escape_until:
                    for term in gen_kwargs.get("until"):
                        if len(term) > 0:
                            s = s.split(term)[0]
                res.append(s)
                bar.update(1)
            req.append(contexts)
        bar.close()
        return res
    
    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        try:
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )
        except jinja2.exceptions.TemplateError:
            eval_logger.warning(
                "Failed to apply chat template. removing the system role in chat history."
            )
            chat_history = [msg for msg in chat_history if msg["role"] != "system"]
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )

        return chat_templated
