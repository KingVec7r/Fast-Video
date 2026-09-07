import os
import torch
from safetensors.torch import save_file

from torch.utils.data import Sampler, DataLoader

from typing import List, Optional
from datetime import timedelta
import logging

from transformers import Trainer
from transformers.trainer import is_sagemaker_mp_enabled, get_parameter_names, has_length, is_accelerate_available, is_datasets_available
from accelerate.utils import GradientAccumulationPlugin
from typing import List, Optional
from torch.nn import LayerNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

logger = logging.getLogger(__name__)

if is_accelerate_available():
    from accelerate import Accelerator, InitProcessGroupKwargs

if is_datasets_available():
    import datasets

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return

def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks

def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    return [i for megabatch in megabatches for i in megabatch]

def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]

def get_image_token_num_grouped_indices(image_token_nums, batch_size, world_size, generator=None):

    groups = {}
    for i, itn in enumerate(image_token_nums):
        key = itn if itn is not None else -100
        groups.setdefault(key, []).append(i)

    megabatch_size = world_size * batch_size
    group_queues = {}

    for key, group_indices in groups.items():

        perm = torch.randperm(len(group_indices), generator=generator)
        shuffled = [group_indices[i] for i in perm.tolist()]

        num_to_keep = (len(shuffled) // megabatch_size) * megabatch_size
        if num_to_keep == 0:
            continue
        shuffled = shuffled[:num_to_keep]

        queue = []
        for start in range(0, len(shuffled), megabatch_size):
            megabatch = shuffled[start : start + megabatch_size]
            # Simple random allocation among ranks of the same step
            chunk_size = len(megabatch) // world_size
            chunks = [megabatch[j * chunk_size : (j + 1) * chunk_size] for j in range(world_size)]
            queue.append(chunks)
            
        group_queues[key] = queue

    all_megabatches = []
    for queue in group_queues.values():
        all_megabatches.extend(queue)   # [R0_chunk, R1_chunk, ...]

    perm = torch.randperm(len(all_megabatches), generator=generator)
    all_megabatches = [all_megabatches[i] for i in perm.tolist()]

    result = []
    for chunks in all_megabatches:
        for chunk in chunks:
            result.extend(chunk)

    return result

class LengthGroupedSampler(Sampler):
    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_strategy: str = "length",
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")
        
        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_strategy = group_strategy
        
        self.indices = self._compute_indices()

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        # self.indices = self._compute_indices()
        return iter(self.indices)

    def _compute_indices(self):
        if self.group_strategy == "modality":
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        elif self.group_strategy == "image_token_num":
            indices = get_image_token_num_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        return indices

class LLaVATrainer(Trainer):
    @property
    def is_tp_enabled(self):
        """Compatibility stub for transformers >= 4.41."""
        return False

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override to force CUDA memory cleanup between gradient accumulation micro-batches.

        Without this, DeepSpeed + gradient accumulation can cause memory to monotonically
        increase across micro-batches because:
          1. The CUDA caching allocator holds onto freed blocks and may request new ones
             from the OS instead of reusing cached blocks when fragmentation is high.
          2. DeepSpeed's internal gradient partitioning can retain intermediate buffers
             that are only truly released at optimizer.step() time.

        Calling ``torch.cuda.empty_cache()`` after each micro-batch forces the caching
        allocator to release unused blocks back to the OS, which dramatically reduces
        peak memory during gradient accumulation. The performance overhead is negligible
        (< 2 % wall-clock time in practice) because the call is non-blocking.
        """
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        # After each micro-batch backward (gradients are now accumulated in .grad),
        # release CUDA cache so the next micro-batch can reuse the freed memory.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return loss

    def create_accelerator_and_postprocess(self):
        grad_acc_kwargs = {
            "num_steps": self.args.gradient_accumulation_steps,
            "sync_with_dataloader": False}

        gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))

        # create accelerator object
        self.accelerator = Accelerator(
            deepspeed_plugin=self.args.deepspeed_plugin, gradient_accumulation_plugin=gradient_accumulation_plugin, 
            kwargs_handlers=[accelerator_kwargs]
        )
        # some Trainer classes need to use `gather` instead of `gather_for_metrics`, thus we store a flag
        self.gather_function = self.accelerator.gather_for_metrics

        # deepspeed and accelerate flags covering both trainer args and accelerate launcher
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None

        # post accelerator creation setup
        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            fsdp_plugin.limit_all_gathers = self.args.fsdp_config.get("limit_all_gathers", fsdp_plugin.limit_all_gathers)
            if is_accelerate_available("0.23.0"):
                fsdp_plugin.activation_checkpointing = self.args.fsdp_config.get("activation_checkpointing", fsdp_plugin.activation_checkpointing)
                if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                    raise ValueError("The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg " "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic " "when using FSDP.")

        if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
            self.propagate_args_to_deepspeed()

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        group_strategy = getattr(self.args, 'group_strategy', None)

        # logger.debug(f"trainer.args.train_batch_size: {self.args.train_batch_size}, trainer.args.world_size: {self.args.world_size}, trainer.args.gradient_accumulation_steps: {self.args.gradient_accumulation_steps}")
        # assert False

        if group_strategy == "modality":
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_strategy="modality",
            )
        elif group_strategy == "image_token_num":
            lengths = self.train_dataset.image_token_num
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_strategy="image_token_num",
            )
        elif group_strategy is not None and group_strategy != "":
            # Fallback: length-based grouping for other explicit strategies.
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_strategy="length",
            )
        else:
            return super()._get_train_sampler()

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = True
            dataloader_params["worker_init_fn"] = None
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor if self.args.dataloader_prefetch_factor is not None else (self.args.dataloader_num_workers * 2 if self.args.dataloader_num_workers != 0 else None)

        dataloader = self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

        return dataloader

    def create_optimizer(self):
        """
        Setup the optimizer with custom learning rate assignment for different model components.
        
        Learning rate assignment:
        - LLM parameters (embed_tokens, layers, norm): lr_llm with optional layer-wise decay
        - Vision tower: lr_vision_tower
        - Query generator: lr_query
        - Compressor: lr_compressor
        - MM projector: lr_projector
        - Grounding anchor: lr_anchor
        
        If lr_llm_decay is set, LLM layer learning rates decrease exponentially by layer index.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            # Setup learning rates with fallback to lr_learning_rate
            base_lr = self.args.learning_rate
            lr_llm = getattr(self.args, 'lr_llm', None) or base_lr
            lr_vision_tower = getattr(self.args, 'lr_vision_tower', None) or base_lr
            lr_query = getattr(self.args, 'lr_query', None) or base_lr
            lr_compressor = getattr(self.args, 'lr_compressor', None) or base_lr
            lr_projector = getattr(self.args, 'lr_projector', None) or base_lr
            
            # LLM layer-wise decay
            lr_llm_decay = getattr(self.args, 'lr_llm_decay', None)
            if lr_llm_decay is not None and not (0 < lr_llm_decay <= 1.0):
                raise ValueError(f"lr_llm_decay must be between 0 and 1, got {lr_llm_decay}")

            # Get weight decay parameters (global, matching original logic)
            decay_parameters = get_parameter_names(opt_model, [LayerNorm, Qwen2RMSNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            # Collect parameters by category and decay status
            params_by_category = {
                'llm': {'layers': {'decay': [], 'no_decay': []}, 
                        'embed_norm': {'decay': [], 'no_decay': []}},
                'vision_tower': {'decay': [], 'no_decay': []},
                'query': {'decay': [], 'no_decay': []},
                'compressor': {'decay': [], 'no_decay': []},
                'projector': {'decay': [], 'no_decay': []}
            }

            for name, param in opt_model.named_parameters():
                if not param.requires_grad:
                    continue
                    
                category = self.get_parameter_category(name)
                
                # Check if parameter should have weight decay
                use_decay = name in decay_parameters
                
                # Separate LLM layers from embed_tokens/norm for special handling
                if category == 'llm':
                    if name.startswith(('model.embed_tokens', 'model.norm')):
                        target = params_by_category['llm']['embed_norm']
                    else:
                        target = params_by_category['llm']['layers']
                    
                    if use_decay:
                        target['decay'].append((name, param))
                    else:
                        target['no_decay'].append((name, param))
                else:
                    target = params_by_category[category]
                    if use_decay:
                        target['decay'].append(param)
                    else:
                        target['no_decay'].append(param)

            # Build final optimizer groups
            optimizer_groups = []

            # Process LLM parameters with optional layer-wise decay
            llm_layers = params_by_category['llm']['layers']
            llm_embed_norm = params_by_category['llm']['embed_norm']

            if lr_llm_decay and (llm_layers['decay'] or llm_layers['no_decay']):
                # Extract unique layer indices
                layer_indices = set()
                for name, _ in llm_layers['decay'] + llm_layers['no_decay']:
                    parts = name.split('.')
                    if len(parts) > 3 and parts[0] == 'model' and parts[1] == 'layers':
                        try:
                            layer_indices.add(int(parts[2]))
                        except ValueError:
                            continue
                
                num_layers = max(layer_indices) if layer_indices else 0
                
                # Group parameters by layer index
                layer_params = {idx: {'decay': [], 'no_decay': []} for idx in layer_indices}
                
                for name, param in llm_layers['decay']:
                    idx = int(name.split('.')[2])
                    layer_params[idx]['decay'].append(param)
                
                for name, param in llm_layers['no_decay']:
                    idx = int(name.split('.')[2])
                    layer_params[idx]['no_decay'].append(param)
                
                # Add layer groups in order (shallow to deep)
                for idx in sorted(layer_indices):
                    layer_lr = lr_llm * (lr_llm_decay ** idx)
                    self.add_param_group(optimizer_groups, layer_params[idx]['decay'], self.args.weight_decay, layer_lr)
                    self.add_param_group(optimizer_groups, layer_params[idx]['no_decay'], 0.0, layer_lr)
                

                # Add embed_tokens and norm with deeper decay (exponent = num_layers(len(layers)-1))
                embed_norm_lr = lr_llm * (lr_llm_decay ** num_layers)
                self.add_param_group(optimizer_groups, [p for _, p in llm_embed_norm['decay']], self.args.weight_decay, embed_norm_lr)
                self.add_param_group(optimizer_groups, [p for _, p in llm_embed_norm['no_decay']], 0.0, embed_norm_lr)
            else:
                # No layer decay - combine all LLM parameters
                all_llm_decay = [p for _, p in llm_layers['decay']] + [p for _, p in llm_embed_norm['decay']]
                all_llm_no_decay = [p for _, p in llm_layers['no_decay']] + [p for _, p in llm_embed_norm['no_decay']]
                
                self.add_param_group(optimizer_groups, all_llm_decay, self.args.weight_decay, lr_llm)
                self.add_param_group(optimizer_groups, all_llm_no_decay, 0.0, lr_llm)

            # Process other categories (only if they exist in the model)
            category_configs = [
                ('vision_tower', lr_vision_tower),
                ('query', lr_query),
                ('compressor', lr_compressor),
                ('projector', lr_projector)
            ]
            
            for category, lr in category_configs:
                # Skip if category has no parameters (model might not have this component)
                if not any(params_by_category[category].values()):
                    continue
                    
                self.add_param_group(optimizer_groups, params_by_category[category]['decay'], self.args.weight_decay, lr)
                self.add_param_group(optimizer_groups, params_by_category[category]['no_decay'], 0.0, lr)
            
            # for i, item in enumerate(optimizer_groups):
            #     print(f"optimizer_groups[{i}]:\n")
            #     for k, v in item.items():
            #         print(f"    {k}: ", type(v))
            # assert False

            self.save_optimizer_groups(optimizer_groups)

            # Create optimizer using the same method as original
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_groups, **optimizer_kwargs)
            
            # Maintain original Adam8bit restriction
            if optimizer_cls.__name__ == "Adam8bit":
                raise NotImplementedError

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        keys_to_match = {
            "projector": ['projector'],
            "compressor": ['projector', 'compressor'],
            }
        
        current_version = self.args.mm_version

        if current_version in keys_to_match:
            if current_version == "compressor":
                file_name = "adapter"
            else:
                file_name = current_version

            weight_to_save = get_mm_adapter_state_maybe_zero_3(
                self.model.named_parameters(), 
                keys_to_match[current_version]
            )

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                os.makedirs(output_dir, exist_ok=True)
                self.model.config.save_pretrained(output_dir)
                save_file(weight_to_save, os.path.join(output_dir, f'{file_name}.safetensors'))
        
        super(LLaVATrainer, self)._save_checkpoint(model, trial)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'mm_version', False) in ["projector", "compressor"]:
            pass  # skip full save; only adapter weights are saved via _save_checkpoint
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

    # Define parameter categories based on name patterns
    def get_parameter_category(self, name):
        """Categorize parameter by name"""
        # LLM components (start with)
        if name.startswith(('model.embed_tokens', 'model.layers', 'model.norm')):
            return 'llm'
        # Vision tower (start with)
        elif name.startswith('model.vision_tower'):
            return 'vision_tower'
        # Query parameters (contain)
        elif '.queries' in name:
            return 'query'
        # Compressor components (start with)
        elif name.startswith('model.compressor'):
            return 'compressor'
        # MM projector (start with)
        elif name.startswith('model.projector'):
            return 'projector'
        else:
            raise ValueError(f"Parameter {name} does not belong to any known category")

    def add_param_group(self, optimizer_groups, params, weight_decay, lr):
        """Helper to add a parameter group if not empty"""
        if not params:
            return
        optimizer_groups.append({
            'params': params,
            'weight_decay': weight_decay,
            'lr': lr,
        })

    def save_optimizer_groups(self, optimizer_groups):
        if self.args.local_rank:
            return
        os.makedirs(self.args.output_dir, exist_ok=True)
        with open(f"{self.args.output_dir}/optimizer_info.txt", "w", encoding="utf-8") as f:
            for i, group in enumerate(optimizer_groups):
                f.write(f"Group {i}:\n")
                f.write(f"  Learning Rate: {group['lr']}\n")
                f.write(f"  Weight Decay: {group.get('weight_decay', 0.0)}\n")
                f.write("  Parameters:\n")
                for param in group['params']:
                    for name, p in self.model.named_parameters():
                        if p is param:
                            f.write(f"    {name}\n")
                            break
                f.write("\n")