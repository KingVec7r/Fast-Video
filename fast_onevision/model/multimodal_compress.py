import os
import torch
import torch.nn as nn
from typing import Optional, List, Dict
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding
import logging

from fast_onevision.constants import IGNORE_INDEX, V_START_ID, V_END_ID, IMG_PAD_ID, VID_PAD_ID, SEQ_PAD_ID, USER_ID, SPLIT_ID, START_ID, END_ID, MAX_MM_ENCODE_LOOP
from fast_onevision.utils import init_layers_from_pretrain

logger = logging.getLogger(__name__)

class IntentModeler(nn.Module):
    """
    Intent Modeling module.

    Distills the core semantic objectives from a textual query Q into a compact
    set of learnable intent tokens Q_intent.  These tokens serve as a conditional
    prior that steers the downstream aggregation toward task-critical visual
    information.

    Formulation:
      Q_intent = IntentModeler([E_txt ; Q_learn] ; Phi)

    where:
      - E_txt  : text embeddings of the user query (shape [B, L, d])
      - Q_learn: K learnable query tokens (shape [K, d]), stored in ``self.queries``
      - Phi    : Rotary Positional Embeddings (RoPE)
      - l      : number of Transformer layers (``compress_intent_modeler_layer_num``)

    The module uses *bidirectional* (non-causal) attention so that every token
    can attend to every other token, enabling holistic intent distillation.
    """
    def __init__(self, config, rotary_emb):
        super().__init__()
        self.config = config
        self.rotary_emb = rotary_emb

        self.register_buffer("user_start_ids", torch.tensor([START_ID, USER_ID], dtype=torch.long), persistent=False)
        self.register_buffer("user_end_ids", torch.tensor([END_ID, SPLIT_ID], dtype=torch.long), persistent=False)
        
        self.img_pad_id = IMG_PAD_ID
        self.vid_pad_id = VID_PAD_ID
        
        # K learnable intent queries Q_learn  (K = compress_intent_token_num = 4)
        self.query_num = config.compress_intent_token_num
        self.queries = nn.Parameter(torch.randn(self.query_num, config.hidden_size))
        
        # l Transformer layers for intent refinement (l = 2)
        self.generate_layers = nn.ModuleList([
            Qwen3DecoderLayer(config, layer_idx=i) 
            for i in range(config.compress_intent_modeler_layer_num)
        ])
        
        # Bidirectional (full) attention — allows holistic cross-token interaction
        for layer in self.generate_layers:
            layer.self_attn.is_causal = False

    def forward(
        self, 
        text_embeddings: torch.Tensor,   # E_txt: [B, L, d]
        input_ids: torch.Tensor,          # token ids for locating user-region boundaries
        pad_embedding: torch.Tensor       # padding embedding (e.g., zero vector)
    ) -> torch.Tensor:
        """
        Returns:
            Q_intent: [B, K, d]  — distilled intent tokens
        """

        batch_size, seq_len, _ = text_embeddings.shape
        device = text_embeddings.device
        if self.user_start_ids.device != device:
            self.user_start_ids = self.user_start_ids.to(device)
            self.user_end_ids = self.user_end_ids.to(device)
        start_len = self.user_start_ids.size(0)
        end_len = self.user_end_ids.size(0)

        start_matches = (input_ids.unfold(1, start_len, 1) == self.user_start_ids).all(dim=-1)
        all_end_matches = (input_ids.unfold(1, end_len, 1) == self.user_end_ids).all(dim=-1)

        end_match_counts = all_end_matches.cumsum(dim=1)
        
        # The ending ids at odd positions belong to "user", while those at even positions belong to "assistant" (there is no "system prompt" in the training data)
        user_end_matches = all_end_matches & (end_match_counts % 2 != 0)

        # User area init
        diff = torch.zeros(batch_size, seq_len + 1, device=device, dtype=torch.int32)
        
        start_indices = torch.where(start_matches)
        diff[start_indices[0], start_indices[1]] += 1
        end_indices = torch.where(user_end_matches)
        diff[end_indices[0], end_indices[1] + end_len] -= 1
        
        # Mask (cumsum)
        in_region_mask = (diff.cumsum(dim=1)[:, :seq_len] > 0)

        is_pad = (input_ids == self.img_pad_id) | (input_ids == self.vid_pad_id)
        is_dup = torch.zeros_like(is_pad, dtype=torch.bool)
        is_dup[:, 1:] = is_pad[:, 1:] & (input_ids[:, 1:] == input_ids[:, :-1])
        
        valid_mask = in_region_mask & (~is_dup)

        valid_counts = valid_mask.sum(dim=1)

        # avoid using .item() for possible dynamic shape in torch.compile
        # max_valid_len = valid_counts.max().item()
        # total_out_len = self.query_num + max_valid_len
        total_out_len = self.query_num + seq_len

        out_embeds = pad_embedding.view(1, 1, -1).repeat(batch_size, total_out_len, 1)
        
        out_embeds[:, :self.query_num] = self.queries.unsqueeze(0)

        row_idx, col_idx = torch.where(valid_mask)
        # valid token should be place at which pos of out_embeds
        # cumsum -> relative position
        local_pos = torch.cumsum(valid_mask, dim=1) 
        dest_col_idx = local_pos[row_idx, col_idx] + (self.query_num - 1)

        out_embeds[row_idx, dest_col_idx] = text_embeddings[row_idx, col_idx]

        # Mask and Position IDs ──
        pos_range = torch.arange(total_out_len, device=device).unsqueeze(0)
        # [batch, total_out_len]
        padding_mask = (pos_range < (self.query_num + valid_counts).unsqueeze(1))

        position_ids = pos_range.expand(batch_size, -1)

        cos, sin = self.rotary_emb(out_embeds, position_ids)
        
        hidden_states = out_embeds
        for layer in self.generate_layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=padding_mask,
                position_embeddings=(cos, sin),
            )

        return hidden_states[:, :self.query_num, :]

    def init_weights(self, pretrained_path):
        """Initialize IntentModeler based on pre-trained Qwen3"""
        logger.info("Initializing IntentModeler......")
        nn.init.normal_(self.queries, mean=0.0, std=self.config.initializer_range)
        init_layers_from_pretrain(self.generate_layers, pretrained_path)
        

class GuidanceGenerator(nn.Module):
    """
    Guidance Generator.

    Produces a frame-specific guidance vector g_t that steers the Aggregator
    to extract *novel* (non-redundant) information from the current frame's
    patch tokens V_t.

    Formulation:
      g_t = GuidanceGenerator([Q_intent ; f_{<t} ; Q_guide] ; Phi)

    where:
      - Q_intent : intent tokens from the IntentModeler
      - f_{<t}   : history of previously compressed frame tokens (autoregressive context)
      - Q_guide  : a single learnable query token (stored in ``self.queries``)
      - Phi      : Rotary Positional Embeddings (RoPE)
      - m        : number of Transformer layers (``compress_guidance_gen_layer_num``)

    Key design choices:
      - Uses *causal* (autoregressive) attention so the generator can be
        conditioned on f_{<t} without seeing future frames.
      - Maintains a KV-cache (``past_key_values``) across time steps so that
        the intent tokens Q_intent are only encoded once, and each subsequent
        call only processes the newly compressed tokens.
      - The output at the Q_guide position is extracted as the guidance vector g_t.
    """
    def __init__(self, config, rotary_emb):
        super().__init__()
        self.config = config
        self.rotary_emb = rotary_emb
        
        # Single learnable guide query Q_guide  (1 token)
        self.query_num = 1
        self.queries = nn.Parameter(torch.randn(self.query_num, config.hidden_size))
        
        # m Transformer layers (m = 2); default Qwen3DecoderLayer is causal
        self.generate_layers = nn.ModuleList([
            Qwen3DecoderLayer(config, layer_idx=i) 
            for i in range(config.compress_guidance_gen_layer_num)
        ])

    def forward(
        self, 
        source_embeddings: torch.Tensor,                          # [Q_intent ; f_{<t}]: [B, S, d]
        past_key_values: Optional[DynamicCache] = None,           # KV-cache for autoregressive decoding
        use_cache: bool = True
    ):
        """
        Args:
            source_embeddings: concatenation of intent tokens and previously
                               compressed frame tokens f_{<t}.
            past_key_values:   accumulated KV-cache from prior time steps.
            use_cache:         whether to return updated KV-cache.

        Returns:
            guidance_token:    [B, 1, d] — guidance vector g_t
            past_key_values:   updated KV-cache (or None).
        """
        batch_size, _, _ = source_embeddings.shape
        device = source_embeddings.device

        # Append the learnable guide query Q_guide at the end
        q_embeds = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        input_embeds = torch.cat([source_embeddings, q_embeds], dim=1)
        current_seq_len = input_embeds.size(1)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        cache_position = torch.arange(
            past_length,
            past_length + current_seq_len,
            dtype=torch.long,
            device=device,
        )

        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        attention_mask = None
        if past_length > 0:
            attention_mask = torch.ones(
                (batch_size, past_length + current_seq_len),
                dtype=torch.bool,
                device=device,
            )

        cos, sin = self.rotary_emb(input_embeds, position_ids)

        hidden_states = input_embeds
        for layer in self.generate_layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=(cos, sin),
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        # Extract hidden state at Q_guide position → guidance vector g_t
        guidance_token = hidden_states[:, -self.query_num:, :]

        # Crop KV-cache to exclude Q_guide — not needed for future steps
        if use_cache and past_key_values is not None:
            past_key_values.crop(-self.query_num)

        return guidance_token, past_key_values if use_cache else None

    def init_weights(self, pretrained_path):
        """Initialize GuidanceGenerator based on pre-trained Qwen3"""
        logger.info("Initializing GuidanceGenerator......")
        nn.init.normal_(self.queries, mean=0.0, std=self.config.initializer_range)
        init_layers_from_pretrain(self.generate_layers, pretrained_path)
        
class Compressor(nn.Module):
    """
    Aggregator / Compressor.

    Condenses the N dense patch tokens V_t of a frame into a single compressed
    token f_t, guided by the frame-specific guidance vector g_t.

    Formulation:
      f_t = Aggregator([V_t ; g_t] ; Phi)

    where:
      - V_t : patch tokens of the t-th frame (shape [N, d])
      - g_t : guidance vector from the GuidanceGenerator (shape [1, d])
      - Phi : Rotary Positional Embeddings — applied to patch tokens to
              preserve their spatial-positional relationships
      - n   : number of Transformer layers (``compress_layer_num``)

    Key design choices:
      - Uses *bidirectional* (full) attention so every patch can attend
        to every other patch and to the guidance token.
      - RoPE position IDs are assigned sequentially to the concatenated
        [g_t ; V_t] sequence, encoding spatial structure.
      - The hidden state at the g_t position (first ``query_num`` positions)
        is taken as the compressed frame representation f_t.
    """
    def __init__(self, config, rotary_emb, chunk_inference: bool = False, max_frames_per_chunk: int = 64):
        super().__init__()
        self.config = config
        self.rotary_emb = rotary_emb
        self.chunk_inference = chunk_inference
        self.max_frames_per_chunk = max_frames_per_chunk
        
        # n Transformer layers (n = 2, bidirectional / full attention)
        self.generate_layers = nn.ModuleList([
            Qwen3DecoderLayer(config, layer_idx=i) 
            for i in range(config.compress_layer_num)
        ])
        # Bidirectional attention: guidance token and patches interact freely
        for layer in self.generate_layers:
            layer.self_attn.is_causal = False
        
    def forward(
        self,
        guidance_token: torch.Tensor,    # g_t: [B, query_num, d]
        source_embeddings: torch.Tensor, # V_t: [B, T_chunk, N, d] — patch tokens
    ) -> torch.Tensor:
        """
        Args:
            guidance_token:    guidance vector(s) for the current chunk of frames.
            source_embeddings: patch tokens for a chunk of frames.

        Returns:
            f_t: [B, T_chunk, query_num, d] — compressed frame tokens.
                 When query_num=1, this is one token per frame (STPF paradigm).
        """
        batch_size, num_frames, num_patches, hidden_size = source_embeddings.shape
        query_num = guidance_token.shape[1]
        device = source_embeddings.device

        # Replicate g_t for each frame in the chunk: [B, T_chunk, query_num, d]
        guidance_token = guidance_token.unsqueeze(1).expand(-1, num_frames, -1, -1)
        
        # Concatenate [g_t ; V_t] along the token dimension
        # Result: [B, T_chunk, query_num + N, d]
        combined_embeds = torch.cat([guidance_token, source_embeddings], dim=2)
        
        # [batch_size * num_frames, total_seq_len, hidden_size]
        total_seq_len = query_num + num_patches
        hidden_states = combined_embeds.view(batch_size * num_frames, total_seq_len, hidden_size)

        total_frames = batch_size * num_frames

        if self.chunk_inference and total_frames > self.max_frames_per_chunk:
            # Chunked inference: split along the flattened frames dimension (dim=0)
            # to limit peak memory. Each chunk is processed independently through all layers.
            all_outputs = []
            for chunk_start in range(0, total_frames, self.max_frames_per_chunk):
                chunk_end = min(chunk_start + self.max_frames_per_chunk, total_frames)
                chunk_hidden = hidden_states[chunk_start:chunk_end]

                # Position IDs and Rotary Embeddings for this chunk
                chunk_position_ids = torch.arange(total_seq_len, device=device).unsqueeze(0)
                chunk_position_ids = chunk_position_ids.expand(chunk_end - chunk_start, -1)

                cos, sin = self.rotary_emb(chunk_hidden, chunk_position_ids)

                for layer in self.generate_layers:
                    chunk_hidden = layer(
                        chunk_hidden,
                        attention_mask=None,  # Full Attention
                        position_embeddings=(cos, sin),
                    )

                # [chunk_frames, query_num, hidden_size]
                all_outputs.append(chunk_hidden[:, :query_num, :])

            # [batch_size * num_frames, query_num, hidden_size]
            compressed_output = torch.cat(all_outputs, dim=0)
        else:
            # Sequential position IDs encode spatial structure (RoPE Phi)
            # Position 0 → guidance token, positions 1..N → patch tokens
            position_ids = torch.arange(total_seq_len, device=device).unsqueeze(0)
            position_ids = position_ids.expand(batch_size * num_frames, -1)

            cos, sin = self.rotary_emb(hidden_states, position_ids)

            for layer in self.generate_layers:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=None,  # Full Attention
                    position_embeddings=(cos, sin),
                )

            # Hidden state at g_t position → compressed token f_t
            compressed_output = hidden_states[:, :query_num, :]

        # Reshape: [B*T_chunk, query_num, d] → [B, T_chunk, query_num, d]
        compressed_output = compressed_output.view(batch_size, num_frames, query_num, hidden_size)
        return compressed_output

    def init_weights(self, pretrained_path):
        """Initialize Compressor based on pre-trained Qwen3"""
        logger.info("Initializing Compressor......")
        init_layers_from_pretrain(self.generate_layers, pretrained_path)
        
class MultimodalCompressLayer(nn.Module):
    """
    Top-level AR-QSA module.

    Orchestrates the full Autoregressive Query-Guided Semantic Aggregation
    pipeline, which consists of three sub-modules:

      1. IntentModeler      — distills task intent Q_intent
      2. GuidanceGenerator   — produces frame-specific guidance g_t
      3. Compressor          — aggregates patches into f_t

    The autoregressive factorization:
      p(F | V, Q) = prod_{t=1..T} p(f_t | V_t, f_{<t}, Q)

    is implemented as a loop over chunks of frames.  At each step:
      a) GuidanceGenerator takes [Q_intent ; f_{<t}] and produces g_t.
      b) Compressor takes [V_t ; g_t] and produces f_t.
      c) f_t is appended to the history buffer for the next iteration.

    For images (single-frame input), the same frame is compressed multiple
    times (up to ``max_encode_loop``), simulating a static video to leverage
    the autoregressive mechanism for more thorough semantic extraction.

    Training strategy:
      - Stage 1 (Modality Alignment): AR-QSA is bypassed (``simple_replace=True``).
      - Stage 2 (Structural Adaptation): AR-QSA is activated; static videos
        synthesized via 16-fold image replication.
      - Stage 3–4: Full autoregressive pipeline with real video data.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.img_pad_id = IMG_PAD_ID
        self.vid_pad_id = VID_PAD_ID

        self.max_encode_loop =  getattr(config, "mm_max_compress_loop", None)

        rotary_emb = Qwen3RotaryEmbedding(config)

        # Three sub-modules of the AR-QSA pipeline
        self.intent_modeler = IntentModeler(config, rotary_emb)          # Intent Modeling
        self.guidance_generator = GuidanceGenerator(config, rotary_emb)  # Guidance Generation
        self.compressor = Compressor(config, rotary_emb)                 # Aggregation

    def init_weights(self, init_llm): 
        self.intent_modeler.init_weights(init_llm)
        self.guidance_generator.init_weights(init_llm)
        self.compressor.init_weights(init_llm)

    def forward(
        self,
        text_embeddings: torch.Tensor,                              # E_txt: [B, L, d]
        input_ids: torch.Tensor,                                     # for locating visual placeholder positions
        image_embeddings: torch.Tensor,                              # V: [B, T, N, d] — patch tokens from vision encoder
        pad_embedding: Optional[torch.Tensor] = None,
        img_compressed_len: Optional[int] = None,                    # target compressed length for single images
        simple_replace: bool = False                                 # if True, bypass AR-QSA (Stage 1 training)
    ) -> torch.Tensor:
        """
        Returns:
            text_embeddings with visual placeholders replaced by
            compressed tokens F = {f_1, ..., f_T}.
        """
        batch_size, num_frames, _, hidden_size = image_embeddings.shape
        device = image_embeddings.device
        dtype = image_embeddings.dtype

        if simple_replace:
            # Stage 1 (Modality Alignment): bypass AR-QSA
            compressed_embeddings = image_embeddings.view(batch_size, -1, hidden_size)
            target_len = image_embeddings.shape[-2]
        else:
            # ═══════════════════════════════════════════════════════════
            #  Step 1: Intent Modeling
            #  Q_intent = IntentModeler([E_txt ; Q_learn] ; Phi)
            # ═══════════════════════════════════════════════════════════
            intent_token = self.intent_modeler(text_embeddings, input_ids, pad_embedding)
            intent_len = intent_token.shape[1]  # K
            
            # Determine the target compressed sequence length
            if num_frames > 1:
                # Video mode: one compressed token per frame (STPF paradigm)
                target_len = num_frames
            else:
                # Image mode: compress the same frame multiple times to leverage
                # the autoregressive mechanism for richer semantic extraction
                target_len = img_compressed_len if img_compressed_len is not None else self.max_encode_loop
            
            # Chunked processing: split target_len into at most max_encode_loop chunks
            # to bound the GuidanceGenerator's context window
            num_chunks = min(target_len, self.max_encode_loop)
            base_len = target_len // num_chunks
            remainder = target_len % num_chunks

            # ═══════════════════════════════════════════════════════════
            #  History buffer: stores [Q_intent ; f_1 ; f_2 ; ... ; f_T]
            #  Implements the autoregressive conditioning f_{<t}
            #  p(f_t | V_t, f_{<t}, Q)
            # ═══════════════════════════════════════════════════════════
            total_history_len = intent_len + target_len
            history_buffer = torch.zeros(
                batch_size, total_history_len, hidden_size, 
                dtype=dtype, device=device
            )
            
            # Pre-fill with intent tokens (constant across all time steps)
            history_buffer[:, :intent_len, :] = intent_token

            past_key_values = None
            current_compressed_pos = intent_len  # write pointer in history_buffer

            # ═══════════════════════════════════════════════════════════
            #  Autoregressive loop: for t = 1 .. T
            # ═══════════════════════════════════════════════════════════
            for i in range(num_chunks):
                current_chunk_len = base_len + (1 if i < remainder else 0)
                
                # Build input for GuidanceGenerator: [Q_intent ; f_{<t}]
                # At i=0, only Q_intent is available (f_{<1} = empty set)
                if i == 0:
                    gen_input = history_buffer[:, :intent_len, :]
                else:
                    # Use the most recently compressed chunk as autoregressive context
                    prev_start = current_compressed_pos - (base_len + (1 if (i-1) < remainder else 0))
                    gen_input = history_buffer[:, prev_start:current_compressed_pos, :]
                
                # ── Step 2: Guidance Generation ──
                # g_t = GuidanceGenerator([Q_intent ; f_{<t} ; Q_guide] ; Phi)
                guidance_token, past_key_values = self.guidance_generator(
                    gen_input, 
                    past_key_values=past_key_values, 
                    use_cache=True
                )

                if num_frames > 1:
                    # ── Step 3a: Video aggregation ──
                    # f_t = Compressor([V_t ; g_t] ; Phi)
                    # Each frame in the chunk is independently compressed with the same g_t
                    start_frame = current_compressed_pos - intent_len
                    chunk_img = image_embeddings[:, start_frame : start_frame + current_chunk_len, :, :]
                    current_compressed = self.compressor(guidance_token, chunk_img)
                else:
                    # ── Step 3b: Image mode — repeated aggregation of the same frame ──
                    # The autoregressive GuidanceGenerator causes g_t to drift across
                    # iterations, extracting complementary semantics from the same V
                    single_frame_compress = self.compressor(guidance_token, image_embeddings)
                    if current_chunk_len == 1:
                        current_compressed = single_frame_compress
                    else:
                        current_compressed = single_frame_compress.repeat(1, current_chunk_len, 1, 1)

                # Append f_t to history buffer → becomes part of f_{<t} for next iteration
                history_buffer[:, current_compressed_pos : current_compressed_pos + current_chunk_len, :] = \
                    current_compressed.view(batch_size, current_chunk_len, hidden_size)
                
                current_compressed_pos += current_chunk_len

            # Extract compressed tokens F = {f_1, ..., f_T} (excluding Q_intent prefix)
            # This is the video-level latent sequence
            compressed_embeddings = history_buffer[:, intent_len:, :]

        # ── Replace visual placeholder tokens (<|video_pad|> / <|image_pad|>) ──
        # with the compressed tokens F, producing the final multimodal sequence
        # that is fed into the LLM backbone (loss computed on p(Y | F, Q))
        visual_mask = (input_ids == self.img_pad_id) | (input_ids == self.vid_pad_id)
        
        _, indices = torch.topk(visual_mask.to(torch.float32), k=target_len, dim=1, sorted=False)
        indices = indices.sort(dim=1).values 
        scatter_indices = indices.unsqueeze(-1).expand(-1, -1, hidden_size)
        
        full_visual_buffer = torch.zeros_like(text_embeddings)
        full_visual_buffer = full_visual_buffer.scatter(1, scatter_indices, compressed_embeddings)
        text_embeddings = torch.where(visual_mask.unsqueeze(-1), full_visual_buffer, text_embeddings)

        return text_embeddings