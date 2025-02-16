# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import glob
import json
import re
from pathlib import Path

import torch
from accelerate import init_empty_weights
from huggingface_hub import file_exists, hf_hub_download, snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file

from transformers import (
    AddedToken,
    AutoConfig,
    AutoImageProcessor,
    AutoTokenizer,
    JanusConfig,
    JanusForConditionalGeneration,
    LlavaProcessor,
    SiglipVisionConfig, AutoModelForCausalLM,
)

EPILOG_TXT = """Example:
    python transformers/src/transformers/models/janus/convert_janus_weights_to_hf.py --text_model_id lmsys/vicuna-7b-v1.5 --vision_model_id openai/clip-vit-large-patch14-336 --output_hub_path org/janus-v1.5-7b-conv --old_state_dict_id liuhaotian/janus-v1.5-7b

Example for creating the old state dict file with Python:

    import torch
    from janus.model.language_model.janus_llama import JanusLlamaForCausalLM

    # load model
    kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
    model = JanusLlamaForCausalLM.from_pretrained("liuhaotian/janus-v1.5-7b", low_cpu_mem_usage=True, **kwargs)

    # load vision tower
    model.get_vision_tower().load_model()

    # Save state dict
    torch.save(model.state_dict(), "tmp/hf_models/janus-v1.5-7b/model_state_dict.bin")
"""

KEYS_TO_MODIFY_MAPPING = {
    r"^gen_vision_model\.": "gen_vision.",
}


def load_original_state_dict(model_id):
    directory_path = snapshot_download(repo_id=model_id, allow_patterns=["*.safetensors"])

    original_state_dict = {}
    for path in glob.glob(f"{directory_path}/*"):
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    original_state_dict[key] = f.get_tensor(key)

    # tied wieghts so lm.head is not saved. Let's clone to load state dict
    if "lm_head.weight" not in original_state_dict:
        original_state_dict["lm_head.weight"] = original_state_dict["model.embed_tokens.weight"].clone()

    if "model.image_newline" in original_state_dict:
        # not used in the original implementation because "merge_type=flat"
        del original_state_dict["model.image_newline"]
    return original_state_dict


def load_sharded_safetensors(model_dir):
    # Initialize an empty state dict
    state_dict = {}

    # Load all shards
    for shard_file in Path(model_dir).glob('model-*.safetensors'):
        current_shard = load_file(shard_file)
        state_dict.update(current_shard)

    return state_dict


# used only for janus-interlave
# for ex: Qwen/Qwen1.5-0.5B-Chat google/siglip-so400m-patch14-384 lmms-lab/janus-next-interleave-qwen-0.5b
def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith(".inv_freq"):
            continue
        for old_pattern, new_pattern in KEYS_TO_MODIFY_MAPPING.items():
            key = re.sub(old_pattern, new_pattern, key)

        new_state_dict[key] = value
    return new_state_dict


def convert_janus_llama_to_hf(text_model_id, vision_model_id, old_janus_folder, output_hub_path, output_dir):
    torch.set_default_dtype(torch.float16)
    text_config = AutoConfig.from_pretrained(text_model_id)

    tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)
    if "Qwen" not in text_model_id:  # qwen already has a pad token
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    image_processor = AutoImageProcessor.from_pretrained(vision_model_id)
    processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)

    if "siglip" in vision_model_id:
        vision_config = SiglipVisionConfig(
            hidden_size=1152,
            image_size=384,
            intermediate_size=4304,
            num_attention_heads=16,
            num_hidden_layers=26,
            patch_size=14,
            vision_use_head=False,
        ).to_dict()
    else:
        vision_config = None

    """
    TMP_COMMENT: I added .to_dict() here because **text_config was not working for text_config of type LlamaConfig. 
    Maybe change later.
    """
    config = JanusConfig(
        text_config=text_config.to_dict() if text_config else None,
        encoder_vision_config=vision_config.to_dict() if vision_config else None,
    )

    # llms-lab interleeave models do not use any selection startegy except for last hidden state
    if "Qwen" in text_model_id:
        config.image_token_index = 151646
        if "siglip" in vision_model_id:
            config.vision_feature_select_strategy = "full"
            config.vision_feature_layer = -1
    else:
        config.pad_token_id = 32001
        config.image_token_index = 32000

    with init_empty_weights():
        model = JanusForConditionalGeneration(config)

    state_dict = load_sharded_safetensors(old_janus_folder)
    state_dict = convert_state_dict_to_hf(state_dict)

    # TMP_COMMENT: strict is False, but this is meant to be temporary
    model.load_state_dict(state_dict, assign=True, strict=False)
    if output_dir is not None:
        model.save_pretrained(output_dir, safe_serialization=True)

    if output_hub_path is not None:
        model.push_to_hub(output_hub_path)
        processor.push_to_hub(output_hub_path)


def main():
    parser = argparse.ArgumentParser(
        epilog=EPILOG_TXT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--text_model_id",
        help="Hub location of the text model",
    )
    parser.add_argument(
        "--vision_model_id",
        help="Hub location of the vision model",
    )
    parser.add_argument(
        "--output_hub_path",
        help="Location on the hub of the converted model",
    )
    parser.add_argument(
        "--old_janus_folder",
        help="Local folder with the original model saved in pytorch safetensors format",
    )

    parser.add_argument(
        "--output_dir",
        help="Location on the hub of the original model",
    )
    args = parser.parse_args()
    convert_janus_llama_to_hf(args.text_model_id, args.vision_model_id,
                              args.old_janus_folder, args.output_hub_path,
                              args.output_dir
                              )


if __name__ == "__main__":
    main()
