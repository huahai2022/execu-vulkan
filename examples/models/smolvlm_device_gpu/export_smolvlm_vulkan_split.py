import argparse
import importlib.util
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
BASE_EXPORT_PATH = THIS_DIR / "export_smolvlm_vulkan.py"

_BASE_SPEC = importlib.util.spec_from_file_location(
    "smolvlm_vulkan_base", BASE_EXPORT_PATH
)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise RuntimeError(f"Failed to load base exporter from {BASE_EXPORT_PATH}")
base = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(base)


def _build_exported_programs(args):
    base.patch_leafspec_bug()

    model = base.AutoModelForVision2Seq.from_pretrained(
        args.model_dir,
        local_files_only=True,
        device_map="cpu",
        torch_dtype=torch.float32,
    ).eval()

    base.activate_et_shims()

    text_state_dict = base.convert_text_state_dict(model.state_dict())
    text_model = base.build_text_model(
        text_state_dict,
        args.max_seq_len,
        args.max_context_len,
    )

    vision_encoder = base.SmolVLMVisionEncoder(model).eval()
    token_embedding = base.SmolVLMTokenEmbedding(model).eval()
    text_decoder = base.SmolVLMTextDecoder(text_model).eval()

    image_input = (torch.zeros(1, 3, 512, 512, dtype=torch.float32),)
    example_token_len = args.max_context_len
    token_input = (
        torch.arange(1, example_token_len + 1, dtype=torch.long).unsqueeze(0),
    )
    embedding_input = (
        torch.zeros(1, example_token_len, 960, dtype=torch.float32),
        torch.tensor([0], dtype=torch.long),
    )

    token_dim = torch.export.Dim("token_dim", min=1, max=args.max_context_len)
    token_shapes = ({1: token_dim},)
    embedding_shapes = ({1: token_dim}, {0: 1})

    with torch.no_grad():
        vision_ep = torch.export.export(vision_encoder, image_input, strict=False)
        token_ep = torch.export.export(
            token_embedding,
            token_input,
            dynamic_shapes=token_shapes,
            strict=False,
        )
        text_ep = torch.export.export(
            text_decoder,
            embedding_input,
            dynamic_shapes=embedding_shapes,
            strict=False,
        )
        text_ep = base.normalize_custom_sdpa_cache_args(text_ep)

    return {
        "vision_encoder": vision_ep,
        "token_embedding": token_ep,
        "text_decoder": text_ep,
    }


def _text_constant_methods(args):
    return {
        "get_max_seq_len": args.max_seq_len,
        "get_max_context_len": args.max_context_len,
        "get_vocab_size": 49280,
        "get_bos_id": 1,
        "get_eos_ids": [49279],
        "use_kv_cache": True,
        "use_sdpa_with_kv_cache": True,
        "enable_dynamic_shape": True,
    }


def _storage_type(name):
    from executorch.backends.vulkan.serialization.vulkan_graph_schema import (
        VkStorageType,
    )

    return VkStorageType.BUFFER if name == "buffer" else VkStorageType.TEXTURE_3D


def _lower_single_method(args, method_name, exported_program, storage_name, dynamic_shapes):
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    partitioner = {
        method_name: [
            VulkanPartitioner(
                {
                    "require_dynamic_shapes": dynamic_shapes,
                    "force_fp16": args.vulkan_force_fp16,
                    "storage_type_override": _storage_type(storage_name),
                }
            )
        ]
    }

    constant_methods = _text_constant_methods(args) if method_name == "text_decoder" else {}

    edge_manager = to_edge_transform_and_lower(
        {method_name: exported_program},
        partitioner=partitioner,
        constant_methods=constant_methods,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
    )
    if args.print_delegation_summary:
        base.print_delegation_summary(edge_manager)
    return edge_manager


def _to_executorch(edge_manager):
    from executorch.exir import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass
    from executorch.exir.passes.sym_shape_eval_pass import HintBasedSymShapeEvalPass

    return edge_manager.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            sym_shape_eval_pass=HintBasedSymShapeEvalPass(),
        )
    )


def _write_program(program, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        program.write_to_file(f)
    print(f"wrote {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--max-context-len", type=int, default=256)
    parser.add_argument("--vulkan-force-fp16", action="store_true")
    parser.add_argument(
        "--vulkan-storage",
        choices=("texture3d", "buffer"),
        default="buffer",
    )
    parser.add_argument(
        "--vulkan-vision-storage",
        choices=("texture3d", "buffer"),
        default=None,
    )
    parser.add_argument(
        "--print-delegation-summary",
        action="store_true",
        help="Print a per-method summary of delegate lowering before writing each .pte.",
    )
    args = parser.parse_args()

    export_programs = _build_exported_programs(args)

    output_dir = Path(args.output_dir)
    vision_storage = args.vulkan_vision_storage or args.vulkan_storage
    method_specs = [
        ("vision_encoder", vision_storage, False),
        ("token_embedding", args.vulkan_storage, True),
        ("text_decoder", args.vulkan_storage, True),
    ]

    for method_name, storage_name, dynamic_shapes in method_specs:
        edge_manager = _lower_single_method(
            args,
            method_name,
            export_programs[method_name],
            storage_name,
            dynamic_shapes,
        )
        executorch_program = _to_executorch(edge_manager)
        output_path = output_dir / f"{method_name}_vulkan.pte"
        _write_program(executorch_program, output_path)


if __name__ == "__main__":
    main()
