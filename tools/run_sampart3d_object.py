import argparse
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass


DEFAULT_CONFIG_TEMPLATE = pathlib.Path("configs/sampart3d/sampart3d-trainmlp-render16views.py")


@dataclass(frozen=True)
class RunSpec:
    object_name: str
    exp_name: str
    mesh_path: pathlib.Path
    render_dir: pathlib.Path
    exp_dir: pathlib.Path
    config_path: pathlib.Path
    results_dir: pathlib.Path
    vis_dir: pathlib.Path
    backbone_weight: pathlib.Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render, train, and evaluate SAMPart3D for a single GLB object."
    )
    parser.add_argument("--glb", required=True, help="Path to the input .glb file.")
    parser.add_argument("--exp-name", default=None, help="Experiment name. Defaults to the GLB stem.")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value to use. Default: 0")
    parser.add_argument("--num-gpus", default="1", help="Number of GPUs passed to training/eval. Default: 1")
    parser.add_argument(
        "--blender",
        default=None,
        help="Path to the Blender executable. Defaults to <repo>/blender-4.0.0-linux-x64/blender",
    )
    parser.add_argument(
        "--backbone-weight",
        default=None,
        help="Path to ptv3-object.pth. Defaults to <repo>/ckpt/ptv3-object.pth",
    )
    parser.add_argument(
        "--config-template",
        default=str(DEFAULT_CONFIG_TEMPLATE),
        help="Template config used to generate the per-object config.",
    )
    parser.add_argument(
        "--weight-name",
        default="5000",
        help="Checkpoint name used at eval time, without .pth suffix. Default: 5000",
    )
    parser.add_argument("--skip-render", action="store_true", help="Skip Blender rendering.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation.")
    return parser


def derive_run_spec(
    repo_root: pathlib.Path,
    glb_path: pathlib.Path,
    exp_name: str | None,
    backbone_weight: pathlib.Path,
    weight_name: str = "5000",
) -> RunSpec:
    object_name = glb_path.stem
    exp_name = exp_name or object_name
    exp_dir = repo_root / "exp" / "sampart3d" / exp_name
    return RunSpec(
        object_name=object_name,
        exp_name=exp_name,
        mesh_path=repo_root / "mesh_root" / f"{object_name}.glb",
        render_dir=repo_root / "data_root" / object_name,
        exp_dir=exp_dir,
        config_path=exp_dir / "config.py",
        results_dir=exp_dir / "results" / weight_name,
        vis_dir=exp_dir / "vis_pcd" / weight_name,
        backbone_weight=backbone_weight,
    )


def ensure_glb_in_mesh_root(source_glb: pathlib.Path, target_glb: pathlib.Path) -> None:
    target_glb.parent.mkdir(parents=True, exist_ok=True)
    if source_glb.resolve() == target_glb.resolve():
        return
    shutil.copy2(source_glb, target_glb)


def generate_config(template_path: pathlib.Path, spec: RunSpec, repo_root: pathlib.Path) -> None:
    template = template_path.read_text(encoding="utf-8")

    base_runtime = (repo_root / "configs" / "_base_" / "default_runtime.py").as_posix()
    rendered = template.replace(
        '_base_ = ["../_base_/default_runtime.py"]',
        f'_base_ = ["{base_runtime}"]',
    )

    rendered = rendered.replace(
        'data_root = ""',
        f'data_root = "{(repo_root / "data_root").as_posix()}"'
    )
    rendered = rendered.replace(
        'mesh_root = ""',
        f'mesh_root = "{(repo_root / "mesh_root").as_posix()}"'
    )
    rendered = rendered.replace(
        'backbone_weight_path = ""',
        f'backbone_weight_path = "{spec.backbone_weight.as_posix()}"',
    )

    spec.exp_dir.mkdir(parents=True, exist_ok=True)
    spec.config_path.write_text(rendered, encoding="utf-8")



def run_command(command: list[str], cwd: pathlib.Path, env: dict[str, str] | None = None) -> None:
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"\n>>> {printable}\n", flush=True)
    merged_env = os.environ.copy()
    if env is not None:
        merged_env.update(env)
    subprocess.run(command, cwd=str(cwd), env=merged_env, check=True)


def render_object(repo_root: pathlib.Path, blender_path: pathlib.Path, spec: RunSpec) -> None:
    spec.render_dir.mkdir(parents=True, exist_ok=True)

    command = [
        str(blender_path),
        "-b",
        "-P",
        str(repo_root / "tools" / "blender_render_16views.py"),
        str(spec.mesh_path),
        "glb",
        str(spec.render_dir),
    ]

    try:
        run_command(command, cwd=repo_root / "tools")
        return
    except subprocess.CalledProcessError as exc:
        meta_path = spec.render_dir / "meta.json"
        render_count = len(list(spec.render_dir.glob("render_*.webp")))
        depth_count = len(list(spec.render_dir.glob("depth_*.exr")))

        if meta_path.exists() and render_count >= 16 and depth_count >= 16:
            print(
                "\nBlender returned a non-zero exit code, "
                "but the expected render outputs are present. Continuing.\n",
                flush=True,
            )
            print(f"meta.json: {meta_path}")
            print(f"render count: {render_count}")
            print(f"depth count: {depth_count}")
            return

        print("\nBlender render failed and outputs are incomplete.\n", flush=True)
        print(f"meta.json exists: {meta_path.exists()}")
        print(f"render count: {render_count}")
        print(f"depth count: {depth_count}")
        raise exc



def train_object(repo_root: pathlib.Path, spec: RunSpec, num_gpus: str, visible_gpus: str) -> None:
    env = {
        "CUDA_VISIBLE_DEVICES": visible_gpus,
        "PYTHONPATH": str(repo_root),
    }
    run_command(
        [
            sys.executable,
            "launch/train.py",
            "--config-file",
            str(spec.config_path),
            "--num-gpus",
            num_gpus,
            "--options",
            f"save_path={spec.exp_dir.as_posix()}",
            f"oid={spec.object_name}",
            "label=",
        ],
        cwd=repo_root,
        env=env,
    )


def eval_object(repo_root: pathlib.Path, spec: RunSpec, num_gpus: str, visible_gpus: str, weight_name: str) -> None:
    env = {
        "CUDA_VISIBLE_DEVICES": visible_gpus,
        "PYTHONPATH": str(repo_root),
    }
    run_command(
        [
            sys.executable,
            "launch/eval.py",
            "--config-file",
            str(spec.config_path),
            "--num-gpus",
            num_gpus,
            "--options",
            f"save_path={spec.exp_dir.as_posix()}",
            f"weight={(spec.exp_dir / 'model' / f'{weight_name}.pth').as_posix()}",
        ],
        cwd=repo_root,
        env=env,
    )


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    source_glb = pathlib.Path(args.glb).expanduser().resolve()
    if source_glb.suffix.lower() != ".glb":
        parser.error("--glb must point to a .glb file")
    if not source_glb.exists():
        parser.error(f"GLB file does not exist: {source_glb}")

    blender_path = pathlib.Path(args.blender).expanduser().resolve() if args.blender else (
        repo_root / "blender-4.0.0-linux-x64" / "blender"
    )
    backbone_weight = pathlib.Path(args.backbone_weight).expanduser().resolve() if args.backbone_weight else (
        repo_root / "ckpt" / "ptv3-object.pth"
    )
    template_path = pathlib.Path(args.config_template).expanduser().resolve()

    if not template_path.exists():
        parser.error(f"Config template does not exist: {template_path}")
    if not args.skip_render and not blender_path.exists():
        parser.error(f"Blender executable does not exist: {blender_path}")
    if not backbone_weight.exists():
        parser.error(f"Backbone weight does not exist: {backbone_weight}")

    spec = derive_run_spec(
        repo_root=repo_root,
        glb_path=source_glb,
        exp_name=args.exp_name,
        backbone_weight=backbone_weight,
        weight_name=args.weight_name,
    )

    ensure_glb_in_mesh_root(source_glb, spec.mesh_path)
    generate_config(template_path, spec, repo_root)

    if not args.skip_render:
        render_object(repo_root, blender_path, spec)
    if not args.skip_train:
        train_object(repo_root, spec, num_gpus=args.num_gpus, visible_gpus=args.gpu)
    if not args.skip_eval:
        eval_object(repo_root, spec, num_gpus=args.num_gpus, visible_gpus=args.gpu, weight_name=args.weight_name)

    print("\nRun complete.")
    print(f"Config:   {spec.config_path}")
    print(f"Renders:  {spec.render_dir}")
    print(f"Results:  {spec.results_dir}")
    print(f"Meshes:   {spec.vis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
