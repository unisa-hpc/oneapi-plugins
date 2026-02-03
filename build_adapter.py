#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
import base64
import glob


def run(cmd, cwd=None, env=None):
    print(f"[cmd] {' '.join(cmd)}" + (f"  (cwd={cwd})" if cwd else ""))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def git_clone_or_update(repo_url: str, dest: Path, branch: str, depth: int = 1, update: bool = True):
    if not dest.exists():
        run(["git", "clone", repo_url, "-b", branch, f"--depth={depth}", str(dest)])
        return

    if not update:
        print(f"[*] Repo exists at {dest}, update disabled.")
        return

    # Best-effort update to requested branch
    run(["git", "fetch", "--all", "--tags"], cwd=str(dest))
    run(["git", "checkout", branch], cwd=str(dest))
    run(["git", "pull", "--ff-only"], cwd=str(dest))


def cmake_build_ur(
    llvm_dir: Path,
    build_dir: Path,
    install_prefix: Path,
    enable_cuda: bool,
    enable_hip: bool,
    build_type: str,
    jobs: int,
    clean: bool,
):
    ur_src = llvm_dir / "unified-runtime"
    if not ur_src.exists():
        raise RuntimeError(f"Cannot find unified-runtime at {ur_src}. Is this the intel/llvm repo?")

    if clean and build_dir.exists():
        print(f"[*] Cleaning build dir: {build_dir}")
        shutil.rmtree(build_dir)

    ensure_dir(build_dir)
    ensure_dir(install_prefix)

    # UR CMake options
    cmake_args = [
        "cmake",
        "-S", str(ur_src),
        "-B", str(build_dir),
        "-DUR_BUILD_TESTS=OFF",
        "-DUR_ENABLE_TRACING=ON",
        "-DUR_BUILD_ADAPTER_OPENCL=ON",
        f"-DUR_BUILD_ADAPTER_CUDA={'ON' if enable_cuda else 'OFF'}",
        f"-DUR_BUILD_ADAPTER_HIP={'ON' if enable_hip else 'OFF'}",
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
    ]

    run(cmake_args, cwd=str(llvm_dir))
    run(["cmake", "--build", str(build_dir), "-j", str(jobs)], cwd=str(llvm_dir))
    run(["cmake", "--install", str(build_dir)], cwd=str(llvm_dir))


def make_payload_tar_gz(ur_install: Path, out_tar_gz: Path):
    libdir = ur_install / "lib"
    if not libdir.exists():
        raise RuntimeError(f"UR install lib dir not found: {libdir}")

    # Include all libur*.so* (symlinks + versioned libs)
    libs = sorted(Path(p) for p in glob.glob(str(libdir / "libur*.so*")))
    if not libs:
        raise RuntimeError(f"No libur*.so* found in {libdir}. Build/install likely failed.")

    # Create temp payload layout: payload/lib/<libs>
    with tempfile.TemporaryDirectory() as tmp:
        payload_root = Path(tmp) / "payload"
        payload_lib = payload_root / "lib"
        ensure_dir(payload_lib)

        for lib in libs:
            # Preserve symlinks if present
            dest = payload_lib / lib.name
            if lib.is_symlink():
                target = os.readlink(lib)
                dest.symlink_to(target)
            else:
                shutil.copy2(lib, dest)

        # Tar it up (preserve symlinks)
        ensure_dir(out_tar_gz.parent)
        with tarfile.open(out_tar_gz, "w:gz") as tf:
            tf.add(payload_root, arcname="")

    print(f"[+] Created payload: {out_tar_gz}")


def build_installer_from_stub(stub_path: Path, payload_tar_gz: Path, out_installer: Path):
    if not stub_path.exists():
        raise RuntimeError(f"Stub installer not found: {stub_path}")
    if not payload_tar_gz.exists():
        raise RuntimeError(f"Payload not found: {payload_tar_gz}")

    stub_text = stub_path.read_text(encoding="utf-8", errors="strict")
    marker = "__PAYLOAD_B64_BELOW__"
    if marker not in stub_text:
        raise RuntimeError(f"Stub missing marker line: {marker}")

    payload_bytes = payload_tar_gz.read_bytes()
    payload_b64 = base64.b64encode(payload_bytes).decode("ascii")

    ensure_dir(out_installer.parent)
    out_installer.write_text(stub_text, encoding="utf-8")
    with out_installer.open("a", encoding="utf-8") as f:
        # Ensure payload starts on a new line after marker
        if not stub_text.endswith("\n"):
            f.write("\n")
        f.write(payload_b64)

    # Make executable
    out_installer.chmod(out_installer.stat().st_mode | 0o111)
    print(f"[+] Created installer: {out_installer}")


def main():
    ap = argparse.ArgumentParser(
        description="Build UR adapter (CUDA or HIP) from intel/llvm and generate a self-extracting installer."
    )
    backend = ap.add_mutually_exclusive_group(required=True)
    backend.add_argument("--cuda", action="store_true", help="Build UR CUDA adapter")
    backend.add_argument("--hip", action="store_true", help="Build UR HIP adapter")

    ap.add_argument("-b", "--branch", required=True,
                    help="intel/llvm branch or tag (e.g., v6.3.0)")
    ap.add_argument("--workdir", default="work_ur_build",
                    help="Working directory to place source/build/install artifacts")
    ap.add_argument("--repo-url", default="https://github.com/intel/llvm",
                    help="Git repo URL for intel/llvm")
    ap.add_argument("--no-update", action="store_true",
                    help="Do not git fetch/pull if repo already exists")
    ap.add_argument("--clean", action="store_true",
                    help="Delete build directory before building")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 8,
                    help="Parallel build jobs")
    ap.add_argument("--build-type", default="Release", choices=["Release", "RelWithDebInfo", "Debug"],
                    help="CMake build type")

    ap.add_argument("--stub", required=True,
                    help="Path to installer stub script (must contain __PAYLOAD_B64_BELOW__)")

    ap.add_argument("--oneapi-version", default="2025.3",
                    help="Used only to name the output installer file (you can customize naming)")
    ap.add_argument("--os", dest="os_name", default="linux", help="Used only for naming")
    ap.add_argument("--arch", default="x86_64", help="Used only for naming")
    ap.add_argument("--vendor", default=None,
                    help="Override vendor label used in naming (default: nvidia for cuda, amd for hip)")
    ap.add_argument("--out", default=None,
                    help="Output installer path. If omitted, auto-named in workdir/out/")

    args = ap.parse_args()

    workdir = Path(args.workdir).resolve()
    src_dir = workdir / "llvm"
    build_dir = workdir / "build"
    install_dir = workdir / "ur-install"
    out_dir = workdir / "out"
    ensure_dir(workdir)

    enable_cuda = bool(args.cuda)
    enable_hip = bool(args.hip)

    default_vendor = "nvidia" if enable_cuda else "amd"
    vendor = args.vendor or default_vendor
    backend_name = "cuda" if enable_cuda else "hip"

    if args.out:
        out_installer = Path(args.out).resolve()
    else:
        out_installer = out_dir / f"oneapi-ur-{backend_name}-{args.os_name}-{args.arch}-{args.oneapi_version}.sh"

    payload_tar = out_dir / f"ur_adapter_payload_{backend_name}_{args.oneapi_version}.tar.gz"

    print("[*] Settings")
    print(f"    backend: {backend_name} (vendor={vendor})")
    print(f"    branch/tag: {args.branch}")
    print(f"    workdir: {workdir}")
    print(f"    stub: {Path(args.stub).resolve()}")
    print(f"    output installer: {out_installer}")

    git_clone_or_update(
        repo_url=args.repo_url,
        dest=src_dir,
        branch=args.branch,
        depth=1,
        update=not args.no_update
    )

    # Build/install UR
    cmake_build_ur(
        llvm_dir=src_dir,
        build_dir=build_dir,
        install_prefix=install_dir,
        enable_cuda=enable_cuda,
        enable_hip=enable_hip,
        build_type=args.build_type,
        jobs=args.jobs,
        clean=args.clean
    )

    # Payload + installer
    make_payload_tar_gz(install_dir, payload_tar)
    build_installer_from_stub(Path(args.stub).resolve(), payload_tar, out_installer)

    print("[+] Done.")
    print(f"    Installer: {out_installer}")
    print(f"    Payload:   {payload_tar}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
