# oneAPI Unified Runtime Adapter Installers

This repository provides a **reproducible workflow** to build and distribute
**oneAPI Unified Runtime (UR) adapter installers** as **self-contained scripts**.

The installers embed the required shared libraries (`.so`) as a **base64-encoded payload**
and install them directly into a versioned oneAPI compiler directory.

## Repository layout

```
.
├── installer.stub.sh        # Installer stub (NO payload, safe to edit)
├── docs/                    # GitHub Pages site
│   ├── index.html           # Download UI
│   └── manifest.json        # Available adapters (source of truth)
└── README.md
```

> The **final installer script with payload** is **not committed** to the repository.
> It is published as a **GitHub Release asset**.

---

## How the installer works

1. The installer script contains:
   - shell logic
   - validation
   - installation steps
2. The UR adapter libraries are:
   - archived (`tar.gz`)
   - base64-encoded
   - appended after a marker line
3. At runtime the installer:
   - decodes the payload
   - extracts the libraries
   - installs them into:
     ```
     <ONEAPI_ROOT>/compiler/<ONEAPI_VERSION>/lib
     ```

---

## Build the UR CUDA adapter

The CUDA adapter is built from Intel’s `llvm` repository under `unified-runtime`.

Example:

```bash
git clone https://github.com/intel/llvm -b v6.3.0 --depth=1
cd llvm

cmake -S unified-runtime -B build   -DUR_BUILD_TESTS=OFF   -DUR_ENABLE_TRACING=ON   -DUR_BUILD_ADAPTER_OPENCL=ON   -DUR_BUILD_ADAPTER_CUDA=ON   -DCMAKE_BUILD_TYPE=Release   -DCMAKE_INSTALL_PREFIX="$PWD/ur-install"

cmake --build build -j
cmake --install build
```

---

## Create the payload archive

```bash
UR_PREFIX="$PWD/ur-install"

rm -rf payload
mkdir -p payload/lib

# Copy libraries including symlinks
cp -a "${UR_PREFIX}/lib/libur"*.so* payload/lib/

tar -czf ur_adapter_payload.tar.gz -C payload .
```

---

## Generate the final installer (base64 payload)

Edit **only** the stub script (`installer.stub.sh`).

The stub must end with a marker line:

```
__PAYLOAD_B64_BELOW__
```

Create the final installer:

```bash
OUT="oneapi-ur-cuda-linux-x86_64-2025.3.sh"

cat installer.stub.sh <(base64 -w 0 ur_adapter_payload.tar.gz) > "$OUT"
chmod +x "$OUT"
```

> ⚠️ Do **not** open or edit the final installer after appending the payload.

---

## Run the installer

```bash
./oneapi-ur-cuda-linux-x86_64-2025.3.sh
```

Custom oneAPI root:

```bash
./oneapi-ur-cuda-linux-x86_64-2025.3.sh --oneapi-root /custom/intel/oneapi
```

Verify:

```bash
source /opt/intel/oneapi/setvars.sh
sycl-ls --verbose
```

---

## Distribution model

- **GitHub Releases**: installer scripts (binary artifacts)
- **GitHub Pages**: download UI and manifest

---

## License

This project is licensed under the Apache License, Version 2.0.
See the [LICENSE](LICENSE) file for details.
