# Colab in VS Code — Demo Notebook

This repository demonstrates how to run **Google Colab runtimes directly inside Visual Studio Code** using the official Colab extension, while keeping a **Git-first, reproducible workflow**.

The included notebook serves as a lightweight bootstrap for:
- Verifying GPU and CUDA availability
- Inspecting the Colab runtime environment
- Cloning and running project code without relying on Google Drive mounts

---

## Why this workflow?

Google Colab provides powerful, on-demand compute, but its browser-based environment can be limiting for development. Running Colab inside VS Code combines the best of both worlds:

- **VS Code** for editing, navigation, refactoring, and extensions  
- **Colab** for free or low-cost GPU/TPU-backed compute  
- **Git-first workflows** instead of Drive-dependent setups  
- **Ephemeral, reproducible environments** you can recreate anytime

This setup is ideal for demos, experiments, and short-lived research or prototyping tasks.

---

## Contents

- `colab_in_vscode.ipynb` — Demo notebook for validating and bootstrapping a Colab runtime inside VS Code

---

## Prerequisites

### Required
- A Google account with access to **Google Colab**
- **Visual Studio Code**
- Internet access (for authentication and cloning repositories)

### VS Code extensions
Install the following:
- **Google Colab** (official extension)
- **Jupyter** (dependency; VS Code may prompt you to install it)

---

## Setup: Running Colab in VS Code

1. **Install the Google Colab extension**
   - Open VS Code → Extensions (`Ctrl+Shift+X`)
   - Search for **Google Colab**
   - Install the official extension

2. **Open the notebook**
   - Open `colab_in_vscode.ipynb` in VS Code

3. **Select a Colab runtime**
   - Use the kernel selector (top-right of the notebook)
   - Choose a **Colab runtime** exposed by the extension
   - Authenticate with your Google account if prompted

4. **(Optional) Enable GPU**
   - Select a GPU-backed runtime if your workload requires it
   - GPU availability depends on your Colab tier and current capacity

---

## What the demo notebook does

The notebook is intentionally minimal and focuses on environment validation rather than application logic.

### 1. GPU availability check
Runs:
```bash
nvidia-smi
```
Confirms whether a GPU is attached and visible to the runtime.

### 2. CUDA toolkit check
Runs:
```bash
nvcc --version
```
Verifies whether the CUDA compiler is available (useful for CUDA-dependent builds).

### 3. Runtime filesystem inspection
Runs:
```bash
ls -la
```
Shows the current working directory (typically `/content`) and existing files.

### 4. Clone your project repository
Runs:
```bash
git clone <your-repo-url>
```
Pulls your code into the ephemeral Colab runtime so it can be executed immediately.
> Replace `<your-repo-url>` with your actual repository URL.

## Recommended next steps (after cloning)

Most real workflows will add a few follow-up cells, such as:
```bash
# Move into the project directory
cd <repo-folder>

# Install dependencies
pip install -r requirements.txt
```

Optional diagnostics:
```bash
python --version
pip --version
df -h
free -h
```

## Important notes

### Ephemeral runtime

- Colab runtimes are temporary

- Files stored locally can be lost on restart or disconnect

- Commit changes to Git or push outputs elsewhere if they must be preserved

### Why Git instead of Google Drive?

This demo intentionally avoids Drive mounts to:

- Reduce setup friction

- Improve reproducibility

- Treat the runtime as disposable compute, not long-term storage

## Troubleshooting

### GPU not detected

- Ensure a GPU runtime is selected

- Restart the runtime and try again

- GPU availability may be limited on free tiers

### `nvcc` not found

- Not all Colab images include the CUDA compiler

- Install additional CUDA components only if your project requires them

### `git clone` authentication issues

- For private repos, HTTPS + access token is recommended

- SSH keys are possible but less convenient in ephemeral environments

## Intended audience

This repository is intended for:

- Developers and researchers prototyping on Colab

- Teams demonstrating Colab + VS Code workflows

- Workshops or internal demos where fast setup matters

## License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this code, with or without modification, for any purpose, provided that the original copyright
notice and license text are included in all copies or substantial portions of the software.

See the [`LICENSE`](LICENSE) file for full details.