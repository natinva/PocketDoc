# PocketDoc

This repository provides several medical imaging and analysis tools written in Python. Each tool can run on a Raspberry Pi using the same set of dependencies.

## Setup (Raspberry Pi)

1. **Install Python and `venv`**
   ```bash
   sudo apt-get update
   sudo apt-get install python3 python3-venv python3-pip
   ```
2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv pdoc-venv
   source pdoc-venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set the OpenAI API key** (required for the PatientSum application)
   ```bash
   export OPENAI_API_KEY=your_key_here
   ```
5. **Run an application**
   You can launch a small menu interface that lets you pick any tool:
   ```bash
   python3 launcher.py
   ```
   Alternatively, you may run a script directly by navigating to its folder.

Model weight files (`*.pt`) are expected under the `Modeller/` directory. If a script fails to find its model, place the appropriate file there.

## Notes
- Heavy packages such as `torch` may require ARM wheels on Raspberry Pi. Consult the PyTorch website for installation options if the default installation fails.
- Each script uses relative paths to locate models, so ensure you run them from within the repository.
