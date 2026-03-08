# SafeAgree-OPP115Loader

SafeAgree-OPP115Loader is a data processing pipeline designed to parse, format, and upload the OPP-115 (Online Privacy Policies) dataset into a structured Hugging Face dataset. 

This repository was used to generate the official dataset available at: [exiort/SafeAgree-OPP115](https://huggingface.co/datasets/exiort/SafeAgree-OPP115).

The tool processes raw HTML privacy policies, aligns them with their respective CSV annotations, and formats the output into a JSON-string target suitable for training and fine-tuning Large Language Models (LLMs). 

## Features

* **Automated Parsing:** Uses `BeautifulSoup` to extract clean text from sanitized HTML policy segments.
* **Data Alignment:** Matches sanitized policies with `annotations` and `pretty_print` files to ensure accurate feature mapping.
* **HF Dataset Conversion:** Converts the processed segments into a Hugging Face `Dataset` object containing `input_text` and `target_json_string` features, with optional `policy_id` and `segment_id` metadata.
* **Automated Splits:** Automatically splits the dataset into `train` (80%), `validation` (10%), and `test` (10%) sets using a fixed seed (31) before uploading.
* **Environment Validation:** Includes a robust validation script (`environment_validation_check.py`) to verify your local LLM training environment (checking PyTorch, CUDA, Unsloth, bitsandbytes, and xformers).

## Prerequisites

Ensure you have the required Python packages installed:
```bash
pip install -r requirements.txt
```

## Directory Structure Requirement

For the `load_opp115` command to work, your base data directory must contain the following three subdirectories:
* `annotations/`
* `sanitized_policies/`
* `pretty_print/`

## Usage

The `main.py` script acts as the entry point and accepts two primary commands: `load_opp115` and `upload_opp115`.

### 1. Process and Save the Dataset Locally
Run the loader command to parse the raw files and save them as a Hugging Face dataset to your local disk.

```bash
python main.py load_opp115
```
**Prompts:**
* `BasePath:` The path to the folder containing the required subdirectories.
* `SavePath:` The destination path where the Hugging Face dataset will be saved.
* `IncludeMetadata(Y/N):` Choose 'Y' to include `policy_id` and `segment_id` in the output features.

### 2. Upload to Hugging Face Hub
Run the upload command to apply train/test/validation splits and push the dataset to a Hugging Face repository.

```bash
python main.py upload_opp115
```
**Prompts:**
* `DatasetPath:` The local path where you saved the dataset in step 1.
* `RepoID:` Your Hugging Face repository ID (e.g., `your-username/SafeAgree-OPP115`).
* `Token:` Your Hugging Face write access token.
* `CommitMessage:` A message for the commit.
* `IsPrivate(Y/N):` Whether the repository should be private.

### 3. Environment Validation (Optional)
If you plan to use the resulting dataset to fine-tune an LLM locally using Unsloth or PEFT, run the validation script to ensure your CUDA environment and dependencies are correctly configured:

```bash
python environment_validation_check.py
```

---

## 📜 License

This project is licensed under the MIT License. Copyright (c) 2026 Buğrahan İmal. You are free to use, copy, modify, merge, publish, and distribute this software as per the license conditions.
