#!/usr/bin/env python3
"""
MinerU API PDF Processing Script

This script uses the MinerU cloud API to extract content from PDF files.
API documentation: https://mineru.net/apiManage/docs

Workflow:
1. Request presigned upload URLs via /api/v4/file-urls/batch
2. Upload PDF files to presigned URLs via PUT
3. Create extraction task via /api/v4/extract/task
4. Poll for results via /api/v4/extract-results/batch/{batch_id}
5. Download extracted content (markdown, JSON)

Usage:
    python process_pdfs.py
"""

import os
import time
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Helper function to resolve paths (relative to script or absolute)
def resolve_path(path_str: str) -> Path:
    """Resolve a path string - if relative, make it relative to script directory."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(__file__).parent / path

# Configuration from .env
API_BASE_URL = os.getenv("MINERU_API_URL", "https://mineru.net/api/v4")
API_KEY = os.getenv("MINERU_API_KEY")
PDF_DIR = resolve_path(os.getenv("PDF_INPUT_DIR", "pdf"))
OUTPUT_DIR = resolve_path(os.getenv("OUTPUT_DIR", "output"))

# Extraction options
ENABLE_OCR = os.getenv("ENABLE_OCR", "false").lower() == "true"
ENABLE_FORMULA = os.getenv("ENABLE_FORMULA", "true").lower() == "true"
ENABLE_TABLE = os.getenv("ENABLE_TABLE", "true").lower() == "true"
DOCUMENT_LANGUAGE = os.getenv("DOCUMENT_LANGUAGE", "en")
LAYOUT_MODEL = os.getenv("LAYOUT_MODEL", "doclayout_yolo")

# Polling settings
POLL_TIMEOUT = int(os.getenv("POLL_TIMEOUT", "600"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "10"))


def is_pdf_processed(pdf_path: Path) -> bool:
    """
    Check if a PDF file has already been processed.

    A PDF is considered processed if its output directory exists
    and contains a full.md file.
    Preserves the original directory structure.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        True if already processed, False otherwise
    """
    # Calculate relative path from PDF_DIR to preserve directory structure
    relative_path = pdf_path.relative_to(PDF_DIR)
    output_relative_dir = relative_path.parent
    output_filename_stem = relative_path.stem

    # Build the output directory path preserving structure
    output_dir = OUTPUT_DIR / output_relative_dir / output_filename_stem
    markdown_file = output_dir / "full.md"
    return markdown_file.exists()


def get_presigned_urls(file_names: list[str]) -> dict:
    """
    Get presigned URLs for uploading files.
    
    Args:
        file_names: List of file names to get upload URLs for
    
    Returns:
        API response containing presigned URLs and batch_id
    """
    url = f"{API_BASE_URL}/file-urls/batch"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "files": [{"name": name} for name in file_names]
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()


def upload_file_to_presigned_url(presigned_url: str, file_path: Path, max_retries: int = 3):
    """
    Upload a file to a presigned URL using PUT with retry logic.
    
    Args:
        presigned_url: The presigned URL to upload to
        file_path: Path to the local file
        max_retries: Maximum number of retry attempts (default: 3)
    """
    _last_exception = None  # noqa: F841
    
    for attempt in range(max_retries + 1):
        try:
            # Don't specify Content-Type - must match whatever was used to generate the presigned URL
            with open(file_path, "rb") as f:
                response = requests.put(presigned_url, data=f, timeout=300)  # 5 min timeout for large files
                response.raise_for_status()
                return  # Success
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            _last_exception = e  # noqa: F841
            if attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                print(f"   ⚠ Upload failed (attempt {attempt + 1}/{max_retries + 1}): {type(e).__name__}")
                print(f"   Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise  # Re-raise on final attempt


def create_extraction_task(file_urls: list[dict], is_ocr: bool = None, enable_formula: bool = None, enable_table: bool = None) -> dict:
    """
    Create a PDF extraction task using MinerU API.
    
    Args:
        file_urls: List of dicts with file information (url, name, etc.)
        is_ocr: Whether to enable OCR mode (uses global ENABLE_OCR if not specified)
        enable_formula: Whether to extract formulas (uses global ENABLE_FORMULA if not specified)
        enable_table: Whether to extract tables (uses global ENABLE_TABLE if not specified)
    
    Returns:
        API response containing batch_id for polling
    """
    # Use global config values if not specified
    if is_ocr is None:
        is_ocr = ENABLE_OCR
    if enable_formula is None:
        enable_formula = ENABLE_FORMULA
    if enable_table is None:
        enable_table = ENABLE_TABLE
    
    url = f"{API_BASE_URL}/extract/task"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    # The API expects an array of file objects, each with url and optional parameters
    # Based on MinerU API documentation, the format is:
    # [{"url": "...", "is_ocr": false, ...}, ...]
    payload = []
    for file_info in file_urls:
        file_payload = {
            "url": file_info["url"],
            "is_ocr": is_ocr,
            "enable_formula": enable_formula,
            "enable_table": enable_table,
            "layout_model": LAYOUT_MODEL,
            "language": DOCUMENT_LANGUAGE
        }
        if "name" in file_info:
            file_payload["data_id"] = file_info["name"]  # Use name as data_id for tracking
        payload.append(file_payload)
    
    response = requests.post(url, headers=headers, json=payload)
    print(f"   Request payload: {json.dumps(payload, indent=2)}")
    print(f"   Response status: {response.status_code}")
    print(f"   Response body: {response.text}")
    response.raise_for_status()
    return response.json()


def get_extraction_result(batch_id: str) -> dict:
    """
    Get the result of an extraction task.
    
    Args:
        batch_id: The batch ID returned when creating the task
    
    Returns:
        API response containing the extraction result
    """
    url = f"{API_BASE_URL}/extract-results/batch/{batch_id}"
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def wait_for_extraction(batch_id: str, timeout: int = None, poll_interval: int = None) -> dict:
    """
    Wait for an extraction task to complete.
    
    Args:
        batch_id: The batch ID returned when creating the task
        timeout: Maximum time to wait in seconds
        poll_interval: Time between polling requests in seconds
    
    Returns:
        The final extraction result
    """
    # Use global config values if not specified
    if timeout is None:
        timeout = POLL_TIMEOUT
    if poll_interval is None:
        poll_interval = POLL_INTERVAL
    
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        result = get_extraction_result(batch_id)
        data = result.get("data", {})
        extract_results = data.get("extract_result", [])
        
        if not extract_results:
            print("  Waiting for processing to start...")
            time.sleep(poll_interval)
            continue
        
        # Check the state of each file
        all_done = True
        any_failed = False
        for item in extract_results:
            state = item.get("state", "unknown")
            file_name = item.get("file_name", "unknown")
            if state != "done":
                all_done = False
                print(f"  {file_name}: {state}")
                if state == "failed":
                    any_failed = True
                    print(f"    Error: {item.get('err_msg', 'Unknown error')}")
            else:
                print(f"  {file_name}: ✓ done")
        
        if all_done:
            return result
        elif any_failed:
            raise RuntimeError(f"Some files failed extraction: {extract_results}")
        
        time.sleep(poll_interval)
    
    raise TimeoutError(f"Extraction timed out after {timeout} seconds")


def download_result(result_url: str, output_path: Path):
    """
    Download the extraction result to a local file.
    
    Args:
        result_url: URL of the result file
        output_path: Local path to save the file
    """
    response = requests.get(result_url)
    response.raise_for_status()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    print(f"    Downloaded: {output_path}")


def process_pdfs(pdf_files: list[Path]):
    """
    Process multiple PDF files through the MinerU API.

    Args:
        pdf_files: List of paths to PDF files
    """
    file_names = [pdf.name for pdf in pdf_files]

    # Create a mapping from file name to full path for later use
    file_path_map = {pdf.name: pdf for pdf in pdf_files}

    # Step 1: Get presigned URLs for upload
    print(f"\n1. Getting presigned URLs for {len(pdf_files)} files...")
    presigned_response = get_presigned_urls(file_names)
    print(f"   Response: {json.dumps(presigned_response, indent=2)}")

    data = presigned_response.get("data", {})
    batch_id = data.get("batch_id")
    file_urls = data.get("file_urls", [])

    if not batch_id or not file_urls:
        raise ValueError(f"Failed to get presigned URLs: {presigned_response}")

    print(f"   Batch ID: {batch_id}")
    print(f"   Got {len(file_urls)} presigned URLs")

    # Step 2: Upload files to presigned URLs
    # Note: file_urls is a list of URL strings, not objects
    print("\n2. Uploading files to presigned URLs...")
    for pdf_path, presigned_url in zip(pdf_files, file_urls):
        print(f"   Uploading: {pdf_path.name}")
        upload_file_to_presigned_url(presigned_url, pdf_path)
        print(f"   ✓ Uploaded: {pdf_path.name}")

    # Step 3: Wait for extraction to complete
    # After uploading to the presigned URLs, MinerU automatically processes the files
    # We use the batch_id from the upload step to poll for results
    print("\n3. Waiting for extraction to complete...")
    print(f"   Using batch_id: {batch_id}")
    result = wait_for_extraction(batch_id)

    # Step 4: Download results
    print("\n4. Downloading results...")
    data = result.get("data", {})
    extract_result = data.get("extract_result", [])

    for item in extract_result:
        file_name = item.get("file_name", "unknown")
        state = item.get("state", "unknown")

        if state != "done":
            print(f"   Skipping {file_name}: state is {state}")
            continue

        print(f"   Processing results for: {file_name}")

        # Get the original PDF path to preserve directory structure
        original_pdf_path = file_path_map.get(file_name)
        if not original_pdf_path:
            print(f"    Warning: Could not find original path for {file_name}")
            continue

        # Calculate relative path from PDF_DIR to preserve directory structure
        relative_path = original_pdf_path.relative_to(PDF_DIR)
        output_relative_dir = relative_path.parent  # Get parent directories
        output_filename_stem = relative_path.stem  # Filename without extension

        # Download the full ZIP file containing all extracted content
        zip_url = item.get("full_zip_url")
        if zip_url:
            # Preserve directory structure in output
            output_zip_path = OUTPUT_DIR / output_relative_dir / f"{output_filename_stem}.zip"
            extract_dir = OUTPUT_DIR / output_relative_dir / output_filename_stem

            download_result(zip_url, output_zip_path)

            # Extract the ZIP file
            import zipfile
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(output_zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"    Extracted to: {extract_dir}")
        else:
            print("    No download URL available")

    print("\n✓ All files processed successfully!")


def main():
    """Main entry point."""
    if not API_KEY:
        print("Error: API key not found in .env file")
        print("Please ensure your .env file contains: apikey=your_api_key_here")
        return
    
    print("MinerU PDF Processing")
    print("=" * 60)
    print(f"API Base URL: {API_BASE_URL}")
    print(f"PDF Directory: {PDF_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("\nExtraction Settings:")
    print(f"  OCR: {ENABLE_OCR}")
    print(f"  Formula: {ENABLE_FORMULA}")
    print(f"  Table: {ENABLE_TABLE}")
    print(f"  Language: {DOCUMENT_LANGUAGE}")
    print(f"  Layout Model: {LAYOUT_MODEL}")
    print("\nPolling Settings:")
    print(f"  Timeout: {POLL_TIMEOUT}s")
    print(f"  Interval: {POLL_INTERVAL}s")
    
    # Find all PDF files recursively in subdirectories
    pdf_files = list(PDF_DIR.rglob("*.pdf"))
    
    if not pdf_files:
        print(f"\nNo PDF files found in {PDF_DIR}")
        return
    
    print(f"\nFound {len(pdf_files)} PDF files:")
    
    # Check for already processed files (resume functionality)
    pending_files = []
    skipped_files = []
    
    for pdf in pdf_files:
        if is_pdf_processed(pdf):
            skipped_files.append(pdf)
            print(f"  ✓ {pdf.name} (already processed, skipping)")
        else:
            pending_files.append(pdf)
            print(f"  - {pdf.name} ({pdf.stat().st_size / 1024:.1f} KB)")
    
    if skipped_files:
        print(f"\nSkipping {len(skipped_files)} already processed files.")
    
    if not pending_files:
        print("\nAll files have already been processed. Nothing to do.")
        return
    
    print(f"\nProcessing {len(pending_files)} pending files...")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process all pending PDFs as a batch
    try:
        process_pdfs(pending_files)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
