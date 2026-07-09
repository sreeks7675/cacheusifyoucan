"""
Helpers for validating and persisting uploaded images to disk.
"""

import os
import uuid

from fastapi import UploadFile, HTTPException, status

from app.config import ALLOWED_IMAGE_EXTENSIONS, MAX_UPLOAD_SIZE_MB, UPLOAD_DIR
from app.utils.logger import logger


def validate_extension(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_IMAGE_EXTENSIONS)}",
        )
    return ext


async def save_upload(file: UploadFile) -> dict:
    """
    Validates and writes an UploadFile to disk under a UUID-based name
    (avoids collisions / path traversal from user-supplied filenames).
    Returns metadata needed to create the Investigation DB row.
    """
    ext = validate_extension(file.filename)

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({size_mb:.1f} MB). Max is {MAX_UPLOAD_SIZE_MB} MB.",
        )

    stored_filename = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join(UPLOAD_DIR, stored_filename)

    try:
        with open(file_path, "wb") as f:
            f.write(contents)
    except OSError as e:
        logger.error(f"Failed writing upload to disk: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save uploaded file.",
        )

    logger.info(f"Saved upload '{file.filename}' -> {file_path} ({size_mb:.2f} MB)")

    return {
        "original_filename": file.filename,
        "stored_filename": stored_filename,
        "file_path": file_path,
        "content_type": file.content_type,
    }
