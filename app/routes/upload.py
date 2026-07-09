from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.schemas import UploadResponse
from app.services.investigation_service import create_investigation
from app.utils.file_handler import save_upload
from app.utils.logger import logger

router = APIRouter(tags=["Upload"])


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Accepts an image file, validates it, stores it on disk, and creates
    an Investigation record with status 'uploaded'. Call /investigate
    next with the returned investigation_id to run the agent pipeline.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided."
        )

    try:
        upload_meta = await save_upload(file)
        investigation = create_investigation(db, upload_meta)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error while processing upload.",
        )

    return UploadResponse(
        investigation_id=investigation.id,
        original_filename=investigation.original_filename,
        status=investigation.status,
    )
