"""Windows folder service."""
from pathlib import Path
from typing import List
import os


class FolderService:
    """Windows folder service."""
    
    def get_pdf_files(self, folder_path: str) -> List[Path]:
        """
        Get all PDF and image files from a Windows folder.
        
        Args:
            folder_path: Folder path
        
        Returns:
            List of PDF and image file paths (PDF, JPG, PNG)
        """
        try:
            folder = Path(folder_path)
            
            if not folder.exists():
                raise FileNotFoundError(f"Folder does not exist: {folder_path}")
            
            if not folder.is_dir():
                raise ValueError(f"Path is not a folder: {folder_path}")
            
            # Recursively search for all PDF and image files.
            pdf_files = list(folder.rglob("*.pdf"))
            jpg_files = list(folder.rglob("*.jpg"))
            jpeg_files = list(folder.rglob("*.jpeg"))
            png_files = list(folder.rglob("*.png"))
            
            all_files = pdf_files + jpg_files + jpeg_files + png_files
            
            return all_files
            
        except Exception as e:
            raise Exception(f"Error while reading folder: {str(e)}")
    
    def validate_folder_path(self, folder_path: str) -> bool:
        """
        Validate whether a folder path is valid.
        
        Args:
            folder_path: Folder path
        
        Returns:
            True if valid
        """
        try:
            path = Path(folder_path)
            return path.exists() and path.is_dir()
        except Exception:
            return False

