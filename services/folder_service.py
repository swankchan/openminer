"""Windows folder service."""
from pathlib import Path
from typing import List
import os


class FolderService:
    """Windows folder service."""
    
    def get_pdf_files(self, folder_path: str) -> List[Path]:
        """
        Get all PDF files from a Windows folder.
        
        Args:
            folder_path: Folder path
        
        Returns:
            List of PDF file paths
        """
        try:
            folder = Path(folder_path)
            
            if not folder.exists():
                raise FileNotFoundError(f"Folder does not exist: {folder_path}")
            
            if not folder.is_dir():
                raise ValueError(f"Path is not a folder: {folder_path}")
            
            # Recursively search for all PDF files.
            pdf_files = list(folder.rglob("*.pdf"))
            
            return pdf_files
            
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

