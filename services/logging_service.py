"""
Logging Service for PDF Processing History

This module provides SQLite-based logging for tracking PDF processing operations,
including input/output files, timestamps, and processing statistics.
"""

import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

# Default database path
DEFAULT_DB_PATH = Path(__file__).parent.parent / "processing_logs.db"


class LoggingService:
    """
    Service for logging and querying PDF processing history.
    Thread-safe SQLite operations with connection pooling.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or DEFAULT_DB_PATH)
        self._local = threading.local()
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Get a thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        try:
            yield self._local.conn
        except Exception:
            self._local.conn.rollback()
            raise

    def _init_db(self):
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processing_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    input_filename TEXT NOT NULL,
                    source_path TEXT,
                    processed_path TEXT,
                    output_searchable_pdf TEXT,
                    output_json TEXT,
                    output_csv TEXT,
                    status TEXT DEFAULT 'success',
                    error_message TEXT,
                    processing_time_ms INTEGER,
                    file_size_bytes INTEGER
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON processing_logs(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON processing_logs(status)
            """)
            conn.commit()

    def log_processing(
        self,
        input_filename: str,
        source_path: Optional[str] = None,
        processed_path: Optional[str] = None,
        output_searchable_pdf: Optional[str] = None,
        output_json: Optional[str] = None,
        output_csv: Optional[str] = None,
        status: str = "success",
        error_message: Optional[str] = None,
        processing_time_ms: Optional[int] = None,
        file_size_bytes: Optional[int] = None,
    ) -> int:
        """
        Log a PDF processing operation.
        
        Returns:
            The ID of the inserted log record.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO processing_logs (
                    input_filename, source_path, processed_path,
                    output_searchable_pdf, output_json, output_csv,
                    status, error_message, processing_time_ms, file_size_bytes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    input_filename,
                    source_path,
                    processed_path,
                    output_searchable_pdf,
                    output_json,
                    output_csv,
                    status,
                    error_message,
                    processing_time_ms,
                    file_size_bytes,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_logs(
        self,
        limit: int = 50,
        offset: int = 0,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get processing logs with pagination and optional filtering.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            date_from: Filter by start date (ISO format)
            date_to: Filter by end date (ISO format)
            status: Filter by status ('success' or 'failed')
        
        Returns:
            Dictionary with 'items', 'total', 'limit', 'offset'
        """
        conditions = []
        params = []

        if date_from:
            conditions.append("timestamp >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("timestamp <= ?")
            params.append(date_to + " 23:59:59")
        if status:
            conditions.append("status = ?")
            params.append(status)

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        with self._get_connection() as conn:
            # Get total count
            count_sql = f"SELECT COUNT(*) FROM processing_logs{where_clause}"
            total = conn.execute(count_sql, params).fetchone()[0]

            # Get paginated results
            query_sql = f"""
                SELECT * FROM processing_logs
                {where_clause}
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """
            rows = conn.execute(query_sql, params + [limit, offset]).fetchall()

            items = [dict(row) for row in rows]

        return {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def get_log_by_id(self, log_id: int) -> Optional[Dict[str, Any]]:
        """Get a single log record by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM processing_logs WHERE id = ?", (log_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_stats(self, period: str = "daily") -> Dict[str, Any]:
        """
        Get processing statistics for a given period.
        
        Args:
            period: 'daily', 'monthly', or 'yearly'
        
        Returns:
            Dictionary with period-based statistics
        """
        now = datetime.now()

        if period == "daily":
            # Last 30 days
            start_date = now - timedelta(days=30)
            date_format = "%Y-%m-%d"
            group_by = "date(timestamp)"
        elif period == "monthly":
            # Last 12 months
            start_date = now - timedelta(days=365)
            date_format = "%Y-%m"
            group_by = "strftime('%Y-%m', timestamp)"
        else:  # yearly
            # Last 5 years
            start_date = now - timedelta(days=365 * 5)
            date_format = "%Y"
            group_by = "strftime('%Y', timestamp)"

        with self._get_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT 
                    {group_by} as period,
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    AVG(processing_time_ms) as avg_time_ms
                FROM processing_logs
                WHERE timestamp >= ?
                GROUP BY {group_by}
                ORDER BY period DESC
                """,
                (start_date.strftime("%Y-%m-%d"),),
            ).fetchall()

            data = [dict(row) for row in rows]

        return {
            "period": period,
            "data": data,
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics: today, this month, this year, all time.
        """
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        month_start = now.strftime("%Y-%m-01")
        year_start = now.strftime("%Y-01-01")

        with self._get_connection() as conn:
            # Today
            today_count = conn.execute(
                "SELECT COUNT(*) FROM processing_logs WHERE date(timestamp) = ?",
                (today,),
            ).fetchone()[0]

            today_success = conn.execute(
                "SELECT COUNT(*) FROM processing_logs WHERE date(timestamp) = ? AND status = 'success'",
                (today,),
            ).fetchone()[0]

            # This month
            month_count = conn.execute(
                "SELECT COUNT(*) FROM processing_logs WHERE timestamp >= ?",
                (month_start,),
            ).fetchone()[0]

            month_success = conn.execute(
                "SELECT COUNT(*) FROM processing_logs WHERE timestamp >= ? AND status = 'success'",
                (month_start,),
            ).fetchone()[0]

            # This year
            year_count = conn.execute(
                "SELECT COUNT(*) FROM processing_logs WHERE timestamp >= ?",
                (year_start,),
            ).fetchone()[0]

            year_success = conn.execute(
                "SELECT COUNT(*) FROM processing_logs WHERE timestamp >= ? AND status = 'success'",
                (year_start,),
            ).fetchone()[0]

            # All time
            total_count = conn.execute(
                "SELECT COUNT(*) FROM processing_logs"
            ).fetchone()[0]

            total_success = conn.execute(
                "SELECT COUNT(*) FROM processing_logs WHERE status = 'success'"
            ).fetchone()[0]

            # Average processing time
            avg_time = conn.execute(
                "SELECT AVG(processing_time_ms) FROM processing_logs WHERE status = 'success'"
            ).fetchone()[0]

        return {
            "today": {"total": today_count, "success": today_success},
            "month": {"total": month_count, "success": month_success},
            "year": {"total": year_count, "success": year_success},
            "all_time": {"total": total_count, "success": total_success},
            "avg_processing_time_ms": round(avg_time) if avg_time else None,
        }

    def delete_old_logs(self, days: int = 365) -> int:
        """
        Delete logs older than specified days.
        
        Returns:
            Number of deleted records.
        """
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM processing_logs WHERE timestamp < ?", (cutoff,)
            )
            conn.commit()
            return cursor.rowcount


# Global instance
_logging_service: Optional[LoggingService] = None


def get_logging_service() -> LoggingService:
    """Get the global LoggingService instance."""
    global _logging_service
    if _logging_service is None:
        _logging_service = LoggingService()
    return _logging_service
