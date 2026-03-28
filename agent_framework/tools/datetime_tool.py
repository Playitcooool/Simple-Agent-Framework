"""
DateTime tool — get current date and time.
"""

from dataclasses import dataclass
from datetime import datetime
import os


@dataclass
class DateTimeTool:
    """
    Get current date and time.

    Attributes:
        timezone: Timezone name (e.g., 'America/New_York', 'Asia/Shanghai').
                  If None, uses local time.
    """

    timezone: str = None

    def run(self, format: str = "%Y-%m-%d %H:%M:%S", timezone: str = None) -> str:
        """
        Get current datetime.

        Args:
            format: strftime format string (default: '%Y-%m-%d %H:%M:%S')
            timezone: Timezone name (overrides self.timezone if provided)

        Returns:
            Formatted datetime string or error message
        """
        tz = timezone or self.timezone

        try:
            if tz:
                # Try to get timezone-aware datetime
                os.environ['TZ'] = tz
                import time
                time.tzset()
                now = datetime.now()
                # Reset to local
                os.environ['TZ'] = time.tzname[0] if time.tzname else ''
            else:
                now = datetime.now()

            return now.strftime(format)

        except Exception as e:
            return f"[Error: {e}]"


# Decorator-based tool
def datetime_tool(format: str = "%Y-%m-%d %H:%M:%S", timezone: str = None) -> str:
    """Get current datetime. Args: format (str), timezone (str, optional)."""
    tool = DateTimeTool()
    return tool.run(format, timezone)
