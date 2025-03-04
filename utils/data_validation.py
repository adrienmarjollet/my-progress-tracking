import json
from typing import Dict, Any, Optional

def validate_metadata(metadata: str) -> Optional[Dict[str, Any]]:
    """
    Validates metadata JSON string and ensures it has required fields.
    
    Args:
        metadata: JSON string containing question metadata
        
    Returns:
        Parsed metadata dictionary or None if invalid
    """
    try:
        if not metadata:
            return None
            
        metadata_dict = json.loads(metadata)
        
        # Validate difficulty if present
        if 'difficulty' in metadata_dict:
            difficulty = metadata_dict['difficulty']
            
            # Ensure difficulty is a number
            if not isinstance(difficulty, (int, float)):
                return None
                
            # Ensure difficulty is within reasonable bounds (assuming 1-10 scale)
            if difficulty < 0 or difficulty > 10:
                metadata_dict['difficulty'] = max(0, min(10, difficulty))
        
        return metadata_dict
        
    except (json.JSONDecodeError, TypeError):
        return None
