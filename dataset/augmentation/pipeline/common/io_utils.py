#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IO utility functions"""

import json
import os
from typing import List, Dict, Any


def load_json(path: str) -> List[Dict[str, Any]]:
    """
    Load JSON file
    
    Args:
        path: JSON file path
        
    Returns:
        JSON data (must be list format)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError("Input JSON must be a list of dictionaries.")


def save_json(path: str, data: List[Dict[str, Any]]):
    """
    Save JSON file (using temp file for atomicity)
    
    Args:
        path: Output file path
        data: Data to save (list format)
    """
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
