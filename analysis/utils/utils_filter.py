import re

class EntityNormalizer:
    def __init__(self, prefix_rules=None, suffix_rules=None):
        """
        Initialize the cleaner
        :param prefix_rules: Prefix regular list (loaded from configuration)
        :param suffix_rules: Suffix regular list (loaded from configuration)
        """
        self.prefix_rules = prefix_rules if prefix_rules else []
        self.suffix_rules = suffix_rules if suffix_rules else []
        self.QUALITY_LIMIT = 100
        
        # Define the reject words set (hardcoded universal noise)
        self.reject_words = {
            "none", "n/a", "unknown", "null", "nil", "not available", "no answer"
        }

    def _universal_clean(self, text):
        """
        [Generic method] Revised version: use white list mode, thoroughly clean parentheses and quotes
        """
        if not text:
            return None
            
        # 0. [New feature] Line break check
        # If the entity contains a line break, it means that multiple lines of text have been extracted, which is usually incorrect
        if "\n" in text or "\r" in text:
            return None
        # 1. Convert to lowercase
        text = text.lower()
        
        # 2. Reject word check
        if text.strip() in self.reject_words:
            return None
            
        # 3. [Revised logic] White list cleaning
        # Explanation:
        # [^\w\s\.\-] matches all "not" (word characters, spaces, dots, hyphens) things
        # That is, all parentheses (), quotes "", asterisks * etc. are replaced with empty
        text = re.sub(r'[^\w\s\.\-]', '', text)
        
        # 4. Normalize whitespace (to prevent leaving extra spaces after deleting punctuation)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _specific_clean(self, text):
        """
        [Special method] Depend on external configuration for prefix and suffix processing
        """
        if not text:
            return None

        # 1. Process prefixes (Prefixes)
        for pattern in self.prefix_rules:
            # Execute replacement
            new_text = re.sub(pattern, "", text).strip()
            # Only when the remaining part length > 1 is effective (to prevent deleting "A" like this single word)
            if len(new_text) > 1:
                text = new_text

        # 2. Process suffixes (Suffixes)
        for pattern in self.suffix_rules:
            # Execute replacement
            new_text = re.sub(pattern, "", text).strip()
            # [Safe check]: Only when the remaining length > 2 after removing the suffix is accepted
            # To prevent: "Gap Inc" -> "Gap" (OK), "IT Inc" -> "IT" (Dangerous), "A Inc" -> "A" (Rejected)
            if len(new_text) > 2:
                text = new_text
            
        return text

    def normalize(self, raw_text):
        """
        Main entry function
        """
        if raw_text == "NOT_ATTEMPTED":
            return "not_attempted"
        # First stage: Universal cleaning
        text = self._universal_clean(raw_text)
        # If the universal cleaning is already None, return directly
        if not text:
            return None
        if len(text) > self.QUALITY_LIMIT:
            return None
            
        # Second stage: Specific cleaning (load configuration)
        text = self._specific_clean(text)
        
        return text

# ==========================================
#  Configuration Area
# ==========================================

# 1. Prefix rules (regular list)
# Note: Regular usually starts with ^, \s+ means followed by space
MY_PREFIXES = [
    r"^dr\.?\s+",          # Dr. / Dr
    r"^prof\.?\s+",        # Prof.
    r"^mr\.?\s+",          # Mr.
    r"^mrs\.?\s+",         # Mrs.
    r"^sir\s+",            # Sir
    r"^the\s+",            # The (Universal article)
    r"^machine for\s+",    # For your patent problem: "Machine for..."
    r"^method of\s+"       # For patent/academic problem: "Method of..."
]

# 2. Suffix rules (regular list)
# Note: Regular usually ends with $, \s+ means preceded by space
MY_SUFFIXES = [
    r"\s+inc\.?$",         # Inc. / Inc
    r"\s+ltd\.?$",         # Ltd.
    r"\s+corp\.?$",        # Corp.
    r"\s+project$",        # Project (For your Hadoop case)
    r"\s+system$",         # System (For machine/software)
    r"\s+group$",          # Group
]

# ==========================================
#  Run demonstration (Execution)
# ==========================================

if __name__ == "__main__":
    # Instantiate the cleaner, inject configuration
    normalizer = EntityNormalizer(
        prefix_rules=MY_PREFIXES, 
        suffix_rules=MY_SUFFIXES
    )

    # Simulate some dirty data
    test_data = [
        "NOT_ATTEMPTED",
        "  The Berlin Zoo  ",           # -> berlin zoo (Remove spaces + Remove The)
        "Dr. Michael S. Waterman",      # -> michael s. waterman (Remove title)
        "Apache Hadoop Project",        # -> apache hadoop (Remove suffix)
        "Machine for Making Bags",      # -> making bags (Remove specific machine prefix)
        "(3350) Kapitsa",               # -> 3350 kapitsa (Remove punctuation)
        "Gap Inc.",                     # -> gap (Safe removal)
        "My Inc.",                      # -> my inc. (Because there are only 2 letters left, the protection mechanism is triggered, not removed)
        "unknown"                       # -> None
    ]
    import json
    with open("/data3/xuhaoming/Confidence/confidence/attack/data/better_extract_responded_with_entities.json", "r") as f:
        data = json.load(f)
    
    for item in data:
        new_data = item.get("original_extracted_entities", [])
        test_data.extend(new_data)

    print(f"{'Original':<30} | {'Cleaned Entity':<30}")
    print("-" * 65)
    
    for raw in test_data:
        cleaned = normalizer.normalize(raw)
        # To make it look good, None is converted to string
        display = cleaned if cleaned else "NULL"
        if raw.lower() != display.lower() and raw != "NOT_ATTEMPTED":
            print(f"{raw:<10} | {display:<10}")