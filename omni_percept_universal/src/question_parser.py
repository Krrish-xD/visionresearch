import re

def parse_question_type(question, detected_objects=None):
    """
    Parses the critical question into a category and extracts relevant target objects.
    """
    question_lower = question.lower()
    category = "GENERAL"
    target_noun = None
    
    # Categorization heuristics
    if any(w in question_lower for w in ["how many", "number of", "count the"]):
        match = re.search(r'(?:how many|number of|count the)\s+([^?.!,]+)', question_lower)
        if match:
            category = "COUNTING"
            extracted = match.group(1).strip()
            words = extracted.split()
            stop_words = ["can", "are", "do", "is", "there", "in", "on", "the", "you"]
            filtered = []
            for w in words:
                if w in stop_words:
                    break
                filtered.append(w)
            target_noun = " ".join(filtered) if filtered else words[0]
            
            if target_noun and target_noun.endswith('s'):
                blacklist = ["bus", "gas", "glass", "lens", "cross", "boss", "fuss", "miss"]
                if target_noun.lower() not in blacklist and len(target_noun) > 3:
                    target_noun = target_noun[:-1]
    elif any(w in question_lower for w in ["is there a", "do you see a", "is there any", "can you see a", "can you see any"]):
        match = re.search(r'(?:is there a|do you see a|is there any|can you see a|can you see any)\s+([^?.!,]+)', question_lower)
        if match:
            category = "PRESENCE"
            extracted = match.group(1).strip()
            words = extracted.split()
            stop_words = ["in", "on", "the", "at", "around", "behind"]
            filtered = []
            for w in words:
                if w in stop_words:
                    break
                filtered.append(w)
            target_noun = " ".join(filtered) if filtered else words[0]
    elif any(w in question_lower for w in ["facing left", "facing right", "point left", "point right", "pointing"]):
        category = "ABSOLUTE_ORIENTATION"
    elif "facing" in question_lower or "look at" in question_lower or "looking at" in question_lower:
        category = "RELATIVE_ORIENTATION"
    elif any(w in question_lower for w in ["on the left", "on the right", "left side", "right side", "left or right", "right or left"]):
        category = "RELATIVE_LOCATION"
        subject = "object"
        reference = "main object"
        match_sub = re.search(r'(?:is the|are the|does the)\s+(.+?)\s+(?:on the left|on the right|left side|right side|left or right|right or left)', question_lower)
        if match_sub:
            raw_sub = match_sub.group(1).strip()
            subject = " ".join([w for w in raw_sub.split() if not w.endswith("'s") and not w.endswith("’s")])
            if not subject: subject = raw_sub
            
        match_ref = re.search(r'(?:of the|of)\s+([^?.!,]+)', question_lower)
        if match_ref:
            raw_ref = match_ref.group(1).strip()
            reference = " ".join([w for w in raw_ref.split() if not w.endswith("'s") and not w.endswith("’s")])
            if not reference: reference = raw_ref
            
        target_noun = {"subject": subject, "reference": reference}
    elif any(w in question_lower for w in ["color", "what kind", "material", "texture", "breed", "type"]):
        category = "ATTRIBUTE"
    elif any(w in question_lower for w in ["behind", "occluded", "cover", "hide"]):
        category = "OCCLUSION"
    elif any(w in question_lower for w in ["how far", "distance", "close", "near", "next to"]):
        category = "DISTANCE"
        
    # Extract relevant objects from detected_objects
    relevant_objects = []
    if detected_objects:
        for obj in detected_objects:
            if obj['label'].lower() in question_lower:
                relevant_objects.append(obj)
                
        # Fallback: if no objects explicitly matched the label but we have generic terms
        if not relevant_objects and any(generic in question_lower for generic in ["animal", "object", "person", "item", "it", "bird", "vehicle"]):
            relevant_objects = detected_objects
            
    return category, relevant_objects, target_noun
