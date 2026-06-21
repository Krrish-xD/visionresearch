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
        category = "COUNTING"
        match = re.search(r'(?:how many|number of|count the)\s+([^?.!,]+)', question_lower)
        if match:
            extracted = match.group(1).strip()
            words = extracted.split()
            stop_words = ["can", "are", "do", "is", "there", "in", "on", "the", "you"]
            filtered = []
            for w in words:
                if w in stop_words:
                    break
                filtered.append(w)
            target_noun = " ".join(filtered) if filtered else words[0]
    elif any(w in question_lower for w in ["is there a", "do you see a", "is there any", "can you see a", "can you see any"]):
        category = "PRESENCE"
        match = re.search(r'(?:is there a|do you see a|is there any|can you see a|can you see any)\s+([^?.!,]+)', question_lower)
        if match:
            extracted = match.group(1).strip()
            words = extracted.split()
            stop_words = ["in", "on", "the", "at", "around", "behind"]
            filtered = []
            for w in words:
                if w in stop_words:
                    break
                filtered.append(w)
            target_noun = " ".join(filtered) if filtered else words[0]
    elif any(w in question_lower for w in ["color", "what kind", "material", "texture", "breed", "type"]):
        category = "ATTRIBUTE"
    elif any(w in question_lower for w in ["facing left", "facing right", "point left", "point right", "pointing"]):
        category = "ABSOLUTE_ORIENTATION"
    elif "facing" in question_lower or "look at" in question_lower or "looking at" in question_lower:
        category = "RELATIVE_ORIENTATION"
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
            
    return category, relevant_objects, target_noun
