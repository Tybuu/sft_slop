def format_target(target_raw: str) -> str:
    target = target_raw.strip()
    
    # Ensure it starts with Thought:
    if not target.lower().startswith("thought:"):
        target = f"Thought: {target}"
        
    # Ensure Action: starts on a new line
    target = target.replace("Action:", "\nAction:")
    target = target.replace("Action Input:", "\nAction Input:")
    
    # Fix missing JSON brackets in Action Input
    # If Action Input is followed by a " but no {, we wrap it.
    if "Action Input:" in target:
        parts = target.split("Action Input:")
        new_target = parts[0]
        for part in parts[1:]:
            content = part.strip()
            if content and not content.startswith("{"):
                # Wrap the rest of the line or string in brackets
                # This is a bit heuristic but works for most cases
                new_target += "Action Input: {" + content + "}"
            else:
                new_target += "Action Input: " + part
        target = new_target

    # Fix double newlines
    while "\n\n" in target:
        target = target.replace("\n\n", "\n")
        
    return target
