from typing import List, Dict
import re

def transform_text_to_list(full_text, add_lines=False):
    """
    Transforms the full text into a list of dictionaries with section numbers and texts.

    This version handles:
    - Numbering with or without a trailing dot.
    - Optional leading whitespace.
    - Roman numerals (I, II, III, etc.) and Arabic numerals.

    Args:
        full_text (str): The complete text of the document.

    Returns:
        list: A list of dictionaries with 'number' and 'text' keys.
    """
    # Preprocess the text to insert line breaks before numbering
    preprocessed_text = full_text
    if add_lines:
        preprocessed_text = re.sub(r'(?<!\n)(\b\d+(?:\.\d+)*)\.?\s+', r'\n\1. ', full_text)

    # Enhanced regex pattern with mandatory trailing dot
    pattern = r'(?m)^\s*(?P<number>(?:\d+|[IVXLCDM]+)(?:\.\d+)*)\.\s+(?P<text>.+)'

    # Find all matches in the text
    matches = list(re.finditer(pattern, preprocessed_text))

    # Initialize the list to hold the result
    result = []

    # If there is text before the first match, consider it as preamble with number '0'
    if matches:
        first_match_start = matches[0].start()
        preamble = full_text[:first_match_start].strip()
        if preamble:
            result.append({"number": "0", "text": preamble})

    # Iterate through all matches and add them to the result list
    for match in matches:
        number = match.group('number')
        text = match.group('text').strip()
        # Prepend the numbering to the text to match desired output
        full_section_text = f"{number}. {text}"
        result.append({"number": number, "text": full_section_text})

    return result


from collections import OrderedDict
#from typing import List, Dict

def group_paragraphs(paragraphs: List[Dict[str, str]], num_digits: int) -> List[Dict[str, str]]:
    """
    Groups paragraphs based on the specified number of digits in their numbering.
    Ensures that only exact matches or those starting with 'group_key.' are included in each group.
    Includes parent texts up to (num_digits - 1) levels in the concatenated 'text' field of each group.

    Args:
        paragraphs (List[Dict[str, str]]): List of paragraphs with 'number' and 'text'.
        num_digits (int): Number of digits (levels) to group by.

    Returns:
        List[Dict[str, str]]: Grouped paragraphs with 'number' and concatenated 'text'.
    """
    if not isinstance(num_digits, int) or num_digits < 1:
        raise ValueError("num_digits must be a positive integer.")
    
    # Create a mapping from 'number' to 'text' for easy parent text retrieval
    number_to_text = {para['number']: para['text'] for para in paragraphs}
    
    # Determine unique group keys based on num_digits
    group_keys = []
    for para in paragraphs:
        number_parts = para['number'].split('.')
        if len(number_parts) >= num_digits:
            group_key = '.'.join(number_parts[:num_digits])
            if group_key not in group_keys:
                group_keys.append(group_key)
    
    # Additionally, include group_keys for paragraphs with less than num_digits levels
    # but have no sub-items (i.e., no paragraphs start with their number + '.')
    for para in paragraphs:
        number = para['number']
        number_parts = number.split('.')
        if len(number_parts) < num_digits:
            # Check if no other paragraph starts with this number + '.'
            has_sub_items = any(p['number'].startswith(number + '.') for p in paragraphs)
            if not has_sub_items and number not in group_keys:
                group_keys.append(number)
    
    # Initialize an ordered dictionary to maintain the insertion order
    grouped = OrderedDict()
    
    for group_key in group_keys:
        # Determine parent keys up to (num_digits - 1) levels
        parent_keys = []
        if num_digits > 1:
            for i in range(1, num_digits):
                parent = '.'.join(group_key.split('.')[:i])
                if parent in number_to_text:
                    parent_keys.append(parent)
        
        # Retrieve parent texts
        parent_texts = [number_to_text[parent] for parent in parent_keys]
        
        # Collect paragraphs that belong to this group
        if len(group_key.split('.')) >= num_digits:
            # Group key has at least num_digits levels; include paragraphs where
            # 'number' == group_key or starts with 'group_key.'
            group_paras = [
                para['text']
                for para in paragraphs
                if para['number'] == group_key or para['number'].startswith(group_key + '.')
            ]
        else:
            # Group key has less than num_digits levels; include only the exact paragraph
            group_paras = [
                para['text']
                for para in paragraphs
                if para['number'] == group_key
            ]
        
        # Concatenate parent texts and group paragraphs
        combined_text = '\n'.join(parent_texts + group_paras)
        
        # Add the grouped data to the ordered dictionary
        grouped[group_key] = {
            'number': group_key,
            'text': combined_text
        }
    
    return list(grouped.values())