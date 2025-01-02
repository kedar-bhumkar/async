import spacy

# Load the English language model
nlp = spacy.load("en_core_web_lg")

def remove_phi_pii_spacy(text: str) -> str:
    """
    Remove PHI/PII from text using spaCy's named entity recognition.
    """
    # Process the text
    doc = nlp(text)
    
    # Define entity mappings (spaCy's entity labels to our replacement text)
    entity_mappings = {
        'PERSON': 'PERSON',
        'GPE': 'LOCATION',    # Countries, cities, states
        'LOC': 'LOCATION',    # Non-GPE locations
        'FAC': 'LOCATION',    # Buildings, airports, highways, etc.
        'ORG': 'ORGANIZATION',
        'CARDINAL': 'NUMBER',
        'DATE': 'DATE',
        'TIME': 'TIME',
        'MONEY': 'MONEY',
    }
    
    # Sort entities by start position (reversed to process from end to start)
    entities = sorted(doc.ents, key=lambda x: x.start_char, reverse=True)
    
    # Create a mutable version of the text
    modified_text = text
    
    # Replace entities with their anonymous versions
    for ent in entities:
        if ent.label_ in entity_mappings:
            replacement = entity_mappings[ent.label_]
            modified_text = (
                modified_text[:ent.start_char] +
                replacement +
                modified_text[ent.end_char:]
            )
    
    return modified_text

# Example usage
if __name__ == "__main__":
    #test_text = """John Smith lives in New York and works at Microsoft. 
    #His phone number is 555-123-4567 and he earns $150,000 per year."""

    test_text = ''' Transcript: Reviewed the following with the member on 29-07-2024.  Geriatric syndrome was assessed. member reports overall health to be Very good No change in Self-assessed mental health Pain assessment completed verbally. Verbal pain scale reported as  0 Constitutional Reviewed and negative. Eyes Assessed.  members Uses glasses Nose and throat Assessed.  member reports  difficulty swallowing. Respiratory was reviewed and negative. Cardiovascular was Reviewed and negative. Gastrointestinal was reviewed and negative. Genitourinary was Assessed.  member reports  difficulty urinating. Cognitive impairment was not seen NEUROLOGICAL was assessed .  The member said she has had Numbness and tingling with Prickling sensationIt feels like Pins and needles Musculoskeletal  assessed and gait disturbances were seen. Reports history of fractures on Left femur.  The last fracture was in 1996. member informed that he is Non-diabetic. Endocrine was assessed.  The patient has hot and cold intolerance and has excessive thirst and hunger Psychological assessment was done.  She Reports depression.  Manages it with activities. Some additional notes about the member - He was once incacerated in jail for 2 weeks. {missing_sections} 
    '''
    
    cleaned_text = remove_phi_pii_spacy(test_text)
    print(cleaned_text)
