from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_analyzer.nlp_engine import SpacyNlpEngine, NerModelConfiguration
from backend.core.utility.LLMConfig import LLMConfig

# Initialize Presidio Analyzer and Anonymizer
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def remove_phi_pii_presidio(text: str) -> str:
    
    # Specify the entities you want to detect, excluding 'DATE' and 'US_SOCIAL_SECURITY_NUMBER'
    #entities_to_detect = ["PERSON","PHONE_NUMBER", "EMAIL_ADDRESS","LOCATION","US_SSN","CREDIT_CARD","ADDRESS"]  # Add other entities as needed
    #allow_list = ["Prickling", "numbness", "tingling"]

    entities_to_detect = LLMConfig.get_config()['phi_allowed_entities']
    allow_list = LLMConfig.get_config()['phi_allowed_list']
    

    print(f"entities_to_detect-{entities_to_detect}")
    print(f"allow_list-{allow_list}")
    # Define which model to use
    model_config = [{"lang_code": "en", "model_name": "en_core_web_lg"}]

    ner_model_configuration = NerModelConfiguration(default_score = 0.95)

    # Create the NLP Engine based on this configuration
    spacy_nlp_engine = SpacyNlpEngine(models= model_config, ner_model_configuration=ner_model_configuration)

    analyzer = AnalyzerEngine(nlp_engine=spacy_nlp_engine)    



    # Analyze the text to find PII entities
    results = analyzer.analyze(text=text.lower(), language="en", entities=entities_to_detect, allow_list=allow_list)
    print(results)
    
    if not results:
        return text
        
    # Create operator config for each entity type
    operators = {
        "PERSON": OperatorConfig("replace", {"new_value":"PERSON"}),
        "PHONE_NUMBER": OperatorConfig("replace", {"new_value":"PHONE"}),
        "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value":"EMAIL"}),       
        "LOCATION": OperatorConfig("replace", {"new_value":"LOCATION"}),
        "US_SSN": OperatorConfig("replace", {"new_value":"SSN"}),
        "CREDIT_CARD": OperatorConfig("replace", {"new_value":"CREDIT_CARD"}),
        "ADDRESS": OperatorConfig("replace", {"new_value":"ADDRESS"})
    }

    # Anonymize the text
    try:
        result = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators
        )
        return result.text
    except Exception as e:
        print(f"Anonymization error: {e}")
        return text

# Example usage
if __name__ == "__main__":
    #text = """ Visit Type is Comprehensive Member New or Established is New Date of Visit is 05/18/2023 Place of Service is Assisted Living Facility How was visit Completed is Face to Face CC/Reason for Program Admission is Follow up on chronic conditions and management HPI is Member seen today for an initial comprehensive visit. He lives alone in an independent senior apartment. He has two cats for company. His son lives near by but he doesn't see him or talk often but does talk more often to his daughter in law. Denies falls in over 2 years. 1. HTN, CKD3A, afib, secondary hypercoagulability, anemia in CKD- continues on iron and vitamin C daily, coumadin managed by Holy Family coumadin clinic 2. Moderate protein calorie malnutrition, esophageal obstruction, hiatal hernia, vitamin D deficiency- swallowing ok now, has had dilation procedures in the past, continues on famotidine and vitamin d supplements. 3. BPH, obstructive uropathy- continues to self cath several times daily. 4. MDD, insomnia, RLS- manages depressive symptoms by planning daily outing on the bus. RLS managed with gabapentin, insomnia with melatonin with good effect. 5. OA- infrequent pain, but acetaminophen is effective when needed. Uses cane in the apartment and walker when out of apartment. No falls in more than 2 years.
    #  """

    text = ''' Transcript: Reviewed the following with the Mr. John Hill on 29-07-2024.  Geriatric syndrome was assessed. member reports overall health to be Very good No change in Self-assessed mental health Pain assessment completed verbally. Verbal pain scale reported as  0 Constitutional Reviewed and negative. Eyes Assessed.  members Uses glasses Nose and throat Assessed.  member reports  difficulty swallowing. Respiratory was reviewed and negative. Cardiovascular was Reviewed and negative. Gastrointestinal was reviewed and negative. Genitourinary was Assessed.  member reports  difficulty urinating. Cognitive impairment was not seen NEUROLOGICAL was assessed .  The member said she has had Numbness and tingling with Prickling sensationIt feels like Pins and needles Musculoskeletal  assessed and gait disturbances were seen. Reports history of fractures on Left femur.  The last fracture was in 1996. member informed that he is Non-diabetic. Endocrine was assessed.  The patient has hot and cold intolerance and has excessive thirst and hunger Psychological assessment was done.  She Reports depression.  Manages it with activities. Some additional notes about the member - He was once incacerated in jail for 2 weeks. 
    '''
    
    cleaned_text = remove_phi_pii_presidio(text)
    print(cleaned_text)