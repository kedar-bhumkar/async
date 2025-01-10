from typing import Dict, Type, Literal, Optional
from backend.core.model.pydantic_models import *
from backend.core.logging.custom_logger import logger
from backend.core.utility.constants import prompts_file
from backend.core.utility.fuzzy_matching import check_word_in_transcript
from backend.core.utility.util import *
from backend.core.utility.shared import shared_data_instance
import yaml

def ros_pe_formatter(data: ros):    
    logger.critical('Inside ros_pe_formatter ...')
    # Get field types dictionary
    field_dict = get_field_types(ros)

    # TODO: Need to optimiz this as the loops are common in these functions
    if shared_data_instance.get_data('validation_errors')!=None:
        data = fix_validation_errors(data, field_dict)        
    data = init_Reviewed_and_Negative(data, field_dict)
    data = init_fields_to_na_for_not_assessed_sections(data, field_dict)

    return data

# init field value to 'NA' if the field value is in validation_errors
def fix_validation_errors(data, field_dict):
    print(f"Inside fix_valiation_errors ...")

    validation_errors:map = shared_data_instance.get_data('validation_errors')
    print(f"***** validation_errors-{validation_errors}")

    for section in data.dict():
        section_data = getattr(data, section)
        if section!='Reviewed_with' and section!= 'additional_notes': 
            section_data = getattr(data, section)
            section_model = field_dict.get(section)
            attributes = get_pydantic_attributes(globals()[section_model])        
            for attrib in attributes:    
                if attrib in validation_errors:                                    
                    value = (validation_errors.get(attrib)).get('input_value')
                    attrib_value = getattr(section_data, attrib, None)
                    if value == attrib_value:
                        print(f"***** Correction: validation error correction for {attrib} to na")                        
                        setattr(section_data, attrib, 'na')
    return data


# init Reviewed_and_Negative to 'false' if the assessed field value is 'Assessed' and any of sub field values are either 'false' or 'true'
def init_Reviewed_and_Negative(data, field_dict):
    logger.critical(F'Inside init_Reviewed_and_Negative ...{data}')
    for section in data.dict():
        section_data = getattr(data, section)
        if section!='Reviewed_with' and section!= 'additional_notes': 
            section_model = field_dict.get(section)
            attributes = get_pydantic_attributes(globals()[section_model])
            assessed_field_name = find_assessed_field(globals()[section_model], Literal["Assessed", "Not Assessed"])
            if assessed_field_name!=None:
                assessed_field_value = getattr(section_data,assessed_field_name,None)      
                if assessed_field_value!=None and (assessed_field_value=='Not Assessed' or assessed_field_value.casefold()=='notassessed'):
                    print(f"***** Correction: Setting Reviewed_and_Negative to true for {section.casefold()} - {assessed_field_name.casefold()}")
                    setattr(section_data, 'Reviewed_and_Negative', 'na')                
                elif assessed_field_value!='None':                
                    for attrib in attributes:                    
                        field_value =   getattr(section_data,attrib,None)  
                        if field_value!=None and field_value.casefold() !='na' and attrib != 'Reviewed_and_Negative' and section.casefold() not in attrib.casefold():
                            print(f"***** Correction:  Setting Reviewed_and_Negative to false for {section.casefold()} - {attrib.casefold()}")
                            setattr(section_data, 'Reviewed_and_Negative', 'false')
                            break
    return data                    

# init field value to 'NA' if the assessed field value is 'Not Assessed' and the field value is not 'NA'
def init_fields_to_na_for_not_assessed_sections(data, field_dict):
    for section in data.dict():
        section_data = getattr(data, section)          
        if section!='Reviewed_with' and section!= 'additional_notes':           
            section_model = field_dict.get(section)
            attributes = get_pydantic_attributes(globals()[section_model])
            assessed_field_name = find_assessed_field(globals()[section_model], Literal["Assessed", "Not Assessed"])
            if assessed_field_name!=None: 
                assessed_field_value = getattr(section_data,assessed_field_name,None)      
                for attrib in attributes:                    
                    field_value =   getattr(section_data,attrib,None)                                            
                    if(field_value!=None and field_value!='NA'):                                           
                        if(assessed_field_value == "Not Assessed"): 
                            if(field_value == "false"):
                                print(f"***** Correction: Setting field value of {attrib} to 'NA'for a 'Not assessed' section")
                                setattr(section_data, attrib, 'NA')     
                    elif(assessed_field_value == "Not Assessed" and attrib == 'Not_Assessed_Reason' and field_value == None):    
                                print(f"***** Correction: Setting 'Not_Assessed_Reason' to 'Other'")
                                setattr(section_data, 'Not_Assessed_Reason', 'Other')   

    shared_data_instance.set_data('confidence_map', '')
    return data

# Function to dynamically fetch all attributes of a Pydantic class
def get_pydantic_attributes(model: BaseModel) -> Dict[str, str]:
    print(f"***** Inside get_pydantic_attributes")
    print(f"***** model- {model.model_fields.items()}")
    return {field_name: field for field_name, field in model.model_fields.items()}


# Function to dynamically get and return field types as a dictionary
def get_field_types(model: BaseModel) -> Dict[str, str]:


    field_types = {}
    for field_name, field in model.model_fields.items():
        field_type = field.annotation
        if hasattr(field_type, "__name__"):
            field_type_name = field_type.__name__
        else:
            field_type_name = str(field_type)
        field_types[field_name] = field_type_name
    return field_types


# Function to find the field with the specified type
def find_assessed_field(cls: Type[BaseModel], literal_type: Literal["Assessed", "Not Assessed"]) -> Optional[str]:
    for field_name, field in cls.model_fields.items():
        print(f"***** field_name- {field_name}")
        print(f"***** field- {field}")
        print(f"***** field.annotation- {field.annotation}")
        print(f"***** literal_type- {literal_type}")
        
        if field.annotation == literal_type:
            return field_name
    return None


field_dict = get_field_types(ros)
section_model = field_dict.get('CONSTITUTIONAL')
#print(f'section_model-{section_model}')
#attributes = get_pydantic_attributes(globals()[section_model])
##print(attributes)

# Literal type to check
literal_type = Literal["Assessed", "Not Assessed"]
#field_name = find_assessed_field(Constitutional, literal_type)
##print(f"field_name -{field_name}")


def getConfig(file_path):
    # Define the path to the YAML file
    yaml_file_path = file_path

    # Read the YAML file
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
