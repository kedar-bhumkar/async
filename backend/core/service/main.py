from openai import  OpenAI
import time
from backend.core.logging.custom_logger import *
from backend.core.model.pydantic_models import *
from backend.core.utility.constants import *
from backend.core.utility.util import *
from backend.core.db.db_layer import *
from backend.core.db.db_stats import *
from backend.core.utility.fuzzy_matching import *
from backend.core.utility.phi_remover import *
from backend.core.utility.shared import *
from backend.core.utility.LLMConfig import *
import pandas as pd
from io import StringIO


clientSync = any
theModel = any
theLlmParameters = any
theFormatter = None
thePrompt = None
theSystemPrompt= None
theUserPrompt = None
run_mode = None
run_count= None
run_id = None
theIdealResponse = None
accuracy_check = None
test_map = {}

# Serial

db_data= []
i_data={}




def generate(client,count,prompt,page):    
    print("Entering method: generate")
    input_token_count = num_tokens_from_string(''.join([theSystemPrompt, prompt]), 'cl100k_base', "input")

    completion_params = {
       "messages": [
           {"role": "system", "content": theSystemPrompt},
           {"role": "user", "content": prompt}
       ],
       "model": theModel
   }

    completion_params.update(theLlmParameters)
    print(f"modified completion_params-{completion_params}")
   
   
    start = time.perf_counter()
    chat_completion = client.chat.completions.create(**completion_params)   
    
    llm_latency = time.perf_counter() - start
    print(f"Time taken to get real LLM response - {llm_latency:0.2f} seconds.")

    response = chat_completion.choices[0].message.content
    
    print(f"chat_completion-total_tokens-{chat_completion.usage.total_tokens}")
    print(f"chat_completion-completion_tokens-{chat_completion.usage.completion_tokens}")
    print(f"chat_completion-prompt_tokens-{chat_completion.usage.prompt_tokens}")
    print(f"chat_completion-prompt_tokens_details_cached_tokens-{chat_completion.usage.prompt_tokens_details.cached_tokens}")
    print(f"chat_completion.system_fingerprint-{chat_completion.system_fingerprint}")

    shared_data_instance.set_data('fingerprint', chat_completion.system_fingerprint)
    output_token_count = num_tokens_from_string(response,'cl100k_base', "output")
    shared_data_instance.set_data('input_token_count', input_token_count)
    shared_data_instance.set_data('output_token_count', output_token_count)
    shared_data_instance.set_data('llm_latency', llm_latency)

    return response
 
# Serial
def generate_serially( page, prompt):  
    print("Entering method: generate_serially")
    global clientSync, thePrompt     
    thePrompt = prompt_constrainer(page,prompt,-1)
    return [generate(clientSync, 0, thePrompt, page)]


def init_AI_client(model_family, model):
    print("Entering method: init_AI_client")
    global clientSync, theModel,theLlmParameters
    theModel = model

    llm_config = LLMConfig.get_config()
    theLlmParameters = json.loads(llm_config['parameters'])  

    config = getConfig(config_file)
    print(f"config-{config}")

    clientSync = OpenAI(
        api_key  = config[model_family]['key'],
        base_url = config[model_family]['url'],
    ) 

def init_prompts(usecase, page):
    print("Entering method: init_prompts")
    global thePrompt,theSystemPrompt,theIdealResponse,theUserPrompt
    config = getConfig(prompts_file)     

    theSystemPrompt = get_system_prompt(usecase, page)
    theUserPrompt = get_user_prompt(usecase, page)
    print(f"shared_data_instance.get_data('request_type')-{shared_data_instance.get_data('request_type')}")
    if(shared_data_instance.get_data('request_type') != "api"):
        theIdealResponse = config[usecase]['user_prompt'][page]['serial']['ideal_response']
    else:
        theIdealResponse = shared_data_instance.get_data('theIdealResponse')
  

def prompt_constrainer(page,thePrompt, count=None):
    print("Entering method: prompt_constrainer")
    global theUserPrompt
    negativePrompt = ''
    thePrompt = replace_dates(thePrompt)
    
    #remove phi/pii    
    if(shared_data_instance.get_data('phi_detection') == True):
        print(f"Inside phi detection ")
        thePrompt = remove_phi_pii_presidio(thePrompt)
    else:
        print(f"phi detection is off")
    
    sharedPrompt = thePrompt
    print(f"sharedPrompt-{sharedPrompt}")
    #Create copy so that we just get what the user said minus constraints 
    shared_data_instance.set_data('thePrompt', sharedPrompt)   
    
    try:
        cls = globals()[page]
    except:
        logger.info(f"No pydantic model defined")
    else:    
        logger.info(f"pydantic model defined")
        response_schema_dict = cls.model_json_schema()
        response_schema_json = json.dumps(response_schema_dict, indent=2)         
                
        if(shared_data_instance.get_data('negative_prompt')== True):        
            negativePrompt = fuzzyMatch(thePrompt)   
            logger.critical(f"negativePrompt-{negativePrompt}")                 
            thePrompt = theUserPrompt.format(constraints=response_schema_json,transcript=thePrompt,missing_sections=negativePrompt)    
        else:
            thePrompt = theUserPrompt.format(constraints=response_schema_json,transcript=thePrompt,missing_sections='')    
        
        print(f"theNewPrompt ****** -{thePrompt}")
        
    finally:
        return thePrompt
    



def sync_async_runner(message:Message):    
    print("Entering method: sync_async_runner")
    global theFormatter, i_data, db_data
    db_data = []
    

    theFormatter = message.formatter
    
    init_AI_client(message.family, message.model)
    print(f"Calling  init_prompts")
    init_prompts(message.usecase, message.page)

    if (message.mode == "serial" or  message.mode == "dual"):   
        # Serial invoker
        start = time.perf_counter()
        response = generate_serially(message.page, message.prompt)        
        end = time.perf_counter() - start        
        response = response[0]
        
        logger.info(f"Serial Program finished in {end:0.2f} seconds.")
    # Parallel invoker code not supported anymore
    
    if(message.run_mode !=None):
        response = log(message, response, end)
        #logger.critical(f"f-response-{response}")
   
    time.sleep(float(message.sleep))
  
    return response


def init_test_result(message:Message, result,formatted_ideal_response,formatted_real_response):
    print("Entering method: init_test_result")
    
    test_result = {}
    test_result['matches_idealResponse'] = result.is_match
    test_result['idealResponse_changes'] = result.changes
    test_result['accuracy_difflib_similarity'] = result.metrics
    test_result['matched_tokens'] = result.matched_tokens
    test_result['mismatched_tokens'] = result.mismatched_tokens
    test_result['mismatch_percentage'] = round(result.mismatch_percentage,2)
    test_result['ideal_response'] = formatted_ideal_response
    test_result['actual_response'] = formatted_real_response
    test_result['original_response'] = shared_data_instance.get_data('original_response')  
    test_result['original_run_no'] = shared_data_instance.get_data('original_run_no')
    test_result['original_prompt'] = shared_data_instance.get_data('original_prompt')
    test_result['fingerprint'] = shared_data_instance.get_data('fingerprint')
    test_result['page'] = message.page
    test_result['status'] = 'success'
    
    return test_result

def init_run_stats(message:Message,result:ComparisonResult,time,formatted_ideal_response,formatted_real_response):    
    print("Entering method: init_run_stats")
    global run_id, theSystemPrompt,thePrompt,theModel
    _prompt = shared_data_instance.get_data('thePrompt')
    print(f"thePrompt-{_prompt}")
    print(f"formatted_ideal_response-{formatted_ideal_response}")
    print(f"formatted_real_response-{formatted_real_response}")
    run_stats = {
            'usecase': message.usecase,
            'mode':message.mode,
            'functionality':message.page,
            'llm':theModel,
            'llm_parameters': LLMConfig.get_config()['parameters'],
            'isBaseline': True,
            'run_no': run_id,
            'system_prompt': theSystemPrompt,
            'user_prompt': _prompt,
            'response': formatted_real_response,
            'ideal_response':formatted_ideal_response,
            'execution_time': time,
            'matches_baseline': True,
            'matches_ideal':result.is_match,
            'difference': '',
            'ideal_response_difference': result.changes,
            'similarity_metric':result.metrics,
            'use_for_training': shared_data_instance.get_data('use_for_training'),
            'fingerprint': shared_data_instance.get_data('fingerprint'),
            'input_token_count': shared_data_instance.get_data('input_token_count'),
            'output_token_count': shared_data_instance.get_data('output_token_count'),
            'llm_latency': round(shared_data_instance.get_data('llm_latency'),2)
        }

    return run_stats

def log(message:Message,response, time):    
    print("Entering method: log")
    #logger.critical(f"logging in db ...mode={mode}, response ={response}")
    global theFormatter, i_data, db_data, run_id,theIdealResponse,thePrompt,test_map    
    run_mode = shared_data_instance.get_data('run_mode')    
    formatted_ideal_response = ""
    
    formatted_real_response = get_Pydantic_Filtered_Response(message.page,response,theFormatter,response_type='actual')
    print(f"theIdealResponse-{theIdealResponse}")
    if(theIdealResponse != None and  theIdealResponse != ''):
        print("Computing ideal response")
        formatted_ideal_response = get_Pydantic_Filtered_Response(message.page,theIdealResponse, None)       


    result = compare(formatted_ideal_response, formatted_real_response)
    shared_data_instance.set_data('theIdealResponse', formatted_ideal_response)

    if(run_mode == 'cli-test-llm' or run_mode == 'eval-test-llm'):
         logger.info(f"Running {run_mode} test")         
         print(f"key while saving-{shared_data_instance.get_data('run_no')}")
         test_map[shared_data_instance.get_data('run_no')] = init_test_result(message,result,formatted_ideal_response,formatted_real_response)
    else:
        i_data = init_run_stats(message,result,time,formatted_ideal_response,formatted_real_response)    
        db_data.append(i_data)
    
    return formatted_real_response



def init_defaults(message: Message):
    message.mode = message.mode or LLMConfig.get_default("mode")
    message.run_mode = message.run_mode or LLMConfig.get_default("run_mode")
    message.family = message.family or LLMConfig.get_default("family")
    message.formatter = message.formatter or LLMConfig.get_default("formatter")
    message.run_count = message.run_count or LLMConfig.get_default("run_count")
    message.accuracy_check = message.accuracy_check or LLMConfig.get_default("accuracy_check")
    message.use_for_training = message.use_for_training or LLMConfig.get_default("use_for_training")
    message.error_detection = message.error_detection or LLMConfig.get_default("error_detection")
    message.phi_detection = message.phi_detection or LLMConfig.get_default("phi_detection")
    message.negative_prompt = message.negative_prompt or LLMConfig.get_default("negative_prompt")
    message.sleep = message.sleep or LLMConfig.get_default("sleep")
    message.model = message.model or LLMConfig.get_default("model")
    message.prompt = message.prompt or LLMConfig.get_default("prompt")
    message.ideal_response = message.ideal_response or None
    message.usecase = message.usecase or LLMConfig.get_default("usecase")
    message.page = message.page or LLMConfig.get_default("page")
    message.test_size_limit = message.test_size_limit or 10
    logger.critical(f"message-{message}")
    return message

def process_request(message: Message):
    global run_id, theIdealResponse, test_map, db_data
    message = init_defaults(message)
    
    # Set shared data
    shared_data = {
        'negative_prompt': message.negative_prompt,
        'use_for_training': message.use_for_training,
        'error_detection': message.error_detection,
        'run_mode': message.run_mode,
        'phi_detection': message.phi_detection,
        'theIdealResponse': message.ideal_response,
        'prompt': message.prompt
    }    


    for key, value in shared_data.items():
            shared_data_instance.set_data(key, value)

    run_id = getRunID(8)
    config = getConfig(prompts_file)    

    # if file_name is provided, process each prompt in the file in serial mode on same llm
    if message.file_name is not None:
        return _process_prompts_via_file(message)

    else:
        # if file_name is not provided, process the prompt based on below logic        
        if message.prompt is None and (message.run_mode != "cli-test-llm" or  message.run_mode != "eval-test-llm"):
            prompt = config[message.usecase]['user_prompt'][message.page][message.mode]['input']
            if isinstance(prompt, str):
                    prompt = add_space_after_punctuation(prompt)     
        

        # Process based on run mode
        if message.run_mode == "multiple-llm":
            return _process_multiple_llm(message)

        elif message.run_mode == "cli-test-llm":
            return _process_test_llm(message)

        elif message.run_mode == "eval-test-llm":
            return _process_eval_test_llm(message)

        elif message.run_mode == "bulk_transcript":
            return _process_bulk_transcript(message)

        else:  # same-llm mode
            return _process_same_llm(message)    

def _process_prompts_via_file(message:Message):
    df_prompts = load_prompt_from_file(message.file_name)        
    if df_prompts is not None:
        for index, row in df_prompts.iterrows():
            prompt = row['Transcript']
            if isinstance(prompt, str):
                prompt = add_space_after_punctuation(prompt)
                _process_same_llm(message)
    else:
        logger.error(f"No prompts found in the file: {message.file_name}")

def _process_multiple_llm(message:Message):
    """Handle multiple LLM processing mode"""
    result = parse_models(getConfig(config_file))
    responses = []
    
    for model_family, models in result.items():
        for model in models:
            response = sync_async_runner(message)
            responses.append(response)
    
    return responses

def _process_test_llm(message:Message):
    """Handle test LLM processing mode"""
    result = get_test_data(message.test_size_limit,message.consistency_request.page)
    
    # Check if result is empty or None
    if result is None or result.empty:
        logger.error(f"No test data found for page: {message.page}")
        raise ValueError(f"No test data available for page: {message.page}")
    
    for count, row in result.iterrows():
        logger.critical(f"Running test {count+1} of total {len(result)}")
        if row['user_prompt'] is not None:
            # hack visit later
            row['ideal_response'] = row['response']
            _setup_test_data(row)
            prompt = ''.join([
                'Return_data_constraints: {constraints} ',
                row['user_prompt'],
                '{missing_sections}'
            ])
            start_time = time.time()
            message.prompt = prompt
            message.page = row['functionality']
            message.usecase = row['usecase']
            sync_async_runner(message)
            print(f"key while extracting-{shared_data_instance.get_data('run_no')}")
            test_result =test_map[shared_data_instance.get_data('run_no')] 
            test_result['execution_time'] = round(time.time() - start_time, 2)
            test_result['llm_latency'] = shared_data_instance.get_data('llm_latency')

    return _generate_test_summary('consistency','consistency-eval-test')


def _process_eval_test_llm(message:Message):
    print("Entering method: _process_eval_test_llm")
    
    eval_file_data = message.eval_request.csv_data
    print(f"eval_file_data-{eval_file_data}")
    if not eval_file_data:
        logger.error("No evaluation file data found")
    try:
        # Convert to string if it's bytes
        if isinstance(eval_file_data, bytes):
            eval_file_data = eval_file_data.decode('utf-8-sig')
     

            
        try:
            #df = pd.read_csv(StringIO(decoded_string))
            print("reading csv")
            df = pd.read_csv(StringIO(eval_file_data))
            print("reading csv done")
        except Exception as e:
            logger.error(f"CSV parsing error: {str(e)}")
            return {"error": f"Could not parse CSV data: {str(e)}"}
        
        # Validate required columns
        required_columns = ['user_prompt', 'ideal_response']
        if not all(col in df.columns for col in required_columns):
            logger.error("CSV file missing required columns: user_prompt, ideal_response")
            return
        
        results = []
        total_rows = len(df)
        print(f"total_rows-{total_rows}")
        exit
        for index, row in df.iterrows():
            logger.critical(f"Processing evaluation {index + 1} of {total_rows}")
            
            if pd.notna(row['user_prompt']):
                # Construct prompt
                prompt = ''.join([
                    'Return_data_constraints: {constraints} ',
                    row['user_prompt'],
                    '{missing_sections}'
                ])
                row['run_no'] = index
                _setup_test_data(row)         
                # Execute test
                start_time = time.time()
                message.prompt = prompt
                message.page = message.eval_request.page                
                sync_async_runner(message)
                
                # Store test results
                test_result = test_map[shared_data_instance.get_data('run_no')]
                test_result['llm_latency'] = round(shared_data_instance.get_data('llm_latency'),2)
                test_result['execution_time'] = round(time.time() - start_time, 2)
                results.append(test_result)
        eval_name = message.eval_request.evalName    
        print(f"eval_name-{eval_name}")
        return _generate_test_summary('eval', eval_name)
    except Exception as e:
        logger.error(f"Error processing evaluation file: {str(e)}")
        return {"error": str(e)}


def _process_bulk_transcript(message:Message):
    print("Entering method: _process_bulk_transcript")
    
    bulk_file_data = message.eval_request.csv_data
    print(f"bulk_file_data-{bulk_file_data}")
    if not bulk_file_data:
        logger.error("No evaluation file data found")
    try:
        # Convert to string if it's bytes
        if isinstance(bulk_file_data, bytes):
            bulk_file_data = bulk_file_data.decode('utf-8-sig')
     

            
        try:
            #df = pd.read_csv(StringIO(decoded_string))
            print("reading csv")
            df = pd.read_csv(StringIO(bulk_file_data))
            print("reading csv done")
        except Exception as e:
            logger.error(f"CSV parsing error: {str(e)}")
            return {"error": f"Could not parse CSV data: {str(e)}"}
        
        # Validate required columns
        required_columns = ['user_prompt']
        if not all(col in df.columns for col in required_columns):
            logger.error("CSV file missing required columns: user_prompt")
            return
        
        results = []
        total_rows = len(df)
        print(f"total_rows-{total_rows}")
        exit
        for index, row in df.iterrows():
            logger.critical(f"Processing evaluation {index + 1} of {total_rows}")
            
            if pd.notna(row['user_prompt']):
                # Construct prompt
                prompt = ''.join([
                    'Return_data_constraints: {constraints} ',
                    row['user_prompt'],
                    '{missing_sections}'
                ])
                message.prompt = prompt
                message.page = message.eval_request.page   
                message.run_mode = "bulk_transcript"             
                response = sync_async_runner(message)                
                insert(db_data)            
                
  
    except Exception as e:
        logger.error(f"Error processing evaluation file: {str(e)}")
        return {"error": str(e)}



def _process_same_llm(message:Message):
    global db_data  
    """Handle same LLM processing mode"""
    response = [sync_async_runner(message) for _ in range(int(message.run_count))] 
    insert(db_data)
    return {"response": response, "ideal_response": shared_data_instance.get_data('theIdealResponse'), "prompt": shared_data_instance.get_data('prompt')}



def _setup_test_data(row):
    """Helper to set up shared data for test runs"""
    shared_data = {
        'theIdealResponse': row.get('ideal_response') ,
        'run_no': row.get('run_no') ,
        'ideal_response': row.get('ideal_response'), 
        'original_response': row.get('response') ,
        'usecase': row.get('usecase') ,
        'original_run_no': row.get('run_no'),
        'original_prompt': row.get('user_prompt') 
    } 
    
    for key, value in shared_data.items():
        shared_data_instance.set_data(key, value)

def _generate_test_summary(test_type, eval_name):
    """Generate and log test results summary"""
    total_tests = len(test_map)
    passed_tests = sum(1 for test in test_map.values() if test['matches_idealResponse'])
    failed_tests = total_tests - passed_tests
    pass_rate = f"{(passed_tests/total_tests)*100:.2f}%"
    average_execution_time = round(sum(test['execution_time'] for test in test_map.values()) / total_tests, 2)
    accuracy = 100 -sum(test['mismatch_percentage'] for test in test_map.values()) / total_tests
    print(f"accuracy-{accuracy}")
    summary = {
        "AI model": theModel,
        "Tests Passed": passed_tests,
        "Tests Failed": failed_tests,
        "Total Tests": total_tests,
        "Pass Rate": pass_rate,
        "Average Execution Time": average_execution_time,
        "Accuracy": round(accuracy,2)
    }
    
    logger.critical("\nTest Suite Results:")
    logger.critical("==================")
    for key, value in summary.items():
        logger.critical(f"{key}: {value}")
  
    test_run_no = save_test_results(test_map, theModel, total_tests, passed_tests, failed_tests, pass_rate, average_execution_time, test_type, eval_name, accuracy)
    print(f"test_run_no-{test_run_no}")
    print(f'summary-{summary}')
    return test_run_no

def handleRequest(message: Message):
    """Handle API requests"""
    if isinstance(message.prompt, str):
        message.prompt = add_space_after_punctuation(message.prompt)
        

    return process_request(message)



