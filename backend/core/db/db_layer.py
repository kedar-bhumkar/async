import warnings
warnings.filterwarnings("ignore")
import psycopg2
from backend.core.utility.constants import *
from backend.core.utility.util import getConfig
import pandas as pd
from datetime import datetime
from backend.core.logging.custom_logger import *
from functools import lru_cache

def connect():
    config = getConfig(db_conn_file)
    #print(f"config-{config['db']['postgres']}")
   
    conn = psycopg2.connect(**config['db']['postgres'])
    cursor = conn.cursor()   
    #print(f"config-2")

    return conn, cursor



def insert(data_list):    
    conn, cursor = connect()
    try:                
        # SQL query to insert data into the table
        insert_query = INSERT_QUERY
        # Prepare the values to be inserted
        values_list = [
            (
                data['usecase'],               
                data['functionality'],
                data['llm'],
                data['llm_parameters'],
                data['isBaseline'],
                data['run_no'],
                data['system_prompt'],
                data['user_prompt'],
                data['response'],
                data['ideal_response'],
                data['execution_time'],
                data['matches_baseline'],
                data['matches_ideal'],
                data['difference'],
                data['ideal_response_difference'],
                data['mode'],
                data['similarity_metric'],                
                datetime.now(),  # Current date for run_date
                data['use_for_training'],
                data['fingerprint'],
                data['input_token_count'],
                data['output_token_count'],
                data['llm_latency']
            ) 
            for data in data_list  
        ]
     
        # Execute the insert query
        cursor.executemany(insert_query, values_list)        
        # Commit the transaction
        conn.commit()
        
        print("Data inserted successfully")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Close the cursor and connection
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def read(q):    
    print(f"query-{q}")
    conn, cursor = connect()
    
    try:
        # Query the data from the table        
        df = pd.read_sql_query(q, conn)
        
        return df
    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Close the cursor and connection
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def readWithGroupFilter(run_id):
    return read("".join([READ_QUERY, f"where run_no='{run_id}'"]))


def get_test_data(test_size_limit,page):
    if(test_size_limit == None):
        return read("".join([TEST_QUERY, f" and functionality='{page}' ORDER BY user_prompt, run_no DESC"]))
    else:
        return read("".join([TEST_QUERY, f" and functionality='{page}' ORDER BY user_prompt, run_no DESC LIMIT {test_size_limit}  "]))



def insert_test_results_data(model:str,total_tests: int, tests_passed: int, tests_failed: int, pass_rate: str, average_execution_time: float, test_type: str,eval_name: str,accuracy: float):
    conn, cursor = connect()
    try:
        insert_query = INSERT_TEST_RESULTS_QUERY
        
        cursor.execute(insert_query, (total_tests, tests_passed, tests_failed, pass_rate, average_execution_time, test_type, eval_name,accuracy))
        test_run_no = cursor.fetchone()[0]
    
        conn.commit()
        print(f"Test results inserted successfully with run number: {test_run_no}")
        return test_run_no

    except Exception as e:
        print(f"Error inserting test results: {e}")
        return None
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def insert_test_results_detail_data(model:str,test_run_no: int,original_response: str,actual_response: str,ideal_response: str,difference: str,original_run_no: int,original_prompt: str, execution_time: float,fingerprint: str,matched_tokens: bool,mismatched_tokens: bool,mismatch_percentage: float, page: str,status: str,llm_latency: float):
    #logger.critical(f"insert_test_results_detail_data - {model},{test_run_no},{original_response},{actual_response},{ideal_response},{difference},{original_run_no},{original_prompt}")
    conn, cursor = connect()
    try: 
        insert_query = INSERT_TEST_RESULTS_DETAIL_QUERY

        cursor.execute(insert_query, (test_run_no, original_response, actual_response, ideal_response, difference,original_run_no,original_prompt, execution_time,fingerprint,matched_tokens,mismatched_tokens,mismatch_percentage, page,status,llm_latency))
        conn.commit()
        print(f"Test results detail inserted successfully with run number: {test_run_no}")

    except Exception as e:
        print(f"Error inserting test results detail: {e}")

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def save_test_results(test_map,model,total_tests,passed_tests,failed_tests,pass_rate,average_execution_time,test_type,eval_name,accuracy):
    print(f"average_execution_time-{average_execution_time}")
    
    test_run_no = insert_test_results_data(model,total_tests,passed_tests,failed_tests,pass_rate,average_execution_time,test_type,eval_name,accuracy)
    for test in test_map.values():
        print(f"test['execution_time']-{test['execution_time']}")
        insert_test_results_detail_data(model,test_run_no,test['original_response'],test['actual_response'],test['ideal_response'],test['idealResponse_changes'],test['original_run_no'],test['original_prompt'], test['execution_time'], test['fingerprint'],test['matched_tokens'],test['mismatched_tokens'],test['mismatch_percentage'], test['page'],test['status'],test['llm_latency'])

    return test_run_no
    
def get_test_results(test_run_no):
    return read("".join([TEST_RESULTS_QUERY, f" Where test_run_no='{test_run_no}'"]))
    return read("".join([TEST_RESULTS_DETAIL_QUERY, f" '"]))


def get_test_results_detail(test_run_no,test_result_id):
    if(test_result_id):
        return read("".join([TEST_RESULTS_DETAIL_QUERY, f" Where trd.test_run_no='{test_run_no}' and trd.test_results_detail_no='{test_result_id}'"]))
    else:
        return read("".join([TEST_RESULTS_DETAIL_QUERY, f" Where trd.test_run_no='{test_run_no}'"]))


def get_test_names():
    return read("".join([TEST_NAMES_QUERY]))


@lru_cache(maxsize=100)
def get_system_prompt(usecase: str, page: str) -> str:
    df = read("".join([SYSTEM_PROMPT_QUERY, f" where usecase='{usecase}' and page='{page}' and isactive=true"]))
    if df.empty:
        return None
    return df.to_dict('records')[0]['system_prompt']


def get_llm_config():
    df = read("".join([LLM_CONFIG_QUERY, f" LIMIT 1"]))
    if df.empty:
        raise RuntimeError("No active LLM configuration found in database")
    else:
        return df.to_dict('records')[0]


