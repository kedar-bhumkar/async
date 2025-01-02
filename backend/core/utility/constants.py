#model = "gpt-3.5-turbo-0125"
#model="gpt-4o"
#model="meta-llama/llama-3-8b",
#model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF"
standard_model = "gpt-3.5-turbo-0125" 

config_file =".\\backend\\core\\config\\config.yaml"
prompts_file=".\\backend\\core\\config\\prompts.yaml"
db_conn_file='.\\backend\\core\\config\\db_config.yaml'




 


INSERT_QUERY = """ 
        INSERT INTO Run_stats (
            usecase,
            functionality,
            llm,
            llm_parameters,
            isBaseline,
            run_no,
            system_prompt,
            user_prompt,
            response,
            ideal_response,
            execution_time,
            matches_baseline,
            matches_ideal,
            difference,
            ideal_response_difference,
            mode,
            similarity_metric,
            run_date,
            use_for_training,
            fingerprint,
            input_token_count,
            output_token_count,
            llm_latency
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
READ_QUERY = """
        SELECT 
            run_no,
            execution_time,
            usecase,
            functionality,
            isbaseline,
            matches_baseline,
            matches_ideal,
            difference,
            ideal_response_difference,
            response 
        FROM Run_stats
        """

TEST_QUERY = """
        SELECT DISTINCT ON (user_prompt)
            run_no,
            execution_time,
            usecase,
            functionality,                        
            user_prompt,
            response,
            ideal_response            
        FROM Run_stats 
        WHERE use_for_testing = 'true'
       
        """

TEST_RESULTS_DETAIL_QUERY = """
        SELECT 
            test_run_no, 
            test_results_detail_no, 
            original_response, 
            trd.ideal_response, 
            actual_response, 
            original_prompt, 
            trd.fingerprint as trd_fingerprint, 
            rs.fingerprint as rs_fingerprint,
            trd.matched_tokens, 
            trd.mismatched_tokens, 
            trd.mismatch_percentage, 
            trd.execution_time,             
            trd.page, 
            trd.status
        FROM public.test_results_detail trd
        LEFT JOIN public.run_stats rs 
            ON trd.original_run_no = rs.run_no
        """

TEST_RESULTS_QUERY = """
    SELECT         
        test_run_date,
        total_tests,
        tests_passed,
        tests_failed,
        tests_pass_rate,
        average_execution_time,
        test_type,
        accuracy
    FROM test_results 
    
"""

TEST_NAMES_QUERY = """
    SELECT eval_name, test_run_no
    FROM test_results
    where status = 'active'
"""

VIEW_TEST_RESULTS_DETAIL_QUERY = """
    SELECT 
    FROM test_results_detail    
"""

SYSTEM_PROMPT_QUERY = """
    SELECT system_prompt
    FROM prompt_config
    
"""

LLM_CONFIG_QUERY = """
    SELECT *
    FROM llm_config         
"""

 