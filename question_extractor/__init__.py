import re
import os
import asyncio
import openai
import json
from pathlib import Path
import logging

from tenacity import (
    retry,
    wait_random_exponential,
)  
from openai import OpenAIError
from aiolimiter import AsyncLimiter
from langchain.chat_models import ChatOpenAI
from contextlib import asynccontextmanager
from .markdown import load_markdown_files_from_db,load_markdown_files_from_directory, split_markdown
from .token_counting import count_tokens_text, count_tokens_messages, get_available_tokens, are_tokens_available_for_both_conversations
from .prompts import create_answering_conversation_messages, create_extraction_conversation_messages
from service.AgentConfigManager import AgentConfigManager
from service.TrackedChatOpenAIWrapper import TrackedChatOpenAIWrapper
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(filename='agent.log', level=logging.INFO)
db_manager = AgentConfigManager()

# replace the "Key" with your own API key, you can provide multiply APIs in the list
api_key_lock = asyncio.Lock()
api_key_index = 0
#---------------------------------------------------------------------------------------------
# QUESTION PROCESSING 

# Ensure we do not run too many concurent requests
model_rate_limits = 2000
max_concurent_request = int(model_rate_limits * 0.75)
throttler = asyncio.Semaphore(max_concurent_request)
import logging


def flatten_nested_lists(nested_lists):
    """
    Takes a list of lists as input and returns a flattened list containing all elements.
    
    Args:
        nested_lists (list of lists): A list containing one or more sublists.

    Returns:
        list: A flattened list containing all elements from the input nested lists.
    """
    flattened_list = []

    # Iterate through the nested lists and add each element to the flattened_list
    for sublist in nested_lists:
        flattened_list.extend(sublist)

    return flattened_list

@retry(
    wait=wait_random_exponential(min=15, max=40),
)
async def run_model(messages):
    """
    Asynchronously runs the chat model with as many tokens as possible on the given messages.
    
    Args:
        messages (list): A list of input messages to be processed by the model.

    Returns:
        str: The model-generated output text after processing the input messages.
    """
    async with api_key_lock:  # Ensure that the rotation is thread-safe
        global api_key_index
        #os.environ['OPENAI_API_KEY'] = API_KEYS[api_key_index]
        #api_key_index = (api_key_index + 1) % len(API_KEYS)
    # Count the number of tokens in the input messages
    num_tokens_in_messages = count_tokens_messages(messages)

    # Calculate the number of tokens available for processing
    num_tokens_available = get_available_tokens(num_tokens_in_messages)

    # Create an instance of the ChatOpenAI model with minimum imagination (temperature set to 0)
    #model =  ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=num_tokens_available)
    model = TrackedChatOpenAIWrapper(agent_id=9, model="gpt-4o-mini", temperature=0.0, max_tokens=num_tokens_available)
 
    try:
        # Use a semaphore to limit the number of simultaneous calls
        async with throttler:
            try:
                # Asynchronously run the model on the input messages
                output = await model.call(messages) 
            except Exception as e:
                logging.exception(f"ERROR ({e}): Could not generate text for an input.")
                return 'ERROR'
    except openai.error.RateLimitError as e:
        logging.info(f"ERROR ({e}): Rate limit exceeded, retrying.")
        raise  # Re-raise the exception to allow tenacity to handle the retry
    except openai.error.APIConnectionError as e:
        logging.info(f"ERROR ({e}): Could not connect, retrying.")
        raise  # Re-raise the exception to allow tenacity to handle the retry
    except Exception as e:
        logging.exception(f"ERROR ({e}): Could not generate text for an input.")
        return 'ERROR'
    logging.info("dovrebbe aver finito il run model")
    # Extract and return the generated text from the model output
    return output.generations[0].text.strip()

def extract_questions_from_output(output):
    """
    Takes a numbered list of questions as a string and returns them as a list of strings.
    The input might have prefixes/suffixes that are not questions or incomplete questions.

    Args:
        output (str): A string containing a numbered list of questions.

    Returns:
        list of str: A list of extracted questions as strings.
    """
    # Define a regex pattern to match questions (lines starting with a number followed by a dot and a space)
    question_pattern = re.compile(r"^\s*\d+\.\s*(.+)$", re.MULTILINE)

    # Find all the questions matching the pattern in the input text
    questions = question_pattern.findall(output)

    # Check if the last question is incomplete (does not end with punctuation or a parenthesis)
    if (len(questions) > 0) and (not re.search(r"[.!?)]$", questions[-1].strip())):
        logging.info(f"WARNING: Popping incomplete question: '{questions[-1]}'")
        questions.pop()

    return questions


async def extract_questions_from_text(file_path, text):
    """
    Asynchronously extracts questions from the given text.
    
    Args:
        file_path (str): The file path of the markdown file.
        text (str): The text content of the markdown file.

    Returns:
        list of tuple: A list of tuples, each containing the file path, text, and extracted question.
    """
    # Ensure the text can be processed by the model
    #global api_key_index
    text = text.strip()
    num_tokens_text = count_tokens_text(text)

    if not are_tokens_available_for_both_conversations(num_tokens_text):
        # Split text and call function recursively
        logging.info(f"WARNING: Splitting '{file_path}' into smaller chunks.")

        # Build tasks for each subsection of the text
        tasks = []
        for sub_title, sub_text in split_markdown(text):
            sub_file_path = file_path + '/' + sub_title.replace('# ', '#').replace(' ', '-').lower()
            task = extract_questions_from_text(sub_file_path, sub_text)
            tasks.append(task)

        # Asynchronously run tasks and gather outputs
        tasks_outputs = await asyncio.gather(*tasks)
        logging.info("arrivo alla fine della estrazione delle domande dal testo")

        # Flatten and return the results
        return flatten_nested_lists(tasks_outputs)
    else:
        # Run the model to extract questions
        messages = create_extraction_conversation_messages(text)
        output = await run_model(messages)
        questions = extract_questions_from_output(output)
        logging.info("estraggo questions")
        # Associate questions with source information and return as a list of tuples
        outputs = [(file_path, text, question.strip()) for question in questions]
        return outputs


async def generate_answer(question, source):
    """
    Asynchronously generates an answer to a given question using the provided source text.
    
    Args:
        question (str): The question to be answered.
        source (str): The text containing relevant information for answering the question.

    Returns:
        str: The generated answer to the question.
    """
    # Create the input messages for the chat model
    messages = create_answering_conversation_messages(question, source)
    # Asynchronously run the chat model with the input messages
    logging.info(f"run model di generate_answer")
    answer = await run_model(messages)
    logging.info(f"finita run model di generate_answer")
    return answer

#---------------------------------------------------------------------------------------------
# FILE PROCESSING

async def process_file(page_id, file_path, text, progress_counter, db_manager, verbose=True, max_qa_pairs=300):
    """
    Asynchronously processes a file, extracting questions and generating answers concurrently,
    checking and saving results to the database.
    
    Args:
        page_id (int): The ID of the page to process.
        file_path (str): The file path of the markdown file.
        text (str): The text content of the markdown file.
        progress_counter (dict): A dictionary containing progress information ('nb_files_done' and 'nb_files').
        db_manager (DBManager): A database manager instance to interact with the database.
        verbose (bool): If True, logging.info progress information. Default is True.
        max_qa_pairs (int): Maximum number of question-answer pairs to process.

    Returns:
        list: A list of dictionaries containing source, question, and answer information.
    """
    # Step 1: Check if questions and answers for the page_id already exist in the database
    query = "SELECT scraped_page_id, path, question, answer FROM qa_blocks WHERE scraped_page_id = %s"
    existing_records =  db_manager.fetch(query, (page_id,))

    # If records exist, return them
    if existing_records:
        logging.info("Records already exist in the database.")
        #return existing_records 
    
    # Step 2: Extract questions from the text if not found in the database
    questions = await extract_questions_from_text(file_path, text)
    questions = questions[:max_qa_pairs]  # Limit to max_qa_pairs if needed

    logging.info("genero risposte e domande.")
    # Step 3: Generate answers asynchronously
    tasks = [generate_answer(question, text) for _, text, question in questions]
    tasks_outputs = await asyncio.gather(*tasks)

    # Step 4: Prepare the results and save them in the database
    result = []
    for (sub_file_path, sub_text, question), answer in zip(questions, tasks_outputs):
        logging.info(f" answer: {answer}, question: {question}")
           
        result_entry = {'page_id': page_id, 'source': sub_file_path, 'question': question, 'answer': answer}
        result.append(result_entry)
        """ try:
            insert_qa_block(db_manager,page_id,sub_file_path,question,answer)
        except Exception as e:
            logging.info(f"errore in qa_block: {str(e)}") """
        logging.info(f"faccio question")

        
        # Insert each Q&A pair into the database
        #insert_query = """
        #INSERT INTO qa_blocks (scraped_page_id, source, question, answer)
        #VALUES (%s, %s, %s, %s)
        #"""
        #await db_manager.execute(insert_query, (page_id, sub_file_path, question, answer))

    # Step 5: Update progress and logging.info information if verbose is True
    progress_counter['nb_files_done'] += 1
    if verbose:
        logging.info(f"{progress_counter['nb_files_done']}/{progress_counter['nb_files']}: File '{file_path}' processed!")

    logging.info(f"\n result di process_file: \n {result}")
    return result

def insert_qa_block(db_manager, page_id, sub_file_path, question, answer):
    """
    Inserisce un blocco QA nella tabella `qa_blocks`.
    """
    query = """
    INSERT INTO qa_blocks (scraped_page_id, path, question, answer, created_at, updated_at)
    VALUES (%s, %s, %s, %s, now()::timestamp(0), now()::timestamp(0))
    """
    
    params = (page_id, sub_file_path, question, answer)
    
    # Esegui l'inserimento utilizzando db_manager
    db_manager.execute(query, params)

async def process_files(files, db_manager, verbose=True):
    """
    Asynchronously processes a list of files, extracting questions and generating answers concurrently.
    
    Args:
        files (list): A list of tuples containing file paths and their respective text content.
        verbose (bool): If True, logging.info progress information. Default is True.

    Returns:
        list: A merged list of dictionaries containing source, question, and answer information.
    """
    # Set up progress information for display
    nb_files = len(files)
    progress_counter = {'nb_files': nb_files, 'nb_files_done': 0}
    if verbose: logging.info(f"Starting question extraction on {nb_files} files.")

    # Build and run tasks for each file concurrently
    tasks = []
    #logging.info("file zero:")
    #logging.info(files[0])
    for id, file_path, text in files:
        task = process_file(id, file_path, text, progress_counter,db_manager, verbose=verbose)
        tasks.append(task)

    tasks_outputs = await asyncio.gather(*tasks)

    # Merge results from all tasks
    return flatten_nested_lists(tasks_outputs)

#---------------------------------------------------------------------------------------------
# MAIN

async def extract_questions_from_db(agent_stream_id, db_manager=None,verbose=True):
    """
    Extracts questions and answers from all markdown files from db.

    Args:
        agent_stream_id (str): Agent id.
        db_manager(Object): Class db_manager
        verbose (bool): If True, logging.info progress information. Default is True.

    Returns:
        list: A list of dictionaries containing path, source, question, and answer information.
    """
    # Load input files from the folder
    if verbose: logging.info(f"Loading files from '{agent_stream_id}'.")
    logging.info("carico i file dal db")
    files = load_markdown_files_from_db(agent_stream_id, db_manager)
   

    #logging.info(files)
    # Run question extraction tasks
   # loop = asyncio.get_event_loop()
    #if loop:
    #    results = loop.run_until_complete(process_files(files, verbose=verbose))
    #else:
    logging.info("processo i files")

    results = await process_files(files,db_manager, verbose=verbose)
    logging.info(f"\n risultato di process_files: \n {results}")


    if verbose: logging.info(f"Done, {len(results)} question/answer pairs have been generated!")
    return results