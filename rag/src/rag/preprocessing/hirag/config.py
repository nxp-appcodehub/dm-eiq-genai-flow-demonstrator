# Copyright 2024-2025 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

from dataclasses import dataclass


@dataclass
class Config:
    llm_name: str = "llama3.1-8B"  # Among the following list ["llama3.1-8B", "llama2-7B"]
    chunking_method: str = "SpaCy"  # Among the following list ["recursive", "fixed", "NLTK", "SpaCy"]
    chunk_size: int = 1000    # We recommend 1000
    chunk_overlap: int = 200  # We recommend 200
    global_chunk_size: int = 7000  # We recommend 7000
    global_chunk_overlap: int = 500  # We recommend 500
    max_QA_pair_limit: int = 15
    prompt: str = ("Give me as much question-answer pairs as needed to cover the whole following text. "
                   "Question (Q) and answer (A) must be short and precise. The pairs must be formatted as: Q:, A:."
                   )
    parsing_QA_pattern: str = r"Q:\s*(.*?),?\s*A:\s*(.*?)(?=\s*Q:|\s*Note: I|$)"

    global_understanding_QA_pair_limit: int = 25
    global_understanding_prompt = """
            Given the following text, generate a comprehensive serie of question-answer pairs that demonstrate a thorough understanding of the global content. 
            The questions should be clear and concise, and the answers should provide a brief summary or explanation. 
            Avoid inventing information that is not stated in the text. 
            Use the text to support the answers, and avoid making assumptions. 
            When answering the questions, prioritize conciseness while ensuring each response is at least one sentence long. 
            If a question has multiple possible answers, provide multiple question-answer pairs.
            
            
            Example:
            Useful questions: \"What is this document about?\", \"Who is speaking?\", \"What is the title of this document?\", 
            
            Or if the LLM were given:
            \"
            Menu:
            - Hamburger (bread, steak, salad, tomato, onion) 16$
            - Cheeseburger (bread, steak, cheese, salad, tomato, onion) 17$
            - Rice bowl eggs (rice, egg, beans) 12$
            \"
            The expected output would be something like:
            Q: What is the most expensive meal on the menu?, A: The cheeseburger (17$) is the most expensive meal.
            Q: What is the cheapest meal?, A: The rice bowl egg (12$) is the cheapest meal.
            Q: What are the vegetarian meals?, A: The rice bowl egg is the only vegetarian meal.
        """
    
    local_QA_pair_limit: int = 25
    local_understanding_prompt = """
        Extract a list of concise question-answer pairs from the text, covering all main topics and entities 
        mentioned. Ensure each pair is accurate, relevant, and well-formatted as Q: ..., A: ...
    
        When creating the pairs, consider the following guidelines:
    
        Entity-based questions: Extract questions about the main entities mentioned in the text, such as people, 
        places, organizations, and objects. For example, \"Who is...\", \"What is...\", \"Where is...\".
    
        Action-based questions: Identify actions, events, or processes mentioned in the text and ask questions about
        them, like \"What is happening...\", \"What is being done...\", or \"What action is...\".
    
        Attribute-based questions: Extract questions about attributes or characteristics of entities mentioned in 
        the text, such as \"What color is...\", \"How old is...\", or \"What is the name of...\".
    
        Contextual questions: Formulate questions that provide context or additional information about the main 
        topic, like \"Why is this happening...\", \"What is the purpose of...\", or \"How does this relate to...\".
    
        When answering the questions, prioritize conciseness while ensuring each response is at least one sentence 
        long. Replace acronyms and technical words by their definition if they are clearly defined in the text. Avoid 
        inventing information not explicitly stated in the text. 
    
        Example of the output for the given text \"The white dog is running.\":
        Q: Who is running?, A: The white dog is running.
        Q: What color is the dog?, A: The dog is white.
        Q: What is the dog doing?, A: The dog is running.
    """
    
    data_augmentation_QA_pair_limit: int = 5
    data_augmentation_prompt = """
        Given a Question-Answer (Q-A) pair, perform data augmentation by generating new Q-A pairs with the following modifications:
        - Synonym replacement: Replace key words or phrases with their synonyms or their definition, while preserving the original meaning.
        - Passive voice conversion: Convert the answer to the passive voice, where possible, to create alternative expressions.
        - Sentence rephrasing: Rephrase the answer to provide different ways of expressing the same idea, without altering the core meaning.
        
        Input format:
        Q: <original question>, A: <original answer>
        Output format:
        Q: <new question1>, A: <new answer1>
        Q: <new question2>, A: <new answer2>
        
        Example of the output for the given text \"Q: What causes global warming?, A: Greenhouse gas emissions trap heat in the atmosphere.\":
        Q: What leads to climate change?, A: Heat is retained in the atmosphere due to greenhouse gases.
        Q: What is global warming caused by?, A: Heat is trapped in the atmosphere by greenhouse gas emissions.
        Q: What factors contribute to rising global temperatures?, A: The atmosphere is warmed due to the accumulation of greenhouse gases.
        Q: Why is the Earth getting hotter?, A: Gases like CO2 keep heat from escaping into space.
    """
