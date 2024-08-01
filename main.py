import uvicorn
from typing import List, Dict, Any
import glob
import openai
from langchain.llms import OpenAI
from prompt import prompt_template,system_template,new_prompt
from langchain.prompts import PromptTemplate
from fastapi import FastAPI, UploadFile, Request,HTTPException
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi.middleware.cors import CORSMiddleware
from langchain.docstore.document import Document
from dotenv import load_dotenv
from pydantic import BaseModel
import mysql.connector
import os
import re
from langchain.vectorstores import Pinecone
import pinecone
from langchain.embeddings import OpenAIEmbeddings


load_dotenv()

#PineCone details
pinecone_key = os.getenv('pinecone_key')
pinecone_env = os.getenv('pinecone_env')
index_name = "ailegalbot"
# nameSpace = "test-namespace"

pinecone.init(
    api_key=pinecone_key,
    environment=pinecone_env,
)

# OpenAI details
os.environ["OPENAI_API_KEY"] = os.getenv("openai_key")  ## client api
openai.api_key = os.getenv("openai_key")


base_dir = os.getcwd()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]                           
)

class vector(BaseModel):
    database_name:str
    id: int

class Highlight(BaseModel):
    word: str | None = None
    database_name:str


class Comparision(BaseModel):
    updated_paragraph: str
    original_paragraph: str

class all_quesans(BaseModel):
    database_name:str
    query: str


class delete(BaseModel):
    database_name:str
    id: int

class ques_ans(BaseModel):
    database_name:str
    query: str
    name_space: str


# Function to store the embeddings into the PineCone DataBase
def insertion(database_name,sub_name, texts):
    namespace_nm = f'{database_name}_{sub_name}'

    if not texts:
        return f"There's no data in the contract"
    else:

        embed = OpenAIEmbeddings()
        index = pinecone.Index(index_name)
        stats = index.describe_index_stats()
        if namespace_nm in stats.namespaces:
            return f'Clause created successfully'
        else:

            nameSpace = f'{database_name}_All_Pdf'
            Pinecone.from_documents(documents=texts, embedding=embed, index_name=index_name,
                                    namespace=nameSpace)
            Pinecone.from_documents(documents=texts, embedding=embed, index_name=index_name,
                                    namespace=namespace_nm)
        return f'Clause created successfully'


# Function to query and store the PDf Text
@app.post('/vector_stores/')
async def vector_stores(item:vector):
    items = item.dict()
    database_name=items['database_name']
    id = items['id']

    try:
        db_config = {
            'host': os.getenv('db_host'),
            'user': os.getenv('db_user'),
            'password': os.getenv('db_password'),
            'database': database_name
        }
        connection = mysql.connector.connect(**db_config)
    except mysql.connector.Error as conn_err:
        print(conn_err)
        raise HTTPException(status_code=400, detail=f'Something went wrong')

    try:
        cursor = connection.cursor()
        query_ = (f'''SELECT section_id ,clause, pdf_name,pdf_id,title,
                CASE WHEN row_num = 1 THEN CONCAT(title, ' ', clause) ELSE clause END AS merge_column
                FROM (
                    SELECT section_clauses.section_id,section_clauses.clause,contracts.name AS pdf_name,contracts.id AS pdf_id,contract_details.title,
                    ROW_NUMBER() OVER (PARTITION BY section_clauses.section_id ORDER BY section_clauses.clause) AS row_num
                    FROM section_clauses
                     JOIN contract_details ON contract_details.id=section_clauses.section_id
                     JOIN contracts ON contracts.id=contract_details.contract_id
                    WHERE contracts.id = {id}
                    ) AS data; ''')

        cursor.execute(query_)
        result = cursor.fetchall()

        clean_text = []
        for item in result:
            par = str(item[5])
            # Check if i is a string before applying re.sub
            if isinstance(par, str):
                plain_text = re.sub(r'<.*?>', '', par)
                clean_text.append(Document(page_content=plain_text,
                                           metadata={"Pdf_Name": item[2], "Pdf_Id": item[3], "Section_Id": item[0]}))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_text = text_splitter.split_documents(clean_text)
        txt = insertion(database_name, f'{id}', split_text)

    except mysql.connector.Error as err:
        print(err)
        raise HTTPException(status_code=400, detail=f'Something went wrong')

    finally:
        if connection:
            cursor.close()
            connection.close()
    return {'success': True ,'status':200,'message': txt}


# Function to question answer from All pdfs in that Database
@app.post("/question_answer")
async def question_answer(item: all_quesans):
    items = item.dict()
    database_name=items['database_name']
    query = items['query']

    try:
        embed = OpenAIEmbeddings()
        index = pinecone.Index(index_name)
        stats = index.describe_index_stats()
        print(stats,"STAT")
        nameSpace = f"{database_name}_All_Pdf"
        if nameSpace in stats.namespaces:
            vectorstore = Pinecone.from_existing_index(index_name='ailegalbot', embedding=embed, text_key="text",
                                                       namespace=nameSpace)

            retriever = vectorstore.as_retriever(search_type="similarity")  # search_kwargs={"k": 13}
            llm = OpenAI(temperature=0.4, model_name='gpt-3.5-turbo')
            # llm = ChatOpenAI(model_name='gpt-3.5-turbo')

            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=prompt_template,
            )
            question_answers = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type='stuff',
                retriever=retriever,
                verbose=False,
                return_source_documents=True,
                chain_type_kwargs={
                    "verbose": False,
                    "prompt": prompt,
                }
            )
            response = question_answers({"query": query})
            detail_info = response['source_documents']

            unique_pdf_names = set(doc.metadata["Pdf_Name"] for doc in detail_info)
            return {'success': True, 'status': 200, 'result': response["result"], 'Pdf_Name': unique_pdf_names,
                    'query': query}

        else:
            return {'success': False ,'status':400,'message':'No data found'}

    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=400, detail=f'Something went wrong')


# Function to question answer from particular PDF of given database
@app.post("/particular_qa_file")
async def particular_qa_file(item: ques_ans):
    items = item.dict()
    database_name=items['database_name']
    query = items['query']
    # name_space = items['name_space']
    name_space = f"{database_name}_{items['name_space']}"

    try:

        embed = OpenAIEmbeddings()
        vectorstore = Pinecone.from_existing_index(index_name='ailegalbot', embedding=embed, text_key="text",
                                                   namespace=name_space)
        retriever = vectorstore.as_retriever(search_type="similarity", )  # search_kwargs={"k": 13}
        llm = OpenAI(temperature=0.4, model_name='gpt-3.5-turbo')
        # llm = ChatOpenAI(model_name='gpt-3.5-turbo')

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template,
        )
        question_answers = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=retriever,
            verbose=False,
            return_source_documents=True,
            chain_type_kwargs={
                "verbose": False,
                "prompt": prompt,
            }
        )
        response = question_answers({"query": query})
        output = response["result"]
        detail_info = response['source_documents']
        pdf_name = set(doc.metadata['Pdf_Name'] for doc in detail_info)

        if not pdf_name:
            output = 'No data found'
            return {'success': False, 'status': 400,'message': output}

        else:
            return {'success': True,'status': 200,'result': output, 'query': query}

    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=400, detail=f'Something went wrong')


# Function to simplify/paraphrase the text
@app.post("/simply_language")
async def simplify_text(input_text: List[str]):
    print(input_text[0])
    try:
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=f"Simplify the following text in a simple language:\n{input_text[0]}\n",
            max_tokens=500,
            n=1,
            stop=None,  # You can provide a list of strings to use as stopping criteria
        )
        # Extract and return the simplified text
        return response['choices'][0]['text'].strip()

    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=400, detail=f'Something went wrong')



# Function to comparison of the text
@app.post("/comparision_text")
async def comparision(item: Comparision):
    parameter = item.dict()
    length_u = parameter['updated_paragraph'].split()
    length_o = parameter['original_paragraph'].split()
    if len(length_u) > len(length_o):
        if ' '.join(length_o) == ' '.join(length_u[:len(length_o)]):
            print(' '.join(length_u[:len(length_o)]), ' '.join(length_u[len(length_o):]))  # change #green
            res = f'<span style="background-color: green;">{" ".join(length_u[len(length_o):])}</span>'

            result = ' '.join([' '.join(length_u[:len(length_o)]), res])
        else:
            result = f'<span style="background-color: orange;">{" ".join(length_u)}</span>'  # green #change #orange
    elif len(length_u) == len(length_o):
        if ' '.join(length_o) == ' '.join(length_u):
            result = ' '.join(length_u)
        else:
            result = f'<span style="background-color: orange;">{" ".join(length_u)}</span>'  # orange
    return {'comparision_text': result}


# Function to delete the PDF from the vector database
@app.post('/delete/')
async def delete(item: delete):
    items = item.dict()
    database_name=items['database_name']
    id = items['id']
    dict={}
    try:
        index = pinecone.Index(index_name)
        index_stats = index.describe_index_stats()
        namespace_names = [str(name) for name in index_stats["namespaces"].keys()]
        print(namespace_names, 'Namespace')
        namespace_id = f"{database_name}_{id}"

        if namespace_id in namespace_names:
            index.delete(filter={"Pdf_Id": id}, namespace=f'{database_name}_All_Pdf')
            index.delete(delete_all=True, namespace=namespace_id)
            txt = f'Contract deleted successfully.'
        else:
            txt = f'Contract does not exist.'
        dict['success'] = True
        dict['status'] = 200
        dict['message'] = txt


    except Exception as e:
        print(e)
        txt = "Something went wrong"
        dict['success'] = False
        dict['status'] = 400
        dict['message'] = txt

    return dict





# Function to do the highlight of the text
@app.post("/highlight")
async def highlight(item: Highlight):
    # parameter = item.dict()
    parameter = item.model_dump()
    database_name=parameter['database_name']
    try:
        
        db_config = {
            'host': os.getenv('db_host'),
            'user': os.getenv('db_user'),
            'password': os.getenv('db_password'),
            'database': database_name
        }
        connection = mysql.connector.connect(**db_config)
    except mysql.connector.Error as conn_err:
        print(conn_err)
        raise HTTPException(status_code=400, detail='Something went wrong')

    highlight_dict = {}
    try:

        cursor = connection.cursor()
        query = f'''SELECT contract_details.title,section_clauses.clause,contracts.name,contracts.id,section_clauses.number FROM section_clauses
                 JOIN contract_details ON contract_details.id=section_clauses.section_id
                 JOIN contracts ON contracts.id=contract_details.contract_id;'''

        cursor.execute(query)
        result = cursor.fetchall()
        # print(result,"oldresult")
        # Initialize the data structure
        highlight_dict = {}

        # Process the result set
        for row in result:
            contract_name = row[2]
            contract_id = row[3]
            section_title = row[0]
            clause_text = row[1]
            clause_number = row[4]

            # Highlight the specified word
            if len(re.findall(re.escape(parameter['word']), clause_text, flags=re.IGNORECASE)) > 0:
                if 'p class=' in clause_text:
                    text_parts = clause_text.split('p class=')
                    highlight_text_ = re.sub(re.escape(parameter['word']),
                                             '<span style="background-color: yellow;">\\g<0></span>', text_parts[1],
                                             flags=re.IGNORECASE)
                    highlight_text = "p class=".join([text_parts[0], highlight_text_])
                else:
                    highlight_text = re.sub(re.escape(parameter['word']),
                                            '<span style="background-color: yellow;">\\g<0></span>', clause_text,
                                            flags=re.IGNORECASE)

                # Create the contract entry if it doesn't exist
                if contract_name not in highlight_dict:
                    highlight_dict[contract_name] = {
                        "contract_name": contract_name,
                        "contract_id": contract_id,
                        "sections": []
                    }

                # Find the section within the contract
                section_exists = False
                for section in highlight_dict[contract_name]["sections"]:
                    if section["section_title"] == section_title:
                        section["clauses"].append({
                            "number": clause_number,
                            "clause": highlight_text
                        })
                        section_exists = True
                        break

                # If the section does not exist, create a new section
                if not section_exists:
                    highlight_dict[contract_name]["sections"].append({
                        "section_title": section_title,
                        "clauses": [{
                            "number": clause_number,
                            "clause": highlight_text
                        }]
                    })

        # Convert the dictionary to the desired list format
        highlight_list = [{
            "contract_name": contract["contract_name"],
            "contract_id": contract["contract_id"],
            "sections": contract["sections"]
        } for contract in highlight_dict.values()]

    finally:
        cursor.close()
        connection.close()

    return {'highlight_text': highlight_list if parameter['word'] and not parameter['word'].isspace() else []}





if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port = 7001)
