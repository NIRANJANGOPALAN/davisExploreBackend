from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import os
import json
from sqlalchemy import create_engine, inspect
from pymongo import MongoClient
from bson.json_util import dumps
from sqlalchemy import text
import logging
logging.basicConfig(level=logging.ERROR)

import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate



app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "https://davis-connect.vercel.app/"}})
CORS(app, resources={r"/auth/*": {"origins": "https://davis-connect.vercel.app"}})

PASSWORD = "Test01"

# In-memory storage for authenticated sessions
authenticated_sessions = {}

@app.route('/auth/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if password == PASSWORD:
        session_id = str(uuid.uuid4())
        authenticated_sessions[session_id] = username
        return jsonify({"status": "success", "session_id": session_id, "username": username}), 200
    else:
        return jsonify({"status": "failed", "message": "Invalid password"}), 401

@app.route('/auth/logout', methods=['POST'])
def logout():
    session_id = request.json.get('session_id')
    if session_id in authenticated_sessions:
        del authenticated_sessions[session_id]
        return jsonify({"status": "success", "message": "Logged out successfully"}), 200
    return jsonify({"status": "failed", "message": "Invalid session"}), 401

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/process-file', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        # Create the uploads directory if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            headers = df.columns.tolist()
            data_types = df.dtypes.apply(lambda x: x.name).tolist()
            
            # Correlation matrix for numerical columns
            numeric_df = df.select_dtypes(include=[np.number])
            correlation_matrix = numeric_df.corr().round(2).fillna(0).to_dict()
            
            # NEW CODE: Prepare data for charts
            chart_data = {}
            for header in headers:
                if df[header].dtype in ['int64', 'float64']:
                    data = df[header].tolist()
                    
                    # Scatter plot data
                    scatter_data = [{'x': i, 'y': val} for i, val in enumerate(data)]
                    
                    # Bar chart data
                    unique_values, counts = np.unique(data, return_counts=True)
                    bar_data = {
                        'labels': [str(val) for val in unique_values],
                        'datasets': [{
                            'label': header,
                            'data': counts.tolist(),
                        }]
                    }
                    
                    # Histogram data
                    hist, bin_edges = np.histogram(data, bins='auto')
                    histogram_data = {
                        'labels': [f'{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}' for i in range(len(bin_edges)-1)],
                        'datasets': [{
                            'label': header,
                            'data': hist.tolist(),
                        }]
                    }
                    
                    chart_data[header] = {
                        'scatter': scatter_data,
                        'bar': bar_data,
                        'histogram': histogram_data
                    }
            
            # MODIFIED: Added chart_data to the result
            result = {
                "headers": [{"name": name, "type": dtype} for name, dtype in zip(headers, data_types)],
                "correlation_matrix": correlation_matrix,
                "chart_data": chart_data
            }
            
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            # Attempt to remove the file, but don't raise an error if it fails
            try:
                os.remove(file_path)
            except OSError:
                pass  # Ignore errors when trying to remove the file
    else:
        return jsonify({"error": "File type not allowed"}), 400

# NEW ROUTE: Added to fetch specific chart data if needed
@app.route('/get-chart-data', methods=['POST'])
def get_chart_data():
    data = request.json
    header = data.get('header')
    chart_data = data.get('chart_data')
    
    if not header or not chart_data or header not in chart_data:
        return jsonify({"error": "Invalid header or missing chart data"}), 400
    
    return jsonify(chart_data[header])

def create_db_engine(db_type, username, password, host, port, database):
    if db_type == 'postgresql':
        return create_engine(f'postgresql://{username}:{password}@{host}:{port}/{database}')
    elif db_type == 'mysql':
        return create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')
    elif db_type == 'oracle':
        return create_engine(f'oracle+cx_oracle://{username}:{password}@{host}:{port}/{database}')
    elif db_type == 'sqlserver':
        return create_engine(f'mssql+pyodbc://{username}:{password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server')
    else:
        raise ValueError('Unsupported database type')

@app.route('/api/connect', methods=['POST'])
def connect_db():
    data = request.json
    db_type = data['dbType']
    host = data['host']
    port = data['port']
    username = data['username']
    password = data['password']
    database = data['database']

    CLUSTER_NAME = "testCluster01"
    CLUSTER_PWD = "Discover42"

    try:
        engine = create_db_engine(db_type, username, password, host, port, database)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        return jsonify({'tables': tables})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
def _convert_to_native_type(value):
    """
    Convert various data types to JSON-serializable native Python types.
    """
    if pd.isna(value):
        return None
    elif isinstance(value, (pd.Timestamp, np.datetime64)):
        return value.isoformat()
    elif hasattr(value, 'item'):
        return value.item()
    return value

@app.route('/api/table-details', methods=['POST'])
def get_table_details():
    data = request.json
    db_type = data['dbType']
    host = data['host']
    port = data['port']
    username = data['username']
    password = data['password']
    database = data['database']
    table_name = data['tableName']

    try:
        engine = create_db_engine(db_type, username, password, host, port, database)
        inspector = inspect(engine)

        # Get column details for the specified table
        columns = inspector.get_columns(table_name)
        column_details = []
        for col in columns:
            if isinstance(col, dict):
                column_details.append({
                    'name': col.get('name', 'Unknown'),
                    'type': str(col.get('type', 'Unknown'))
                })
            else:
                app.logger.warning(f"Unexpected column data type: {type(col)}")

        # Get primary key information
        pk = inspector.get_pk_constraint(table_name)
        primary_key = pk.get('constrained_columns', []) if isinstance(pk, dict) else []

        # Get foreign key information
        fks = inspector.get_foreign_keys(table_name)
        foreign_keys = []
        for fk in fks:
            if isinstance(fk, dict):
                foreign_keys.append({
                    'columns': fk.get('constrained_columns', []),
                    'referredTable': fk.get('referred_table', 'Unknown')
                })

        # Get index information
        indexes = inspector.get_indexes(table_name)
        index_details = []
        for idx in indexes:
            if isinstance(idx, dict):
                index_details.append({
                    'name': idx.get('name', 'Unknown'),
                    'columns': idx.get('column_names', [])
                })

        # Records Count
        with engine.connect() as connection:
            result = connection.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            record_count = result.scalar()

        # Get descriptive summary
        with engine.connect() as connection:
            df = pd.read_sql_table(table_name, connection)

             # Identify column types
            numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_columns = df.select_dtypes(include=['datetime64', 'timedelta64']).columns.tolist()
            print("numerical_columns", numerical_columns)
            print("categorical_columns",categorical_columns)
            print("datetime_columns",datetime_columns)

            # Numerical Summary
            numerical_summary = {}
            if numerical_columns:
                numerical_summary = df[numerical_columns].describe().to_dict()
                # Convert to native Python types
                for col in numerical_summary:
                    for stat in numerical_summary[col]:
                        numerical_summary[col][stat] = _convert_to_native_type(numerical_summary[col][stat])

            print("numerical_summary",numerical_summary)
            summary = df.describe(include='all').to_dict()

            # Convert numpy types to Python native types for JSON serialization
            for col in summary:
                for stat in summary[col]:
                    if isinstance(summary[col][stat], pd.Timestamp):
                        summary[col][stat] = summary[col][stat].isoformat()
                    elif pd.isna(summary[col][stat]):
                        summary[col][stat] = None
                    else:
                        summary[col][stat] = summary[col][stat].item() if hasattr(summary[col][stat], 'item') else summary[col][stat]

        return jsonify({
            'tableName': table_name,
            'columns': column_details,
            'primaryKey': primary_key,
            'foreignKeys': foreign_keys,
            'indexes': index_details,
            'recordCount': record_count,
            'summary': summary
        })

    except Exception as e:
        app.logger.error(f"Error in get_table_details: {str(e)}")
        app.logger.exception("Exception traceback:")
        return jsonify({'error': str(e)}), 500

@app.route('/api/table-records', methods=['POST'])
def get_table_records():
    data = request.json
    db_type = data['dbType']
    host = data['host']
    port = data['port']
    username = data['username']
    password = data['password']
    database = data['database']
    table_name = data['tableName']
    page = data.get('page', 1)
    per_page = data.get('perPage', 100)

    try:
        engine = create_db_engine(db_type, username, password, host, port, database)
        
        with engine.connect() as connection:
            # Get total count of records
            count_query = text(f"SELECT COUNT(*) FROM {table_name}")
            total_records = connection.execute(count_query).scalar()

            # Calculate offset
            offset = (page - 1) * per_page

            # Fetch paginated records
            query = text(f"SELECT * FROM {table_name} LIMIT :limit OFFSET :offset")
            result = connection.execute(query, {"limit": per_page, "offset": offset})
            
            # Convert to list of dictionaries
            records = [dict(row._mapping) for row in result]

        return jsonify({
            'records': records,
            'totalRecords': total_records,
            'page': page,
            'perPage': per_page
        })

    except Exception as e:
        logging.error(f"Error in get_table_details: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
# Set up Google API key (consider using environment variables for security)
os.environ["GOOGLE_API_KEY"] = "AIzaSyC_IfQGhoWi03gsMAlhSJyCd1LXx8i_xbA"

# Custom prompt template with all required variables
custom_prompt = PromptTemplate(
    input_variables=["input", "top_k", "table_info"],
    template="""You are an expert SQL developer. Given an input question, create a syntactically correct SQL query to run.
    
    Use the following context about the database schema:
    {table_info}
    
    # Use the top {top_k} results if applicable.
    Human: {input}
    AI: Based on the given schema and your question, here's the SQL query to answer your request:
    ```sql
    """
)

@app.route('/api/generate-sql-query', methods=['POST'])
def generate_query():
    # Parse incoming JSON data
    data = request.json
    question = data.get('question')
    
    # Default database connection details (you may want to make these configurable)
    db_type = "postgresql"
    host = "localhost"
    port = "5432"
    username = "postgres"
    password = "Database42"
    database = "TestDB"
    
    # Validate input
    if not question:
        return jsonify({"error": "Missing question"}), 400
    
    try:
        # Create the database connection
        db_uri = f"{db_type}://{username}:{password}@{host}:{port}/{database}"
        db = SQLDatabase.from_uri(db_uri)
        
        # Initialize Gemini AI model
        llm = GoogleGenerativeAI(model="gemini-pro", temperature=0)
        
        # Create the SQL query chain
        chain = create_sql_query_chain(llm, db, prompt=custom_prompt)
        
        # Fetch table schema information
        print("Fetching table schema information...")
        table_info = db.get_table_info()
        print("Schema retrieved successfully. Generating query...")
        
        # Invoke the chain to generate the SQL query
        sql_query = chain.invoke({
            "question": question,
            "top_k": 5,  # Adjustable top results
            "table_info": table_info
        })
        
        return jsonify({"query": sql_query}), 200
    
    except Exception as e:
        print(f"Error generating SQL query: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/column-headers', methods=['POST'])
def get_column_headers():
    data = request.json
    db_type = data['dbType']
    host = data['host']
    port = data['port']
    username = data['username']
    password = data['password']
    database = data['database']
    table_name = data['tableName']

    try:
        engine = create_db_engine(db_type, username, password, host, port, database)
        inspector = inspect(engine)

        # Get columns for the specified table
        columns = inspector.get_columns(table_name)
        column_names = [col['name'] for col in columns]
        print(column_names)

        return jsonify({'columnHeaders': column_names})
    
    except Exception as e:
        app.logger.error(f"Error in get_column_headers: {str(e)}")
        return jsonify({'error': str(e)}), 500
# # MongoDB connection
# CLUSTER_NAME = "testCluster01"
# CLUSTER_PWD = "Discover42"
# uri = f""

# client = MongoClient(uri)
# db = client['user_feedback']  # database name

# # Ensure the collection exists
# if 'userqueries' not in db.list_collection_names():
#     db.create_collection('userqueries')

# collection = db['userqueries']  # get reference to existing collection

# @app.route('/api/submit-query', methods=['POST'])
# def submit_query():
#     try:
#         data = request.json
#         # Validate data
#         if not all(key in data for key in ['name', 'email', 'query']):
#             return jsonify({"error": "Missing required fields"}), 400

#         # Insert data into existing MongoDB collection
#         result = collection.insert_one(data)

#         if result.inserted_id:
#             return jsonify({"message": "Query submitted successfully"}), 200
#         else:
#             return jsonify({"error": "Failed to submit query"}), 500

#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return jsonify({"error": "Internal server error"}), 500

# @app.route('/api/get-queries', methods=['GET'])
# def get_queries():
#     try:
#         queries = list(collection.find())
#         return dumps(queries), 200
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))