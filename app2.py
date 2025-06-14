import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import requests
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import hashlib
import uuid
from datetime import datetime, timedelta
from google.cloud import storage
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from google.oauth2 import service_account
import io
import json
import PyPDF2
import docx

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Pinecone configuration
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.environ.get('PINECONE_ENVIRONMENT')
PINECONE_INDEX = os.environ.get('PINECONE_INDEX')
HUGGINGFACE_TOKEN = os.environ.get('HUGGINGFACE_TOKEN')

# Google Cloud Storage configuration
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "zelio.json"
BUCKET_NAME = os.environ.get('BUCKET_NAME')

# Google Drive folder IDs
AUTO_IMPORT_DRIVE_FOLDER_ID = os.environ.get('AUTO_IMPORT_DRIVE_FOLDER_ID')
GOOGLE_DRIVE_CREDENTIALS_FILE = "drive_credentials.json"
GOOGLE_DRIVE_DELETE_FOLDER_ID = os.environ.get('GOOGLE_DRIVE_DELETE_FOLDER_ID')
SOURCE_DRIVE_FOLDER_ID = os.environ.get('SOURCE_DRIVE_FOLDER_ID')
DEST_DRIVE_FOLDER_ID = os.environ.get('DEST_DRIVE_FOLDER_ID')

# LLM configuration
OPENROUTER_API_KEY = os.environ.get('openrouter_secret_key')
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Initialize Pinecone and Google Cloud Storage
pc = None
index = None
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

try:
    if PINECONE_API_KEY:
        pc = Pinecone(api_key=PINECONE_API_KEY)

        # Check if index exists, create if it doesn't
        try:
            # Try to describe the index first to see if it exists
            index_info = pc.describe_index(PINECONE_INDEX)
            app.logger.info(f"Index {PINECONE_INDEX} already exists")
        except Exception:
            # Index doesn't exist, create it
            app.logger.info(f"Creating index {PINECONE_INDEX}")
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=384,  # all-MiniLM-L6-v2 embedding dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='gcp',
                    region='us-west1'
                )
            )
            # Wait for index to be ready
            import time
            time.sleep(10)

        # Get the index
        index = pc.Index(PINECONE_INDEX)

        app.logger.info(f"Successfully connected to Pinecone index: {PINECONE_INDEX}")
    else:
        app.logger.warning("PINECONE_API_KEY not found in environment variables")
except Exception as e:
    app.logger.error(f"Failed to initialize Pinecone: {str(e)}")
    index = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_hash(filepath):
    """Generate hash for file to use as unique ID"""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_embedding(text, title=None):
    """Get embedding from Hugging Face hosted service, using both title and description if available."""
    try:
        # Combine title and description for richer embedding
        if title:
            full_text = f"{title}\n\n{str(text)}"
        else:
            full_text = str(text)
        # Use Gradio client for Hugging Face embedding service with authentication
        from gradio_client import Client

        client = Client("ZQqqqygy/embeddingservice", hf_token=HUGGINGFACE_TOKEN)
        result = client.predict(
            text=full_text,
            api_name="/predict"
        )

        app.logger.debug(f"Embedding service returned: {type(result)} - {result}")

        # Handle different result formats from Hugging Face service
        if isinstance(result, list):
            return result
        elif isinstance(result, dict):
            # If it's a dict, try to extract the embedding data
            if 'data' in result:
                return result['data']
            elif 'embedding' in result:
                return result['embedding']
            elif len(result) == 1:
                # If dict has one key, return its value
                key = list(result.keys())[0]
                return result[key]
            else:
                app.logger.error(f"Dictionary result without expected keys: {result}")
                return None
        else:
            app.logger.error(f"Unexpected embedding result format: {type(result)} - {result}")
            return None

    except Exception as e:
        app.logger.error(f"Error getting embedding: {str(e)}")
        return None

@app.route('/')
def index_route():
    """Home page - redirect to upload"""
    return redirect(url_for('upload'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handle multiple document uploads directly to Google Drive"""
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No files selected'})

        files = request.files.getlist('file')
        if not files or files[0].filename == '':
            return jsonify({'success': False, 'error': 'No files selected'})

        uploaded_files = []
        error_files = []

        for file in files:
            if file and file.filename and allowed_file(file.filename):
                try:
                    file_id = str(uuid.uuid4())
                    original_filename = file.filename
                    safe_filename = f"{file_id}_{secure_filename(original_filename)}"
                    
                    # Save file temporarily
                    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
                    file.save(temp_path)

                    try:
                        creds = service_account.Credentials.from_service_account_file(
                            GOOGLE_DRIVE_CREDENTIALS_FILE,
                            scopes=['https://www.googleapis.com/auth/drive.file']
                        )
                        service = build('drive', 'v3', credentials=creds)
                        
                        # Upload the document
                        file_metadata = {
                            'name': safe_filename,
                            'parents': [SOURCE_DRIVE_FOLDER_ID]
                        }
                        
                        with open(temp_path, 'rb') as f:
                            media = MediaIoBaseUpload(f, mimetype='application/octet-stream', resumable=True)
                            uploaded_file = service.files().create(
                                body=file_metadata,
                                media_body=media,
                                fields='id'
                            ).execute()
                        
                        # Create initial JSON metadata file
                        json_data = {
                            "description": "",  # Empty description, will be updated later
                            "original_filename": original_filename,
                            "upload_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_"),
                            "drive_file_id": uploaded_file.get('id'),
                            "id": file_id
                        }
                        
                        json_filename = f"{file_id}.json"
                        json_metadata = {
                            'name': json_filename,
                            'parents': [SOURCE_DRIVE_FOLDER_ID]
                        }
                        
                        json_content = json.dumps(json_data, indent=2)
                        media = MediaIoBaseUpload(
                            io.BytesIO(json_content.encode()),
                            mimetype='application/json',
                            resumable=True
                        )
                        
                        json_file = service.files().create(
                            body=json_metadata,
                            media_body=media,
                            fields='id'
                        ).execute()

                        uploaded_files.append(original_filename)

                        # Start background processing
                        def process_file_async():
                            try:
                                # Extract content for LLM
                                with open(temp_path, 'rb') as f:
                                    content = extract_document_content(f, original_filename)

                                # Get description from LLM
                                if content:
                                    description = get_document_description(content, original_filename)
                                    if description:
                                        # Update JSON with description
                                        json_data['description'] = description
                                        updated_content = json.dumps(json_data, indent=2)
                                        media = MediaIoBaseUpload(
                                            io.BytesIO(updated_content.encode()),
                                            mimetype='application/json',
                                            resumable=True
                                        )
                                        service.files().update(
                                            fileId=json_file['id'],
                                            media_body=media
                                        ).execute()
                            finally:
                                # Clean up temp file
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)

                        # Start background processing in a new thread
                        from threading import Thread
                        Thread(target=process_file_async).start()
                            
                    except Exception as e:
                        error_files.append(original_filename)
                        app.logger.error(f"Drive upload error for {original_filename}: {str(e)}")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                
                except Exception as e:
                    error_files.append(file.filename)
                    app.logger.error(f"Error handling file {file.filename}: {str(e)}")
            else:
                error_files.append(file.filename)

        response = {
            'success': True,
            'uploaded': uploaded_files,
            'errors': error_files,
            'message': f"Successfully uploaded {len(uploaded_files)} file(s)"
        }
        if error_files:
            response['message'] += f" ({len(error_files)} failed)"
        
        return jsonify(response)

    return render_template('upload.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    """Handle document search"""
    results = []
    query = ''

    if request.method == 'POST':
        query = request.form.get('query', '').strip()

        if not query:
            flash('Please enter a search query', 'error')
        elif not index:
            flash('Search service is not available. Please check configuration.', 'error')
        else:
            try:
                # Get embedding for search query
                query_embedding = get_embedding(query)

                if query_embedding:
                    # Search in Pinecone
                    search_results = index.query(
                        vector=query_embedding,
                        top_k=50,
                        include_metadata=True,
                        include_values=True
                    )

                    for idx, match in enumerate(search_results['matches'], 1):
                        relevance_score = match['score']
                        if relevance_score > 0.2:
                            metadata = match['metadata']
                            results.append({
                                'rank': idx,
                                'filename': metadata.get('original_filename', 'Unknown'),
                                'description': metadata.get('description', ''),
                                'score': round(relevance_score, 3),
                                'upload_date': metadata.get('upload_date', ''),
                                'download_filename': metadata.get('filename', ''),
                                'file_size': metadata.get('file_size', 0),
                                'gcs_path': metadata.get('gcs_path', '')
                            })

                    results.sort(key=lambda x: x['score'], reverse=True)

                    if not results:
                        flash('No relevant documents found for your query', 'info')
                        app.logger.info(f"No results found for query: '{query}'")
                    else:
                        app.logger.info(f"Search completed: {len(results)} results (>20% match) for query '{query}'")

            except Exception as e:
                app.logger.error(f"Error during search: {str(e)}")
                flash('An error occurred while searching', 'error')

    return render_template('search.html', results=results, query=query)

@app.route('/preview/<filename>')
def preview_file(filename):
    """
    Preview/download file from Google Cloud Storage using a signed URL.
    This can be used for in-browser preview or download.
    """
    try:
        blob = bucket.blob(filename)
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=15),
            method="GET"
        )
        return redirect(url)
    except Exception as e:
        app.logger.error(f"Error previewing file {filename}: {str(e)}")
        flash('File not found or cannot be previewed/downloaded', 'error')
        return redirect(url_for('search'))

@app.route('/admin/preview-document/<file_id>')
def admin_preview_document(file_id):
    """
    Get a preview of a document and its metadata from Google Drive
    """
    try:
        creds = service_account.Credentials.from_service_account_file(
            GOOGLE_DRIVE_CREDENTIALS_FILE,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        service = build('drive', 'v3', credentials=creds)
        
        # Download document content
        request = service.files().get_media(fileId=file_id)
        doc_stream = io.BytesIO()
        downloader = MediaIoBaseDownload(doc_stream, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        doc_stream.seek(0)
        
        # Get file metadata
        file_metadata = service.files().get(fileId=file_id, fields='name,mimeType').execute()
        
        # Extract content based on file type
        content = extract_document_content(doc_stream, file_metadata['name'])
        
        # If it's a PDF, we might want to limit the preview
        if file_metadata['mimeType'] == 'application/pdf':
            content = content[:2000] + '...' if len(content) > 2000 else content
        
        return jsonify({
            'success': True,
            'content': content,
            'metadata': file_metadata
        })
        
    except Exception as e:
        app.logger.error(f"Error previewing document {file_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'pinecone_connected': index is not None,
        'upload_folder_exists': os.path.exists(UPLOAD_FOLDER)
    }
    return jsonify(status)

@app.errorhandler(413)
def too_large(e):
    flash('File is too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('upload'))

@app.errorhandler(404)
def not_found(e):
    return render_template('base.html', 
                         title='Page raNot Found',
                         content='<div class="text-center"><h1 class="text-2xl font-bold mb-4">Page Not Found</h1><p>The page you are looking for does not exist.</p></div>'), 404

@app.errorhandler(500)
def server_error(e):
    app.logger.error(f"Server error: {str(e)}")
    return render_template('base.html',
                         title='Server Error', 
                         content='<div class="text-center"><h1 class="text-2xl font-bold mb-4">Server Error</h1><p>An internal server error occurred.</p></div>'), 500

@app.route('/admin/import-drive', methods=['POST'])
def admin_import_drive():
    """
    Trigger the scan_and_import_drive_folder process and return logs.
    """
    log = []
    def log_fn(msg):
        log.append(msg)
        app.logger.info(msg)

    try:
        scan_and_import_drive_folder(log_fn=log_fn)
        return jsonify(success=True, log=log)
    except Exception as e:
        app.logger.error(f"Admin import failed: {e}")
        return jsonify(success=False, error=str(e)), 500

@app.route('/admin/clear-drive-folder', methods=['POST'])
def admin_clear_drive_folder():
    """
    Delete all files in the specified Google Drive folder.
    """
    log = []
    def log_fn(msg):
        log.append(msg)
        app.logger.info(msg)
    try:
        delete_all_files_in_drive_folder(GOOGLE_DRIVE_DELETE_FOLDER_ID, log_fn=log_fn)
        return jsonify(success=True, log=log)
    except Exception as e:
        app.logger.error(f"Admin clear drive folder failed: {e}")
        return jsonify(success=False, error=str(e)), 500

@app.route('/admin')
def admin():
    """Admin page for importing documents from Google Drive."""
    return render_template('admin.html')

@app.route('/admin/list-drive-docs', methods=['GET'])
def admin_list_drive_docs():
    """
    List all document+json pairs in the source Google Drive folder.
    """
    try:
        creds = service_account.Credentials.from_service_account_file(
            GOOGLE_DRIVE_CREDENTIALS_FILE,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        service = build('drive', 'v3', credentials=creds)
        query = f"'{SOURCE_DRIVE_FOLDER_ID}' in parents and trashed = false"
        files = []
        page_token = None
        while True:
            results = service.files().list(
                q=query,
                fields="nextPageToken, files(id, name, mimeType)",
                pageToken=page_token
            ).execute()
            files.extend(results.get('files', []))
            page_token = results.get('nextPageToken', None)
            if not page_token:
                break

        # Group files by id prefix (before underscore for docs, before .json for jsons)
        docs = {}
        jsons = {}
        for f in files:
            if f['name'].endswith('.json'):
                base_id = f['name'].replace('.json', '')
                jsons[base_id] = f
            elif '_' in f['name']:
                doc_id = f['name'].split('_')[0]
                docs[doc_id] = f

        # Only show pairs where both doc and json exist
        pairs = []
        for doc_id, doc_file in docs.items():
            json_file = jsons.get(doc_id)
            if json_file:
                pairs.append({
                    "id": doc_id,
                    "doc": {"id": doc_file['id'], "name": doc_file['name']},
                    "json": {"id": json_file['id'], "name": json_file['name']}
                })
        return jsonify(success=True, pairs=pairs)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

@app.route('/admin/move-drive-docs', methods=['POST'])
def admin_move_drive_docs():
    """
    Move selected doc+json pairs from source to destination folder.
    Expects JSON: { "ids": [id1, id2, ...] }
    """
    data = request.get_json()
    ids = data.get('ids', [])
    log = []
    try:
        creds = service_account.Credentials.from_service_account_file(
            GOOGLE_DRIVE_CREDENTIALS_FILE,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        service = build('drive', 'v3', credentials=creds)
        # List all files in source folder
        query = f"'{SOURCE_DRIVE_FOLDER_ID}' in parents and trashed = false"
        files = []
        page_token = None
        while True:
            results = service.files().list(
                q=query,
                fields="nextPageToken, files(id, name)",
                pageToken=page_token
            ).execute()
            files.extend(results.get('files', []))
            page_token = results.get('nextPageToken', None)
            if not page_token:
                break
        docs = {}
        jsons = {}
        for f in files:
            if f['name'].endswith('.json'):
                base_id = f['name'].replace('.json', '')
                jsons[base_id] = f
            elif '_' in f['name']:
                doc_id = f['name'].split('_')[0]
                docs[doc_id] = f

        moved_files = []
        for doc_id in ids:
            doc_file = docs.get(doc_id)
            json_file = jsons.get(doc_id)
            for f in [doc_file, json_file]:
                if f:
                    try:
                        # Move file to destination folder
                        service.files().update(
                            fileId=f['id'],
                            addParents=DEST_DRIVE_FOLDER_ID,
                            removeParents=SOURCE_DRIVE_FOLDER_ID,
                            fields='id, parents'
                        ).execute()
                        moved_files.append(doc_id)
                        log.append(f"Moved: {f['name']}")
                    except Exception as e:
                        log.append(f"Failed to move {f['name']}: {e}")

        # After moving files, trigger import process for moved files
        if moved_files:
            import_result = import_moved_documents(moved_files, service, log)
            log.extend(import_result)

        return jsonify(success=True, log=log)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

def import_moved_documents(doc_ids, service, log):
    """
    Import moved documents into GCS and Pinecone.
    """
    import_logs = []
    
    for doc_id in doc_ids:
        try:
            # List files in destination folder
            query = f"'{DEST_DRIVE_FOLDER_ID}' in parents and trashed = false"
            results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
            files = results.get('files', [])

            # Find matching document and JSON files
            doc_file = None
            json_file = None
            for f in files:
                if f['name'].startswith(doc_id):
                    if f['name'].endswith('.json'):
                        json_file = f
                    else:
                        doc_file = f

            if not doc_file or not json_file:
                import_logs.append(f"Skipped import for {doc_id}: Missing files")
                continue

            # Download document
            doc_stream = io.BytesIO()
            request = service.files().get_media(fileId=doc_file['id'])
            downloader = MediaIoBaseDownload(doc_stream, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            doc_stream.seek(0)

            # Download JSON metadata
            json_stream = io.BytesIO()
            request = service.files().get_media(fileId=json_file['id'])
            downloader = MediaIoBaseDownload(json_stream, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            json_stream.seek(0)
            metadata = json.load(json_stream)

            # Prepare metadata fields
            description = metadata.get('description', '')
            original_filename = metadata.get('original_filename', doc_file['name'])
            upload_timestamp = metadata.get('upload_timestamp', '')
            file_hash = doc_id

            # Save document to a temp file for GCS upload
            gcs_filename = doc_file['name']
            temp_path = os.path.join(UPLOAD_FOLDER, gcs_filename)
            with open(temp_path, 'wb') as f:
                f.write(doc_stream.read())

            # Upload document to Google Cloud Storage
            blob = bucket.blob(gcs_filename)
            blob.upload_from_filename(temp_path)
            os.remove(temp_path)

            # Get embedding for description and title
            embedding = get_embedding(description, title=original_filename)

            if embedding and index:
                pinecone_metadata = {
                    'filename': gcs_filename,
                    'original_filename': original_filename,
                    'description': description,
                    'upload_date': upload_timestamp,
                    'file_size': blob.size,
                    'gcs_path': f"gs://{BUCKET_NAME}/{gcs_filename}"
                }
                
                index.upsert(
                    vectors=[{
                        "id": file_hash,
                        "values": embedding,
                        "metadata": pinecone_metadata
                    }]
                )
                import_logs.append(f"Successfully imported: {gcs_filename}")
            else:
                blob.delete()
                import_logs.append(f"Failed to import {gcs_filename}: Embedding or index error")

        except Exception as e:
            import_logs.append(f"Error importing {doc_id}: {str(e)}")

    return import_logs

def scan_and_import_drive_folder(log_fn=None):
    """
    Scan the Google Drive folder for new documents and their JSON metadata,
    upload to Google Cloud Storage, and index in Pinecone.
    Only process if both a document and its matching JSON exist (by ID).
    Accepts log_fn for logging progress.
    """
    # Authenticate and build Drive service
    creds = service_account.Credentials.from_service_account_file(
        GOOGLE_DRIVE_CREDENTIALS_FILE,
        scopes=['https://www.googleapis.com/auth/drive']
    )
    service = build('drive', 'v3', credentials=creds)

    # List all files in the folder
    query = f"'{AUTO_IMPORT_DRIVE_FOLDER_ID}' in parents and trashed = false"
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    files = results.get('files', [])

    # Group files by Drive file ID (before underscore for docs, or id for json)
    docs = {}
    jsons = {}
    for f in files:
        if f['name'].endswith('.json'):
            # JSON file: name is {id}.json
            base_id = f['name'].replace('.json', '')
            jsons[base_id] = f
        else:
            # Document file: name is {id}_{filename}
            if '_' in f['name']:
                doc_id = f['name'].split('_')[0]
                docs[doc_id] = f

    # Only process pairs where both doc and json exist
    for doc_id, doc_file in docs.items():
        json_file = jsons.get(doc_id)
        if not json_file:
            continue  # skip if no metadata

        # Download document
        doc_stream = io.BytesIO()
        request = service.files().get_media(fileId=doc_file['id'])
        downloader = MediaIoBaseDownload(doc_stream, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        doc_stream.seek(0)

        # Download JSON metadata
        json_stream = io.BytesIO()
        request = service.files().get_media(fileId=json_file['id'])
        downloader = MediaIoBaseDownload(json_stream, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        json_stream.seek(0)
        metadata = json.load(json_stream)

        # Prepare metadata fields
        description = metadata.get('description', '')
        original_filename = metadata.get('original_filename', doc_file['name'])
        upload_timestamp = metadata.get('upload_timestamp', '')
        file_hash = doc_id  # use doc_id as unique id

        # Save document to a temp file for GCS upload
        gcs_filename = doc_file['name']
        temp_path = os.path.join(UPLOAD_FOLDER, gcs_filename)
        with open(temp_path, 'wb') as f:
            f.write(doc_stream.read())

        # Upload document to Google Cloud Storage
        blob = bucket.blob(gcs_filename)
        blob.upload_from_filename(temp_path)
        os.remove(temp_path)

        # Get embedding for description and title
        embedding = get_embedding(description, title=original_filename)

        if embedding and index:
            pinecone_metadata = {
                'filename': gcs_filename,
                'original_filename': original_filename,
                'description': description,
                'upload_date': upload_timestamp,
                'file_size': blob.size,
                'gcs_path': f"gs://{BUCKET_NAME}/{gcs_filename}"
            }
            try:
                index.upsert(
                    vectors=[{
                        "id": file_hash,
                        "values": embedding,
                        "metadata": pinecone_metadata
                    }]
                )
                msg = f"Imported: {gcs_filename} (ID: {file_hash})"
                if log_fn:
                    log_fn(msg)
                else:
                    app.logger.info(msg)
            except Exception as pinecone_error:
                app.logger.error(f"Pinecone upsert failed for auto-import: {str(pinecone_error)}")
                blob.delete()
                if log_fn:
                    log_fn(f"Failed: {gcs_filename} (Pinecone error)")
                continue
        else:
            blob.delete()
            msg = f"Failed: {gcs_filename} (embedding or index error)"
            if log_fn:
                log_fn(msg)
            else:
                app.logger.warning(msg)

        # Optionally: Move or mark processed files in Drive to avoid re-importing

    # After processing, robustly delete all files in the Google Drive folder 1bvEJdHzySYFsR5xkQSjtk3le6hcjzRNH
    try:
        creds = service_account.Credentials.from_service_account_file(
            GOOGLE_DRIVE_CREDENTIALS_FILE,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        service = build('drive', 'v3', credentials=creds)
        folder_id = GOOGLE_DRIVE_DELETE_FOLDER_ID
        # Use the correct mimeType for files and include trashed=false
        query = f"'{folder_id}' in parents and trashed = false"
        deleted_count = 0
        page_token = None
        all_files = []
        while True:
            results = service.files().list(
                q=query,
                fields="nextPageToken, files(id, name, mimeType, owners)",
                pageToken=page_token,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True
            ).execute()
            files = results.get('files', [])
            all_files.extend(files)
            page_token = results.get('nextPageToken', None)
            if not page_token:
                break
        if log_fn:
            log_fn(f"Found {len(all_files)} files to delete in Drive folder {folder_id}.")
        else:
            app.logger.info(f"Found {len(all_files)} files to delete in Drive folder {folder_id}.")
        for f in all_files:
            # Only delete if the service account is the owner of the file
            owners = f.get('owners', [])
            is_owner = any(owner.get('me', False) for owner in owners)
            try:
                service.files().delete(fileId=f['id'], supportsAllDrives=True).execute()
                msg = f"Deleted from Drive folder {folder_id}: {f['name']} ({f['id']})"
                deleted_count += 1
                if log_fn:
                    log_fn(msg)
                else:
                    app.logger.info(msg)
            except Exception as e:
                msg = f"Failed to delete {f['name']} ({f['id']}) from Drive folder {folder_id}: {e}"
                if log_fn:
                    log_fn(msg)
                else:
                    app.logger.warning(msg)
        if log_fn:
            log_fn(f"Total files deleted from Google Drive folder {folder_id} after import: {deleted_count}")
        else:
            app.logger.info(f"Total files deleted from Google Drive folder {folder_id} after import: {deleted_count}")
    except Exception as e:
        msg = f"Error deleting files in Drive folder {GOOGLE_DRIVE_DELETE_FOLDER_ID} after import: {e}"
        if log_fn:
            log_fn(msg)
        else:
            app.logger.error(msg)

def delete_all_files_in_drive_folder(folder_id, log_fn=None):
    """
    Delete all files in the specified Google Drive folder.
    """
    try:
        creds = service_account.Credentials.from_service_account_file(
            GOOGLE_DRIVE_CREDENTIALS_FILE,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        service = build('drive', 'v3', credentials=creds)
        query = f"'{folder_id}' in parents and trashed = false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])
        for f in files:
            try:
                service.files().delete(fileId=f['id']).execute()
                msg = f"Deleted: {f['name']} ({f['id']})"
                if log_fn:
                    log_fn(msg)
                else:
                    app.logger.info(msg)
            except Exception as e:
                msg = f"Failed to delete {f['name']} ({f['id']}): {e}"
                if log_fn:
                    log_fn(msg)
                else:
                    app.logger.warning(msg)
        if log_fn:
            log_fn("All files deleted from Google Drive folder.")
        else:
            app.logger.info("All files deleted from Google Drive folder.")
    except Exception as e:
        msg = f"Error deleting files in Drive folder: {e}"
        if log_fn:
            log_fn(msg)
        else:
            app.logger.error(msg)

def extract_document_content(file_stream, filename):
    """Extract text content from supported document types"""
    content = ""
    file_ext = filename.lower().split('.')[-1]
    
    try:
        if file_ext == 'pdf':
            pdf_reader = PyPDF2.PdfReader(file_stream)
            for page in pdf_reader.pages:
                content += page.extract_text() + "\n"
        elif file_ext in ['doc', 'docx']:
            doc = docx.Document(file_stream)
            content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        elif file_ext == 'txt':
            content = file_stream.read().decode('utf-8')
        return content.strip()
    except Exception as e:
        app.logger.error(f"Error extracting content from {filename}: {str(e)}")
        return ""

def get_document_description(content, filename):
    """Get document description from LLM"""
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        system_prompt = """You are a document analyzer focused on educational content. 
        Analyze the given document and create a detailed description that includes:
        - Subject area and specific topic
        - IB course level (HL/SL) if identifiable
        - Type of content (practice questions, notes, essay, etc.)
        - Year/grade level if mentioned
        - Any assessment or grade information if present
        - Key topics or concepts covered
        Make the description detailed but concise (max 200 words) and focus on making it searchable."""
        
        # Truncate content if too long
        max_chars = 14000  # Adjust based on model context window
        content = content[:max_chars]
        
        payload = {
            "model": "deepseek/deepseek-r1-0528:free",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Filename: {filename}\n\nContent: {content}\n\nCreate a detailed description for this document."}
            ]
        }
        
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        description = response.json()['choices'][0]['message']['content'].strip()
        return description
    
    except Exception as e:
        app.logger.error(f"Error getting description from LLM: {str(e)}")
        return None

def update_empty_descriptions():
    """Scan for and update JSON files with empty descriptions"""
    try:
        creds = service_account.Credentials.from_service_account_file(
            GOOGLE_DRIVE_CREDENTIALS_FILE,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        service = build('drive', 'v3', credentials=creds)
        
        # List all files in source folder
        query = f"'{SOURCE_DRIVE_FOLDER_ID}' in parents and trashed = false"
        all_files = []
        page_token = None
        while True:
            results = service.files().list(
                q=query,
                fields="nextPageToken, files(id, name, mimeType)",
                pageToken=page_token
            ).execute()
            all_files.extend(results.get('files', []))
            page_token = results.get('nextPageToken', None)
            if not page_token:
                break
        
        # Group files
        docs = {}
        jsons = {}
        for f in all_files:
            if f['name'].endswith('.json'):
                base_id = f['name'].replace('.json', '')
                jsons[base_id] = f
            elif '_' in f['name']:
                doc_id = f['name'].split('_')[0]
                docs[doc_id] = f
        
        # Process files with empty descriptions
        for file_id, json_file in jsons.items():
            # Download and check JSON content
            json_stream = io.BytesIO()
            request = service.files().get_media(fileId=json_file['id'])
            downloader = MediaIoBaseDownload(json_stream, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            json_stream.seek(0)
            
            json_data = json.load(json_stream)
            
            # Check if description is empty
            if not json_data.get('description'):
                doc_file = docs.get(file_id)
                if not doc_file:
                    continue
                
                # Download document content
                doc_stream = io.BytesIO()
                request = service.files().get_media(fileId=doc_file['id'])
                downloader = MediaIoBaseDownload(doc_stream, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                doc_stream.seek(0)
                
                # Extract content and get description
                content = extract_document_content(doc_stream, doc_file['name'])
                if content:
                    description = get_document_description(content, doc_file['name'])
                    if description:
                        # Update JSON file
                        json_data['description'] = description
                        
                        # Upload updated JSON
                        updated_content = json.dumps(json_data, indent=2)
                        media = MediaIoBaseUpload(
                            io.BytesIO(updated_content.encode()),
                            mimetype='application/json',
                            resumable=True
                        )
                        service.files().update(
                            fileId=json_file['id'],
                            media_body=media
                        ).execute()
                        
                        app.logger.info(f"Updated description for {doc_file['name']}")
                    
        return True
    except Exception as e:
        app.logger.error(f"Error updating descriptions: {str(e)}")
        return False

# Add a new endpoint to trigger description updates
@app.route('/admin/update-descriptions', methods=['POST'])
def admin_update_descriptions():
    """Endpoint to trigger empty description updates"""
    try:
        success = update_empty_descriptions()
        if success:
            return jsonify(success=True, message="Descriptions updated successfully")
        else:
            return jsonify(success=False, message="Failed to update descriptions"), 500
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

@app.route('/admin/get-preview-url/<file_id>')
def admin_get_preview_url(file_id):
    """Generate a URL for previewing a document in Google Drive"""
    try:
        creds = service_account.Credentials.from_service_account_file(
            GOOGLE_DRIVE_CREDENTIALS_FILE,
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        service = build('drive', 'v3', credentials=creds)
        
        # Get file metadata
        file_metadata = service.files().get(
            fileId=file_id, 
            fields='webViewLink',
            supportsAllDrives=True
        ).execute()
        
        # Use the webViewLink which is more reliable for previews
        preview_url = file_metadata.get('webViewLink', '').replace('/view', '/preview')
        
        if not preview_url:
            preview_url = f"https://drive.google.com/file/d/{file_id}/preview"
        
        return jsonify({
            'success': True,
            'url': preview_url
        })
        
    except Exception as e:
        app.logger.error(f"Error generating preview URL for document {file_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)