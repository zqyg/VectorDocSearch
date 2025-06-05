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
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
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
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', 'pcsk_3j8EYZ_aXgqmciqjA3fhuBgq8bBz2G1cYFbZ4PRdnrMwwres9UsRUPdYjgyKKHH2a7Uz3')
PINECONE_ENVIRONMENT = os.environ.get('PINECONE_ENVIRONMENT', 'us-west1-gcp')
PINECONE_INDEX = os.environ.get('PINECONE_INDEX', 'ragdata')
HUGGINGFACE_TOKEN = os.environ.get('HUGGINGFACE_TOKEN', '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')

# Initialize Pinecone
pc = None
index = None

try:
    if PINECONE_API_KEY:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index exists, create if it doesn't
        existing_indexes = pc.list_indexes().names()
        if PINECONE_INDEX not in existing_indexes:
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=384,  # all-MiniLM-L6-v2 embedding dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='gcp',
                    region='us-west1'
                )
            )
        
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

def get_embedding(text):
    """Get embedding from Hugging Face hosted service"""
    try:
        # Use Gradio client for Hugging Face embedding service with authentication
        from gradio_client import Client
        
        client = Client("ZQqqqygy/embeddingservice", hf_token=HUGGINGFACE_TOKEN)
        result = client.predict(
            text=text,
            api_name="/predict"
        )
        
        # Result should be a list of embeddings
        if isinstance(result, list):
            return result
        else:
            app.logger.error(f"Unexpected embedding result format: {type(result)}")
            return None
            
    except Exception as e:
        app.logger.error(f"Error getting embedding: {str(e)}")
        return None

@app.route('/')
def index():
    """Home page - redirect to upload"""
    return redirect(url_for('upload'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handle document upload"""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        description = request.form.get('description', '').strip()
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if not description:
            flash('Please provide a description for the document', 'error')
            return redirect(request.url)
        
        if file and file.filename and allowed_file(file.filename):
            try:
                # Secure the filename
                original_filename = file.filename or "unknown"
                filename = secure_filename(original_filename)
                
                # Add timestamp to prevent filename conflicts
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
                filename = timestamp + filename
                
                # Save file
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Get file hash for unique ID
                file_hash = get_file_hash(filepath)
                
                # Get embedding for description
                embedding = get_embedding(description)
                
                if embedding and index:
                    # Store in Pinecone
                    metadata = {
                        'filename': filename,
                        'original_filename': file.filename,
                        'description': description,
                        'upload_date': datetime.now().isoformat(),
                        'file_size': os.path.getsize(filepath)
                    }
                    
                    # Upsert to Pinecone
                    index.upsert([(file_hash, embedding, metadata)])
                    
                    flash(f'Document "{file.filename}" uploaded successfully!', 'success')
                    app.logger.info(f"Document uploaded and indexed: {filename}")
                else:
                    flash('Document uploaded but indexing failed. Search may not work properly.', 'warning')
                    app.logger.warning(f"Document uploaded but not indexed: {filename}")
                
                return redirect(url_for('upload'))
                
            except Exception as e:
                app.logger.error(f"Error uploading file: {str(e)}")
                flash('An error occurred while uploading the file', 'error')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload PDF, DOC, DOCX, or TXT files only.', 'error')
            return redirect(request.url)
    
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
                        top_k=10,
                        include_metadata=True
                    )
                    
                    # Process results
                    for match in search_results['matches']:
                        if match['score'] > 0.7:  # Only show relevant results
                            metadata = match['metadata']
                            results.append({
                                'filename': metadata.get('original_filename', 'Unknown'),
                                'description': metadata.get('description', ''),
                                'score': round(match['score'], 3),
                                'upload_date': metadata.get('upload_date', ''),
                                'download_filename': metadata.get('filename', ''),
                                'file_size': metadata.get('file_size', 0)
                            })
                    
                    if not results:
                        flash('No relevant documents found for your query', 'info')
                    else:
                        app.logger.info(f"Search completed: {len(results)} results for query '{query}'")
                else:
                    flash('Failed to process search query', 'error')
                    
            except Exception as e:
                app.logger.error(f"Error during search: {str(e)}")
                flash('An error occurred while searching', 'error')
    
    return render_template('search.html', results=results, query=query)

@app.route('/download/<filename>')
def download_file(filename):
    """Download uploaded file"""
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
    except Exception as e:
        app.logger.error(f"Error downloading file {filename}: {str(e)}")
        flash('File not found or cannot be downloaded', 'error')
        return redirect(url_for('search'))

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
                         title='Page Not Found',
                         content='<div class="text-center"><h1 class="text-2xl font-bold mb-4">Page Not Found</h1><p>The page you are looking for does not exist.</p></div>'), 404

@app.errorhandler(500)
def server_error(e):
    app.logger.error(f"Server error: {str(e)}")
    return render_template('base.html',
                         title='Server Error', 
                         content='<div class="text-center"><h1 class="text-2xl font-bold mb-4">Server Error</h1><p>An internal server error occurred.</p></div>'), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
