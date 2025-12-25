"""
xserver_gae.py - Google App Engine Standard Environment Version
Image/Video Gallery Server using Google Cloud Storage

Bucket name: xserver
Favorites: Cookie-based (max 50 items, under 4KB)
"""

from flask import Flask, render_template, Response, abort, jsonify, request, make_response, redirect, url_for
from google.cloud import storage
from PIL import Image
import io
import json
import hashlib
import cv2
import numpy as np
import tempfile
import os
import functools

app = Flask(__name__)

# Configuration
GCS_BUCKET_NAME = 'xserver'
THUMBNAIL_PREFIX = 'thumbnails/'
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.mp4', '.webm', '.mov', '.avi', '.mkv'}
VIDEO_EXTENSIONS = {'.mp4', '.webm', '.mov', '.avi', '.mkv'}
MAX_FAVORITES = 50
FAVORITES_COOKIE_NAME = 'xserver_favorites'
AUTH_COOKIE_NAME = 'xserver_auth'
PASSWORD = '18'

# Initialize GCS client (uses default credentials in GAE)
storage_client = storage.Client()


def get_bucket():
    """Get the GCS bucket."""
    return storage_client.bucket(GCS_BUCKET_NAME)


# ============================================================================
# FAVORITES - Cookie Based (Max 50, under 4KB)
# ============================================================================

def load_favorites_from_cookie():
    """Load favorites from cookie."""
    try:
        cookie_value = request.cookies.get(FAVORITES_COOKIE_NAME, '')
        if not cookie_value:
            return set()
        # Decode from JSON
        favorites_list = json.loads(cookie_value)
        return set(favorites_list[:MAX_FAVORITES])  # Ensure max limit
    except (json.JSONDecodeError, TypeError):
        return set()


def create_favorites_response(response, favorites):
    """Add favorites cookie to response."""
    # Convert to list and limit to MAX_FAVORITES
    favorites_list = sorted(list(favorites))[:MAX_FAVORITES]
    cookie_value = json.dumps(favorites_list)
    
    # Set cookie with 1 year expiry
    response.set_cookie(
        FAVORITES_COOKIE_NAME,
        cookie_value,
        max_age=365 * 24 * 60 * 60,  # 1 year
        httponly=False,  # Allow JS access for UI updates
        samesite='Lax',
        secure=True  # GAE uses HTTPS
    )
    return response


def is_favorite(image_path):
    """Check if an image is in favorites."""
    favorites = load_favorites_from_cookie()
    return image_path in favorites


# ============================================================================
# GCS File Operations
# ============================================================================

def list_blobs_with_prefix(prefix='', delimiter='/'):
    """List blobs in bucket with given prefix."""
    bucket = get_bucket()
    blobs = bucket.list_blobs(prefix=prefix, delimiter=delimiter)
    return blobs


def get_subfolders():
    """Get all top-level subdirectories in the bucket."""
    try:
        bucket = get_bucket()
        # List with delimiter to get "folders"
        iterator = bucket.list_blobs(delimiter='/')
        
        # Consume the iterator to populate prefixes
        list(iterator)
        
        folders = []
        for prefix in iterator.prefixes:
            folder_name = prefix.rstrip('/')
            # Skip thumbnails folder and hidden folders
            if not folder_name.startswith('.') and folder_name != 'thumbnails':
                folders.append(folder_name)
        
        return sorted(folders)
    except Exception as e:
        print(f"Error listing folders: {e}")
        return []


def get_items_in_folder(folder_path):
    """Get all images/videos and subfolders in a specific folder."""
    try:
        bucket = get_bucket()
        prefix = folder_path.rstrip('/') + '/' if folder_path else ''
        
        iterator = bucket.list_blobs(prefix=prefix, delimiter='/')
        
        images = []
        for blob in iterator:
            # Get just the filename
            name = blob.name[len(prefix):]
            if name and not name.startswith('.'):
                ext = os.path.splitext(name)[1].lower()
                if ext in ALLOWED_EXTENSIONS:
                    images.append(name)
        
        # Get subfolders
        subfolders = []
        for sub_prefix in iterator.prefixes:
            subfolder_name = sub_prefix[len(prefix):].rstrip('/')
            if not subfolder_name.startswith('.') and subfolder_name != 'thumbnails':
                subfolders.append(subfolder_name)
        
        return sorted(images), sorted(subfolders)
    except Exception as e:
        print(f"Error listing folder {folder_path}: {e}")
        return [], []


def get_blob(path):
    """Get a blob from GCS."""
    bucket = get_bucket()
    return bucket.blob(path)


def blob_exists(path):
    """Check if a blob exists."""
    blob = get_blob(path)
    return blob.exists()


def download_blob_to_memory(path):
    """Download blob content to memory."""
    blob = get_blob(path)
    return blob.download_as_bytes()


def upload_blob_from_memory(path, data, content_type='image/jpeg'):
    """Upload blob from memory."""
    blob = get_blob(path)
    blob.upload_from_string(data, content_type=content_type)


def delete_blob(path):
    """Delete a blob from GCS."""
    blob = get_blob(path)
    if blob.exists():
        blob.delete()
        return True
    return False


def get_blob_content_type(filename):
    """Get content type based on file extension."""
    ext = os.path.splitext(filename)[1].lower()
    content_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.mp4': 'video/mp4',
        '.webm': 'video/webm',
        '.mov': 'video/quicktime',
        '.avi': 'video/x-msvideo',
        '.mkv': 'video/x-matroska',
    }
    return content_types.get(ext, 'application/octet-stream')


def is_video(filename):
    """Check if file is a video."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in VIDEO_EXTENSIONS


# ============================================================================
# Thumbnail Generation
# ============================================================================

def generate_thumbnail(folder_path, filename):
    """Generate or retrieve a thumbnail for an image or video."""
    try:
        # Source and thumbnail paths
        if folder_path:
            source_path = f"{folder_path}/{filename}"
        else:
            source_path = filename
        
        thumb_filename = os.path.splitext(filename)[0] + '.jpg'
        if folder_path:
            thumb_path = f"{THUMBNAIL_PREFIX}{folder_path}/{thumb_filename}"
        else:
            thumb_path = f"{THUMBNAIL_PREFIX}{thumb_filename}"
        
        # Check if thumbnail already exists
        if blob_exists(thumb_path):
            return thumb_path
        
        # Download source file
        source_data = download_blob_to_memory(source_path)
        
        # Generate thumbnail
        size = (400, 400)
        
        if is_video(filename):
            # Handle video - extract first frame
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1], delete=False) as tmp:
                tmp.write(source_data)
                tmp_path = tmp.name
            
            try:
                cam = cv2.VideoCapture(tmp_path)
                ret, frame = cam.read()
                cam.release()
                
                if ret:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    return None
            finally:
                os.unlink(tmp_path)
        else:
            # Handle image
            img = Image.open(io.BytesIO(source_data))
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
        
        # Resize
        img.thumbnail(size, Image.Resampling.LANCZOS)
        
        # Save to bytes
        thumb_bytes = io.BytesIO()
        img.save(thumb_bytes, 'JPEG', quality=85)
        thumb_bytes.seek(0)
        
        # Upload thumbnail
        upload_blob_from_memory(thumb_path, thumb_bytes.getvalue(), 'image/jpeg')
        
        return thumb_path
        
    except Exception as e:
        print(f"Error generating thumbnail for {filename}: {e}")
        return None


# ============================================================================
# AUTHENTICATION
# ============================================================================

def check_auth(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if request.cookies.get(AUTH_COOKIE_NAME) == PASSWORD:
            return f(*args, **kwargs)
        return redirect(url_for('login', next=request.url))
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form.get('password')
        age_check = request.form.get('age_check')
        next_url = request.form.get('next') or url_for('index')
        
        if password == PASSWORD and age_check:
            resp = make_response(redirect(next_url))
            resp.set_cookie(
                AUTH_COOKIE_NAME,
                PASSWORD,
                max_age=365 * 24 * 60 * 60,
                httponly=True,
                samesite='Lax',
                secure=True
            )
            return resp
        elif not age_check:
             return render_template('login.html', error='You must confirm you are 18 or older', next=next_url)
        else:
            return render_template('login.html', error='Invalid password', next=next_url)
            
    return render_template('login.html', next=request.args.get('next', ''))


# ============================================================================
# Routes
# ============================================================================

@app.route('/')
@check_auth
def index():
    """Main page showing all subfolders."""
    folders = get_subfolders()
    return render_template('index.html', 
                          folders=folders, 
                          bucket_name=GCS_BUCKET_NAME)


@app.route('/folder/<folder_name>')
@app.route('/folder/<folder_name>/<subfolder_name>')
@check_auth
def view_folder(folder_name, subfolder_name=None):
    """Display all images/videos in a specific folder or subfolder."""
    # Security check - prevent directory traversal
    if '..' in folder_name:
        abort(404)
    
    if subfolder_name:
        if '..' in subfolder_name:
            abort(404)
        
        folder_path = f"{folder_name}/{subfolder_name}"
        display_name = f"{folder_name} / {subfolder_name}"
        parent_folder = folder_name
    else:
        folder_path = folder_name
        display_name = folder_name
        parent_folder = None
    
    # Get images and subfolders
    images, subfolders = get_items_in_folder(folder_path)
    
    if not images and not subfolders:
        # Check if folder exists at all
        bucket = get_bucket()
        iterator = bucket.list_blobs(prefix=folder_path + '/', max_results=1)
        if not list(iterator):
            abort(404)
    
    # Get favorites status for all images
    favorites = load_favorites_from_cookie()
    image_favorites = {}
    for image in images:
        image_path = f"{folder_path}/{image}"
        image_favorites[image_path] = image_path in favorites
    
    return render_template('folder.html',
                          folder_name=display_name,
                          folder_path=folder_path,
                          images=images,
                          image_count=len(images),
                          subfolders=subfolders if not subfolder_name else [],
                          parent_folder=parent_folder,
                          favorites=image_favorites)


@app.route('/thumbnail/<path:file_path>')
@check_auth
def serve_thumbnail(file_path):
    """Serve a thumbnail, generating it if necessary."""
    # Security check
    if '..' in file_path:
        abort(404)
    
    # Split path into folder and filename
    parts = file_path.rsplit('/', 1)
    if len(parts) == 2:
        folder_path, filename = parts
    else:
        folder_path = ''
        filename = parts[0]
    
    # Generate/get thumbnail
    thumb_path = generate_thumbnail(folder_path, filename)
    
    if thumb_path:
        try:
            data = download_blob_to_memory(thumb_path)
            response = Response(data, mimetype='image/jpeg')
            response.headers['Cache-Control'] = 'public, max-age=86400'
            return response
        except Exception:
            pass
    
    # Fallback to original image
    return serve_image(file_path)


@app.route('/image/<path:file_path>')
@check_auth
def serve_image(file_path):
    """Serve individual image/video files."""
    # Security check
    if '..' in file_path:
        abort(404)
    
    try:
        data = download_blob_to_memory(file_path)
        content_type = get_blob_content_type(file_path)
        
        response = Response(data, mimetype=content_type)
        response.headers['Cache-Control'] = 'public, max-age=3600'
        return response
    except Exception as e:
        print(f"Error serving image {file_path}: {e}")
        abort(404)


@app.route('/api/favorite', methods=['POST'])
@check_auth
def handle_favorite():
    """API endpoint to add or remove favorites (cookie-based)."""
    try:
        data = request.get_json()
        
        action = data.get('action')
        image_path = data.get('image_path')
        
        if not action or not image_path:
            return jsonify({
                'success': False,
                'error': 'Missing action or image_path'
            }), 400
        
        favorites = load_favorites_from_cookie()
        
        if action == 'add':
            if len(favorites) >= MAX_FAVORITES:
                return jsonify({
                    'success': False,
                    'error': f'Maximum of {MAX_FAVORITES} favorites allowed'
                }), 400
            favorites.add(image_path)
        elif action == 'remove':
            favorites.discard(image_path)
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid action. Use "add" or "remove"'
            }), 400
        
        # Create response with updated cookie
        response = make_response(jsonify({
            'success': True,
            'action': action,
            'image_path': image_path,
            'favorites_count': len(favorites)
        }))
        
        return create_favorites_response(response, favorites)
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/favorites')
@check_auth
def view_favorites():
    """Display all favorited images."""
    favorites = load_favorites_from_cookie()
    
    if not favorites:
        return render_template('favorites.html',
                              folder_name="Favorites",
                              folder_path="favorites",
                              favorite_images=[],
                              image_count=0,
                              max_favorites=MAX_FAVORITES,
                              favorites={})
    
    # Parse favorite paths and verify they exist
    favorite_images = []
    image_favorites = {}
    
    for fav_path in favorites:
        # Verify the image still exists in GCS
        if blob_exists(fav_path):
            filename = os.path.basename(fav_path)
            folder = os.path.dirname(fav_path)
            favorite_images.append({
                'name': filename,
                'path': fav_path,
                'folder': folder
            })
            image_favorites[fav_path] = True
    
    return render_template('favorites.html',
                          folder_name="Favorites",
                          folder_path="favorites",
                          favorite_images=favorite_images,
                          image_count=len(favorite_images),
                          max_favorites=MAX_FAVORITES,
                          favorites=image_favorites)


@app.route('/api/delete', methods=['POST'])
@check_auth
def handle_delete():
    """API endpoint to delete favorited images from GCS and remove from favorites."""
    try:
        data = request.get_json()
        
        if not data or 'image_path' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: image_path'
            }), 400
        
        image_path = data['image_path']
        
        # Security check
        if '..' in image_path:
            return jsonify({
                'success': False,
                'error': 'Invalid image path'
            }), 400
        
        # Verify the image exists in favorites
        favorites = load_favorites_from_cookie()
        if image_path not in favorites:
            return jsonify({
                'success': False,
                'error': 'Image is not in favorites'
            }), 404
        
        # Delete from GCS
        deleted = delete_blob(image_path)
        
        # Also delete thumbnail if exists
        thumb_path = f"{THUMBNAIL_PREFIX}{os.path.dirname(image_path)}/{os.path.splitext(os.path.basename(image_path))[0]}.jpg"
        delete_blob(thumb_path)
        
        # Remove from favorites
        favorites.discard(image_path)
        
        response = make_response(jsonify({
            'success': True,
            'message': 'File deleted successfully and removed from favorites' if deleted else 'File not found but removed from favorites',
            'image_path': image_path
        }))
        
        return create_favorites_response(response, favorites)
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        }), 500


@app.route('/api/clear_favorites', methods=['POST'])
@check_auth
def clear_favorites():
    """Clear all favorites from cookie."""
    response = make_response(jsonify({
        'success': True,
        'message': 'Favorites cleared'
    }))
    response.delete_cookie(FAVORITES_COOKIE_NAME)
    return response


# Entry point for GAE
if __name__ == '__main__':
    # For local development
    app.run(debug=True, host='127.0.0.1', port=8080)
