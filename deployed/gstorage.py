from google.cloud import storage
import os

# Explicitly set service account path relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
service_account_path = os.path.join(current_dir, "service_account.json")
if os.path.exists(service_account_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path

# Try to load .env file for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed or not needed (e.g. production)

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

def upload_blob_from_memory(bucket_name, contents, destination_blob_name, content_type='text/plain'):
    """Uploads a file to the bucket from a string or bytes."""
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_string(contents, content_type=content_type)

    print(f"{destination_blob_name} uploaded to {bucket_name}.")

def check_or_create_folder(bucket_name, folder_name):
    """Checks if a folder exists in the bucket, creates it if not."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # Ensure folder name ends with /
    if not folder_name.endswith('/'):
        folder_name += '/'

    blobs = list(bucket.list_blobs(prefix=folder_name, max_results=1))
    
    if not blobs:
        print(f"Folder '{folder_name}' does not exist (or is empty). Creating placeholder...")
        blob = bucket.blob(folder_name)
        blob.upload_from_string("")
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")

if __name__ == "__main__":
    bucket_name = "xserver"
    folder_name = "test/"
    file_path = r"C:\Users\jared\Documents\code\local_jarvis\xserver\autogen\dec23_workflow_test2\1766547758_image_1_end_2_video.mp4"
    
    # Check if file exists locally before trying to upload
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
    else:
        destination_blob_name = f"test/{os.path.basename(file_path)}"

        # Initialize client for checking folder
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # Check if folder exists (simulated by checking for prefix)
        # Note: In GCS, folders are just objects with '/' suffix or implied by other objects
        blobs = list(bucket.list_blobs(prefix=folder_name, max_results=1))
        
        # We also check if the folder object itself exists if list_blobs returns empty
        # But list_blobs with prefix should catch it if it exists as a placeholder
        
        if not blobs:
            print(f"Folder '{folder_name}' does not exist (or is empty). Creating placeholder...")
            blob = bucket.blob(folder_name)
            blob.upload_from_string("")
            print(f"Folder '{folder_name}' created.")
        else:
            print(f"Folder '{folder_name}' already exists.")

        # Upload file
        print(f"Uploading {file_path} to {destination_blob_name}...")
        upload_blob(bucket_name, file_path, destination_blob_name)
        print("Done.")
