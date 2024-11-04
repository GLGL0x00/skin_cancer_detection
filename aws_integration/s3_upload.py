import boto3
from boto3.s3.transfer import TransferConfig

# Initialize S3 client
s3_client = boto3.client('s3')

# Define multipart upload configuration
config = TransferConfig(multipart_threshold=1024*25, max_concurrency=10,
                        multipart_chunksize=1024*25, use_threads=True)

# Upload a large file with multipart support
file_path = ""
bucket_name = ""
object_name = ""

try:
    s3_client.upload_file(file_path, bucket_name, object_name, Config=config)
    print("Upload completed successfully.")
except Exception as e:
    print(f"Upload interrupted: {e}")
