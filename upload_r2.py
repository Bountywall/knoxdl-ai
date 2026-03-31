"""
Uploads predictions.json to Cloudflare R2 after the prediction script runs.
R2 credentials are passed via environment variables (GitHub Actions secrets).
"""
import boto3
import os
import json

account_id = os.environ['R2_ACCOUNT_ID']
access_key = os.environ['R2_ACCESS_KEY_ID']
secret_key = os.environ['R2_SECRET_KEY']
bucket     = os.environ['R2_BUCKET']

s3 = boto3.client(
    's3',
    endpoint_url=f'https://{account_id}.r2.cloudflarestorage.com',
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name='auto'
)

s3.upload_file(
    'predictions.json',
    bucket,
    'predictions.json',
    ExtraArgs={
        'ContentType': 'application/json',
        'CacheControl': 'public, max-age=3600',
    }
)

print(f"✅ predictions.json uploaded to R2 bucket: {bucket}")
