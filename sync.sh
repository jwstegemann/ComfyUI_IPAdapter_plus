#! /bin/sh

export AWS_ACCESS_KEY_ID=$S3_ACCESS_KEY
export AWS_SECRET_ACCESS_KEY=$S3_SECRET_KEY
export AWS_DEFAULT_REGION=$S3_REGION

aws s3 sync . s3://yct-backup/backup_2024_10_24/ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus/ --delete --exclude ".git/*"