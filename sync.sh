#! /bin/sh

aws s3 sync . s3://yct-backup/backup_2024_10_24/ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus/ --delete --exclude ".git/*"