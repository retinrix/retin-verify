

Step 1: Configure rclone
rclone config
Just follow these prompts:

Prompt	Your Response
n/s/q>	Type n (new remote)
name>	Type gdrive
Storage>	Type 13 (Google Drive)
client_id>	Press Enter (blank)
client_secret>	Press Enter (blank)
scope>	Type 1 (full access)
root_folder_id>	Press Enter
service_account_file>	Press Enter
y/n> (advanced)	Type n
y/n> (auto config)	Type y

# Create folder structure
rclone mkdir "gdrive:retin-verify/data"
rclone mkdir "gdrive:retin-verify/models"

# Upload your dataset
rclone copy ./data/cnie_dataset_10k "gdrive:retin-verify/data/cnie_dataset_10k" \
    --progress \
    --transfers 8