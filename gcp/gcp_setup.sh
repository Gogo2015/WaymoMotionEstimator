#!/bin/bash
# GCP Compute Engine VM Setup Script for Waymo Project
# This script creates a GPU-enabled VM and sets up the environment

set -e  # Exit on error

# ============================================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================================

PROJECT_ID="waymomotionestimator"  # REQUIRED: Your GCP project ID
ZONE="us-central1-a"              # Region/zone for VM
VM_NAME="waymo-training-vm"       # Name of your VM
MACHINE_TYPE="n1-standard-8"      # 8 vCPUs, 30GB RAM
GPU_TYPE="nvidia-tesla-t4"        # Options: nvidia-tesla-t4, nvidia-tesla-v100, nvidia-tesla-a100
GPU_COUNT=1                       # Number of GPUs
BOOT_DISK_SIZE="200GB"            # Disk size for OS and code
BUCKET_NAME="${PROJECT_ID}-waymo-data"  # GCS bucket for datasets

# ============================================================================
# COLORS FOR OUTPUT
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print_step() {
    echo -e "${BLUE}==>${NC} ${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING:${NC} $1"
}

print_error() {
    echo -e "${RED}ERROR:${NC} $1"
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 is not installed. Please install it first."
        exit 1
    fi
}

# ============================================================================
# PREFLIGHT CHECKS
# ============================================================================

print_step "Running preflight checks..."

# Check if gcloud is installed
check_command gcloud

# Check if user is logged in
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    print_error "Not logged into gcloud. Run: gcloud auth login"
    exit 1
fi

# Validate project ID
if [ "$PROJECT_ID" = "your-gcp-project-id" ]; then
    print_error "Please edit this script and set your PROJECT_ID"
    echo "Find your project ID at: https://console.cloud.google.com/home/dashboard"
    exit 1
fi

print_step "Setting active project to: $PROJECT_ID"
gcloud config set project $PROJECT_ID

# ============================================================================
# ENABLE REQUIRED APIS
# ============================================================================

print_step "Enabling required Google Cloud APIs..."
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com
print_step "APIs enabled successfully"

# ============================================================================
# CREATE GCS BUCKET
# ============================================================================

print_step "Creating Google Cloud Storage bucket: $BUCKET_NAME"
if gsutil ls -b gs://$BUCKET_NAME &> /dev/null; then
    print_warning "Bucket already exists, skipping creation"
else
    gsutil mb -l us-central1 gs://$BUCKET_NAME
    print_step "Bucket created successfully"
fi

# ============================================================================
# CREATE FIREWALL RULE FOR TENSORBOARD
# ============================================================================

print_step "Creating firewall rule for TensorBoard..."
if gcloud compute firewall-rules describe allow-tensorboard &> /dev/null; then
    print_warning "Firewall rule already exists, skipping"
else
    gcloud compute firewall-rules create allow-tensorboard \
        --allow tcp:6006 \
        --source-ranges 0.0.0.0/0 \
        --description "Allow TensorBoard access"
    print_step "Firewall rule created"
fi

# ============================================================================
# CREATE VM STARTUP SCRIPT
# ============================================================================

print_step "Preparing VM startup script..."
cat > /tmp/vm_startup.sh << 'EOF'
#!/bin/bash
# This script runs on VM first boot

set -e

echo "Starting VM initialization..."

# Install system dependencies
apt-get update
apt-get install -y git tmux screen htop ncdu

# Install gcsfuse for mounting GCS buckets
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
apt-get update
apt-get install -y gcsfuse

# Verify TensorFlow can see GPU
su - $USER -c "python3 -c 'import tensorflow as tf; print(\"GPUs Available:\", tf.config.list_physical_devices(\"GPU\"))'"

echo "VM initialization complete!"
EOF

# ============================================================================
# CREATE VM INSTANCE
# ============================================================================

print_step "Creating VM instance: $VM_NAME"
print_step "Machine: $MACHINE_TYPE with $GPU_COUNT x $GPU_TYPE"

if gcloud compute instances describe $VM_NAME --zone=$ZONE &> /dev/null; then
    print_warning "VM already exists!"
    echo -e "${YELLOW}Options:${NC}"
    echo "  1. Delete and recreate (will lose all data on VM)"
    echo "  2. Keep existing VM and skip creation"
    echo "  3. Exit"
    read -p "Enter choice [1-3]: " choice

    case $choice in
        1)
            print_step "Deleting existing VM..."
            gcloud compute instances delete $VM_NAME --zone=$ZONE --quiet
            ;;
        2)
            print_step "Keeping existing VM"
            VM_EXISTS=true
            ;;
        3)
            exit 0
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac
fi

if [ "$VM_EXISTS" != "true" ]; then
    gcloud compute instances create $VM_NAME \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --accelerator=type=$GPU_TYPE,count=$GPU_COUNT \
        --boot-disk-size=$BOOT_DISK_SIZE \
        --boot-disk-type=pd-ssd \
        --image-family=common-cu121-debian-11-py310 \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --metadata-from-file=startup-script=/tmp/vm_startup.sh \
        --metadata="install-nvidia-driver=True"

    print_step "VM created successfully!"
    print_step "Waiting 60 seconds for VM to initialize..."
    sleep 60
fi

# ============================================================================
# PRINT NEXT STEPS
# ============================================================================

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  VM Setup Complete! 🎉${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}VM Details:${NC}"
echo "  Name: $VM_NAME"
echo "  Zone: $ZONE"
echo "  Machine: $MACHINE_TYPE"
echo "  GPU: $GPU_COUNT x $GPU_TYPE"
echo "  Bucket: gs://$BUCKET_NAME"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo ""
echo -e "${YELLOW}1. SSH into your VM:${NC}"
echo "   gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo -e "${YELLOW}2. Clone your repository:${NC}"
echo "   git clone https://github.com/YOUR_USERNAME/WaymoProject.git"
echo "   cd WaymoProject"
echo ""
echo -e "${YELLOW}3. Mount your GCS bucket:${NC}"
echo "   mkdir -p ~/waymo_data"
echo "   gcsfuse --implicit-dirs $BUCKET_NAME ~/waymo_data"
echo ""
echo -e "${YELLOW}4. Install Python dependencies:${NC}"
echo "   pip install -r requirements.txt"
echo ""
echo -e "${YELLOW}5. Upload your data to GCS (from local machine):${NC}"
echo "   gsutil -m cp -r ./training_data gs://$BUCKET_NAME/"
echo "   gsutil -m cp -r ./test_data gs://$BUCKET_NAME/"
echo ""
echo -e "${YELLOW}6. Run training:${NC}"
echo "   screen -S training"
echo "   python train_model.py --config configs/gcp.yaml"
echo "   # Detach: Ctrl+A then D"
echo ""
echo -e "${YELLOW}7. View TensorBoard (from local machine):${NC}"
echo "   gcloud compute ssh $VM_NAME --zone=$ZONE -- -L 6006:localhost:6006"
echo "   # Then on VM: tensorboard --logdir logs/ --host 0.0.0.0"
echo "   # Access at: http://localhost:6006"
echo ""
echo -e "${BLUE}Useful Commands:${NC}"
echo "  Start VM:  gcloud compute instances start $VM_NAME --zone=$ZONE"
echo "  Stop VM:   gcloud compute instances stop $VM_NAME --zone=$ZONE"
echo "  Delete VM: gcloud compute instances delete $VM_NAME --zone=$ZONE"
echo "  SSH:       gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo -e "${YELLOW}💰 Cost Optimization:${NC}"
echo "  - Stop VM when not training (stops compute charges)"
echo "  - Delete VM when project complete (keeps disk snapshot)"
echo "  - Use preemptible VMs for 60-91% savings (may be interrupted)"
echo ""
echo -e "${RED}⚠️  REMEMBER TO STOP YOUR VM WHEN NOT IN USE!${NC}"
echo ""
