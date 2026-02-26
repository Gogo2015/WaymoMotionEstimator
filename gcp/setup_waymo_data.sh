#!/bin/bash
# Setup script to access Waymo Open Motion Dataset on GCP
# Run this on your GCP VM after initialization

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_step() {
    echo -e "${BLUE}==>${NC} ${GREEN}$1${NC}"
}

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Waymo Dataset Setup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# ============================================================================
# OPTION 1: Mount Waymo's Public Bucket (Read-Only)
# ============================================================================

print_step "Option 1: Mount Waymo's public bucket (read-only access)"
echo "This gives you direct access to WOMD without downloading!"
echo ""
read -p "Mount Waymo public bucket? [Y/n]: " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    print_step "Creating mount points..."
    mkdir -p ~/waymo_public/training
    mkdir -p ~/waymo_public/validation
    mkdir -p ~/waymo_public/testing

    print_step "Mounting Waymo training data..."
    gcsfuse --implicit-dirs \
        --only-dir uncompressed/scenario/training \
        waymo_open_dataset_motion_v_1_2_0 \
        ~/waymo_public/training

    print_step "Mounting Waymo validation data..."
    gcsfuse --implicit-dirs \
        --only-dir uncompressed/scenario/validation \
        waymo_open_dataset_motion_v_1_2_0 \
        ~/waymo_public/validation

    print_step "Mounting Waymo testing data..."
    gcsfuse --implicit-dirs \
        --only-dir uncompressed/scenario/testing \
        waymo_open_dataset_motion_v_1_2_0 \
        ~/waymo_public/testing

    echo ""
    print_step "✅ Waymo dataset mounted successfully!"
    echo ""
    echo -e "${BLUE}Data locations:${NC}"
    echo "  Training:   ~/waymo_public/training/"
    echo "  Validation: ~/waymo_public/validation/"
    echo "  Testing:    ~/waymo_public/testing/"
    echo ""
    echo -e "${YELLOW}Note: This is read-only access to Waymo's public bucket${NC}"
fi

# ============================================================================
# OPTION 2: Copy Subset to Your Bucket
# ============================================================================

echo ""
print_step "Option 2: Copy a subset to your own bucket"
echo "Recommended for faster access and experimentation"
echo ""
read -p "Copy a subset to your bucket? [y/N]: " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter your GCS bucket name (e.g., my-project-waymo-data): " BUCKET_NAME
    read -p "How many training files to copy? (e.g., 100): " NUM_FILES

    print_step "Copying $NUM_FILES training files to gs://$BUCKET_NAME/training_data/"

    # Copy first N files
    gsutil -m cp \
        "gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario/training/training.tfrecord-*-of-01000" \
        "gs://$BUCKET_NAME/training_data/" \
        | head -n $NUM_FILES

    print_step "✅ Subset copied to your bucket!"
    echo "Mount it with: gcsfuse $BUCKET_NAME ~/waymo_data"
fi

# ============================================================================
# UPDATE CONFIG FILES
# ============================================================================

echo ""
print_step "Updating config files..."

if [ -d "~/WaymoProject/configs" ]; then
    cd ~/WaymoProject

    # Create new config for public Waymo data
    cat > configs/waymo_public.yaml << 'EOF'
# Waymo Public Dataset Configuration
# Uses data mounted from Waymo's public GCS bucket

experiment:
  name: "waymo_public_experiment"
  description: "Training on Waymo's public dataset via GCS"

data:
  train_dir: "/home/USER/waymo_public/training"
  test_dir: "/home/USER/waymo_public/testing"
  past_steps: 10
  future_steps: 80
  batch_size: 128
  train_split: 0.9

training:
  epochs: 50
  learning_rate: 0.001
  optimizer: "adam"

model:
  name: "ConvMLP"

logging:
  log_dir: "logs"
  save_dir: "trained_models"
  tensorboard: true
EOF

    # Update USER placeholder
    USERNAME=$(whoami)
    sed -i "s/USER/$USERNAME/g" configs/waymo_public.yaml

    print_step "✅ Created configs/waymo_public.yaml"
fi

# ============================================================================
# VERIFY SETUP
# ============================================================================

echo ""
print_step "Verifying setup..."

if [ -d ~/waymo_public/training ]; then
    TRAIN_COUNT=$(ls ~/waymo_public/training/*.tfrecord* 2>/dev/null | wc -l)
    echo "  Training files available: $TRAIN_COUNT"
fi

if [ -d ~/waymo_public/validation ]; then
    VAL_COUNT=$(ls ~/waymo_public/validation/*.tfrecord* 2>/dev/null | wc -l)
    echo "  Validation files available: $VAL_COUNT"
fi

if [ -d ~/waymo_public/testing ]; then
    TEST_COUNT=$(ls ~/waymo_public/testing/*.tfrecord* 2>/dev/null | wc -l)
    echo "  Testing files available: $TEST_COUNT"
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Quick Start:${NC}"
echo ""
echo "1. Navigate to project:"
echo "   cd ~/WaymoProject"
echo ""
echo "2. Run training with Waymo public data:"
echo "   screen -S training"
echo "   python train_model.py --config configs/waymo_public.yaml"
echo "   # Ctrl+A then D to detach"
echo ""
echo "3. Monitor training:"
echo "   screen -r training"
echo "   tensorboard --logdir logs/ --host 0.0.0.0"
echo ""
echo -e "${YELLOW}💡 Tips:${NC}"
echo "  - Public bucket is read-only (can't write to it)"
echo "  - Use your own bucket for saving models/logs"
echo "  - Consider copying a subset for faster iteration"
echo ""
echo -e "${BLUE}Data Stats:${NC}"
echo "  - Training: ~487 files (~1TB)"
echo "  - Validation: ~150 files"
echo "  - Testing: ~150 files"
echo ""
