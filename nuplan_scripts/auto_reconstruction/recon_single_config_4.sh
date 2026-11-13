#!/usr/bin/env bash

set -e  # Exit immediately if any command fails

# Default values
CONFIG=""
NUM_WORKERS=12
NUM_GPUS=1
EXPORT_DIR=""
OUTPUT_DIR=""
MTGS_METHOD="${MTGS_METHOD:-mtgs_st}"

# Parse long options
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --export-dir)
            EXPORT_DIR="$2"
            shift 2
            ;;
        --workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --method)
            MTGS_METHOD="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --config CONFIG --export-dir DIR --output-dir DIR [--workers NUM] [--gpus NUM] [--method mtgs_st|mtgs_deformable]"
            echo "  --config:     Configuration file path (required)"
            echo "  --output-dir: Output directory path (required)"
            echo "  --export-dir: Export directory path (required)"
            echo "  --workers:    Number of workers (default: 12)"
            echo "  --gpus:       Number of GPUs (default: 1)"
            echo "  --method:     MTGS method (default: mtgs_st; use mtgs_deformable for pedestrians)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$CONFIG" ] || [ -z "$EXPORT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --config, --export-dir and --output-dir are required"
    echo "Usage: $0 --config CONFIG --export-dir DIR --output-dir DIR [--workers NUM] [--gpus NUM]"
    exit 1
fi

case "$MTGS_METHOD" in
    mtgs_st|mtgs_deformable)
        ;;
    *)
        echo "Error: --method must be one of: mtgs_st, mtgs_deformable"
        exit 1
        ;;
esac

# =====================================================================
# Cleanup trap disabled for a resumable, staged workflow.
# A failure in this stage must not delete outputs of earlier stages.
# =====================================================================
# Cleanup function to remove output directory on failure
# cleanup_on_failure() {
#     if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ]; then
#         echo "[ERROR] Command failed - cleaning up output directory: $OUTPUT_DIR"
#         mv "$OUTPUT_DIR" "${OUTPUT_DIR}_failed"
#     fi
# }

# Set trap to run cleanup on script failure (DISABLED)
# trap cleanup_on_failure ERR
# =====================================================================

echo "Running with:
    CONFIG=$CONFIG
    OUTPUT_DIR=$OUTPUT_DIR
    EXPORT_DIR=$EXPORT_DIR
    WORKERS=$NUM_WORKERS
    GPUS=$NUM_GPUS
    METHOD=$MTGS_METHOD"

export MTGS_OUTPUT_DIR="$OUTPUT_DIR"
python -m nuplan_scripts.auto_reconstruction.background_reconstruction --config "$CONFIG" --method "$MTGS_METHOD"
python -m nuplan_scripts.auto_reconstruction.render_reconstruction --config "$CONFIG"
# python -m nuplan_scripts.auto_reconstruction.export_reconstruction --config "$CONFIG" --output_dir "$EXPORT_DIR"
# python -m nuplan_scripts.clean_temp_files --config "$CONFIG"
# python -m nuplan_scripts.clean_temp_files --config "$CONFIG"

# If we reach here, all commands succeeded - disable cleanup trap (DISABLED)
# trap - ERR

echo "All commands completed successfully"