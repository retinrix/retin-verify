#!/bin/bash
#
# Run Template Pipeline Script
# Generates synthetic ID documents using run_template_pipeline.py
#
# This script is aligned with server.py to produce the same output format.
#

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values
DOC_TYPE="cnie_front"
DOC_CONFIG="$PROJECT_ROOT/synthetic/configs/cnie_front_custom_final.json"
ARABIC_FONT="$PROJECT_ROOT/synthetic/fonts/ScheherazadeNew-regular.ttf"
TEMPLATE_DIR="$PROJECT_ROOT/synthetic/templates"
NUM_SAMPLES=3
OUTPUT_DIR="$PROJECT_ROOT/data/test_fixed"
FAST_PREVIEW=""
FACE_PHOTOS_DIR="$PROJECT_ROOT/data/vggface2"

# Paired generation configs (optional)
CONFIG_FRONT=""
CONFIG_BACK=""

# Function to display usage
usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Generate synthetic ID documents using the template-based pipeline.

OPTIONS:
    -t, --doc-type TYPE         Document type: passport, cnie_front, cnie_back, 
                                cnie_paired, or all (default: cnie_front)
    -c, --doc-config PATH       Path to document configuration JSON file
    --config-front PATH         Path to CNIE front config (for paired generation)
    --config-back PATH          Path to CNIE back config (for paired generation)
    -f, --arabic-font PATH      Path to Arabic font file
    --template-dir PATH         Path to templates directory (default: synthetic/templates)
    --face-photos-dir PATH      Path to VGGFace2 dataset for real face photos
    -n, --num-samples N         Number of samples to generate (default: 3)
    -o, --output-dir PATH       Output directory (default: data/test_fixed)
    --fast-preview              Enable fast preview mode (skip augmentations)
    --no-real-faces             Use synthetic faces instead of VGGFace2 photos
    -h, --help                  Show this help message

EXAMPLES:
    # Generate 10 CNIE front samples
    $(basename "$0") -t cnie_front -n 10

    # Generate with custom config
    $(basename "$0") -c synthetic/configs/my_config.json -n 5

    # Generate paired CNIE (front + back with same identity)
    $(basename "$0") --config-front synthetic/configs/cnie_front_with_photo.json \\
                     --config-back synthetic/configs/cnie_back_mrz.json \\
                     -t cnie_paired -n 10

    # Fast preview mode (no augmentations)
    $(basename "$0") --fast-preview -n 1

    # Generate all document types
    $(basename "$0") -t all -n 100

    # Use synthetic faces instead of real photos
    $(basename "$0") --no-real-faces -n 10

EOF
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--doc-type)
            DOC_TYPE="$2"
            shift 2
            ;;
        -c|--doc-config)
            DOC_CONFIG="$2"
            shift 2
            ;;
        --config-front)
            CONFIG_FRONT="$2"
            shift 2
            ;;
        --config-back)
            CONFIG_BACK="$2"
            shift 2
            ;;
        -f|--arabic-font)
            ARABIC_FONT="$2"
            shift 2
            ;;
        --template-dir)
            TEMPLATE_DIR="$2"
            shift 2
            ;;
        --face-photos-dir)
            FACE_PHOTOS_DIR="$2"
            shift 2
            ;;
        --no-real-faces)
            FACE_PHOTOS_DIR=""
            shift
            ;;
        -n|--num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --fast-preview)
            FAST_PREVIEW="--fast-preview"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            usage
            exit 1
            ;;
    esac
done

# Validate inputs
echo "==============================================="
echo "Template Pipeline Configuration"
echo "==============================================="
echo "Document type:     $DOC_TYPE"
echo "Document config:   $DOC_CONFIG"
if [[ -n "$CONFIG_FRONT" ]]; then
    echo "Front config:      $CONFIG_FRONT"
fi
if [[ -n "$CONFIG_BACK" ]]; then
    echo "Back config:       $CONFIG_BACK"
fi
echo "Arabic font:       $ARABIC_FONT"
echo "Template dir:      $TEMPLATE_DIR"
if [[ -n "$FACE_PHOTOS_DIR" && -d "$FACE_PHOTOS_DIR" ]]; then
    echo "Face photos:       $FACE_PHOTOS_DIR (VGGFace2)"
else
    echo "Face photos:       synthetic (no VGGFace2)"
fi
echo "Num samples:       $NUM_SAMPLES"
echo "Output dir:        $OUTPUT_DIR"
if [[ -n "$FAST_PREVIEW" ]]; then
    echo "Fast preview:      enabled"
fi
echo "==============================================="
echo ""

# Check if files exist
if [[ ! -f "$DOC_CONFIG" && -z "$CONFIG_FRONT" ]]; then
    echo "Warning: Document config not found: $DOC_CONFIG"
    echo "Using default configuration..."
fi

if [[ ! -f "$ARABIC_FONT" ]]; then
    echo "Warning: Arabic font not found: $ARABIC_FONT"
fi

if [[ ! -d "$TEMPLATE_DIR" ]]; then
    echo "Error: Template directory not found: $TEMPLATE_DIR"
    exit 1
fi

# Check VGGFace2
if [[ -n "$FACE_PHOTOS_DIR" && ! -d "$FACE_PHOTOS_DIR" ]]; then
    echo "Warning: Face photos directory not found: $FACE_PHOTOS_DIR"
    echo "         Using synthetic faces instead."
    FACE_PHOTOS_DIR=""
fi

# Build command array
CMD_ARGS=(
    "python3"
    "$SCRIPT_DIR/run_template_pipeline.py"
    "--doc-type" "$DOC_TYPE"
)

# Add doc-config if not using paired generation with separate configs
if [[ -z "$CONFIG_FRONT" && -f "$DOC_CONFIG" ]]; then
    CMD_ARGS+=("--doc-config" "$DOC_CONFIG")
fi

# Add paired generation configs if provided
if [[ -n "$CONFIG_FRONT" ]]; then
    CMD_ARGS+=("--config-front" "$CONFIG_FRONT")
fi

if [[ -n "$CONFIG_BACK" ]]; then
    CMD_ARGS+=("--config-back" "$CONFIG_BACK")
fi

CMD_ARGS+=(
    "--arabic-font" "$ARABIC_FONT"
    "--template-dir" "$TEMPLATE_DIR"
)

# Add face photos dir if available
if [[ -n "$FACE_PHOTOS_DIR" && -d "$FACE_PHOTOS_DIR" ]]; then
    CMD_ARGS+=("--face-photos-dir" "$FACE_PHOTOS_DIR")
fi

CMD_ARGS+=(
    "--num-samples" "$NUM_SAMPLES"
    "--output-dir" "$OUTPUT_DIR"
)

if [[ -n "$FAST_PREVIEW" ]]; then
    CMD_ARGS+=("$FAST_PREVIEW")
fi

echo "Running command:"
echo "${CMD_ARGS[@]}"
echo ""

# Execute command
"${CMD_ARGS[@]}"

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo "==============================================="
    echo "✅ Pipeline completed successfully!"
    echo "Output directory: $OUTPUT_DIR"
    echo "==============================================="
else
    echo ""
    echo "==============================================="
    echo "❌ Pipeline failed!"
    echo "==============================================="
    exit 1
fi
