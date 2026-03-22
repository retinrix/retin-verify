#!/usr/bin/env python3
"""
Robust Flask server for CNIE tool – with config save/load and batch generation.
Run: python3 server.py
"""

import json
import subprocess
import sys
import traceback
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ----------------------------------------------------------------------
# Path configuration – correctly resolve all directories
# ----------------------------------------------------------------------
# This file is in: retin-verify/synthetic/scripts/server.py
BASE_DIR = Path(__file__).parent                      # synthetic/scripts/
SYNTHETIC_DIR = BASE_DIR.parent                       # synthetic/
PROJECT_ROOT = SYNTHETIC_DIR.parent                   # retin-verify/

TEMPLATE_DIR = SYNTHETIC_DIR / "templates"            # retin-verify/synthetic/templates/
OUTPUT_DIR = PROJECT_ROOT / "data/gui_preview"        # retin-verify/data/gui_preview/
ARABIC_FONT = (SYNTHETIC_DIR / "fonts/ScheherazadeNew-regular.ttf").resolve()
GENERATOR_SCRIPT = BASE_DIR / "template_document_generator.py"
PIPELINE_SCRIPT = BASE_DIR / "run_template_pipeline.py"
BATCH_OUTPUT_DIR = PROJECT_ROOT / "data/cnie_output"  # retin-verify/data/cnie_output/
CONFIG_DIR = SYNTHETIC_DIR / "configs"                # retin-verify/synthetic/configs/
CONFIG_PATH = BASE_DIR / "current_config.json"        # temporary config for preview
CONFIG_FRONT_PATH = BASE_DIR / "current_config_front.json"  # CNIE front config
CONFIG_BACK_PATH = BASE_DIR / "current_config_back.json"    # CNIE back config
VGGFACE2_DEFAULT = PROJECT_ROOT / "data" / "vggface2" # Default VGGFace2 path

# Ensure all directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def get_vggface2_path_from_config() -> Path:
    """Extract and resolve vggface2_path from current config."""
    if not CONFIG_PATH.exists():
        return VGGFACE2_DEFAULT
    
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        vggface2_path = config.get('vggface2_path')
        if vggface2_path:
            # Resolve relative to project root
            path = PROJECT_ROOT / vggface2_path
            if path.exists():
                return path
            # Try as absolute path
            path = Path(vggface2_path)
            if path.exists():
                return path
        
        # Check if photo_placeholders exist - if so, we need VGGFace2
        if config.get('photo_placeholders'):
            print(f"⚠️  Photo placeholders configured but vggface2_path not found. Using default: {VGGFACE2_DEFAULT}")
            return VGGFACE2_DEFAULT
            
    except Exception as e:
        print(f"⚠️  Error reading vggface2_path from config: {e}")
    
    return VGGFACE2_DEFAULT


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.route('/')
def index():
    """Serve the main entry point with navigation menu."""
    gui_path = BASE_DIR / 'gui_tool' / 'index.html'
    return send_file(gui_path)


@app.route('/cnie_front')
def cnie_front():
    """Serve the CNIE Front GUI tool."""
    gui_path = BASE_DIR / 'gui_tool' / 'cnie_tool.html'
    return send_file(gui_path)


@app.route('/cnie_back')
def cnie_back():
    """Serve the CNIE Back GUI tool."""
    gui_path = BASE_DIR / 'gui_tool' / 'cnie_back_tool.html'
    return send_file(gui_path)


@app.route('/passport')
def passport():
    """Serve the Passport placeholder page."""
    gui_path = BASE_DIR / 'gui_tool' / 'index.html'
    return send_file(gui_path)


@app.route('/status', methods=['GET'])
def status():
    """Check if server is alive and dependencies exist."""
    vggface2_path = get_vggface2_path_from_config()
    deps = {
        'generator_script': GENERATOR_SCRIPT.exists(),
        'pipeline_script': PIPELINE_SCRIPT.exists(),
        'template_dir': TEMPLATE_DIR.exists(),
        'arabic_font': ARABIC_FONT.exists(),
        'output_dir': OUTPUT_DIR.exists(),
        'vggface2_path': str(vggface2_path),
        'vggface2_exists': vggface2_path.exists(),
    }
    return jsonify({'status': 'ok', 'dependencies': deps})


def _detect_config_type(data):
    """Detect if config is for cnie_front or cnie_back based on fields."""
    fields = data.get('fields', {})
    if 'mrz_line3' in fields and 'birth_year' in fields:
        return 'cnie_back'
    elif 'national_id' in fields or 'personal_id' in fields:
        return 'cnie_front'
    return 'unknown'


@app.route('/save_config', methods=['POST'])
def save_config():
    """Save current config to temporary file (used for preview/generation)."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        
        # Detect config type and save to appropriate file
        config_type = _detect_config_type(data)
        
        if config_type == 'cnie_back':
            target_path = CONFIG_BACK_PATH
            print(f"📋 Saving CNIE BACK config to {target_path}")
        elif config_type == 'cnie_front':
            target_path = CONFIG_FRONT_PATH
            print(f"📋 Saving CNIE FRONT config to {target_path}")
        else:
            target_path = CONFIG_PATH
            print(f"📋 Saving generic config to {target_path}")
        
        with open(target_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Also save to generic path for backward compatibility
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Verify vggface2_path if photo placeholders are configured
        photo_placeholders = data.get('photo_placeholders', {})
        vggface2_path = data.get('vggface2_path')
        
        if photo_placeholders and vggface2_path:
            resolved_path = PROJECT_ROOT / vggface2_path
            if not resolved_path.exists():
                print(f"⚠️  Warning: vggface2_path does not exist: {resolved_path}")
        
        return jsonify({"status": "ok", "path": str(target_path), "config_type": config_type})
    except Exception as e:
        app.logger.error(f"Save config error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/save_config/<filename>', methods=['POST'])
def save_config_as(filename):
    """
    Save current config to a named file in CONFIG_DIR.
    Filename should not include .json (it will be added automatically).
    """
    try:
        data = request.json
        if not filename.endswith('.json'):
            filename += '.json'
        filepath = CONFIG_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return jsonify({"status": "ok", "path": str(filepath)})
    except Exception as e:
        app.logger.error(f"Save config as error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/list_configs', methods=['GET'])
def list_configs():
    """Return list of .json files in CONFIG_DIR."""
    try:
        files = [f.name for f in CONFIG_DIR.glob('*.json')]
        return jsonify(files)
    except Exception as e:
        app.logger.error(f"List configs error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/load_config/<filename>', methods=['GET'])
def load_config(filename):
    """Load a specific config file from CONFIG_DIR."""
    try:
        if not filename.endswith('.json'):
            filename += '.json'
        filepath = CONFIG_DIR / filename
        if not filepath.exists():
            return jsonify({"error": "File not found"}), 404
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        app.logger.error(f"Load config error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/generate_preview', methods=['POST'])
def generate_preview():
    """Generate a single preview image using the saved config."""
    if not CONFIG_PATH.exists():
        return jsonify({"error": "No config saved"}), 400

    # Get document type from request (default to cnie_front)
    try:
        data = request.get_json(force=True, silent=True) or {}
    except:
        data = {}
    doc_type = data.get('doc_type', 'cnie_front')
    
    # Validate doc_type
    if doc_type not in ['cnie_front', 'cnie_back', 'passport']:
        return jsonify({"error": f"Invalid doc_type: {doc_type}"}), 400
    
    # Get vggface2 path from config
    vggface2_path = get_vggface2_path_from_config()
    
    cmd = [
        sys.executable, str(GENERATOR_SCRIPT),
        '--doc-type', doc_type,
        '--config', str(CONFIG_PATH),
        '--arabic-font', str(ARABIC_FONT),
        '--num-samples', '1',
        '--output-dir', str(OUTPUT_DIR),
        '--template-dir', str(TEMPLATE_DIR),  # Use absolute path
        '--fast-preview'  # Use fast preview mode (no augmentations, but real faces)
    ]
    
    # Always use VGGFace2 for real faces in preview (except cnie_back which has no photo)
    if doc_type != 'cnie_back' and vggface2_path.exists():
        cmd.extend(['--face-photos-dir', str(vggface2_path)])
        print(f"🖼️  Using VGGFace2 photos from: {vggface2_path}")
    else:
        if doc_type == 'cnie_back':
            print(f"ℹ️  CNIE back has no photo placeholder")
        else:
            print(f"⚠️  VGGFace2 not found at: {vggface2_path}, using synthetic faces")
    
    try:
        # Increased timeout for VGGFace2 indexing
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60, cwd=str(PROJECT_ROOT))
        app.logger.info(f"Generator stdout: {result.stdout}")
        if result.stderr:
            app.logger.warning(f"Generator stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        app.logger.error(f"Generator failed (return code {e.returncode}): {e.stderr}")
        return jsonify({"error": e.stderr}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

    sample_dir = OUTPUT_DIR / doc_type / '000000'
    image_path = sample_dir / 'image.jpg'
    if image_path.exists():
        return send_file(image_path, mimetype='image/jpeg')
    else:
        return jsonify({"error": "Image not generated"}), 500


@app.route('/generate_dataset', methods=['POST'])
def generate_dataset():
    """
    Generate a full dataset with the given number of samples
    (using the generator directly, without pipeline splits).
    Includes VGGFace2 photos if configured.
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
    except:
        data = {}
    num_samples = data.get('num_samples', 5)
    doc_type = data.get('doc_type', 'cnie_front')
    
    # Validate doc_type
    if doc_type not in ['cnie_front', 'cnie_back', 'passport']:
        return jsonify({"error": f"Invalid doc_type: {doc_type}"}), 400
    
    if not CONFIG_PATH.exists():
        return jsonify({"error": "No config saved"}), 400

    # Get vggface2 path from config
    vggface2_path = get_vggface2_path_from_config()
    
    cmd = [
        sys.executable, str(GENERATOR_SCRIPT),
        '--doc-type', doc_type,
        '--config', str(CONFIG_PATH),
        '--arabic-font', str(ARABIC_FONT),
        '--num-samples', str(num_samples),
        '--output-dir', str(OUTPUT_DIR),
        '--template-dir', str(TEMPLATE_DIR)  # Use absolute path
    ]
    
    # Add face photos dir if vggface2 exists (and not cnie_back)
    if doc_type != 'cnie_back' and vggface2_path.exists():
        cmd.extend(['--face-photos-dir', str(vggface2_path)])
        print(f"🖼️  Using VGGFace2 photos from: {vggface2_path}")
    else:
        if doc_type == 'cnie_back':
            print(f"ℹ️  CNIE back has no photo placeholder")
        else:
            print(f"⚠️  VGGFace2 not found at: {vggface2_path}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300, cwd=str(PROJECT_ROOT))
        app.logger.info(f"Generator stdout: {result.stdout}")
        if result.stderr:
            app.logger.warning(f"Generator stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        app.logger.error(f"Generator failed: {e.stderr}")
        return jsonify({"error": e.stderr}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

    return jsonify({"status": "ok", "output_dir": str(OUTPUT_DIR), "vggface2_used": vggface2_path.exists()})


@app.route('/run_pipeline', methods=['POST'])
def run_pipeline():
    """
    Run the full pipeline (splits, exports, etc.) using the saved config.
    Expects JSON with optional 'num_samples', 'output_dir', and 'doc_type'.
    Includes VGGFace2 photos if configured.
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
    except:
        data = {}
    num_samples = data.get('num_samples', 5)
    doc_type = data.get('doc_type', 'cnie_front')
    custom_output = data.get('output_dir')
    
    # Validate doc_type
    if doc_type not in ['cnie_front', 'cnie_back', 'passport']:
        return jsonify({"error": f"Invalid doc_type: {doc_type}"}), 400
    
    if custom_output:
        output_dir = Path(custom_output)
    else:
        # Use a timestamped subfolder inside batch output
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = BATCH_OUTPUT_DIR / timestamp

    if not CONFIG_PATH.exists():
        return jsonify({"error": "No config saved"}), 400

    # Get vggface2 path from config
    vggface2_path = get_vggface2_path_from_config()
    
    # Build command for run_template_pipeline.py
    cmd = [
        sys.executable, str(PIPELINE_SCRIPT),
        '--doc-type', doc_type,
        '--doc-config', str(CONFIG_PATH),
        '--arabic-font', str(ARABIC_FONT),
        '--num-samples', str(num_samples),
        '--output-dir', str(output_dir)
    ]
    
    # Add face photos dir if vggface2 exists (and not cnie_back)
    if doc_type != 'cnie_back' and vggface2_path.exists():
        cmd.extend(['--face-photos-dir', str(vggface2_path)])
        print(f"🖼️  Using VGGFace2 photos from: {vggface2_path}")
    else:
        if doc_type == 'cnie_back':
            print(f"ℹ️  CNIE back has no photo placeholder")
        else:
            print(f"⚠️  VGGFace2 not found at: {vggface2_path}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600, cwd=str(PROJECT_ROOT))
        app.logger.info(f"Pipeline stdout: {result.stdout}")
        if result.stderr:
            app.logger.warning(f"Pipeline stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        app.logger.error(f"Pipeline failed: {e.stderr}")
        return jsonify({"error": e.stderr}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "status": "ok", 
        "output_dir": str(output_dir),
        "vggface2_used": vggface2_path.exists(),
        "vggface2_path": str(vggface2_path)
    })


@app.route('/generate_paired', methods=['POST'])
def generate_paired():
    """Generate paired CNIE front and back samples with the same identity."""
    try:
        data = request.get_json(force=True, silent=True) or {}
    except:
        data = {}
    
    num_samples = data.get('num_samples', 5)
    
    # Check for both front and back configs
    # Try specific paths first, then fall back to generic
    front_path = None
    back_path = None
    
    if CONFIG_FRONT_PATH.exists():
        front_path = CONFIG_FRONT_PATH
        print(f"📂 Using front config: {front_path}")
    elif CONFIG_PATH.exists():
        # Check if generic config is actually front config
        try:
            with open(CONFIG_PATH, 'r') as f:
                cfg = json.load(f)
            fields = cfg.get('fields', {})
            if 'national_id' in fields or 'personal_id' in fields:
                front_path = CONFIG_PATH
                print(f"📂 Using generic config as front: {front_path}")
        except:
            pass
    
    if CONFIG_BACK_PATH.exists():
        back_path = CONFIG_BACK_PATH
        print(f"📂 Using back config: {back_path}")
    
    # Validate configs exist
    if not front_path and not back_path:
        return jsonify({"error": "No configs saved. Please configure both CNIE front and back in the GUI tools first."}), 400
    
    if not front_path:
        return jsonify({"error": "CNIE front config not found. Please configure CNIE front first (must have national_id or personal_id fields)."}), 400
    
    if not back_path:
        return jsonify({"error": "CNIE back config not found. Please configure CNIE back first (must have mrz_line3 and birth_year fields)."}), 400
    
    # Get vggface2 path from config
    vggface2_path = get_vggface2_path_from_config()
    
    cmd = [
        sys.executable, str(GENERATOR_SCRIPT),
        '--doc-type', 'cnie_paired',
        '--config-front', str(front_path),
        '--config-back', str(back_path),
        '--arabic-font', str(ARABIC_FONT),
        '--num-samples', str(num_samples),
        '--output-dir', str(OUTPUT_DIR),
        '--template-dir', str(TEMPLATE_DIR)
    ]
    
    # Add face photos dir if vggface2 exists
    if vggface2_path.exists():
        cmd.extend(['--face-photos-dir', str(vggface2_path)])
        print(f"🖼️  Using VGGFace2 photos from: {vggface2_path}")
    else:
        print(f"⚠️  VGGFace2 not found at: {vggface2_path}, using synthetic faces")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300, cwd=str(PROJECT_ROOT))
        app.logger.info(f"Generator stdout: {result.stdout}")
        if result.stderr:
            app.logger.warning(f"Generator stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        app.logger.error(f"Generator failed: {e.stderr}")
        return jsonify({"error": e.stderr}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500
    
    return jsonify({
        "status": "ok",
        "output_dir": str(OUTPUT_DIR / "cnie_pairs"),
        "num_pairs": num_samples,
        "vggface2_used": vggface2_path.exists()
    })


@app.route('/set_vggface2_path', methods=['POST'])
def set_vggface2_path():
    """Set VGGFace2 dataset path and return stats."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        path = data.get('path', '')
        
        if not path:
            return jsonify({"error": "No path provided"}), 400
            
        vggface2_path = Path(path)
        if not vggface2_path.exists():
            return jsonify({"error": f"Path does not exist: {path}"}), 400
        
        # Count identities and images
        num_identities = 0
        num_images = 0
        try:
            for subdir in vggface2_path.iterdir():
                if subdir.is_dir():
                    num_identities += 1
                    num_images += len(list(subdir.glob('*.jpg')))
        except Exception as e:
            return jsonify({"error": f"Error reading directory: {e}"}), 500
        
        # Save path to current config
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
            config['vggface2_path'] = str(vggface2_path)
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
        
        return jsonify({
            "status": "ok",
            "path": str(vggface2_path),
            "num_identities": num_identities,
            "num_images": num_images
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_vggface2_status', methods=['GET'])
def get_vggface2_status():
    """Get current VGGFace2 configuration status."""
    try:
        vggface2_path = get_vggface2_path_from_config()
        
        if not vggface2_path.exists():
            return jsonify({"configured": False})
        
        # Count identities and images
        num_identities = 0
        num_images = 0
        try:
            for subdir in vggface2_path.iterdir():
                if subdir.is_dir():
                    num_identities += 1
                    num_images += len(list(subdir.glob('*.jpg')))
        except:
            pass
        
        return jsonify({
            "configured": True,
            "path": str(vggface2_path),
            "num_identities": num_identities,
            "num_images": num_images
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Print configuration paths for debugging
    print("=" * 60)
    print("Document Configuration Tool Server")
    print("=" * 60)
    print("Configuration:")
    print(f"  PROJECT_ROOT:     {PROJECT_ROOT}")
    print(f"  TEMPLATE_DIR:     {TEMPLATE_DIR}")
    print(f"  OUTPUT_DIR:       {OUTPUT_DIR}")
    print(f"  BATCH_OUTPUT_DIR: {BATCH_OUTPUT_DIR}")
    print(f"  CONFIG_DIR:       {CONFIG_DIR}")
    print(f"  ARABIC_FONT:      {ARABIC_FONT}")
    print(f"  VGGFACE2_DEFAULT: {VGGFACE2_DEFAULT}")
    print("-" * 60)
    print("Scripts:")
    print(f"  GENERATOR_SCRIPT: {GENERATOR_SCRIPT}")
    print(f"  PIPELINE_SCRIPT:  {PIPELINE_SCRIPT}")
    print("=" * 60)

    # Check critical dependencies
    all_ok = True
    if not GENERATOR_SCRIPT.exists():
        print(f"❌ Generator script not found: {GENERATOR_SCRIPT}")
        all_ok = False
    else:
        print(f"✅ Generator script found")
        
    if not PIPELINE_SCRIPT.exists():
        print(f"❌ Pipeline script not found: {PIPELINE_SCRIPT}")
        all_ok = False
    else:
        print(f"✅ Pipeline script found")
        
    if not ARABIC_FONT.exists():
        print(f"⚠️  Arabic font not found: {ARABIC_FONT}")
    else:
        print(f"✅ Arabic font found")
        
    if not TEMPLATE_DIR.exists():
        print(f"⚠️  Template directory not found: {TEMPLATE_DIR}")
    else:
        print(f"✅ Template directory found")
        
    if VGGFACE2_DEFAULT.exists():
        print(f"✅ VGGFace2 dataset found at: {VGGFACE2_DEFAULT}")
    else:
        print(f"⚠️  VGGFace2 dataset not found at: {VGGFACE2_DEFAULT}")
        
    if not all_ok:
        sys.exit(1)
        
    print("=" * 60)
    print("🚀 Server starting at http://127.0.0.1:5000")
    print("📂 GUI Tool: http://127.0.0.1:5000/")
    print("🪪  CNIE Front: http://127.0.0.1:5000/cnie_front")
    print("🪪  CNIE Back: http://127.0.0.1:5000/cnie_back")
    print("🛂 Passport: http://127.0.0.1:5000/passport (coming soon)")
    print("=" * 60)
    app.run(host='127.0.0.1', port=5000, debug=True)
