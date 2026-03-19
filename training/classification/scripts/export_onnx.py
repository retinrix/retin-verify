"""
Export CNIE Classification Model to ONNX Format

This script exports the trained PyTorch model to ONNX format for:
- Cross-platform deployment
- Optimized inference with ONNX Runtime
- Mobile deployment (via ONNX to TFLite conversion)

Usage:
    python export_onnx.py --model-path /path/to/model.pth --output /path/to/model.onnx
"""

import torch
import torch.nn as nn
from pathlib import Path
import argparse
import logging
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from inference_engine import CNIEClassifier, get_model_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def export_to_onnx(
    model_path: str,
    output_path: str,
    input_size: int = 224,
    opset_version: int = 14,
    dynamic_axes: bool = True
):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model_path: Path to PyTorch model (.pth file)
        output_path: Output path for ONNX model
        input_size: Input image size
        opset_version: ONNX opset version
        dynamic_axes: Whether to use dynamic batch/height/width
    """
    logger.info("=" * 60)
    logger.info("Exporting CNIE Classifier to ONNX")
    logger.info("=" * 60)
    
    # Load model
    logger.info(f"Loading PyTorch model from: {model_path}")
    classifier = CNIEClassifier(
        model_path=model_path,
        device='cpu',  # Export on CPU for compatibility
        input_size=input_size
    )
    
    model = classifier.model
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    # Setup dynamic axes for flexible batch size and image dimensions
    if dynamic_axes:
        dynamic_axes_config = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size'}
        }
        logger.info("Dynamic axes enabled: batch_size, height, width")
    else:
        dynamic_axes_config = None
    
    # Export
    logger.info(f"Exporting to ONNX (opset version: {opset_version})...")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes_config,
            verbose=False
        )
        
        logger.info(f"✓ Export successful!")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise
    
    # Verify the export
    output_file = Path(output_path)
    if output_file.exists():
        size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"Output file: {output_path}")
        logger.info(f"File size: {size_mb:.2f} MB")
    
    # Test the exported model
    logger.info("\nValidating exported model...")
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("✓ ONNX model validation passed")
        
        # Print model info
        logger.info(f"\nModel info:")
        logger.info(f"  IR version: {onnx_model.ir_version}")
        logger.info(f"  Opset version: {opset_version}")
        logger.info(f"  Producer: {onnx_model.producer_name}")
        
        # Get input/output info
        for input in onnx_model.graph.input:
            logger.info(f"  Input: {input.name} - {input.type}")
        for output in onnx_model.graph.output:
            logger.info(f"  Output: {output.name} - {output.type}")
            
    except ImportError:
        logger.warning("onnx package not available for validation")
    except Exception as e:
        logger.warning(f"Validation warning: {e}")
    
    return output_path


def test_onnx_inference(onnx_path: str, input_size: int = 224):
    """
    Test ONNX model inference.
    
    Args:
        onnx_path: Path to ONNX model
        input_size: Input image size
    """
    logger.info("\n" + "=" * 60)
    logger.info("Testing ONNX Inference")
    logger.info("=" * 60)
    
    try:
        import onnxruntime as ort
        import numpy as np
        
        # Create inference session
        providers = ort.get_available_providers()
        logger.info(f"Available providers: {providers}")
        
        # Prefer CUDA if available, otherwise CPU
        if 'CUDAExecutionProvider' in providers:
            session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            logger.info("Using CUDA Execution Provider")
        else:
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            logger.info("Using CPU Execution Provider")
        
        # Get input info
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        logger.info(f"Input name: {input_name}")
        logger.info(f"Input shape: {input_shape}")
        
        # Create dummy input
        dummy_input = np.random.randn(1, 3, input_size, input_size).astype(np.float32)
        
        # Run inference
        import time
        num_runs = 10
        
        # Warmup
        for _ in range(3):
            session.run(None, {input_name: dummy_input})
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            outputs = session.run(None, {input_name: dummy_input})
            times.append(time.time() - start)
        
        mean_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        
        logger.info(f"\nInference Performance ({num_runs} runs):")
        logger.info(f"  Mean time: {mean_time:.2f} ms")
        logger.info(f"  Std dev: {std_time:.2f} ms")
        logger.info(f"  FPS: {1000/mean_time:.1f}")
        
        # Get output
        output = outputs[0]
        logger.info(f"\nOutput shape: {output.shape}")
        logger.info(f"Output sample: {output[0][:4]}")
        
        # Apply softmax to get probabilities
        exp_output = np.exp(output - np.max(output))
        probs = exp_output / np.sum(exp_output)
        predicted_class = np.argmax(probs)
        confidence = probs[0][predicted_class]
        
        class_names = ['cnie_front', 'cnie_back', 'other_front', 'other_back']
        logger.info(f"\nPredicted class: {class_names[predicted_class]}")
        logger.info(f"Confidence: {confidence:.4f}")
        
        logger.info("\n✓ ONNX inference test passed")
        
    except ImportError:
        logger.warning("onnxruntime not available for testing")
    except Exception as e:
        logger.error(f"ONNX inference test failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Export CNIE Classifier to ONNX format'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to PyTorch model (.pth file)'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        help='Output path for ONNX model'
    )
    parser.add_argument(
        '--input-size',
        type=int,
        default=224,
        help='Input image size (default: 224)'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=14,
        help='ONNX opset version (default: 14)'
    )
    parser.add_argument(
        '--no-dynamic-axes',
        action='store_true',
        help='Disable dynamic axes (batch/height/width)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test ONNX inference after export'
    )
    
    args = parser.parse_args()
    
    # Get default model path
    if args.model_path is None:
        try:
            model_path = get_model_path()
            args.model_path = str(model_path)
        except FileNotFoundError:
            logger.error("No model found. Please specify --model-path")
            sys.exit(1)
    
    # Set default output path
    if args.output is None:
        input_path = Path(args.model_path)
        args.output = str(input_path.parent / (input_path.stem + '.onnx'))
    
    # Export
    try:
        export_to_onnx(
            model_path=args.model_path,
            output_path=args.output,
            input_size=args.input_size,
            opset_version=args.opset,
            dynamic_axes=not args.no_dynamic_axes
        )
        
        # Test if requested
        if args.test:
            test_onnx_inference(args.output, args.input_size)
        
        logger.info("\n" + "=" * 60)
        logger.info("Export completed successfully!")
        logger.info("=" * 60)
        logger.info(f"ONNX model: {args.output}")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
