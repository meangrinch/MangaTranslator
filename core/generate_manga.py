import argparse
import base64
import os
import time
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

from .utils import (
    detect_speech_bubbles,
    clean_speech_bubbles,
    pil_to_cv2,
    cv2_to_pil,
    call_gemini_api_batch,
    sort_bubbles_manga_order,
    log_message,
    choose_font,
    fit_text_to_bubble,
    torch
)

def save_image_with_compression(image, output_path, jpeg_quality=95, png_compression=6, verbose=False):
    """
    Save an image with specified compression settings.
    
    Args:
        image (PIL.Image): Image to save
        output_path (str or Path): Path to save the image
        original_format (str, optional): Format of the original image
        jpeg_quality (int): JPEG quality (1-100, higher is better quality)
        png_compression (int): PNG compression level (0-9, higher is more compression)
        verbose (bool): Whether to print verbose logging
    
    Returns:
        bool: Whether the save was successful
    """
    output_path = Path(output_path) if not isinstance(output_path, Path) else output_path
    
    extension = output_path.suffix.lower()
    if extension == '.jpg' or extension == '.jpeg':
        output_format = 'JPEG'
        # Convert RGBA to RGB for JPEG by compositing on white background
        if image.mode == 'RGBA':
            log_message(f"Converting RGBA to RGB for JPEG output", verbose=verbose)
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode == 'P': # Handle Palette mode
             log_message(f"Converting P mode to RGB for JPEG output", verbose=verbose)
             image = image.convert('RGB')
    elif extension == '.png':
        output_format = 'PNG'
    elif extension == '.webp':
        output_format = 'WEBP'
    else:
        log_message(f"Warning: Unknown output extension '{extension}'. Saving as PNG.", verbose=verbose, always_print=True)
        output_format = 'PNG'
        output_path = output_path.with_suffix('.png')
    
    try:
        os.makedirs(output_path.parent, exist_ok=True)
        
        save_options = {}
        if output_format == 'JPEG':
            save_options['quality'] = max(1, min(jpeg_quality, 100))
            log_message(f"Saving JPEG image with quality {save_options['quality']} to {output_path}", verbose=verbose)
        elif output_format == 'PNG':
            save_options['compress_level'] = max(0, min(png_compression, 9))
            log_message(f"Saving PNG image with compression level {save_options['compress_level']} to {output_path}", verbose=verbose)
        elif output_format == 'WEBP':
             save_options['lossless'] = True # Reuse png compression for webp lossy
             log_message(f"Saving WEBP image with quality {save_options['quality']} to {output_path}", verbose=verbose)
        else:
             log_message(f"Saving image in {output_format} format to {output_path}", verbose=verbose)

        image.save(str(output_path), format=output_format, **save_options)
        
        return True
    except Exception as e:
        log_message(f"Error saving image to {output_path}: {e}", always_print=True)
        return False

def translate_and_render(image_path, yolo_model_path, api_key, font_dir="./fonts", 
                        input_language="Japanese", output_language="English", 
                        confidence=0.35, output_path=None, dilation_kernel_size=7, dilation_iterations=1,
                        threshold_value=210, min_contour_area=50,
                        closing_kernel_size=7, closing_iterations=1, closing_kernel_shape=cv2.MORPH_ELLIPSE,
                        constraint_erosion_kernel_size=5, constraint_erosion_iterations=1,
                        gemini_model="gemini-2.0-flash", temperature=0.1, top_p=0.95, top_k=1,
                        max_font_size=14, min_font_size=8, line_spacing=1.0,
                        jpeg_quality=95, png_compression=6, verbose=False, image_mode="RGBA", device=None):
    """
    Main function to translate manga speech bubbles and render translations.
    
    Args:
        image_path (str): Path to input image
        yolo_model_path (str): Path to YOLO model
        api_key (str): Gemini API key
        font_dir (str): Directory containing font files
        input_language (str): Source language
        output_language (str): Target language
        confidence (float): Confidence threshold for detections
        output_path (str): Path to save the final image
        dilation_kernel_size (int): Kernel size for dilating initial YOLO mask for ROI.
        dilation_iterations (int): Iterations for dilation.
        threshold_value (int or None): Fixed threshold for isolating bright bubble interior. None uses Otsu.
        min_contour_area (int): Minimum area for a contour to be considered part of the bubble interior.
        closing_kernel_size (int): Kernel size for morphological closing to smooth mask and include text.
        closing_iterations (int): Iterations for closing.
        closing_kernel_shape (int): Shape for the closing kernel (e.g., cv2.MORPH_RECT, cv2.MORPH_ELLIPSE).
        constraint_erosion_kernel_size (int): Kernel size for eroding the original YOLO mask for edge constraint.
        constraint_erosion_iterations (int): Iterations for constraint erosion.
        gemini_model (str): Gemini model to use for inference
        temperature (float): Controls randomness in output
        top_p (float): Nucleus sampling parameter
        top_k (int): Limits to top k tokens
        max_font_size (int): Max font size for rendering text.
        min_font_size (int): Min font size for rendering text.
        line_spacing (float): Line spacing multiplier for rendered text.
        jpeg_quality (int): JPEG compression quality (1-100)
        png_compression (int): PNG compression level (0-9)
        verbose (bool): Whether to print intermediate logs
        image_mode (str): Image mode to use ("RGBA" or "RGB")
        device (torch.device, optional): The device to run models on. Autodetects if None.
    
    Returns:
        PIL.Image: Final translated image
    """
    start_time = time.time()
    
    _device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        pil_original = Image.open(image_path)
    except FileNotFoundError:
        log_message(f"Error: Input image not found at {image_path}", always_print=True)
        raise
    except Exception as e:
        log_message(f"Error opening image {image_path}: {e}", always_print=True)
        raise
    
    original_mode = pil_original.mode
    log_message(f"Original image mode: {original_mode}", verbose=verbose)
    if image_mode == "RGB":
        if pil_original.mode == "RGBA":
            log_message("Converting RGBA to RGB", verbose=verbose)
            background = Image.new('RGB', pil_original.size, (255, 255, 255))
            background.paste(pil_original, mask=pil_original.split()[3])
            pil_image_processed = background
        elif pil_original.mode == "P": # Handle palette mode
             log_message("Converting P to RGB", verbose=verbose)
             pil_image_processed = pil_original.convert("RGB")
        elif pil_original.mode == "LA": # Handle Luminance + Alpha
             log_message("Converting LA to RGB", verbose=verbose)
             background = Image.new('RGB', pil_original.size, (255, 255, 255))
             background.paste(pil_original, mask=pil_original.split()[1])
             pil_image_processed = background
        elif pil_original.mode != "RGB":
             log_message(f"Converting {pil_original.mode} to RGB", verbose=verbose)
             pil_image_processed = pil_original.convert("RGB")
        else:
             pil_image_processed = pil_original # Already RGB
    elif image_mode == "RGBA":
        if pil_original.mode != "RGBA":
            log_message(f"Converting {pil_original.mode} to RGBA", verbose=verbose)
            pil_image_processed = pil_original.convert("RGBA")
        else:
            pil_image_processed = pil_original # Already RGBA
    else: # Default to original mode if image_mode is not RGB or RGBA
        log_message(f"Using original image mode: {original_mode}", verbose=verbose)
        pil_image_processed = pil_original

    original_cv_image = pil_to_cv2(pil_image_processed)
    
    # --- Detection ---
    log_message("Detecting speech bubbles...", verbose=verbose)
    try:
        bubble_data = detect_speech_bubbles(image_path, yolo_model_path, confidence, verbose=verbose, device=_device)
    except Exception as e:
         log_message(f"Error during bubble detection: {e}", always_print=True)
         if output_path:
             save_image_with_compression(pil_image_processed, output_path, jpeg_quality, png_compression, verbose)
         return pil_image_processed
    
    if not bubble_data:
        log_message("No speech bubbles detected in the image", always_print=True)
        if output_path:
            save_image_with_compression(pil_image_processed, output_path, jpeg_quality, png_compression, verbose)
        return pil_image_processed
    
    log_message(f"Detected {len(bubble_data)} potential bubbles.", verbose=verbose)
    
    # --- Cleaning ---
    log_message("Cleaning speech bubbles...", verbose=verbose)
    try:
        actual_threshold_value = -1 if threshold_value < 0 else threshold_value
        
        cleaned_image_cv = clean_speech_bubbles(
            image_path,
            yolo_model_path,
            confidence,
            pre_computed_detections=bubble_data,
            device=_device,
            dilation_kernel_size=dilation_kernel_size,
            dilation_iterations=dilation_iterations,
            threshold_value=actual_threshold_value,
            min_contour_area=min_contour_area,
            closing_kernel_size=closing_kernel_size,
            closing_iterations=closing_iterations,
            closing_kernel_shape=closing_kernel_shape,
            constraint_erosion_kernel_size=constraint_erosion_kernel_size,
            constraint_erosion_iterations=constraint_erosion_iterations
            )
    except Exception as e:
        log_message(f"Error during bubble cleaning: {e}. Proceeding with uncleaned image.", always_print=True)
        cleaned_image_cv = original_cv_image.copy()

    pil_cleaned_image = cv2_to_pil(cleaned_image_cv)
    if pil_cleaned_image.mode != pil_image_processed.mode:
         log_message(f"Converting cleaned image mode from {pil_cleaned_image.mode} to {pil_image_processed.mode}", verbose=verbose)
         pil_cleaned_image = pil_cleaned_image.convert(pil_image_processed.mode)

    draw = ImageDraw.Draw(pil_cleaned_image)
    
    # --- Prepare images for Translation ---
    log_message("Preparing bubble images for translation...", verbose=verbose)
    for bubble in bubble_data:
        x1, y1, x2, y2 = bubble["bbox"]

        bubble_image_cv = original_cv_image[y1:y2, x1:x2].copy()

        try:
            is_success, buffer = cv2.imencode('.jpg', bubble_image_cv)
            if not is_success:
                raise ValueError("cv2.imencode failed")
            image_b64 = base64.b64encode(buffer).decode('utf-8')
            bubble["image_b64"] = image_b64
            log_message(f"Collected bubble at ({x1}, {y1}), size {x2-x1}x{y2-y1}", verbose=verbose)
        except Exception as e:
             log_message(f"Error encoding bubble at {bubble['bbox']}: {e}", verbose=verbose, always_print=True)
             bubble["image_b64"] = None # Mark as failed

    # Filter out bubbles that failed encoding
    valid_bubble_data = [b for b in bubble_data if b.get("image_b64")]
    if not valid_bubble_data:
         log_message("No valid bubble images could be prepared for translation.", always_print=True)
         if output_path:
             save_image_with_compression(pil_cleaned_image, output_path, jpeg_quality, png_compression, verbose)
         return pil_cleaned_image
    
    # --- Sort and Translate ---
    log_message("Sorting bubbles in manga reading order...", verbose=verbose)
    image_width = original_cv_image.shape[1]

    sorted_bubble_data = sort_bubbles_manga_order(valid_bubble_data, image_width)

    bubble_images_b64 = [bubble["image_b64"] for bubble in sorted_bubble_data]
    if not bubble_images_b64:
         log_message("No bubbles to translate after sorting/filtering.", always_print=True)
         if output_path:
             save_image_with_compression(pil_cleaned_image, output_path, jpeg_quality, png_compression, verbose)
         return pil_cleaned_image

    log_message(f"Translating {len(bubble_images_b64)} speech bubbles from {input_language} to {output_language}...",
                always_print=True)

    try:
        translated_texts = call_gemini_api_batch(
            api_key,
            bubble_images_b64,
            input_language,
            output_language,
            gemini_model,
            temperature,
            top_p,
            top_k,
            debug=verbose
        )
    except Exception as e:
         log_message(f"Error during API translation: {e}", always_print=True)
         translated_texts = ["[Translation Error]" for _ in sorted_bubble_data]
    
    # --- Render Translations ---
    log_message("Rendering translations onto image...", verbose=verbose)
    if len(translated_texts) == len(sorted_bubble_data):
        for i, bubble in enumerate(sorted_bubble_data):
            bubble["translation"] = translated_texts[i]
            bbox = bubble["bbox"]
            text = bubble.get("translation", "")

            if not text or text.startswith("API Error") or text == "No text detected or translation failed" or text.startswith("[No translation") or text.startswith("[Translation Error]"):
                log_message(f"Skipping rendering for bubble {bbox} due to empty or error translation: '{text}'", verbose=verbose)
                continue

            log_message(f"Rendering text for bubble {bbox}: '{text[:30]}...'", verbose=verbose)

            font_path = choose_font(text, font_dir, verbose=verbose)
            if font_path == "default":
                 log_message(f"Warning: Using default font for bubble {bbox}", verbose=verbose, always_print=True)

            success = fit_text_to_bubble(
                draw, text, bbox, font_path,
                max_font_size=max_font_size,
                min_font_size=min_font_size,
                line_spacing=line_spacing,
                verbose=verbose
            )
            if not success:
                 log_message(f"Failed to fit text in bubble {bbox}", verbose=verbose, always_print=True)
    else:
         log_message(f"Warning: Mismatch between number of bubbles ({len(sorted_bubble_data)}) and translations ({len(translated_texts)}). Skipping rendering.", always_print=True)
    
    # --- Save Output ---
    if output_path:
        log_message(f"Saving final image to {output_path}", verbose=verbose)
        save_image_with_compression(
            pil_cleaned_image,
            output_path,
            jpeg_quality=jpeg_quality,
            png_compression=png_compression,
            verbose=verbose
        )

    end_time = time.time()
    processing_time = end_time - start_time
    log_message(f"Processing finished in {processing_time:.2f} seconds.", always_print=True)

    return pil_cleaned_image

def batch_translate_images(input_dir, output_dir=None, yolo_model_path=None, api_key=None, 
                          font_dir="./fonts", input_language="Japanese", output_language="English", 
                          confidence=0.35, dilation_kernel_size=7, dilation_iterations=1,
                          threshold_value=210, min_contour_area=50,
                          closing_kernel_size=7, closing_iterations=1, closing_kernel_shape=cv2.MORPH_ELLIPSE,
                          constraint_erosion_kernel_size=5, constraint_erosion_iterations=1,
                          gemini_model="gemini-2.0-flash", temperature=0.1, top_p=0.95, top_k=1, 
                          max_font_size=14, min_font_size=8, line_spacing=1.0,
                          jpeg_quality=95, png_compression=6, verbose=False,
                          progress_callback=None, device=None):
    """
    Process all images in a directory and translate speech bubbles.
    
    Args:
        input_dir (str or Path): Directory containing images to process
        output_dir (str or Path, optional): Directory to save translated images. If None, uses input_dir.
        yolo_model_path (str): Path to YOLO model
        api_key (str): Gemini API key
        font_dir (str): Directory containing font files
        input_language (str): Source language
        output_language (str): Target language
        confidence (float): Confidence threshold for detections
        dilation_kernel_size (int): Kernel size for dilating initial YOLO mask for ROI.
        dilation_iterations (int): Iterations for dilation.
        threshold_value (int or None): Fixed threshold for isolating bright bubble interior. None uses Otsu.
        min_contour_area (int): Minimum area for a contour to be considered part of the bubble interior.
        closing_kernel_size (int): Kernel size for morphological closing to smooth mask and include text.
        closing_iterations (int): Iterations for closing.
        closing_kernel_shape (int): Shape for the closing kernel (e.g., cv2.MORPH_RECT, cv2.MORPH_ELLIPSE).
        constraint_erosion_kernel_size (int): Kernel size for eroding the original YOLO mask for edge constraint.
        constraint_erosion_iterations (int): Iterations for constraint erosion.
        gemini_model (str): Gemini model to use for inference
        temperature (float): Controls randomness in output
        top_p (float): Nucleus sampling parameter
        top_k (int): Limits to top k tokens
        max_font_size (int): Max font size for rendering text.
        min_font_size (int): Min font size for rendering text.
        line_spacing (float): Line spacing multiplier for rendered text.
        jpeg_quality (int): JPEG compression quality (1-100)
        png_compression (int): PNG compression level (0-9)
        verbose (bool): Whether to print intermediate logs
        progress_callback (callable, optional): Function to call with progress updates (0.0-1.0)
        device (torch.device, optional): The device to run models on. Autodetects if None.
        
    Returns:
        dict: Processing results with keys:
            - "success_count": Number of successfully processed images
            - "error_count": Number of images that failed to process
            - "errors": Dictionary mapping filenames to error messages
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        log_message(f"Input path '{input_dir}' is not a directory", always_print=True)
        return {"success_count": 0, "error_count": 0, "errors": {}}
    
    output_dir = Path(output_dir) if output_dir else input_dir / "output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    image_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        log_message(f"No image files found in '{input_dir}'", always_print=True)
        return {"success_count": 0, "error_count": 0, "errors": {}}
    
    results = {
        "success_count": 0,
        "error_count": 0,
        "errors": {}
    }
    
    total_images = len(image_files)
    start_batch_time = time.time()
    
    log_message(f"Starting batch processing of {total_images} images...", always_print=True)
    
    if progress_callback:
        progress_callback(0.0, f"Starting batch processing of {total_images} images...")
    
    for i, img_path in enumerate(image_files):
        try:
            if progress_callback:
                current_progress = i / total_images
                progress_callback(current_progress, f"Processing image {i+1}/{total_images}: {img_path.name}")
            
            output_path = output_dir / f"{img_path.stem}_translated{img_path.suffix}"
            log_message(f"Image {i+1}/{total_images}: Processing {img_path.name}", always_print=True)
            
            output_ext = output_path.suffix.lower()
            if output_ext == '.jpg' or output_ext == '.jpeg':
                image_mode = "RGB"
            else:
                image_mode = "RGBA"
            
            translate_and_render(
                str(img_path),
                yolo_model_path,
                api_key,
                font_dir,
                input_language,
                output_language,
                confidence,
                str(output_path),
                dilation_kernel_size=dilation_kernel_size,
                dilation_iterations=dilation_iterations,
                threshold_value=threshold_value,
                min_contour_area=min_contour_area,
                closing_kernel_size=closing_kernel_size,
                closing_iterations=closing_iterations,
                closing_kernel_shape=closing_kernel_shape,
                constraint_erosion_kernel_size=constraint_erosion_kernel_size,
                constraint_erosion_iterations=constraint_erosion_iterations,
                gemini_model=gemini_model,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_font_size=max_font_size,
                min_font_size=min_font_size,
                line_spacing=line_spacing, 
                jpeg_quality=jpeg_quality,
                png_compression=png_compression,
                verbose=verbose,
                image_mode=image_mode,
                device=device
            )
            
            results["success_count"] += 1
            
            if progress_callback:
                completed_progress = (i + 1) / total_images
                progress_callback(completed_progress, f"Completed {i+1}/{total_images} images")
                
        except Exception as e:
            log_message(f"Error processing {img_path.name}: {str(e)}", always_print=True)
            results["error_count"] += 1
            results["errors"][img_path.name] = str(e)
            
            if progress_callback:
                completed_progress = (i + 1) / total_images
                progress_callback(completed_progress, f"Completed {i+1}/{total_images} images (with errors)")
    
    if progress_callback:
        progress_callback(1.0, "Processing complete")
    
    end_batch_time = time.time()
    total_batch_time = end_batch_time - start_batch_time
    seconds_per_image = total_batch_time / results["success_count"] if results["success_count"] > 0 else 0
    
    log_message(f"\nBatch processing complete: {results['success_count']} of {len(image_files)} images translated in {total_batch_time:.2f} seconds ({seconds_per_image:.2f} seconds/image).", 
                always_print=True)
    if results["error_count"] > 0:
        log_message(f"Failed to process {results['error_count']} images.", always_print=True)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Translate manga/comic speech bubbles")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or directory (if using --batch)")
    parser.add_argument("--output", type=str, required=False, help="Path to save the translated image or directory (if using --batch)")
    parser.add_argument("--batch", action="store_true", help="Process all images in the input directory")
    parser.add_argument("--yolo-model", type=str, required=True, help="Path to YOLO model")
    parser.add_argument("--api-key", type=str, required=True, help="Gemini API key (can also be set via environment variable GOOGLE_API_KEY)")
    parser.add_argument("--gemini-model", type=str, default="gemini-2.0-flash", help="Gemini model to use")
    parser.add_argument("--font-dir", type=str, default="./fonts/CC Wild Words", help="Directory containing font files")
    parser.add_argument("--input-language", type=str, default="Japanese", help="Source language")
    parser.add_argument("--output-language", type=str, default="English", help="Target language")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--dilation-kernel-size", type=int, default=7, help="ROI Dilation Kernel Size")
    parser.add_argument("--dilation-iterations", type=int, default=1, help="ROI Dilation Iterations")
    parser.add_argument("--threshold-value", type=int, default=210, help="Bubble Interior Threshold (0-255, or -1 for auto)")
    parser.add_argument("--min-contour-area", type=int, default=50, help="Min Bubble Contour Area")
    parser.add_argument("--closing-kernel-size", type=int, default=7, help="Mask Closing Kernel Size")
    parser.add_argument("--closing-iterations", type=int, default=1, help="Mask Closing Iterations")
    parser.add_argument("--constraint-erosion-kernel-size", type=int, default=3, help="Edge Constraint Erosion Kernel Size")
    parser.add_argument("--constraint-erosion-iterations", type=int, default=1, help="Edge Constraint Erosion Iterations")
    parser.add_argument("--temperature", type=float, default=0.1, help="Controls randomness in output (0.0-2.0)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling parameter (0.0-1.0)")
    parser.add_argument("--top-k", type=int, default=1, help="Limits to top k tokens")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG compression quality (1-100)")
    parser.add_argument("--png-compression", type=int, default=6, help="PNG compression level (0-9)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.set_defaults(verbose=False, cpu=False)
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
         print("Error: Gemini API key not provided. Set --api-key or GOOGLE_API_KEY environment variable.")
         exit(1)
    
    threshold_val_arg = None if args.threshold_value < 0 else args.threshold_value
    
    target_device = torch.device('cpu') if args.cpu or not torch.cuda.is_available() else torch.device('cuda')
    if not args.cpu and torch.cuda.is_available():
        print("Using CUDA device.")
    else:
        print("Using CPU device.")
    
    if args.batch:
        if not os.path.isdir(args.input):
             print(f"Error: --batch requires --input '{args.input}' to be a directory.")
             exit(1)
        output_dir = args.output
        if not output_dir:
            output_dir = os.path.join(args.input, "output_translated")
            print(f"--output not specified, using default: {output_dir}")
        elif not os.path.exists(output_dir):
             os.makedirs(output_dir)
             print(f"Created output directory: {output_dir}")
        elif not os.path.isdir(output_dir):
             print(f"Error: Specified --output '{output_dir}' is not a directory.") 
             exit(1)
        
        batch_translate_images(
            args.input,
            output_dir,
            args.yolo_model,
            api_key,
            args.font_dir,
            args.input_language,
            args.output_language,
            args.conf,
            dilation_kernel_size=args.dilation_kernel_size,
            dilation_iterations=args.dilation_iterations,
            threshold_value=threshold_val_arg,
            min_contour_area=args.min_contour_area,
            closing_kernel_size=args.closing_kernel_size,
            closing_iterations=args.closing_iterations,
            closing_kernel_shape=cv2.MORPH_ELLIPSE,
            constraint_erosion_kernel_size=args.constraint_erosion_kernel_size,
            constraint_erosion_iterations=args.constraint_erosion_iterations,
            gemini_model=args.gemini_model,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_font_size=args.max_font_size,
            min_font_size=args.min_font_size,
            line_spacing=args.line_spacing,
            jpeg_quality=args.jpeg_quality,
            png_compression=args.png_compression,
            verbose=args.verbose,
            device=target_device
        )
    else:
        if not os.path.isfile(args.input):
            print(f"Error: Input '{args.input}' is not a valid file.")
            exit(1)
        output_path = args.output
        if not output_path:
            input_path = Path(args.input)
            output_path = str(input_path.parent / f"{input_path.stem}_translated{input_path.suffix}")
            print(f"--output not specified, using default: {output_path}")
        
        try:
            log_message(f"Processing {args.input}...", always_print=True)
            
            output_ext = Path(output_path).suffix.lower()
            if output_ext == '.jpg' or output_ext == '.jpeg':
                image_mode = "RGB"
            else:
                image_mode = "RGBA"
            
            translate_and_render(
                args.input,
                args.yolo_model,
                api_key,
                args.font_dir,
                args.input_language,
                args.output_language,
                args.conf,
                output_path,
                dilation_kernel_size=args.dilation_kernel_size,
                dilation_iterations=args.dilation_iterations,
                threshold_value=threshold_val_arg,
                min_contour_area=args.min_contour_area,
                closing_kernel_size=args.closing_kernel_size,
                closing_iterations=args.closing_iterations,
                closing_kernel_shape=cv2.MORPH_ELLIPSE,
                constraint_erosion_kernel_size=args.constraint_erosion_kernel_size,
                constraint_erosion_iterations=args.constraint_erosion_iterations,
                gemini_model=args.gemini_model,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_font_size=args.max_font_size,
                min_font_size=args.min_font_size,
                line_spacing=args.line_spacing,
                jpeg_quality=args.jpeg_quality,
                png_compression=args.png_compression,
                verbose=args.verbose,
                image_mode=image_mode,
                device=target_device
            )
            log_message(f"Translation complete. Result saved to {output_path}", always_print=True)
        except Exception as e:
            log_message(f"Error processing {args.input}: {e}", always_print=True)

if __name__ == "__main__":
    main()
