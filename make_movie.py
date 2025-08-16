#!/usr/bin/env python3
"""
Script to generate band structure movies for multilayer graphene.

Creates individual plots for 1-30 layers and combines them into both
WebM video and animated GIF formats for maximum compatibility.

Usage:
    python make_movie.py [framerate] [stacking_type] [max_layers]
    
Arguments:
    framerate: frames per second (default: 2)
    stacking_type: 'abc' or 'aba' (default: 'abc')
    max_layers: maximum number of layers (default: 30, max: 30)

Examples:
    python make_movie.py 5 abc       # 5 fps, ABC stacking, 30 layers
    python make_movie.py 1.5 aba 20  # 1.5 fps, ABA stacking, 20 layers
    python make_movie.py             # 2 fps, ABC stacking, 30 layers (defaults)
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path

# Import the multilayer graphene module
import multilayer_graphene as mlg

# Try to import PIL for GIF creation
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

def create_gif_from_images(image_dir, output_file, fps, max_layers):
    """Create GIF from PNG images using PIL."""
    if not PIL_AVAILABLE:
        print("  ✗ PIL (Pillow) not available for GIF creation")
        print("    Install with: pip install Pillow")
        return False
    
    # Find all layer images
    image_files = []
    for i in range(1, max_layers + 1):
        img_path = image_dir / f"layer_{i:02d}.png"
        if img_path.exists():
            image_files.append(img_path)
    
    if not image_files:
        print(f"  ✗ No images found in {image_dir}")
        return False
    
    print(f"  Creating GIF from {len(image_files)} images...")
    
    # Load images
    images = []
    for img_path in image_files:
        img = Image.open(img_path)
        # Convert to RGB if needed (GIF doesn't support RGBA well)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        images.append(img)
    
    # Calculate duration per frame (milliseconds)
    duration = int(1000 / fps)
    
    # Save as GIF
    images[0].save(
        output_file,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,  # Infinite loop
        optimize=True
    )
    
    return True

def main():
    # Parse command line arguments
    framerate = 2.0  # Default framerate
    stacking_type = 'abc'  # Default stacking
    max_layers = 30  # Default max layers
    
    if len(sys.argv) > 1:
        try:
            framerate = float(sys.argv[1])
        except ValueError:
            print(f"Error: Invalid framerate '{sys.argv[1]}'. Using default {framerate} fps.")
    
    if len(sys.argv) > 2:
        stacking_type = sys.argv[2].lower()
        if stacking_type not in ['abc', 'aba']:
            print(f"Error: Invalid stacking '{sys.argv[2]}'. Using default 'abc'.")
            stacking_type = 'abc'
    
    if len(sys.argv) > 3:
        try:
            max_layers = min(30, max(1, int(sys.argv[3])))
        except ValueError:
            print(f"Error: Invalid max_layers '{sys.argv[3]}'. Using default {max_layers}.")
    
    print(f"Generating multilayer graphene animations:")
    print(f"  Stacking: {stacking_type.upper()}")
    print(f"  Framerate: {framerate} fps")
    print(f"  Layers: 1-{max_layers}")
    print(f"  Output: WebM video + animated GIF")
    
    # Create output directory for individual frames
    frames_dir = Path(f"frames_{stacking_type}")
    frames_dir.mkdir(exist_ok=True)
    
    # Set up the multilayer graphene parameters
    mlg.set_parameters(stacking=stacking_type)
    
    # Generate individual plots for each layer count
    print(f"\nGenerating {max_layers} individual frames...")
    for N in range(1, max_layers + 1):
        print(f"  Layer {N:2d}/{max_layers}", end=" ", flush=True)
        
        # Calculate bands for this layer count
        mlg.set_parameters(N_layers=N)
        E, k_mag = mlg.calculate_bands()
        
        # Create the plot with consistent styling
        filename = frames_dir / f"layer_{N:02d}.png"
        mlg.plot_bands(
            E=E, 
            k_mag=k_mag, 
            N=N, 
            stacking_type=stacking_type,
            xlim = (-0.2, 0.2), ylim = (-0.6, 0.6),  # Fixed y-limits for consistency
            figsize=(6, 4),    # Slightly larger for movie
            save_as=str(filename),
            highlight_middle=True
        )
        print("✓")
    
    print(f"\nGenerated {len(list(frames_dir.glob('*.png')))} frames in {frames_dir}/")
    
    base_name = f"multilayer_graphene_{stacking_type}_{framerate}fps"
    
    print(f"\nCreating animations: {base_name}.*")
    print(f"  Input: {frames_dir}/layer_*.png")
    print(f"  Framerate: {framerate} fps")
    
    successful_outputs = []
    
    # 1. Create WebM (works reliably)
    webm_file = f"{base_name}.webm"
    print(f"\n1. Creating WebM video: {webm_file}")
    
    if shutil.which('ffmpeg'):
        webm_cmd = [
            'ffmpeg', '-y', '-framerate', str(framerate),
            '-i', str(frames_dir / 'layer_%02d.png'),
            '-c:v', 'libvpx-vp9',
            '-crf', '30',
            '-b:v', '0',  # Variable bitrate
            webm_file
        ]
        
        try:
            result = subprocess.run(webm_cmd, capture_output=True, text=True, check=True)
            file_size = os.path.getsize(webm_file) / (1024 * 1024)  # MB
            successful_outputs.append(f"{webm_file} ({file_size:.1f} MB)")
            print(f"  ✓ WebM created: {webm_file} ({file_size:.1f} MB)")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ WebM creation failed: {e.stderr.strip()}")
    else:
        print("  ✗ ffmpeg not found - skipping WebM")
        print("    Install with: brew install ffmpeg")
    
    # 2. Create animated GIF
    gif_file = f"{base_name}.gif"
    print(f"\n2. Creating animated GIF: {gif_file}")
    
    if create_gif_from_images(frames_dir, gif_file, framerate, max_layers):
        file_size = os.path.getsize(gif_file) / (1024 * 1024)  # MB
        successful_outputs.append(f"{gif_file} ({file_size:.1f} MB)")
        print(f"  ✓ GIF created: {gif_file} ({file_size:.1f} MB)")
    else:
        print(f"  ✗ GIF creation failed")
    
    # Summary
    print(f"\n{'='*60}")
    if successful_outputs:
        print("✓ Successfully created:")
        for output in successful_outputs:
            print(f"  - {output}")
        
        print(f"\nHow to open:")
        print(f"  WebM: Web browser, VLC Media Player, Chrome/Firefox")
        print(f"  GIF:  Any image viewer, web browser, social media")
    else:
        print("✗ No animations were created successfully.")
        print("Check ffmpeg installation and Pillow package.")
    
    print(f"\nFrames available in: {frames_dir}/")
    
    # Clean up frames (optional)
    cleanup = input(f"\nDelete frame images in {frames_dir}/? [y/N]: ").lower().strip()
    if cleanup in ['y', 'yes']:
        shutil.rmtree(frames_dir)
        print(f"✓ Cleaned up {frames_dir}/")
    else:
        print(f"Frames kept in {frames_dir}/")
    
    print(f"\nAnimation generation complete!")

if __name__ == "__main__":
    main()
