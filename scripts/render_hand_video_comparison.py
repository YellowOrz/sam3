"""CPU rendering of paired actual-output video masks, never new inference.

Reference and each prediction stay in separate monochrome panels. Review FPS
is playback speed only; MP4 is lossy and never used for metric computation.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scripts.compare_hand_routes import load_runs
from scripts import evaluate_realsense_full as full
from scripts.compare_realsense_full import decode_rle

CELL = (320, 240)
HEADER = 44
ROW = 276


def compose_frame(rgb, references, predictions, caption):
    """Each row is one side; RGB/reference/models are separate columns."""
    width, height = CELL
    canvas = Image.new('RGB', (width*(2+len(predictions)), HEADER+2*ROW), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 16)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 10), caption, fill='black', font=font)
    rgb = Image.fromarray(rgb).convert('RGB').resize(CELL, Image.Resampling.BILINEAR)
    for row, side in enumerate(full.SIDES):
        def mask_panel(mask):
            if mask is None:
                return Image.new('RGB', CELL, (110, 110, 110))
            return Image.fromarray(mask.astype(np.uint8)*255).convert('RGB').resize(CELL, Image.Resampling.NEAREST)
        panels = [('RGB', rgb), ('reference (assisted)', mask_panel(references[side]))]
        panels += [(label, mask_panel(masks[side])) for label, masks in predictions]
        y = HEADER+row*ROW
        for col, (name, panel) in enumerate(panels):
            draw.text((col*width+6, y+7), side+' | '+name, fill='black', font=font)
            canvas.paste(panel, (col*width, y+36))
    return canvas


def render(data_root, run_paths, output, fps):
    if type(fps) is not int or not 1 <= fps <= 30:
        raise ValueError('Review FPS must be 1..30')
    if output.exists():
        raise ValueError('Require new output directory')
    executable, probe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    if not executable or not probe:
        raise RuntimeError('ffmpeg and ffprobe are required')
    runs = load_runs(run_paths)
    spec = runs[0][1]['protocol']
    if (spec['evaluation_layer'] != 'video_system' or spec['frame_stride'] != 1
            or len(spec['recordings']) != 1 or not 2 <= len(runs) <= 3):
        raise ValueError('Require matching 2..3 method runs of one continuous recording')
    root = data_root.resolve()
    images, annotations, outputs, _, hashes = full.load_publication(root)
    if full.shared.sha256(root/'annotations.json') != spec['annotations_sha256']:
        raise ValueError('Reference annotation identity changed')
    by_id = {im['id']:i for i,im in enumerate(images)}
    selected = [by_id[i] for i in spec['image_ids']]
    if not 1 <= len(selected) <= 6204 or any(images[i]['source_frame_index'] != frame
            or images[i]['recording_id'] != spec['recordings'][0] for frame,i in enumerate(selected)):
        raise ValueError('Frames must be complete ordered prefix starting at zero')
    record_sets = []
    for path, summary in runs:
        rows = [json.loads(line) for line in (path/'records.jsonl').read_text().splitlines()]
        record_sets.append({(r['image_id'],r['prompt_key']):r for r in rows})
        hashes[str(path/'records.jsonl')] = summary['records_sha256']
        hashes[str(path/'summary.json')] = full.shared.sha256(path/'summary.json')
    output.mkdir(parents=True, exist_ok=False)
    width, height = CELL[0]*(len(runs)+2), HEADER+2*ROW
    movie = output/'comparison.mp4'
    cmd = [executable, '-n', '-loglevel', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
        '-video_size', f'{width}x{height}', '-framerate', str(fps), '-i', 'pipe:0', '-an',
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-threads', '1',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(movie)]
    with (output/'ffmpeg.log').open('xb') as log:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
        try:
            for frame, index in enumerate(selected):
                im = images[index]
                asset = outputs[im['id']]['files']['rgb']
                source = full.local_file(root, asset['path'])
                if full.shared.sha256(source) != asset['sha256']:
                    raise ValueError('RGB source changed')
                hashes[str(source)] = asset['sha256']
                with Image.open(source) as handle:
                    rgb = np.asarray(handle.convert('RGB'))
                refs = full.batch_references(root, images, [index], annotations, outputs, hashes)[im['id']]
                predictions = []
                flagged = []
                for (path, summary), rows in zip(runs, record_sets):
                    masks = {side: decode_rle(rows[im['id'],side]['prediction_rle'],
                                             (im['height'], im['width'])) for side in full.SIDES}
                    label = summary['method']+(f" e{summary['epoch']}" if summary['epoch'] else '')
                    predictions.append((label, masks))
                for side in full.SIDES:
                    if full.quality_flags(im,side): flagged.append(side)
                caption = f"{im['recording_id']} | source frame {frame} | review playback {fps} fps"
                if flagged: caption += ' | flagged reference: '+','.join(flagged)
                panel = compose_frame(rgb,refs,predictions,caption)
                if frame in (0,len(selected)//2,len(selected)-1): panel.save(output/f'frame-{frame:06d}.png')
                process.stdin.write(panel.tobytes())
            process.stdin.close()
            if process.wait(timeout=60) != 0:
                raise RuntimeError('ffmpeg failed; partial output retained')
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            raise
    for path, digest in hashes.items():
        if full.shared.sha256(Path(path)) != digest:
            raise ValueError('Input changed during rendering')
    observed = json.loads(subprocess.check_output([probe, '-v', 'error', '-count_frames',
        '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames,width,height', '-of', 'json', str(movie)]))['streams'][0]
    if (int(observed['nb_read_frames']),observed['width'],observed['height']) != (len(selected),width,height):
        raise ValueError('Encoded video frame coverage/size mismatch')
    full.shared.atomic_write_json(output/'manifest.json',dict(status='complete', frames=len(selected),
        image_ids=spec['image_ids'], review_fps=fps, dimensions=[width,height], input_sha256=hashes,
        movie_sha256=full.shared.sha256(movie), ffprobe=observed,
        note='Lossy review visualization, resized; authoritative masks/RLE and metrics remain unchanged. Review speed is not source FPS.'))
    (output/'README.md').write_text('# 连续视频分栏慢放\n\n'
        f'完整{len(selected)}帧，播放{fps}fps（非原视频时间尺度），已用ffprobe核验帧数与尺寸。\n\n'
        '上排左手、下排右手；每排RGB／辅助参考／各模型实际输出。不叠色，不重新推理。灰色为未知参考，不代表无手。\n\n'
        'MP4是缩放、有损的浏览文件，不参与评分；原始RLE与全尺寸PNG仍是准确依据。\n\n'
        '![分栏慢放](comparison.mp4)\n\n[完整核验清单](manifest.json)\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--run', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fps', type=int, default=8)
    args = p.parse_args()
    render(args.data_root, args.run, args.output, args.fps)
