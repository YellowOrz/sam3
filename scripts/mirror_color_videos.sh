#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
用法:
  mirror_color_videos.sh <输入目录> <输出目录>

递归查找输入目录下所有名为 color.mp4 的视频，使用 FFmpeg 左右镜像，
并在输出目录中保留每个视频相对于输入目录的路径。

示例:
  ./scripts/mirror_color_videos.sh /data/dataset /data/dataset_mirrored
EOF
}

if [[ $# -ne 2 ]]; then
  usage >&2
  exit 2
fi

input_arg=$1
output_arg=$2

if [[ ! -d "$input_arg" ]]; then
  echo "错误：输入目录不存在：$input_arg" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "错误：未找到 ffmpeg，请先安装并确保它位于 PATH 中。" >&2
  exit 1
fi

# 使用规范化的绝对路径，确保相对路径计算准确。
input_root=$(cd "$input_arg" && pwd -P)
mkdir -p "$output_arg"
output_root=$(cd "$output_arg" && pwd -P)

if [[ "$input_root" == "$output_root" ]]; then
  echo "错误：输入目录和输出目录不能相同，以免覆盖源视频。" >&2
  exit 1
fi

processed=0
failed=0

while IFS= read -r -d '' source_video; do
  relative_path=${source_video#"$input_root"/}
  output_video="$output_root/$relative_path"

  mkdir -p "$(dirname "$output_video")"
  echo "镜像：$relative_path"

  # FFmpeg 默认会读取标准输入接收交互命令；这里必须禁用，否则它会消耗
  # while 循环中由 find -print0 提供的后续文件路径。
  if ffmpeg -nostdin -hide_banner -loglevel error -y \
    -i "$source_video" \
    -map '0:v?' -map '0:a?' -map_metadata 0 \
    -vf hflip \
    -c:v libx264 -preset medium -crf 18 \
    -c:a copy \
    -movflags +faststart \
    "$output_video"; then
    ((processed += 1))
  else
    echo "失败：$source_video" >&2
    ((failed += 1))
  fi
done < <(
  # 如果输出目录位于输入目录中，跳过它，避免重复处理生成的视频。
  find "$input_root" \
    -path "$output_root" -prune -o \
    -type f -name 'color.mp4' -print0
)

echo "完成：成功 $processed 个，失败 $failed 个。"

if ((failed > 0)); then
  exit 1
fi
