#!/usr/bin/env bash
# Install the Wan 2.2 S2V (speech-to-video) weights on the render box.
#
# Why this is needed: ComfyUI 0.37.0 on 50.50.234.145 ALREADY ships the nodes
# (WanSoundImageToVideo / WanSoundImageToVideoExtend) but none of the weights, and its
# ComfyUI-Manager catalogue is pinned to an offline snapshot that predates Wan 2.2 S2V,
# so the Manager API can only fetch 2 of the 4 files. Hence this script.
#
# NOTE: this deployment is air-gapped BY DESIGN (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1,
# PIP_NO_INDEX=1, Manager network_mode=offline, pull_policy=never). Those are CONTAINER
# settings and do not by themselves stop wget on the HOST -- but if the host has no egress
# either, the preflight below stops the script and you want the staging route at the bottom.
#
# Run it ON the box (needs ~24 GB free). Re-running is safe: -nc skips what is already there.
#
#   bash install_wan_s2v.sh
#
# Then restart ComfyUI so it rescans models/.

set -euo pipefail

# The container's model tree, per the compose bind mount. Override if yours differs.
MODELS="${COMFY_MODELS:-/data/projects/shaheen/comfyui-data/models}"

HF="https://huggingface.co"
W22="$HF/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files"
W21="$HF/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files"

echo "Installing Wan 2.2 S2V weights into $MODELS"
[ -d "$MODELS" ] || { echo "ERROR: $MODELS does not exist - set COMFY_MODELS"; exit 1; }

echo -n "Checking host egress to huggingface.co ... "
if curl -sfI -m 20 "$HF/api/models/Comfy-Org/Wan_2.2_ComfyUI_Repackaged" >/dev/null 2>&1; then
  echo "OK"
else
  echo "NO"
  echo
  echo "This host cannot reach huggingface.co, so it cannot pull the weights itself."
  echo "Use the staging route - see 'IF THE BOX IS AIR-GAPPED' at the end of this script."
  exit 2
fi

get () {  # get <url> <subdir>
  local url="$1" dir="$MODELS/$2"
  mkdir -p "$dir"
  echo "  -> $2/$(basename "$url")"
  wget -nc -q --show-progress -P "$dir" "$url"
}

get "$W22/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors" diffusion_models  # 16.39 GB
get "$W21/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"   text_encoders     #  6.74 GB
get "$W22/vae/wan_2.1_vae.safetensors"                            vae               #  0.25 GB
get "$W22/audio_encoders/wav2vec2_large_english_fp16.safetensors" audio_encoders    #  0.63 GB

echo
echo "Done. Installed:"
for f in diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors \
         text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
         vae/wan_2.1_vae.safetensors \
         audio_encoders/wav2vec2_large_english_fp16.safetensors ; do
  if [ -f "$MODELS/$f" ]; then
    printf '  %-62s %s\n' "$f" "$(du -h "$MODELS/$f" | cut -f1)"
  else
    printf '  %-62s MISSING\n' "$f"
  fi
done
echo
echo "Now restart ComfyUI so it rescans models/, e.g.:"
echo "  curl -X POST http://127.0.0.1:8188/api/manager/reboot"

# ---------------------------------------------------------------------------
# IF THE BOX IS AIR-GAPPED (the preflight above failed)
#
# Download the four files somewhere with internet, then copy them across. The Windows box
# folder C:\Projects\ComfyUI\models already mirrors the server's model tree and looks like
# it exists for exactly this purpose.
#
# On the Windows box (PowerShell):
#
#   $W22 = "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files"
#   $W21 = "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files"
#   $M   = "C:\Projects\ComfyUI\models"
#   New-Item -ItemType Directory -Force "$M\audio_encoders" | Out-Null
#   iwr "$W22/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors" -OutFile "$M\diffusion_models\wan2.2_s2v_14B_fp8_scaled.safetensors"
#   iwr "$W21/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"   -OutFile "$M\text_encoders\umt5_xxl_fp8_e4m3fn_scaled.safetensors"
#   iwr "$W22/vae/wan_2.1_vae.safetensors"                            -OutFile "$M\vae\wan_2.1_vae.safetensors"
#   iwr "$W22/audio_encoders/wav2vec2_large_english_fp16.safetensors" -OutFile "$M\audio_encoders\wav2vec2_large_english_fp16.safetensors"
#
# then sync that tree into the box's bind mount, keeping the same subfolders:
#
#   rsync -av C:/Projects/ComfyUI/models/  <box>:/data/projects/shaheen/comfyui-data/models/
#
# Either way, restart ComfyUI afterwards.
# ---------------------------------------------------------------------------
