$ErrorActionPreference = 'Continue'
$py = 'C:\Users\admin\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe'
$models = 'C:\Users\admin\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\models'
$tmp = 'C:\Users\admin\Desktop\Projects\kidsvid2\Claudetempfiles'
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$log = Join-Path $tmp 'dl_models.log'

"DL start $(Get-Date -Format o)" | Set-Content $log
function D([string]$repo, [string[]]$include, [string]$dest) {
  New-Item -ItemType Directory -Force -Path $dest | Out-Null
  $args = @('hf', 'download', $repo, '--local-dir', $dest)
  foreach ($i in $include) { $args += '--include'; $args += $i }
  "== $repo $($args -join ' ') => $dest" | Add-Content $log
  & $py $args *>> $log
  $code = $LASTEXITCODE
  "   exit=$code" | Add-Content $log
  return $code
}

$rc = 0
# 1) Ref2VA pruned Q4_K (11,4 GB) -> diffusion_models (UnetLoaderGGUF liest dort auch)
$c = D 'unsloth/MiniMax-H3-GGUF' @('minimax_h3_ref2va_pruned-Q4_K.gguf') (Join-Path $models 'diffusion_models')
if ($c -ne 0) { $rc = $c }
# 2) Qwen3-VL-32B minimax TE Q2_K_M (13,1 GB) -> text_encoders
$c = D 'unsloth/MiniMax-H3-GGUF' @('qwen3vl_32b_minimax_h3-Q2_K_M.gguf') (Join-Path $models 'text_encoders')
if ($c -ne 0) { $rc = $c }
# 3) Video + Audio VAE (5,2 + 0,6 GB) -> vae
$c = D 'Comfy-Org/MiniMax-H3' @('vae/*') (Join-Path $models 'vae')
if ($c -ne 0) { $rc = $c }
# 4) Optional: 4-step Turbo LoRA (small) -> loras
$c = D 'Comfy-Org/MiniMax-H3' @('loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors') (Join-Path $models 'loras')
if ($c -ne 0) { $rc = $c }

"DL ENDE $(Get-Date -Format o) rc=$rc" | Add-Content $log
exit $rc