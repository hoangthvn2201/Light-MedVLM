import torch 
from model import LightMedVLM
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="huyhoangt2201/lightmedvlm-iu-mlp-20epochs-tune-vision-encoder",
    local_dir="lightmedvlm"
)

def main():
    ckpt_file="lightmedvlm/checkpoints/checkpoint_epoch15_step4128_bleu0.096789_cider0.022736_renamed.pth"   # Absoluate path to .pth file
    args = {
        "vision_model":"microsoft/swin-base-patch4-window7-224",
        "llm_model":"Qwen/Qwen3-0.6B"
    }
    model = LightMedVLM.load_from_checkpoint(ckpt_file,strict=False, **args)

    image_paths = ['iu_xray/images/CXR1000_IM-0003/0.png', 'iu_xray/images/CXR1000_IM-0003/1.png']
    print(f"Generating report for: {image_path}\n")
    report = model.inference(image_paths)
    print("="*60)
    print("GENERATED REPORT:")
    print("="*60)
    print(report)
    print("="*60)

if __name__ == '__main__':
    main()