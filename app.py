import os
import cv2
import torch
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import argparse

from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return None
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0,0,1,0.4), thickness=1) 

    img = (img * 255).astype(np.uint8)
    return img

def load_model(model_name):
    if model_name == "sam2_hiera_large":
        model = build_sam2(
            config_file="sam2_hiera_l.yaml",
            ckpt_path="segment-anything-2/checkpoints/sam2_hiera_large.pt",
            device ='cuda',
            apply_postprocessing=False
        )
    elif model_name == "sam2_hiera_small":
        model = build_sam2(
            config_file="sam2_hiera_s.yaml",
            ckpt_path="segment-anything-2/checkpoints/sam2_hiera_small.pt",
            device ='cuda', 
            apply_postprocessing=False
        )
    elif model_name == "sam2_hiera_tiny":
        model = build_sam2(
            config_file="sam2_hiera_t.yaml",
            ckpt_path="segment-anything-2/checkpoints/sam2_hiera_tiny.pt",
            device ='cuda', 
            apply_postprocessing=False
        )
    elif model_name == "sam2_hiera_base_plus":
        model = build_sam2(
            config_file="sam2_hiera_b+.yaml",
            ckpt_path="segment-anything-2/checkpoints/sam2_hiera_base_plus.pt",
            device ='cuda', 
            apply_postprocessing=False
        )
    else:
        raise ValueError(f"Model {model_name} not found")
    return model

def segment_anything(mode, input_image, input_video, model_name):
    output_video = None
    if (mode == "SAM2VideoPredictor"):
        return None, output_video
    else:
        model = load_model(model_name)
        image_array = input_image["layers"][0]
        image_pil = Image.fromarray(image_array.astype('uint8'), 'RGB')
        image = np.array(image_pil.convert("RGB"))
        # is_bf16 = image.dtype == torch.bfloat16
        # print("Is image of dtype bfloat16?", is_bf16)
        if (mode == "SAM2AutomaticMaskGenerator"):
            mask_generator = SAM2AutomaticMaskGenerator(model)
            print("model load successfully!")
            masks = mask_generator.generate(image)
            print("masks[0].keys()", masks[0].keys())
            output_image = image
            return output_image, output_video
        else:
            predictor = SAM2ImagePredictor(model)
            predictor.set_image(image)
            masks = mask_generator.generate(image)
            output_image = show_anns(masks)
            return output_image, output_video

def webui(port):
    with gr.Blocks() as Interface:
        with gr.Row():
            with gr.Column():
                with gr.Column():
                    mode_input = gr.Dropdown(
                        choices=["SAM2AutomaticMaskGenerator", "SAM2ImagePredictor", "SAM2VideoPredictor"],
                        label="Mode"
                    )
                    image_input = gr.ImageEditor(image_mode="RGB", label="Input Image", eraser="False", brush=gr.Brush(colors=["#000000"], color_mode="fixed"))
                    video_input = gr.Video(label="Input Video")
                
                    model_input = gr.Dropdown(
                        choices=["sam2_hiera_large", "sam2_hiera_small", "sam2_hiera_tiny", "sam2_hiera_base_plus"],
                        label="Model"
                    )
            with gr.Column():
                with gr.Column():
                    image_outpput = gr.Image(label="Output Image")
                    video_output = gr.Video(label="Output Video")
                    process_masks_button = gr.Button("Processed Masks")

        if mode_input.value == "SAM2VideoPredictor":
            # Preprocess video to extract the first frame
            video = video_input.value
            frame_names = [
                p for p in os.listdir(video)
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
            ]
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
            frame_idx = 0
            image_input.value = Image.open(os.path.join(video, frame_names[frame_idx]))
        
        process_masks_button.click(
            fn=segment_anything,
            inputs=[mode_input, image_input, video_input, model_input],
            outputs=[image_outpput, video_output],
            queue=True,
            api_name=False,
            show_progress=True
        )

    Interface.launch(server_port=port)

def main():
    argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='SAM2 WebUI')
    parser.add_argument('--port', type=int, default=6006, help='Port number')
    parser.parse_args()
    args = parser.parse_args()
    webui(args.port)

if __name__ == '__main__':
    main()
