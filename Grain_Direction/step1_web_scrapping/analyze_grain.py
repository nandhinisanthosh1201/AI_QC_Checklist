import torch

def analyze_grain_direction(image_path, model, processor):
    """
    Analyzes a verified material image to determine if it has a Horizontal or Vertical grain.
    """
    from qwen_vl_utils import process_vision_info
    
    prompt = "Look at this material swatch carefully. Does it have a directional wood grain or pattern? If yes, is the grain direction Horizontal or Vertical? Reply in the format: 'Grain: Yes | Direction: Horizontal/Vertical'. If no grain is visible, reply 'Grain: No'."
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=50)
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    answer = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
    
    return answer
