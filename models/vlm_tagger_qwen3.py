"""
VLM Tagger Module for Facet - Qwen3-VL 2B

Uses Qwen3-VL-2B vision-language model to generate semantic tags for images.
Provides higher quality tagging than CLIP similarity matching, at the cost of
more VRAM (~4GB) and slower inference. Designed for the 8gb profile.
"""

from typing import List, Optional, Dict, Any
import PIL.Image

# Lazy imports
torch = None
Qwen3VLForConditionalGeneration = None
AutoProcessor = None


def _ensure_imports():
    """Lazy load heavy dependencies."""
    global torch, Qwen3VLForConditionalGeneration, AutoProcessor
    if torch is None:
        import torch as _torch
        torch = _torch
    if Qwen3VLForConditionalGeneration is None:
        from transformers import Qwen3VLForConditionalGeneration as _Model
        from transformers import AutoProcessor as _Processor
        Qwen3VLForConditionalGeneration = _Model
        AutoProcessor = _Processor


class Qwen3VLTagger:
    """
    Generate semantic tags using Qwen3-VL-2B vision-language model.

    This tagger provides higher quality semantic understanding compared to
    CLIP embedding similarity, while fitting in the 8gb VRAM profile (~4GB).
    """

    # Same proven prompt as VLMTagger
    PROMPT = """Analyze this photo and provide semantic tags.

Return ONLY a comma-separated list of relevant tags from these categories:
- Scene: landscape, portrait, street, architecture, macro, wildlife, aerial, concert, night, astro, food, sports, travel, fashion, urban
- Subject: person, animal, building, nature, water, sky, mountain, beach, forest, flower, vehicle
- Style: black_and_white, silhouette, long_exposure, dramatic, minimalist, vintage, cinematic, abstract
- Mood: dramatic, peaceful, energetic, intimate, moody

Tags:"""

    def __init__(self, model_config: Dict[str, Any], scoring_config=None):
        """
        Initialize the Qwen3-VL tagger.

        Args:
            model_config: Dict with model settings (model_path, torch_dtype, etc.)
            scoring_config: Optional ScoringConfig instance for loading vocabulary
        """
        self.model_config = model_config
        self.scoring_config = scoring_config
        self.model = None
        self.processor = None

        # Load vocabulary from config for optional filtering
        self.valid_tags = set()
        if scoring_config:
            vocab = scoring_config.get_tag_vocabulary()
            self.valid_tags = set(vocab.keys())

    def load(self):
        """Load the model (deferred until first use)."""
        if self.model is not None:
            return

        _ensure_imports()

        model_path = self.model_config.get('model_path', 'Qwen/Qwen3-VL-2B-Instruct')
        dtype_str = self.model_config.get('torch_dtype', 'bfloat16')
        torch_dtype = getattr(torch, dtype_str, torch.bfloat16)

        print(f"Loading Qwen3-VL-2B from {model_path}...")

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        # Limit image resolution to control VRAM usage during inference.
        # Without this, full-res photos (6000x4000+) create massive vision token
        # sequences that spike VRAM far beyond the model's weight footprint.
        # 512 * 28 * 28 = 401,408 pixels (~633x633 effective resolution)
        max_pixels = self.model_config.get('max_pixels', 512 * 28 * 28)

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            max_pixels=max_pixels,
        )

        print("Qwen3-VL-2B loaded successfully")

    def unload(self):
        """Free VRAM by unloading the model."""
        if self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        _ensure_imports()
        torch.cuda.empty_cache()
        print("Qwen3-VL tagger unloaded")

    def tag_image(self, image: PIL.Image.Image, max_tags: int = 5) -> List[str]:
        """
        Generate tags for a single image.

        Args:
            image: PIL Image to tag
            max_tags: Maximum number of tags to return (default: 5)

        Returns:
            List of tag names
        """
        if self.model is None:
            self.load()

        _ensure_imports()

        max_new_tokens = self.model_config.get('max_new_tokens', 100)

        # Prepare input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.PROMPT},
                ],
            }
        ]

        # Qwen3-VL: apply_chat_template with tokenize=True combines template + tokenization
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Remove token_type_ids if present (not used by Qwen3-VL)
        inputs.pop("token_type_ids", None)

        # Move to device
        inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v
                  for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Trim input tokens from output
        output_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        output_text = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]

        # Parse tags from output
        tags = self._parse_tags(output_text, max_tags)
        return tags

    def tag_batch(self, images: List[PIL.Image.Image], max_tags: int = 5) -> List[List[str]]:
        """
        Generate tags for a batch of images.

        Note: VLM batch processing processes images sequentially.

        Args:
            images: List of PIL Images to tag
            max_tags: Maximum number of tags per image

        Returns:
            List of tag lists, one per image
        """
        return [self.tag_image(img, max_tags) for img in images]

    def _parse_tags(self, text: str, max_tags: int) -> List[str]:
        """
        Parse and validate tags from model output.

        Args:
            text: Raw output text from model
            max_tags: Maximum number of tags to return

        Returns:
            List of cleaned tag names
        """
        # Clean up the text
        text = text.strip()

        # Remove common prefixes the model might add
        for prefix in ['Tags:', 'tags:', 'Here are the tags:', 'The tags are:']:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        # Split by comma and clean each tag
        raw_tags = [t.strip().lower() for t in text.split(',')]

        # Remove empty tags and clean up
        tags = []
        for tag in raw_tags:
            # Remove any numbering or bullets
            tag = tag.lstrip('0123456789.-) ')
            # Remove quotes
            tag = tag.strip('"\'')
            # Replace spaces with underscores for consistency
            tag = tag.replace(' ', '_')

            if tag and len(tag) > 1:
                # Optionally filter to known vocabulary
                if self.valid_tags and tag not in self.valid_tags:
                    # Try to find a close match
                    for valid_tag in self.valid_tags:
                        if tag in valid_tag or valid_tag in tag:
                            tag = valid_tag
                            break

                if tag not in tags:  # Avoid duplicates
                    tags.append(tag)

        return tags[:max_tags]

    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self.model is not None
