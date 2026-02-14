"""
VLM Tagger Module for Facet

Uses Qwen2.5-VL-7B vision-language model to generate semantic tags for images.
Provides higher quality tagging than CLIP similarity matching, at the cost of
more VRAM (~16GB) and slower inference.
"""

from typing import List, Optional, Dict, Any
import PIL.Image

# Lazy imports
torch = None
Qwen2_5_VLForConditionalGeneration = None
AutoProcessor = None


def _ensure_imports():
    """Lazy load heavy dependencies."""
    global torch, Qwen2_5_VLForConditionalGeneration, AutoProcessor
    if torch is None:
        import torch as _torch
        torch = _torch
    if Qwen2_5_VLForConditionalGeneration is None:
        from transformers import Qwen2_5_VLForConditionalGeneration as _Model
        from transformers import AutoProcessor as _Processor
        Qwen2_5_VLForConditionalGeneration = _Model
        AutoProcessor = _Processor


class VLMTagger:
    """
    Generate semantic tags using Qwen2.5-VL-7B vision-language model.

    This tagger provides higher quality semantic understanding compared to
    CLIP embedding similarity, but requires more VRAM (~16GB) and is slower.
    """

    # Default prompt for tag generation
    PROMPT = """Analyze this photo and provide semantic tags.

Return ONLY a comma-separated list of relevant tags from these categories:
- Scene: landscape, portrait, street, architecture, macro, wildlife, aerial, concert, night, astro, food, sports, travel, fashion, urban
- Subject: person, animal, building, nature, water, sky, mountain, beach, forest, flower, vehicle
- Style: black_and_white, silhouette, long_exposure, dramatic, minimalist, vintage, cinematic, abstract
- Mood: dramatic, peaceful, energetic, intimate, moody

Tags:"""

    def __init__(self, model_config: Dict[str, Any], scoring_config=None):
        """
        Initialize the VLM tagger.

        Args:
            model_config: Dict with model settings (model_path, torch_dtype, etc.)
            scoring_config: Optional ScoringConfig instance for loading vocabulary
        """
        self.model_config = model_config
        self.scoring_config = scoring_config
        self.model = None
        self.processor = None
        self.device = 'cuda'

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

        model_path = self.model_config.get('model_path', 'Qwen/Qwen2.5-VL-7B-Instruct')
        dtype_str = self.model_config.get('torch_dtype', 'bfloat16')
        torch_dtype = getattr(torch, dtype_str, torch.bfloat16)

        print(f"Loading Qwen2.5-VL-7B from {model_path}...")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        print("Qwen2.5-VL-7B loaded successfully")

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
        print("VLM tagger unloaded")

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

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )

        # Move to device
        inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v
                  for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
            )

        # Decode output
        generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # Parse tags from output
        tags = self._parse_tags(output_text, max_tags)
        return tags

    def tag_batch(self, images: List[PIL.Image.Image], max_tags: int = 5) -> List[List[str]]:
        """
        Generate tags for a batch of images.

        Note: VLM batch processing is less efficient than CLIP, so this
        processes images sequentially. For better throughput, consider
        using RAM++ or CLIP taggers.

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
                    else:
                        # Keep the tag even if not in vocabulary
                        pass

                if tag not in tags:  # Avoid duplicates
                    tags.append(tag)

        return tags[:max_tags]

    def get_tags_with_scores(self, image: PIL.Image.Image, threshold: float = 0.0) -> Dict[str, float]:
        """
        Get tags with confidence scores.

        Note: VLM taggers don't provide natural confidence scores like
        CLIP similarity. This returns 1.0 for all tags.

        Args:
            image: PIL Image to tag
            threshold: Not used for VLM (included for API compatibility)

        Returns:
            Dict mapping tag names to confidence scores (all 1.0)
        """
        tags = self.tag_image(image)
        return {tag: 1.0 for tag in tags}

    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self.model is not None
