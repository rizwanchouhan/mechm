import re  # Importing regular expression module for text processing

from gpt.common.registry import registry  # Importing registry for registering processors
from gpt.CPU.base_processor import BaseProcessor  # Importing base processor class
from gpt.CPU.randaugment import RandomAugment  # Importing RandomAugment for image augmentation
from omegaconf import OmegaConf  # Importing OmegaConf for configuration management
from torchvision import transforms  # Importing torchvision transforms for image processing
from torchvision.transforms.functional import InterpolationMode  # Importing interpolation mode for image resizing

class BlipImageBaseProcessor(BaseProcessor):
    """
    Base processor for Blip image processing.

    Args:
        mean (tuple): Mean values for normalization (default: (0.48145466, 0.4578275, 0.40821073)).
        std (tuple): Standard deviation values for normalization (default: (0.26862954, 0.26130258, 0.27577711)).
    """
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)


@registry.register_processor("blip_caption")
class BlipCaptionProcessor(BaseProcessor):
    """
    Processor for Blip caption processing.

    Args:
        prompt (str): Prompt text to prepend to captions.
        max_words (int): Maximum number of words in a caption (default: 50).
    """
    def __init__(self, prompt="", max_words=50):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        """
        Method to process a caption.

        Args:
            caption (str): Caption text.

        Returns:
            str: Processed caption.
        """
        caption = self.prompt + self.pre_caption(caption)

        return caption

    @classmethod
    def from_config(cls, cfg=None):
        """
        Class method to create an instance of BlipCaptionProcessor from a configuration.

        Args:
            cfg (OmegaConf): Configuration object (optional).

        Returns:
            BlipCaptionProcessor: Instance of BlipCaptionProcessor.
        """
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)

        return cls(prompt=prompt, max_words=max_words)

    def pre_caption(self, caption):
        """
        Method to preprocess a caption.

        Args:
            caption (str): Caption text.

        Returns:
            str: Preprocessed caption.
        """
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption


@registry.register_processor("blip2_image_train")
class Blip2ImageTrainProcessor(BlipImageBaseProcessor):
    """
    Processor for Blip 2 image training processing.

    Args:
        image_size (int): Size of the images (default: 224).
        mean (tuple): Mean values for normalization.
        std (tuple): Standard deviation values for normalization.
        min_scale (float): Minimum scale for image resizing (default: 0.5).
        max_scale (float): Maximum scale for image resizing (default: 1.0).
    """
    def __init__(self, image_size=224, mean=None, std=None, min_scale=0.5, max_scale=1.0):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size,image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        """
        Method to process an image.

        Args:
            item: Input image.

        Returns:
            torch.Tensor: Processed image.
        """
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        """
        Class method to create an instance of Blip2ImageTrainProcessor from a configuration.

        Args:
            cfg (OmegaConf): Configuration object (optional).

        Returns:
            Blip2ImageTrainProcessor: Instance of Blip2ImageTrainProcessor.
        """
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )


@registry.register_processor("blip2_image_eval")
class Blip2ImageEvalProcessor(BlipImageBaseProcessor):
    """
    Processor for Blip 2 image evaluation processing.

    Args:
        image_size (int): Size of the images (default: 224).
        mean (tuple): Mean values for normalization.
        std (tuple): Standard deviation values for normalization.
    """
    def __init__(self, image_size=224, mean=None, std=None):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        """
        Method to process an image.

        Args:
            item: Input image.

        Returns:
            torch.Tensor: Processed image.
        """
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        """
        Class method to create an instance of Blip2ImageEvalProcessor from a configuration.

        Args:
            cfg (OmegaConf): Configuration object (optional).

        Returns:
            Blip2ImageEvalProcessor: Instance of Blip2ImageEvalProcessor.
        """
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        return cls(image_size=image_size, mean=mean, std=std)
