from omegaconf import OmegaConf  # Importing OmegaConf for configuration management

class BaseProcessor:
    def __init__(self):
        self.transform = lambda x: x  # Initializing a default transformation function
        return

    def __call__(self, item):
        """
        Method to apply the transformation function to an item.

        Args:
            item: Input item to be transformed.

        Returns:
            Transformed item.
        """
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        """
        Class method to create an instance of BaseProcessor from a configuration.

        Args:
            cfg: Configuration object (optional).

        Returns:
            Instance of BaseProcessor.
        """
        return cls()

    def build(self, **kwargs):
        """
        Method to build a BaseProcessor instance from keyword arguments.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            Instance of BaseProcessor.
        """
        cfg = OmegaConf.create(kwargs)  # Creating a configuration object from keyword arguments

        return self.from_config(cfg)  # Creating an instance of BaseProcessor using the configuration
