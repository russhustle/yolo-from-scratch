import torch
import yaml
from pathlib import Path
from ultralytics import YOLO
from ..nn import Yolov8n
import warnings

warnings.filterwarnings("ignore")


class WeightConverter:
    """Convert YOLOv8 official weights to custom model format"""

    def __init__(self):
        # Exact mapping between official YOLOv8 and custom model layer names
        self.exact_mappings = {
            # Backbone - exact matches
            "model.0": "backbone.conv_0",
            "model.1": "backbone.conv_1",
            "model.2": "backbone.c2f_2",
            "model.3": "backbone.conv_3",
            "model.4": "backbone.c2f_4",
            "model.5": "backbone.conv_5",
            "model.6": "backbone.c2f_6",
            "model.7": "backbone.conv_7",
            "model.8": "backbone.c2f_8",
            "model.9": "backbone.sppf",
            # Neck - corrected mappings based on structure analysis
            "model.12": "neck.c2f_1",  # model.12 -> neck.c2f_1
            "model.15": "neck.c2f_2",  # model.15 -> neck.c2f_2
            "model.16": "neck.cv_1",  # model.16 -> neck.cv_1
            "model.18": "neck.c2f_3",  # model.18 -> neck.c2f_3
            "model.19": "neck.cv_2",  # model.19 -> neck.cv_2
            "model.21": "neck.c2f_4",  # model.21 -> neck.c2f_4
        }

    def load_official_weights(self, weight_path):
        """Load official YOLOv8 weights"""
        try:
            # Try to load as YOLO model first
            model = YOLO(weight_path)
            return model.model.state_dict()
        except:
            # Fallback to direct torch load
            checkpoint = torch.load(weight_path, map_location="cpu")
            if "model" in checkpoint:
                return checkpoint["model"].state_dict()
            else:
                return checkpoint

    def copy_weights_exact(self, source_dict, target_dict, source_key, target_key):
        """Copy weights with exact key matching"""
        if source_key in source_dict and target_key not in target_dict:
            target_dict[target_key] = source_dict[source_key].clone()
            return True
        return False

    def convert_conv_weights(
        self, official_dict, custom_dict, official_prefix, custom_prefix
    ):
        """Convert Conv layer weights"""
        conversions = 0

        # Conv2d weights
        conversions += self.copy_weights_exact(
            official_dict,
            custom_dict,
            f"{official_prefix}.conv.weight",
            f"{custom_prefix}.conv.weight",
        )

        # BatchNorm weights
        conversions += self.copy_weights_exact(
            official_dict,
            custom_dict,
            f"{official_prefix}.bn.weight",
            f"{custom_prefix}.bn.weight",
        )
        conversions += self.copy_weights_exact(
            official_dict,
            custom_dict,
            f"{official_prefix}.bn.bias",
            f"{custom_prefix}.bn.bias",
        )
        conversions += self.copy_weights_exact(
            official_dict,
            custom_dict,
            f"{official_prefix}.bn.running_mean",
            f"{custom_prefix}.bn.running_mean",
        )
        conversions += self.copy_weights_exact(
            official_dict,
            custom_dict,
            f"{official_prefix}.bn.running_var",
            f"{custom_prefix}.bn.running_var",
        )
        conversions += self.copy_weights_exact(
            official_dict,
            custom_dict,
            f"{official_prefix}.bn.num_batches_tracked",
            f"{custom_prefix}.bn.num_batches_tracked",
        )

        return conversions

    def convert_c2f_weights(
        self, official_dict, custom_dict, official_prefix, custom_prefix
    ):
        """Convert C2f layer weights"""
        conversions = 0

        # Conv1 and Conv2 - both use cv1/cv2 now
        conversions += self.convert_conv_weights(
            official_dict, custom_dict, f"{official_prefix}.cv1", f"{custom_prefix}.cv1"
        )
        conversions += self.convert_conv_weights(
            official_dict, custom_dict, f"{official_prefix}.cv2", f"{custom_prefix}.cv2"
        )

        # Bottleneck layers - both use .m.{i}
        i = 0
        while f"{official_prefix}.m.{i}.cv1.conv.weight" in official_dict:
            # Convert bottleneck cv1 and cv2
            conversions += self.convert_conv_weights(
                official_dict,
                custom_dict,
                f"{official_prefix}.m.{i}.cv1",
                f"{custom_prefix}.m.{i}.cv1",
            )
            conversions += self.convert_conv_weights(
                official_dict,
                custom_dict,
                f"{official_prefix}.m.{i}.cv2",
                f"{custom_prefix}.m.{i}.cv2",
            )
            i += 1

        return conversions

    def convert_sppf_weights(
        self, official_dict, custom_dict, official_prefix, custom_prefix
    ):
        """Convert SPPF layer weights"""
        conversions = 0

        conversions += self.convert_conv_weights(
            official_dict, custom_dict, f"{official_prefix}.cv1", f"{custom_prefix}.cv1"
        )
        conversions += self.convert_conv_weights(
            official_dict, custom_dict, f"{official_prefix}.cv2", f"{custom_prefix}.cv2"
        )

        return conversions

    def convert_head_weights(self, official_dict, custom_dict, version="n"):
        """Convert detection head weights"""
        conversions = 0

        # Convert box prediction layers (cv2 in official -> box in custom)
        for i in range(3):  # 3 detection scales
            official_prefix = f"model.22.cv2.{i}"
            custom_prefix = f"head.box.{i}"

            # Convert each layer in the sequence
            for j in range(3):  # 3 layers per detection head
                if j < 2:  # First two are Conv layers (with bn)
                    conversions += self.convert_conv_weights(
                        official_dict,
                        custom_dict,
                        f"{official_prefix}.{j}",
                        f"{custom_prefix}.{j}",
                    )
                else:  # Last one is plain Conv2d (no bn)
                    conversions += self.copy_weights_exact(
                        official_dict,
                        custom_dict,
                        f"{official_prefix}.{j}.weight",
                        f"{custom_prefix}.{j}.weight",
                    )
                    conversions += self.copy_weights_exact(
                        official_dict,
                        custom_dict,
                        f"{official_prefix}.{j}.bias",
                        f"{custom_prefix}.{j}.bias",
                    )

        # Convert classification prediction layers (cv3 in official -> cls in custom)
        for i in range(3):  # 3 detection scales
            official_prefix = f"model.22.cv3.{i}"
            custom_prefix = f"head.cls.{i}"

            # Convert each layer in the sequence
            for j in range(3):  # 3 layers per detection head
                if j < 2:  # First two are Conv layers (with bn)
                    conversions += self.convert_conv_weights(
                        official_dict,
                        custom_dict,
                        f"{official_prefix}.{j}",
                        f"{custom_prefix}.{j}",
                    )
                else:  # Last one is plain Conv2d (no bn)
                    conversions += self.copy_weights_exact(
                        official_dict,
                        custom_dict,
                        f"{official_prefix}.{j}.weight",
                        f"{custom_prefix}.{j}.weight",
                    )
                    conversions += self.copy_weights_exact(
                        official_dict,
                        custom_dict,
                        f"{official_prefix}.{j}.bias",
                        f"{custom_prefix}.{j}.bias",
                    )

        # Convert DFL weights
        conversions += self.copy_weights_exact(
            official_dict,
            custom_dict,
            "model.22.dfl.conv.weight",
            "head.dfl.conv.weight",
        )

        return conversions

    def convert_weights(self, official_weight_path, version="n", save_path=None):
        """Main conversion function"""
        print(f"Loading official YOLOv8{version} weights from {official_weight_path}")

        # Load official weights
        official_dict = self.load_official_weights(official_weight_path)

        # Create custom model to get the target structure
        custom_model = Yolov8n(version=version)
        custom_dict = {}

        print("Converting weights...")
        total_conversions = 0

        # Convert backbone and neck weights
        for official_key, custom_key in self.exact_mappings.items():

            # Convert based on layer type
            if "c2f" in custom_key:
                conversions = self.convert_c2f_weights(
                    official_dict, custom_dict, official_key, custom_key
                )
                print(f"  {official_key} -> {custom_key}: {conversions} weights")
                total_conversions += conversions
            elif "sppf" in custom_key:
                conversions = self.convert_sppf_weights(
                    official_dict, custom_dict, official_key, custom_key
                )
                print(f"  {official_key} -> {custom_key}: {conversions} weights")
                total_conversions += conversions
            elif "conv" in custom_key or "cv" in custom_key:
                conversions = self.convert_conv_weights(
                    official_dict, custom_dict, official_key, custom_key
                )
                print(f"  {official_key} -> {custom_key}: {conversions} weights")
                total_conversions += conversions

        # Convert head weights (special handling needed)
        print("  Converting detection head...")
        head_conversions = self.convert_head_weights(
            official_dict, custom_dict, version
        )
        print(f"  Head conversion: {head_conversions} weights")
        total_conversions += head_conversions

        print(f"Total weights converted: {total_conversions}")

        # Load converted weights into custom model
        missing_keys, unexpected_keys = custom_model.load_state_dict(
            custom_dict, strict=False
        )

        if missing_keys:
            print(f"Missing keys (will use random initialization): {len(missing_keys)}")
            # Only show a few examples to avoid spam
            for key in missing_keys[:5]:
                print(f"  - {key}")
            if len(missing_keys) > 5:
                print(f"  ... and {len(missing_keys) - 5} more")

        if unexpected_keys:
            print(f"Unexpected keys (ignored): {len(unexpected_keys)}")

        # Save converted weights
        if save_path:
            torch.save(custom_model.state_dict(), save_path)
            print(f"Converted weights saved to {save_path}")

        return custom_model.state_dict()


def create_compatible_weights(official_weight_path="yolov8n.pt", version="n"):
    """Convenience function to create compatible weights"""
    converter = WeightConverter()

    # Create output filename
    output_path = f"yolov8{version}_custom.pth"

    try:
        converted_weights = converter.convert_weights(
            official_weight_path=official_weight_path,
            version=version,
            save_path=output_path,
        )

        print(f"\n✅ Successfully created compatible weights: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ Error converting weights: {e}")
        print("Falling back to random initialization")
        return None


if __name__ == "__main__":
    # Example usage
    compatible_weights = create_compatible_weights("yolov8n.pt", "n")

    if compatible_weights:
        # Test loading the converted weights
        print("\nTesting converted weights...")
        model = Yolov8n(version="n")
        checkpoint = torch.load(compatible_weights, map_location="cpu")
        model.load_state_dict(checkpoint)
        print("✅ Weights loaded successfully!")

        # Test inference
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ Model inference test passed! Output shape: {output.shape}")
