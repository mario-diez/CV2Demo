from __future__ import annotations
import argparse
from pathlib import Path
import textwrap
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF
try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC


DEFAULT_CHECKPOINTS = (
    "baseline_pix2pix_cityscapes.pt",
    "augmented_pix2pix_cityscapes.pt",
    "label_smoothing_pix2pix_cityscapes.pt",
    "pix2pixHD.pt",
)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False)
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False)
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down6 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1),
            nn.ReLU(),
        )

        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up4 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False)
        self.up5 = Block(features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False)
        self.up6 = Block(features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False)
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)

        bn = self.bottleneck(d7)

        u1 = self.up1(bn)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))

        return self.final_up(torch.cat([u7, d1], dim=1))


class MergeDownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MergeUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class MergeGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.down1 = MergeDownsampleBlock(in_channels, 64, normalize=False)
        self.down2 = MergeDownsampleBlock(64, 128)
        self.down3 = MergeDownsampleBlock(128, 256)
        self.down4 = MergeDownsampleBlock(256, 512)
        self.down5 = MergeDownsampleBlock(512, 512)
        self.down6 = MergeDownsampleBlock(512, 512)
        self.down7 = MergeDownsampleBlock(512, 512)
        self.down8 = MergeDownsampleBlock(512, 512, normalize=False)

        self.up1 = MergeUpsampleBlock(512, 512, dropout=True)
        self.up2 = MergeUpsampleBlock(1024, 512, dropout=True)
        self.up3 = MergeUpsampleBlock(1024, 512, dropout=True)
        self.up4 = MergeUpsampleBlock(1024, 512)
        self.up5 = MergeUpsampleBlock(1024, 256)
        self.up6 = MergeUpsampleBlock(512, 128)
        self.up7 = MergeUpsampleBlock(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))

        return self.final(torch.cat([u7, d1], dim=1))

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.conv_block(x)

class GlobalGenerator(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, ngf=64, n_blocks=9):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_nc, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True),
        ]

        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * 4)]

        model += [
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Pix2PixHDGenerator(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, ngf=64, n_blocks_global=9, n_blocks_local=3):
        super().__init__()
        self.global_gen = GlobalGenerator(in_nc, out_nc, ngf, n_blocks_global)
        self.local_down = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_nc, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
        )
        self.local_blocks = nn.Sequential(*(ResnetBlock(ngf * 2) for _ in range(n_blocks_local)))
        self.local_up = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        )
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        x_low = self.downsample(x)
        g_out = self.global_gen(x_low)
        g_out_upsampled = F.interpolate(g_out, size=x.shape[2:], mode="bilinear", align_corners=False)
        l_features = self.local_down(x)
        l_features = self.local_blocks(l_features)
        out = self.local_up(l_features)
        return (out + g_out_upsampled) / 2.0


def get_device(device_name: str | None = None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("generator", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def detect_generator_architecture(state_dict) -> str:
    if not isinstance(state_dict, dict):
        return "pix2pix"
    keys = list(state_dict.keys())
    if any(key.startswith(("global_gen.", "local_down.", "local_blocks.", "local_up.")) for key in keys):
        return "pix2pixhd"
    if any(key.startswith(("down1.model.", "down8.model.", "final.0.")) for key in keys):
        return "merge_pix2pix"
    if any(key.startswith(("initial_down.", "down1.conv.", "up1.conv.", "final_up.0.")) for key in keys):
        return "pix2pix"
    return "pix2pix"


def build_model_identifier(checkpoint_path: str | Path, architecture: str) -> str:
    checkpoint_stem = Path(checkpoint_path).stem
    return f"{architecture}_{checkpoint_stem}"


def build_output_path(image_path: str | Path, suffix: str, output_path: str | Path | None = None) -> Path:
    image_path = Path(image_path)
    default_name = f"{image_path.stem}_{suffix}.png"
    if output_path is None:
        resolved_path = Path("results") / default_name
    else:
        resolved_path = Path(output_path)
        if resolved_path.exists() and resolved_path.is_dir():
            resolved_path = resolved_path / default_name
        elif resolved_path.suffix == "":
            resolved_path.mkdir(parents=True, exist_ok=True)
            resolved_path = resolved_path / default_name

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return resolved_path


def load_generator(checkpoint_path: str | Path, device: torch.device) -> tuple[nn.Module, str]:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    state_dict = unwrap_state_dict(checkpoint)
    architecture = detect_generator_architecture(state_dict)
    if architecture == "pix2pixhd":
        model = Pix2PixHDGenerator().to(device)
    elif architecture == "merge_pix2pix":
        model = MergeGenerator().to(device)
    else:
        model = Generator().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, architecture


def load_and_prepare_image(image_path: str | Path, paired_image: bool = False):
    raw_image = Image.open(image_path).convert("RGB")
    if paired_image and raw_image.width >= raw_image.height * 2:
        half_width = raw_image.width // 2
        raw_image = raw_image.crop((half_width, 0, raw_image.width, raw_image.height))

    return raw_image.resize((256, 256), RESAMPLE_BICUBIC)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    return (TF.to_tensor(image) * 2.0) - 1.0


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(-1.0, 1.0)
    tensor = (tensor + 1.0) / 2.0
    return TF.to_pil_image(tensor)


def predict_image(generator: nn.Module, model_image: Image.Image, device: torch.device) -> Image.Image:
    input_tensor = image_to_tensor(model_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor = generator(input_tensor)[0]
    return tensor_to_image(output_tensor)


def compose_side_by_side(left_image: Image.Image, right_image: Image.Image) -> Image.Image:
    height = max(left_image.height, right_image.height)
    canvas = Image.new("RGB", (left_image.width + right_image.width, height), color=(255, 255, 255))
    canvas.paste(left_image, (0, 0))
    canvas.paste(right_image, (left_image.width, 0))
    return canvas


def add_caption(image: Image.Image, caption: str) -> Image.Image:
    font = ImageFont.load_default()
    wrapped_caption = "\n".join(textwrap.wrap(caption, width=22)) or caption
    probe = Image.new("RGB", (1, 1), color=(255, 255, 255))
    probe_draw = ImageDraw.Draw(probe)
    text_bbox = probe_draw.multiline_textbbox((0, 0), wrapped_caption, font=font, spacing=2)
    text_height = int(text_bbox[3] - text_bbox[1])
    header_height = text_height + 16

    canvas = Image.new("RGB", (image.width, image.height + header_height), color=(255, 255, 255))
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.multiline_text((8, 8), wrapped_caption, fill=(0, 0, 0), font=font, spacing=2)
    return canvas


def compose_horizontal_strip(images: list[Image.Image]) -> Image.Image:
    if not images:
        raise ValueError("At least one image is required to compose a comparison strip.")

    height = max(image.height for image in images)
    width = sum(image.width for image in images)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))

    current_x = 0
    for image in images:
        canvas.paste(image, (current_x, 0))
        current_x += image.width

    return canvas


def run_checkpoint_demo(
    image_path: str | Path,
    checkpoint_path: str | Path,
    model_image: Image.Image,
    device: torch.device,
    output_path: str | Path | None = None,
):
    generator, architecture = load_generator(checkpoint_path, device)
    generated_image = predict_image(generator, model_image, device)
    comparison_image = compose_side_by_side(model_image, generated_image)
    model_identifier = build_model_identifier(checkpoint_path, architecture)

    output_path = build_output_path(image_path=image_path, suffix=f"{model_identifier}_comparison", output_path=output_path)
    comparison_image.save(output_path)

    return {
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "architecture": architecture,
        "model_identifier": model_identifier,
        "model_image": model_image,
        "generated_image": generated_image,
        "comparison_image": comparison_image,
        "output_path": str(output_path),
    }


def run_demo(
    image_path: str | Path,
    checkpoint_paths: tuple[str | Path, ...],
    output_path: str | Path | None = None,
    device_name: str | None = None,
    paired_image: bool = False,
):
    device = get_device(device_name)
    model_image = load_and_prepare_image(image_path, paired_image=paired_image)
    model_runs = []

    for checkpoint_path in checkpoint_paths:
        result = run_checkpoint_demo(
            image_path=image_path,
            checkpoint_path=checkpoint_path,
            model_image=model_image,
            device=device,
        )
        model_runs.append(result)

    if len(model_runs) == 1:
        return model_runs[0]

    comparison_panels = [add_caption(model_image, "Input")]
    comparison_panels.extend(add_caption(run["generated_image"], run["model_identifier"]) for run in model_runs)

    comparison_image = compose_horizontal_strip(comparison_panels)
    output_path = build_output_path(image_path=image_path, suffix="all_models_comparison", output_path=output_path)
    comparison_image.save(output_path)

    return {
        "device": str(device),
        "model_runs": model_runs,
        "comparison_image": comparison_image,
        "output_path": str(output_path),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Run Pix2Pix inference on a single image.")
    parser.add_argument("image", type=str, help="Input image path")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every local checkpoint and save one combined comparison image",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="baseline_pix2pix_cityscapes.pt",
        help="Path to a trained Pix2Pix checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Where to save the generated image or comparison image; defaults to results/",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force a device name such as cpu, cuda, or mps",
    )
    parser.add_argument(
        "--paired-image",
        action="store_true",
        help="Treat the input as a Cityscapes paired image and use its right half as conditioning input",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.all:
        result = run_demo(
            image_path=args.image,
            checkpoint_paths=DEFAULT_CHECKPOINTS,
            output_path=args.output,
            device_name=args.device,
            paired_image=args.paired_image,
        )
        print(f"Device: {result['device']}")
        print(f"Combined comparison: {result['output_path']}")
        for model_run in result["model_runs"]:
            print(f"Model ID: {model_run['model_identifier']} -> {model_run['output_path']}")
    else:
        result = run_demo(
            image_path=args.image,
            checkpoint_paths=(args.checkpoint,),
            output_path=args.output,
            device_name=args.device,
            paired_image=args.paired_image,
        )

        print(f"Device: {result['device']}")
        print(f"Checkpoint: {result['checkpoint_path']}")
        print(f"Model ID: {result['model_identifier']}")
        print(f"Generated image: {result['output_path']}")


if __name__ == "__main__":
    main()