import os
import mimetypes
import json
import shutil
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper
from comfyui_enums import SAMPLERS, SCHEDULERS

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")
api_json_file = "workflow_api.json"

# Force HF offline
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # Give a list of weights filenames to download during setup
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[],
        )

    def filename_with_extension(self, input_file, prefix):
        extension = os.path.splitext(input_file.name)[1]
        return f"{prefix}{extension}"

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
    ):
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))

    def update_workflow(self, workflow, **kwargs):
        sampler = workflow["3"]["inputs"]
        sampler["seed"] = kwargs["seed"]
        sampler["sampler"] = kwargs["sampler"]
        sampler["scheduler"] = kwargs["scheduler"]

        shift = workflow["10"]["inputs"]
        shift["shift"] = kwargs["shift"]

        prompt = workflow["6"]["inputs"]
        prompt["text"] = kwargs["prompt"]

        negative_prompt = workflow["7"]["inputs"]
        negative_prompt["text"] = f"nsfw, {kwargs['negative_prompt']}"

        if kwargs["image_filename"]:
            # workflow["1"]["inputs"]["image"] = kwargs["image_filename"]
            pass

    def predict(
        self,
        prompt: str = Input(
            default="",
        ),
        negative_prompt: str = Input(
            description="Things you do not want to see in your image",
            default="",
        ),
        image: Path = Input(
            description="An input image",
            default=None,
        ),
        width: int = Input(
            description="The width of the image",
            default=1024,
            ge=512,
            le=2048,
        ),
        height: int = Input(
            description="The height of the image",
            default=1024,
            ge=512,
            le=2048,
        ),
        number_of_images: int = Input(
            description="The number of images to generate",
            default=1,
            ge=1,
            le=10,
        ),
        steps: int = Input(
            description="The number of steps to run the model for (more steps = better image but slower generation. Best results for this model are around 25 steps.)",
            default=25,
            ge=1,
            le=100,
        ),
        cfg: float = Input(
            description="The guidance scale tells the model how similar the output should be to the prompt.",
            default=3.5,
            ge=0,
            le=20,
        ),
        sampler: str = Input(
            default="uni_pc",
            choices=SAMPLERS,
        ),
        scheduler: str = Input(
            default="normal",
            choices=SCHEDULERS,
        ),
        shift: float = Input(
            description="The timestep scheduling shift; shift values higher than 1.0 are better at managing noise in higher resolutions.",
            default=1.73,
            ge=0,
            le=10,
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
        seed: int = seed_helper.predict_seed(),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)
        seed = seed_helper.generate(seed)

        image_filename = None
        if image:
            image_filename = self.filename_with_extension(image, "image")
            self.handle_input_file(image_filename)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_filename=image_filename,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler=sampler,
            scheduler=scheduler,
            shift=shift,
        )

        self.comfyUI.connect()
        self.comfyUI.run_workflow(workflow)

        return optimise_images.optimise_image_files(
            output_format, output_quality, self.comfyUI.get_files(OUTPUT_DIR)
        )
