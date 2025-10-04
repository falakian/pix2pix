from pathlib import Path
from options.test_options import TestOptions
from data import create_dataset
from models.pix2pix_model import Pix2PixModel
import torchvision.utils as vutils
import torch

if __name__ == "__main__":
    opt = TestOptions().parse()

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model: Pix2PixModel = Pix2PixModel(opt)
    model.setup(opt)

    # create a website
    results_dir = Path(opt.dataroot) / opt.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results

        output_tensors = visuals['output_generator']
        real_input = visuals['real_input']
        real_output = visuals['real_output']
        image_names = data['name']
        
        # Ensure image_names is a list, even for batch_size=1
        if isinstance(image_names, str):
            image_names = [image_names]
        

        for j in range(output_tensors.shape[0]):  # Iterate over batch
            grid = torch.cat([
                    real_input[j],
                    output_tensors[j],
                    real_output[j]
                ], dim=2)
            
            image_name = image_names[j]  # Get corresponding name
            

            # Save the image with the input name in results_dir
            output_path = results_dir / f"{image_name}.png"
            vutils.save_image(grid, output_path, normalize=True)